import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import logging
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

OmegaConf.register_new_resolver("eval", eval, replace=True)
logger = logging.getLogger(__name__)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
         # configure training state
        self.global_step = 0
        self.epoch = 1
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ------------------------------
        # DDP initialization
        # ------------------------------
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        self.is_distributed = (world_size > 1 and local_rank >= 0)

        # # debug prints
        # print(f"[pid {os.getpid()}] ENV LOCAL_RANK={os.environ.get('LOCAL_RANK')} RANK={os.environ.get('RANK')} "
        #       f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        #       f" device_count={torch.cuda.device_count()}")

        if self.is_distributed:
            self.local_rank = local_rank
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()
            print(f"[pid {os.getpid()}] Initialized process group: rank {dist.get_rank()}/{dist.get_world_size()}; "
                  f"device {self.device}")
        else:
            self.local_rank = 0
            if torch.cuda.is_available() and ("cuda" in str(cfg.training.device).lower()):
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(cfg.training.device)
            print(f"[pid {os.getpid()}] Non-distributed mode, device {self.device}")

        # ------------------------------
        # Configure model
        # ------------------------------
        self.model = hydra.utils.instantiate(cfg.policy)
        self.model.to(self.device)
        # Wrap model with DDP (if distributed)
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # ------------------------------
        # Configure dataset
        # ------------------------------
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        normalizer.to(self.device)
        # Use DistributedSampler if distributed
        if self.is_distributed:
            train_sampler = DistributedSampler(dataset, shuffle=True)
        else:
            train_sampler = None
        dl_kwargs = dict(cfg.dataloader)
        dl_kwargs.pop("shuffle", None)
        self.train_dataloader = DataLoader(
            dataset=dataset,
            sampler=train_sampler,
            **dl_kwargs
        )
        # Configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        self.val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        # Set normalizer
        self.model.module.set_normalizer(normalizer) if self.is_distributed else self.model.set_normalizer(normalizer)
        # ------------------------------
        # Optimizer, LR scheduler, EMA
        # ------------------------------
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        optimizer_to(self.optimizer, self.device)
        # If resuming and optimizer was created with different param groups, we will skip loading optimizer state later.
        # Ensure param groups have initial_lr for scheduler safety:
        for pg in self.optimizer.param_groups:
            if "initial_lr" not in pg:
                pg["initial_lr"] = pg.get("lr", cfg.optimizer.get("lr", 0.0))

        # Configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(self.train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # Configure ema
        self.ema_model = None
        if cfg.training.use_ema:
            model_ref = self.model.module if self.is_distributed else self.model
            self.ema_model = copy.deepcopy(model_ref)
            self.ema_model.to(self.device)
            self.ema_model.set_normalizer(normalizer)

        self.ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # Resume training (load weights only; skip optimizer state by default)
        if cfg.training.resume:
            ckpt_path = cfg.ckpt_paths
            if ckpt_path and os.path.isfile(ckpt_path):
                # Only load model weights & pickles; skip optimizer state by default to avoid mismatches
                self.load_checkpoint(path=ckpt_path, exclude_keys=("optimizer",), include_keys=None)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost

        # if self.local_rank == 0:
        #     # configure env
        #     env_runner: BaseRunner
        #     env_runner = hydra.utils.instantiate(
        #         cfg.task.env_runner,
        #         output_dir=self.output_dir)

        #     if env_runner is not None:
        #         assert isinstance(env_runner, BaseRunner)
            
        #     # configure logging
        #     cfg.logging.name = str(cfg.logging.name)
        #     cprint("-----------------------------", "yellow")
        #     cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        #     cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        #     cprint("-----------------------------", "yellow")
        #     wandb_run = wandb.init(
        #         dir=str(self.output_dir),
        #         config=OmegaConf.to_container(cfg, resolve=True),
        #         **cfg.logging
        #     )
        #     wandb.config.update(
        #         {
        #             "output_dir": self.output_dir,
        #         }
        #     )

        # # configure checkpoint
        # topk_manager = TopKCheckpointManager(
        #     save_dir=os.path.join(self.output_dir, 'checkpoints'),
        #     **cfg.checkpoint.topk
        # )

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        training_start_time = time.perf_counter()
        for local_epoch_idx in range(cfg.training.num_epochs):
            if self.is_distributed:
                try:
                    self.train_dataloader.sampler.set_epoch(local_epoch_idx)
                except Exception:
                    pass
            step_log = dict()
            train_losses = list()
            with tqdm.tqdm(self.train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                
                model_ref = self.model.module if self.is_distributed else self.model
                model_ref.train()

                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = model_ref.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        self.ema.step(model_ref)
                    t1_4 = time.time()

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': self.lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    logger.info(loss_dict)
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose and self.local_rank == 0:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(self.train_dataloader)-1))
                    if not is_last_batch:
                        # if self.local_rank == 0:
                        #     # log of last step is combined with validation and rollout
                        #     wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            if self.local_rank == 0:
                # policy = self.ema_model if cfg.training.use_ema else model_ref
                # policy.eval()

                # # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                #     t3 = time.time()
                #     # runner_log = env_runner.run(policy, dataset=dataset)
                #     runner_log = env_runner.run(policy)
                #     t4 = time.time()
                #     # print(f"rollout time: {t4-t3:.3f}")
                #     # log all
                #     step_log.update(runner_log)

                # # run validation
                # if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(self.val_dataloader, desc=f"Validation epoch {self.epoch}", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                #                 loss, loss_dict = self.model.compute_loss(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss

                # # run diffusion sampling on a training batch
                # if (self.epoch % cfg.training.sample_every) == 0:
                #     with torch.no_grad():
                #         # sample trajectory from training set, and evaluate difference
                #         batch = dict_apply(train_sampling_batch, lambda x: x.to(self.device, non_blocking=True))
                #         obs_dict = batch['obs']
                #         gt_action = batch['action']
                        
                #         result = policy.predict_action(obs_dict)
                #         pred_action = result['action_pred']
                #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                #         step_log['train_action_mse_error'] = mse.item()
                #         del batch
                #         del obs_dict
                #         del gt_action
                #         del result
                #         del pred_action
                #         del mse

                # if env_runner is None:
                #     step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # # We can't copy the last checkpoint here
                    # # since save_checkpoint uses threads.
                    # # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)

                    # Upload checkpoints to HuggingFace Hub
                    from huggingface_hub import HfApi
                    api = HfApi()
                    repo_id = cfg.hf_repo_id
                    ckpt_path = self.get_checkpoint_path()
                    api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=f"{cfg.task_name}/epoch={self.epoch}_seed={cfg.training.seed}.pth",
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message="Add checkpoints"
                    )

                # ========= eval end for this epoch ==========
                # policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                # wandb_run.log(step_log, step=self.global_step)

            self.global_step += 1
            self.epoch += 1
            del step_log
        
        logger.info(f"Training time: {(time.perf_counter() - training_start_time) / 3600}h")
        if self.is_distributed:    
            dist.destroy_process_group()
 
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if isinstance(value, DDP):
                        value = value.module
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        # Load state_dicts (skip optimizer state if incompatible or explicitly excluded)
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                # skip loading optimizer state by default (it often mismatches across runs)
                if isinstance(self.__dict__.get(key, None), torch.optim.Optimizer):
                    print(f"[load_payload] Skipping optimizer state: {key}")
                    continue
                # if the target object is DDP-wrapped, load into .module
                target = self.__dict__.get(key, None)
                if isinstance(target, DDP):
                    target = target.module
                if target is None:
                    print(f"[load_payload] Warning: target {key} not found in workspace; skipping")
                    continue
                try:
                    target.load_state_dict(value, **kwargs)
                except Exception as e:
                    print(f"[load_payload] Error loading {key}: {e}; skipping.")
                    continue

        # Load pickles
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()



# import os
# import hydra
# import torch
# import dill
# from omegaconf import OmegaConf
# import pathlib
# from torch.utils.data import DataLoader
# import copy
# import random
# import wandb
# import tqdm
# import numpy as np
# from termcolor import cprint
# import shutil
# import time
# import threading
# from hydra.core.hydra_config import HydraConfig
# from diffusion_policy_3d.policy.dp3 import DP3
# from diffusion_policy_3d.dataset.base_dataset import BaseDataset
# from diffusion_policy_3d.env_runner.base_runner import BaseRunner
# from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
# from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
# from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
# from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP

# OmegaConf.register_new_resolver("eval", eval, replace=True)


# class TrainDP3Workspace:
#     include_keys = ['global_step', 'epoch']
#     exclude_keys = tuple()

#     def __init__(self, cfg: OmegaConf, output_dir=None):
#         self.cfg = cfg
#         self._output_dir = output_dir
#         self._saving_thread = None
#          # configure training state
#         self.global_step = 0
#         self.epoch = 1
        
#         # set seed
#         seed = cfg.training.seed
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)

#         # ------------------------------
#         # DDP initialization
#         # ------------------------------
#         if "LOCAL_RANK" in os.environ:
#             dist.init_process_group(backend="nccl")
#             self.local_rank = int(os.environ["LOCAL_RANK"])
#             torch.cuda.set_device(self.local_rank)
#             self.device = torch.device(f"cuda:{self.local_rank}")
#             self.is_distributed = True
#         else:
#             self.local_rank = 0
#             self.device = torch.device(cfg.training.device)
#             self.is_distributed = False

#         # configure model
#         self.model: DP3 = hydra.utils.instantiate(cfg.policy)
#         self.model.to(self.device)

#         # Wrap model with DDP
#         if self.is_distributed:
#             self.model = DDP(
#                 self.model,
#                 device_ids=[self.local_rank],
#                 output_device=self.local_rank,
#                 find_unused_parameters=True
#             )

#         # configure dataset
#         dataset: BaseDataset
#         dataset = hydra.utils.instantiate(cfg.task.dataset)
#         assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
#         normalizer = dataset.get_normalizer()
#         normalizer.to(self.device)

#         # Use DistributedSampler if distributed
#         if self.is_distributed:
#             train_sampler = DistributedSampler(dataset)
#         else:
#             train_sampler = None

#         cfg.dataloader.shuffle = (train_sampler is None)
#         self.train_dataloader = DataLoader(
#             dataset=dataset,
#             sampler=train_sampler,
#             **cfg.dataloader
#         )

#         # configure validation dataset
#         val_dataset = dataset.get_validation_dataset()
#         self.val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

#         self.model.module.set_normalizer(normalizer) if self.is_distributed else self.model.set_normalizer(normalizer)
#         # ------------------------------
#         # Optimizer, LR scheduler, EMA
#         # ------------------------------
#         self.optimizer = hydra.utils.instantiate(
#             cfg.optimizer, params=self.model.parameters())
#         optimizer_to(self.optimizer, self.device)

#         # configure lr scheduler
#         self.lr_scheduler = get_scheduler(
#             cfg.training.lr_scheduler,
#             optimizer=self.optimizer,
#             num_warmup_steps=cfg.training.lr_warmup_steps,
#             num_training_steps=(
#                 len(self.train_dataloader) * cfg.training.num_epochs) \
#                     // cfg.training.gradient_accumulate_every,
#             # pytorch assumes stepping LRScheduler every epoch
#             # however huggingface diffusers steps it every batch
#             last_epoch=self.global_step-1
#         )

#         # configure ema
#         self.ema_model: DP3 = None
#         if cfg.training.use_ema:
#             model_ref = self.model.module if self.is_distributed else self.model
#             self.ema_model = copy.deepcopy(model_ref)
#             self.ema_model.to(self.device)
#             self.ema_model.set_normalizer(normalizer)

#         self.ema: EMAModel = None
#         if cfg.training.use_ema:
#             self.ema = hydra.utils.instantiate(
#                 cfg.ema,
#                 model=self.ema_model
#             )

#         # resume training
#         if cfg.training.resume:
#             ckpt_path = cfg.ckpt_paths
#             if os.path.isfile(ckpt_path):
#                 self.load_checkpoint(path=ckpt_path)

#     def run(self):
#         cfg = copy.deepcopy(self.cfg)
        
#         if cfg.training.debug:
#             cfg.training.num_epochs = 100
#             cfg.training.max_train_steps = 10
#             cfg.training.max_val_steps = 3
#             cfg.training.rollout_every = 20
#             cfg.training.checkpoint_every = 1
#             cfg.training.val_every = 1
#             cfg.training.sample_every = 1
#             RUN_ROLLOUT = True
#             RUN_CKPT = False
#             verbose = True
#         else:
#             RUN_ROLLOUT = True
#             RUN_CKPT = True
#             verbose = False
        
#         RUN_VALIDATION = False # reduce time cost

#         # if self.local_rank == 0:
#         #     # configure env
#         #     env_runner: BaseRunner
#         #     env_runner = hydra.utils.instantiate(
#         #         cfg.task.env_runner,
#         #         output_dir=self.output_dir)

#         #     if env_runner is not None:
#         #         assert isinstance(env_runner, BaseRunner)
            
#         #     # configure logging
#         #     cfg.logging.name = str(cfg.logging.name)
#         #     cprint("-----------------------------", "yellow")
#         #     cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
#         #     cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
#         #     cprint("-----------------------------", "yellow")
#         #     wandb_run = wandb.init(
#         #         dir=str(self.output_dir),
#         #         config=OmegaConf.to_container(cfg, resolve=True),
#         #         **cfg.logging
#         #     )
#         #     wandb.config.update(
#         #         {
#         #             "output_dir": self.output_dir,
#         #         }
#         #     )

#         # # configure checkpoint
#         # topk_manager = TopKCheckpointManager(
#         #     save_dir=os.path.join(self.output_dir, 'checkpoints'),
#         #     **cfg.checkpoint.topk
#         # )

#         # save batch for sampling
#         train_sampling_batch = None

#         # training loop
#         log_path = os.path.join(self.output_dir, 'logs.json.txt')
#         for local_epoch_idx in range(self.epoch - 1, cfg.training.num_epochs):
#             step_log = dict()
#             # ========= train for this epoch ==========
#             train_losses = list()
#             with tqdm.tqdm(self.train_dataloader, desc=f"Training epoch {self.epoch}", 
#                     leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                
#                 model_ref = self.model.module if self.is_distributed else self.model
#                 model_ref.train()

#                 for batch_idx, batch in enumerate(tepoch):
#                     t1 = time.time()
#                     # device transfer
#                     batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
#                     if train_sampling_batch is None:
#                         train_sampling_batch = batch
                
#                     # compute loss
#                     t1_1 = time.time()
#                     raw_loss, loss_dict = model_ref.compute_loss(batch)
#                     loss = raw_loss / cfg.training.gradient_accumulate_every
#                     loss.backward()
                    
#                     t1_2 = time.time()

#                     # step optimizer
#                     if self.global_step % cfg.training.gradient_accumulate_every == 0:
#                         self.optimizer.step()
#                         self.optimizer.zero_grad()
#                         self.lr_scheduler.step()
#                     t1_3 = time.time()
#                     # update ema
#                     if cfg.training.use_ema:
#                         self.ema.step(model_ref)
#                     t1_4 = time.time()

#                     # logging
#                     raw_loss_cpu = raw_loss.item()
#                     tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
#                     train_losses.append(raw_loss_cpu)
#                     step_log = {
#                         'train_loss': raw_loss_cpu,
#                         'global_step': self.global_step,
#                         'epoch': self.epoch,
#                         'lr': self.lr_scheduler.get_last_lr()[0]
#                     }
#                     t1_5 = time.time()
#                     step_log.update(loss_dict)
#                     t2 = time.time()
                    
#                     if verbose and self.local_rank == 0:
#                         print(f"total one step time: {t2-t1:.3f}")
#                         print(f" compute loss time: {t1_2-t1_1:.3f}")
#                         print(f" step optimizer time: {t1_3-t1_2:.3f}")
#                         print(f" update ema time: {t1_4-t1_3:.3f}")
#                         print(f" logging time: {t1_5-t1_4:.3f}")

#                     is_last_batch = (batch_idx == (len(self.train_dataloader)-1))
#                     if not is_last_batch:
#                         # if self.local_rank == 0:
#                         #     # log of last step is combined with validation and rollout
#                         #     wandb_run.log(step_log, step=self.global_step)
#                         self.global_step += 1

#                     if (cfg.training.max_train_steps is not None) \
#                         and batch_idx >= (cfg.training.max_train_steps-1):
#                         break

#             # at the end of each epoch
#             # replace train_loss with epoch average
#             train_loss = np.mean(train_losses)
#             step_log['train_loss'] = train_loss

#             # ========= eval for this epoch ==========
#             if self.local_rank == 0:
#                 policy = self.ema_model if cfg.training.use_ema else model_ref
#                 policy.eval()
#                 # # run rollout
#                 # if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
#                 #     t3 = time.time()
#                 #     # runner_log = env_runner.run(policy, dataset=dataset)
#                 #     runner_log = env_runner.run(policy)
#                 #     t4 = time.time()
#                 #     # print(f"rollout time: {t4-t3:.3f}")
#                 #     # log all
#                 #     step_log.update(runner_log)

#                 # run validation
#                 if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
#                     with torch.no_grad():
#                         val_losses = list()
#                         with tqdm.tqdm(self.val_dataloader, desc=f"Validation epoch {self.epoch}", 
#                                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
#                             for batch_idx, batch in enumerate(tepoch):
#                                 batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
#                                 loss, loss_dict = self.model.compute_loss(batch)
#                                 val_losses.append(loss)
#                                 if (cfg.training.max_val_steps is not None) \
#                                     and batch_idx >= (cfg.training.max_val_steps-1):
#                                     break
#                         if len(val_losses) > 0:
#                             val_loss = torch.mean(torch.tensor(val_losses)).item()
#                             # log epoch average validation loss
#                             step_log['val_loss'] = val_loss

#                 # run diffusion sampling on a training batch
#                 if (self.epoch % cfg.training.sample_every) == 0:
#                     with torch.no_grad():
#                         # sample trajectory from training set, and evaluate difference
#                         batch = dict_apply(train_sampling_batch, lambda x: x.to(self.device, non_blocking=True))
#                         obs_dict = batch['obs']
#                         gt_action = batch['action']
                        
#                         result = policy.predict_action(obs_dict)
#                         pred_action = result['action_pred']
#                         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
#                         step_log['train_action_mse_error'] = mse.item()
#                         del batch
#                         del obs_dict
#                         del gt_action
#                         del result
#                         del pred_action
#                         del mse

#                 # if env_runner is None:
#                 #     step_log['test_mean_score'] = - train_loss
                    
#                 # checkpoint
#                 if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
#                     # checkpointing
#                     if cfg.checkpoint.save_last_ckpt:
#                         self.save_checkpoint()
#                     if cfg.checkpoint.save_last_snapshot:
#                         self.save_snapshot()

#                     # sanitize metric names
#                     metric_dict = dict()
#                     for key, value in step_log.items():
#                         new_key = key.replace('/', '_')
#                         metric_dict[new_key] = value
                    
#                     # # We can't copy the last checkpoint here
#                     # # since save_checkpoint uses threads.
#                     # # therefore at this point the file might have been empty!
#                     # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

#                     # if topk_ckpt_path is not None:
#                     #     self.save_checkpoint(path=topk_ckpt_path)

#                     # Upload to HuggingFace Hub
#                     from huggingface_hub import HfApi
#                     api = HfApi()
#                     repo_id = "HenryWJL/dp3"
#                     ckpt_path = self.get_checkpoint_path()
#                     api.upload_file(
#                         path_or_fileobj=ckpt_path,
#                         path_in_repo=f"{cfg.task_name}/{self.epoch}.pth",
#                         repo_id=repo_id,
#                         repo_type="model",
#                         commit_message="Add checkpoints"
#                     )
#                 # ========= eval end for this epoch ==========
#                 policy.train()

#                 # end of epoch
#                 # log of last step is combined with validation and rollout
#                 # wandb_run.log(step_log, step=self.global_step)

#             self.global_step += 1
#             self.epoch += 1
#             del step_log

#         if self.is_distributed:    
#             dist.destroy_process_group()
        
#     @property
#     def output_dir(self):
#         output_dir = self._output_dir
#         if output_dir is None:
#             output_dir = HydraConfig.get().runtime.output_dir
#         return output_dir
    

#     def save_checkpoint(self, path=None, tag='latest', 
#             exclude_keys=None,
#             include_keys=None,
#             use_thread=False):
#         if path is None:
#             path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
#         else:
#             path = pathlib.Path(path)
#         if exclude_keys is None:
#             exclude_keys = tuple(self.exclude_keys)
#         if include_keys is None:
#             include_keys = tuple(self.include_keys) + ('_output_dir',)

#         path.parent.mkdir(parents=False, exist_ok=True)
#         payload = {
#             'cfg': self.cfg,
#             'state_dicts': dict(),
#             'pickles': dict()
#         } 

#         for key, value in self.__dict__.items():
#             if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
#                 # modules, optimizers and samplers etc
#                 if key not in exclude_keys:
#                     if isinstance(value, DDP):
#                         value = value.module
#                     if use_thread:
#                         payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
#                     else:
#                         payload['state_dicts'][key] = value.state_dict()
#             elif key in include_keys:
#                 payload['pickles'][key] = dill.dumps(value)
#         if use_thread:
#             self._saving_thread = threading.Thread(
#                 target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
#             self._saving_thread.start()
#         else:
#             torch.save(payload, path.open('wb'), pickle_module=dill)
        
#         del payload
#         torch.cuda.empty_cache()
#         return str(path.absolute())
    
#     def get_checkpoint_path(self, tag='latest'):
#         if tag=='latest':
#             return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
#         elif tag=='best': 
#             # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
#             # find the best checkpoint
#             checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
#             all_checkpoints = os.listdir(checkpoint_dir)
#             best_ckpt = None
#             best_score = -1e10
#             for ckpt in all_checkpoints:
#                 if 'latest' in ckpt:
#                     continue
#                 score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
#                 if score > best_score:
#                     best_ckpt = ckpt
#                     best_score = score
#             return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
#         else:
#             raise NotImplementedError(f"tag {tag} not implemented")
            
            

#     def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
#         if exclude_keys is None:
#             exclude_keys = tuple()
#         if include_keys is None:
#             include_keys = payload['pickles'].keys()

#         for key, value in payload['state_dicts'].items():
#             if key not in exclude_keys:
#                 self.__dict__[key].load_state_dict(value, **kwargs)
#         for key in include_keys:
#             if key in payload['pickles']:
#                 self.__dict__[key] = dill.loads(payload['pickles'][key])
    
#     def load_checkpoint(self, path=None, tag='latest',
#             exclude_keys=None, 
#             include_keys=None, 
#             **kwargs):
#         if path is None:
#             path = self.get_checkpoint_path(tag=tag)
#         else:
#             path = pathlib.Path(path)
#         payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
#         self.load_payload(payload, 
#             exclude_keys=exclude_keys, 
#             include_keys=include_keys)
#         return payload
    
#     @classmethod
#     def create_from_checkpoint(cls, path, 
#             exclude_keys=None, 
#             include_keys=None,
#             **kwargs):
#         payload = torch.load(open(path, 'rb'), pickle_module=dill)
#         instance = cls(payload['cfg'])
#         instance.load_payload(
#             payload=payload, 
#             exclude_keys=exclude_keys,
#             include_keys=include_keys,
#             **kwargs)
#         return instance

#     def save_snapshot(self, tag='latest'):
#         """
#         Quick loading and saving for reserach, saves full state of the workspace.

#         However, loading a snapshot assumes the code stays exactly the same.
#         Use save_checkpoint for long-term storage.
#         """
#         path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
#         path.parent.mkdir(parents=False, exist_ok=True)
#         torch.save(self, path.open('wb'), pickle_module=dill)
#         return str(path.absolute())
    
#     @classmethod
#     def create_from_snapshot(cls, path):
#         return torch.load(open(path, 'rb'), pickle_module=dill)


# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.joinpath(
#         'diffusion_policy_3d', 'config'))
# )
# def main(cfg):
#     workspace = TrainDP3Workspace(cfg)
#     workspace.run()

# if __name__ == "__main__":
#     main()
