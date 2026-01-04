import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import torch.nn.functional as F
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
from einops import rearrange, reduce
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.policy.dp3 import DP3
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
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)
        self.model.to(self.device)
        ckpt_path = cfg.ckpt_paths
        if ckpt_path and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=False)['state_dicts']['ema_model'])
            cprint(f"Load pretrained checkpoint {ckpt_path}", color='red')
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.set_normalizer(self.model.normalizer)
        # freeze
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        # Wrap student model with DDP (if distributed)
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
        # ------------------------------
        # Optimizer, EMA
        # ------------------------------
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=5e-5,
            weight_decay=1e-6,
            betas=(0.95, 0.999)
        )
        optimizer_to(self.optimizer, self.device)
        # Configure ema
        self.ema_model = None
        if cfg.training.use_ema:
            model_ref = self.model.module if self.is_distributed else self.model
            self.ema_model = copy.deepcopy(model_ref)
            self.ema_model.to(self.device)
            self.ema_model.set_normalizer(model_ref.normalizer)

        self.ema: EMAModel = None
        if cfg.training.use_ema:
            self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        # training loop
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
                teacher_model = self.teacher_model

                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    # sample timestep
                    batch_size = batch['action'].shape[0]
                    timestep = torch.randint(
                        0, self.teacher_model.noise_scheduler.config.num_train_timesteps, 
                        (batch_size,), device=self.device
                    ).long()
                    # perturb actions
                    batch_perturbed = copy.deepcopy(batch)
                    alpha_bar = teacher_model.noise_scheduler.alphas_cumprod.to(self.device)[timestep]
                    sigma = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
                    batch_perturbed['action'] += sigma * torch.randn_like(batch_perturbed['action'])
                    # predict means
                    mu_student, _, loss_mask = model_ref(batch_perturbed, timestep)
                    with torch.no_grad():
                        mu_teacher, actions, _ = teacher_model(batch, timestep)
                    weight = (actions - mu_teacher).abs().mean(dim=(1, 2), keepdim=True)
                    grad = (mu_student - mu_teacher) / (weight + 1e-8)
                    target = (actions - grad).detach()
                    loss = 0.5 * F.mse_loss(actions, target, reduction='none')
                    loss = loss * loss_mask.type(loss.dtype)
                    loss = reduce(loss, 'b ... -> b (...)', 'mean')
                    loss = loss.mean()
                    loss /= cfg.training.gradient_accumulate_every
                    loss.backward()
                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        # update ema
                        if cfg.training.use_ema:
                            self.ema.step(model_ref)    

                    train_losses.append(loss.item())

                    is_last_batch = (batch_idx == (len(self.train_dataloader)-1))
                    if not is_last_batch:
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            if self.local_rank == 0:    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(exclude_keys=['teacher_model'])
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
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