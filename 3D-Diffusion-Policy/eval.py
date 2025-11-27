import os
import copy
import hydra
import torch
import dill
import pathlib
from omegaconf import OmegaConf
from termcolor import cprint
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.env_runner.base_runner import BaseRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)


class EvalDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # configure model
        self.model: DP3 = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.global_step = 0
        self.epoch = 1

    def eval(self, ckpt_paths):
        cfg = copy.deepcopy(self.cfg)
        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)

        for ckpt_path in ckpt_paths:
            cprint(f"Evaluating checkpoint {ckpt_path}", 'yellow')
            self.load_checkpoint(path=ckpt_path)
            policy = copy.deepcopy(self.model)
            if cfg.training.use_ema:
                policy = copy.deepcopy(self.ema_model)
            policy.eval()
            policy.to(torch.device(cfg.training.device))

            runner_log = env_runner.run(policy)
            
            cprint(f"---------------- Eval Results --------------", 'magenta')
            for key, value in runner_log.items():
                if isinstance(value, float):
                    cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    
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

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                if key == 'model':  # DDP will prepend 'module.' to all keys, now we remove it
                    value = {k[len("module."):]: v for k, v in value.items() if k.startswith("module.")}
                if key in  self.__dict__:
                    self.__dict__[key].load_state_dict(value, **kwargs)
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
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = EvalDP3Workspace(cfg, "outputs")
    workspace.eval(cfg.ckpt_paths)

if __name__ == "__main__":
    main()
    # Example command:
    # python 3D-Diffusion-Policy/eval.py --config-name=dp3.yaml task=robosuite_square training.device="cpu" ckpt_paths=checkpoints/square_3000.pth
