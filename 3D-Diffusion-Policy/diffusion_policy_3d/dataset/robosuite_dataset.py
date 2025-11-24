from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.common.normalize_util import (
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
    robomimic_abs_action_only_normalizer_from_stat
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class RobosuiteDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            shape_meta, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.shape_meta = shape_meta
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=list(shape_meta['obs'].keys()) + ['action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # data = {
        #     'action': self.replay_buffer['action'],
        #     'robot0_eef_pos': self.replay_buffer['state'][...,:],
        #     'point_cloud': self.replay_buffer['point_cloud'],
        # }
        # normalizer = LinearNormalizer()
        # normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # return normalizer
    
        normalizer = LinearNormalizer()
        # action
        stat = array_to_stats(self.replay_buffer['action'])
        normalizer['action'] = robomimic_abs_action_only_normalizer_from_stat(stat)
        # obs
        for key in self.shape_meta['obs'].keys():
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos'):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                normalizer[key] = get_range_normalizer_from_stat(stat)
            elif key.endswith('pc'):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            elif key.endswith('pc_mask'):
                normalizer[key] = get_identity_normalizer_from_stat(stat)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # sample = self.sampler.sample_sequence(idx)
        # data = self._sample_to_data(sample)
        # torch_data = dict_apply(data, torch.from_numpy)
        # return torch_data
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(2)

        obs_dict = dict()
        for key in self.shape_meta['obs'].keys():
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

