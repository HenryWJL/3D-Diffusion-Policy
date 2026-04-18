from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from termcolor import cprint
import copy
import time
import torch_dct
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.transformer import TransformerNoisePredictionNetwIndex
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder


def sample_index(k_min, k_max, batch_size, device, prob=0.2, method="uniform"):
    if method == "uniform":
        u = torch.rand(batch_size, device=device)
        # k = k_min + torch.floor((k_max - k_min + 1) * u).long()
        k = k_min + (k_max - k_min) * u
        k = torch.round(k)
    elif method == "skew":
        # u = torch.rand(batch_size, device=device)
        # k = k_min + (k_max - k_min) * u ** 0.5
        # k = k.long()
        u = torch.rand(batch_size, device=device)
        k = k_min + torch.floor((k_max - k_min + 1) * u ** 0.5).long()
    else:
        raise ValueError(f"Unsupported method {method}")
    # With a probability @prob, k = k_min
    mask = torch.rand(batch_size, device=device) < prob
    k = torch.where(
        mask,
        torch.full_like(k, k_min),
        k
    )
    return k


def dct_reconstruct(trajectory, indices):
    B, H, D = trajectory.shape
    dtype = trajectory.dtype
    device = trajectory.device
    # DCT
    traj_reshaped = trajectory.transpose(1, 2).to(torch.float64)
    dct_coeffs = torch_dct.dct(traj_reshaped, norm="ortho")
    # Masking
    if not torch.is_tensor(indices):
        indices = torch.tensor([indices], dtype=torch.long)
    elif torch.is_tensor(indices) and len(indices.shape) == 0:
        indices = indices[None]
    indices = indices.expand(B).view(B, 1, 1).to(device)
    freq_indices = torch.arange(H, device=device).view(1, 1, H)
    dct_mask = (freq_indices < indices).float()
    dct_mask = dct_mask.expand(B, D, H)
    masked_coeffs = dct_coeffs * dct_mask
    # Inverse DCT
    traj_recons = torch_dct.idct(masked_coeffs, norm="ortho")
    traj_recons = traj_recons.transpose(1, 2).to(dtype)
    return traj_recons


def k_schedule(t, T, k0, k_max, power=1.0):
    """
    t: current diffusion step
    """
    frac = (1 - t / T) ** power
    return torch.round(k0 + frac * (k_max - k0))


def alpha_schedule(t, T, alpha_min=0.0, alpha_max=1.0, power=1.0):
    """
    Controls how strongly refinement is applied
    """
    # frac = (1 - t / T) ** power
    # return alpha_min + frac * (alpha_max - alpha_min)
    return (1 - t / T) ** power


class FGOFlowpolicy(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_train_steps,
            num_inference_steps=None,
            timeshift=1.0,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            index_embed_dim=256,
            condition_type="film",
            encoder_output_dim=256,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            prob=0.2,
            k0_ratio=0.1,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        obs_encoder = DP3Encoder(observation_space=obs_dict,
                                                   img_crop_shape=crop_shape,
                                                out_channel=encoder_output_dim,
                                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                                use_pc_color=use_pc_color,
                                                pointnet_type=pointnet_type,
                                                )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = TransformerNoisePredictionNetwIndex(
            input_len=horizon,
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            timestep_embed_dim=diffusion_step_embed_dim,
            index_embed_dim=index_embed_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.prob = prob
        self.k0_ratio = k0_ratio
        self.kwargs = kwargs

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_train_steps if num_inference_steps is None else num_inference_steps
        timesteps = torch.linspace(1, 0, self.num_inference_steps + 1)
        self.timesteps = (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps)

        print_params(self)

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)
        
        H = condition_data.shape[1]
        k0 = max(1, int(H * self.k0_ratio))
        k_max = H
        T = self.timesteps.max()

        for tcont, tcont_next in zip(self.timesteps[:-1], self.timesteps[1:]):
            kt = k_schedule(tcont_next, T, k0, k_max)
            alpha_t = alpha_schedule(tcont_next, T)
            t = (tcont * self.num_train_steps).long()

            trajectory_k0 = dct_reconstruct(trajectory, k0)
            trajectory_kt = dct_reconstruct(trajectory, kt)

            pred_k0 = self.model(sample=trajectory_k0,
                                timestep=t,
                                index=k0, 
                                global_cond=global_cond)
            pred_kt = self.model(sample=trajectory_kt,
                                timestep=t,
                                index=kt, 
                                global_cond=global_cond)
            noise_pred = (1 - alpha_t) * pred_k0 + alpha_t * pred_kt
            trajectory = trajectory + (tcont_next - tcont) * noise_pred
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        for key, val in nobs.items():
            nobs[key] = val.float()
        nactions = self.normalizer['action'].normalize(batch['action'])
        nactions = nactions.float()
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
         
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn_like(trajectory)

        bsz = trajectory.shape[0]
        # Sample random timestep
        tcont = torch.rand((bsz,), device=trajectory.device)
        t = (tcont * self.num_train_steps).long()

        # Sample a reconstruction index
        k_min = int(self.k0_ratio * horizon)
        # k_max = horizon

        # # linear
        # k_max = k_min + (horizon - k_min) * (1 - timesteps / self.noise_scheduler.config.num_train_timesteps)
        # k_max = torch.round(k_max)
        # quadratic
        k_max = k_min + (horizon - k_min) * torch.sqrt(1 - t / self.num_train_steps)
        k_max = torch.round(k_max)
        # # cosine
        # s = 1 - timesteps / self.noise_scheduler.config.num_train_timesteps
        # k_max = k_min + (horizon - k_min) * torch.sin(math.pi / 2 * s)
        # k_max = torch.round(k_max)
        indices = sample_index(k_min, k_max, batch_size, trajectory.device, self.prob)

        # Reconstruct the trajectory
        trajectory = dct_reconstruct(trajectory, indices)

        # Forward flow step
        direction = noise - trajectory
        noisy_trajectory = (
            trajectory + tcont.view(-1, *[1 for _ in range(trajectory.dim() - 1)]) * direction
        )

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        noise_pred = self.model(noisy_trajectory, t, indices, global_cond)

        # Flow matching loss
        loss = F.mse_loss(noise_pred, direction)
        
        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict