from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import torch_dct
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1DwIndex
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder


def sample_index(k_min, k_max, batch_size, device, prob=0.2):
    mask = torch.rand(batch_size, device=device) < prob
    u = torch.rand(batch_size, device=device)
    k = k_min + (k_max - k_min) * u ** (1 / 2)
    k = k.long()
    # With a probability @prob, k = k_min
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
    indices = indices.view(B, 1, 1).to(device)
    freq_indices = torch.arange(H, device=device).view(1, 1, H)
    dct_mask = (freq_indices < indices).float()
    dct_mask = dct_mask.expand(B, D, H)
    masked_coeffs = dct_coeffs * dct_mask
    # Inverse DCT
    traj_recons = torch_dct.idct(masked_coeffs, norm="ortho")
    traj_recons = traj_recons.transpose(1, 2).to(dtype)
    return traj_recons


# def processingpregt_dct(trajectory, prob=0.2, k0_ratio=0.1):
#     """
#     CFG-style DCT masking with low-k baseline.

#     Args:
#         trajectory: [B, H, D] input action trajectory
#         prob: probability of using unconditional (low-k) baseline
#         k0_ratio: baseline frequency ratio (e.g. 0.2)
#     Returns:
#         out: [B, H, D] processed trajectory
#         core_index: [B] used frequency cutoff indices
#     """
#     B, H, D = trajectory.shape
#     device = trajectory.device

#     # ---- 1. Define low-k baseline ----
#     k0 = max(1, int(k0_ratio * H))  # avoid degenerate zero case

#     # ---- 2. Sample conditional k in [k0, H] ----
#     sampled_k = torch.randint(k0, H + 1, (B,), device=device)

#     # ---- 3. CFG-style dropout mask ----
#     # True → unconditional → use k0
#     cfg_mask = torch.rand(B, device=device) < prob
#     core_index = torch.where(
#         cfg_mask,
#         torch.full_like(sampled_k, k0),
#         sampled_k
#     ).float()

#     # ---- 4. DCT transform ----
#     traj_reshaped = trajectory.transpose(1, 2).to(torch.float64)
#     dct_coeffs = torch_dct.dct(traj_reshaped, norm="ortho")

#     # ---- 5. Frequency masking ----
#     freq_indices = torch.arange(H, device=device).view(1, 1, H)
#     core_thresholds = core_index.view(B, 1, 1)
#     dct_mask = (freq_indices < core_thresholds).float()
#     dct_mask = dct_mask.expand(B, D, H)
#     masked_coeffs = dct_coeffs * dct_mask

#     # ---- 6. Inverse DCT ----
#     idct_result = torch_dct.idct(masked_coeffs, norm="ortho")
#     out = idct_result.transpose(1, 2).to(trajectory.dtype)

#     return out, core_index


# def dct_reconstruct(trajectory, index):
#     """
#     trajectory: [B, H, D]
#     """
#     index = int(index)
#     H = trajectory.shape[1]
#     dtype = trajectory.dtype
#     device = trajectory.device
#     dct_mask = torch.zeros((H,), dtype=torch.float32, device=device)
#     dct_mask[:index] = 1.0
#     dct_mask = dct_mask.view(1, 1, H).to('cpu')
#     traj_reshaped = trajectory.transpose(1, 2).to('cpu').to(torch.float64)
#     dct_coeffs = torch_dct.dct(traj_reshaped, norm="ortho")
#     masked_coeffs = dct_coeffs * dct_mask
#     idct_result = torch_dct.idct(masked_coeffs, norm="ortho")
#     out = idct_result.transpose(1, 2).to(dtype).to(device)
#     return out


# def k_schedule(t, T, k0, k_max, power=2.0):
#     """
#     t: current diffusion step
#     """
#     frac = (1 - t / T) ** power
#     return torch.round(k0 + frac * (k_max - k0))

def k_schedule(t, T, k0, k_max, beta=9.0):
    """
    Fast increase early, slow late.
    beta controls aggressiveness.
    """
    s = 1.0 - t / T
    s = s.clamp(0.0, 1.0)
    frac = torch.log1p(beta * s) / torch.log1p(torch.tensor(beta, device=t.device))
    k = k0 + frac * (k_max - k0)
    return torch.round(k)

# def k_schedule(t, T, k0, k_max, lamb=4.0):
#     s = 1.0 - t / T
#     s = s.clamp(0.0, 1.0)
#     k = k_max - (k_max - k0) * torch.exp(-lamb * s)
#     return torch.round(k)

def delta_k_schedule(t, T, delta_max=4, delta_min=2):
    return torch.round(delta_max * (t / T) + delta_min)

def alpha_schedule(t, T, power=1.0):
    """
    Controls how strongly refinement is applied
    """
    return (1 - t / T) ** power

def alpha_from_k(ks, kl, k0, k_max, gamma=2.0):
    frac = (kl - ks) / (k_max - k0)
    alpha = frac ** gamma
    return alpha


class FGDP(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            index_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            prob=0.2,
            k0_ratio=0.1,
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



        model = ConditionalUnet1DwIndex(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            index_embed_dim=index_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
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

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

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
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        H = condition_data.shape[1]
        k0 = max(1, int(H * self.k0_ratio))
        k_max = H
        T = scheduler.timesteps.max()

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # kt = k_schedule(t, T, k0, k_max)
            # alpha_t = alpha_schedule(t, T)

            # trajectory_k0 = dct_reconstruct(trajectory, k0)
            # trajectory_kt = dct_reconstruct(trajectory, kt)

            # pred_k0 = model(sample=trajectory_k0,
            #                     timestep=t,
            #                     index=k0, 
            #                     local_cond=local_cond, global_cond=global_cond)
            # pred_kt = model(sample=trajectory_kt,
            #                     timestep=t,
            #                     index=kt, 
            #                     local_cond=local_cond, global_cond=global_cond)
            # pred = (1 - alpha_t) * pred_k0 + alpha_t * pred_kt

            kc = k_schedule(t, T, k0, k_max)
            k_delta = delta_k_schedule(t, T)
            ks = torch.clamp(kc - k_delta / 2, k0, k_max)
            kl = torch.clamp(kc + k_delta / 2, k0, k_max)
            alpha_t = alpha_from_k(ks, kl, k0, k_max)

            trajectory_ks = dct_reconstruct(trajectory, ks)
            trajectory_kl = dct_reconstruct(trajectory, kl)

            pred_ks = model(sample=trajectory_ks,
                                timestep=t,
                                index=ks, 
                                local_cond=local_cond, global_cond=global_cond)
            pred_kl = model(sample=trajectory_kl,
                                timestep=t,
                                index=kl, 
                                local_cond=local_cond, global_cond=global_cond)
            pred = (1 - alpha_t) * pred_ks + alpha_t * pred_kl

            # k = k_schedule(t, T, k0, k_max)
            # trajectory = dct_reconstruct(trajectory, k)
            # pred = model(sample=trajectory,
            #                     timestep=t,
            #                     index=k, 
            #                     local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                pred, t, trajectory, ).prev_sample
                
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

        # Sample a random timestep for each action
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()

        # Sample a reconstruction index
        k_min = int(self.k0_ratio * horizon)
        k_max = k_min + (horizon - k_min) * torch.sqrt(1 - timesteps / self.noise_scheduler.config.num_train_timesteps)
        k_max = torch.round(k_max)
        indices = sample_index(k_min, k_max, batch_size, trajectory.device, self.prob)

        # Reconstruct the trajectory
        trajectory = dct_reconstruct(trajectory, indices)

        # Sample noise that we'll add to the actions
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)

        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                        index=indices,
                        local_cond=local_cond, 
                        global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        loss_dict = {
                'bc_loss': loss.item(),
            }
        
        return loss, loss_dict