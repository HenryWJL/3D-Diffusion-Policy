import torch
import numpy as np
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.policy.base_policy import BasePolicy


def svgd_kernel(x, h = -1):
    """
    Computes the RBF kernel and its gradient w.r.t. the first input.
    
    Args:
        x: Tensor of shape [N, D] (particles)
        h: Bandwidth parameter. If < 0, uses the median heuristic.
        
    Returns:
        Kxy: Kernel matrix [N, N]
        dxkxy: Gradient of Kernel [N, D] -> sum_y grad_x k(x, y)
    """
    # 1. Compute Pairwise Squared Distances
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    pairwise_dist = torch.sum(diff**2, dim=-1)  # [N, N]
    # 2. Median Trick for Bandwidth (h)
    if h < 0:
        h = torch.median(pairwise_dist.view(-1)) 
        h = torch.sqrt(0.5 * h / torch.log(torch.tensor(x.shape[0] + 1.0, device=x.device)))
    # 3. Compute RBF Kernel
    Kxy = torch.exp(-pairwise_dist / (2 * h**2))
    # 4. Compute Kernel Gradient (Vectorized)
    # The original loop computes: sum_y [ k(x,y) * (x - y) / h^2 ]
    # Which simplifies to: ( x * sum(k) - K * x ) / h^2
    sum_kxy = torch.sum(Kxy, dim=1, keepdim=True)  # [N, 1]
    # dxkxy = -Matmul(K, x) + x * Sum(K)
    dxkxy = -torch.matmul(Kxy, x) + x * sum_kxy
    dxkxy = dxkxy / (h**2)

    return Kxy, dxkxy


def svgd_update(actions, scores):
    Kxy, dxkxy = svgd_kernel(actions)
    grad_theta = (torch.matmul(Kxy, scores) + dxkxy) / actions.shape[0]


class ActionSampler:

    def __init__(
        self,
        policy: BasePolicy,
        n_action_steps: int,
        action_dim: int,
        max_episode_steps: int
    ) -> None:
        self.policy = policy
        self.n_action_steps = n_action_steps
        # Action buffer
        H = self.n_action_steps
        D = action_dim
        T = max_episode_steps
        self.all_time_actions = torch.zeros(
            (T, T + H, D),
            dtype=torch.float32,
            device=self.policy.device
        )
        self.global_step = 0

    def update(self, obs_dict):
        with torch.no_grad():
            action = self.policy.predict_action(obs_dict)['action']
        # Buffer actions
        t = self.global_step
        H = self.n_action_steps
        self.all_time_actions[[t], t: t + H] = action
        self.global_step += 1

    def step(self, obs_dict):      
        # Get all predictions for the current timestep
        t = self.global_step
        H = self.n_action_steps
        start = max(0, t - H + 1)
        end = t + 1
        nonzero_actions = self.all_time_actions[start: end, t: t + H]
        N = nonzero_actions.shape[0]
        batched_obs_dict = dict_apply(obs_dict, lambda x: x.expand(N, *x.shape[1:]))
        scores = self.policy.compute_score(nonzero_actions, batched_obs_dict)
        current_scores = scores[:, 0]
        current_actions = nonzero_actions[:, 0]
        svgd_update(current_actions, current_scores)

    