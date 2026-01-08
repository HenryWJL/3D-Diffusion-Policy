import torch
import torch.nn.functional as F
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.policy.base_policy import BasePolicy


def svgd_kernel(x, h=None):
    """
    Computes RBF kernel and its gradient w.r.t. x, for trajectories x of shape [N, L, D].
    
    Args:
        x: Tensor of shape [N, L, D] (particles / trajectories)
        h: Bandwidth. If None, uses median heuristic.
    
    Returns:
        Kxy: [N, N] kernel matrix
        dxkxy: [N, L, D] gradient of kernel
    """
    N, L, D = x.shape
    if N <= 1:
        Kxy = torch.ones((N, N), device=x.device, dtype=x.dtype)
        dxkxy = torch.zeros_like(x)
        return Kxy, dxkxy

    # Flatten trajectories for distance computation: [N, L*D]
    x_flat = x.view(N, -1)

    # Pairwise squared distances
    diff = x_flat.unsqueeze(1) - x_flat.unsqueeze(0)  # [N, N, L*D]
    pairwise_dist = torch.sum(diff ** 2, dim=-1)       # [N, N]

    # Median trick for bandwidth
    if h is None:
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        median_dist = torch.median(pairwise_dist[mask])
        median_dist = torch.clamp(median_dist, min=1e-12)
        h = torch.sqrt(0.5 * median_dist / torch.log(torch.tensor(N + 1.0, device=x.device))) + 1e-6

    # RBF kernel
    Kxy = torch.exp(-pairwise_dist / (2 * h ** 2))

    # Kernel gradient (vectorized)
    sum_kxy = Kxy.sum(dim=1, keepdim=True)  # [N, 1]
    dxkxy_flat = -torch.matmul(Kxy, x_flat) + x_flat * sum_kxy
    dxkxy_flat = dxkxy_flat / (h ** 2)

    # Reshape back to [N, L, D]
    dxkxy = dxkxy_flat.view(N, L, D)

    return Kxy, dxkxy


def svgd_gradient(actions, scores, normalize=True):
    """
    Compute SVGD gradient for trajectories.
    
    Args:
        actions: [N, L, D] tensor of particles
        scores: [N, L, D] tensor of gradients of log probability
        normalize: whether to apply RMS normalization per particle
    
    Returns:
        grad: [N, L, D] SVGD gradient
    """
    Kxy, dxkxy = svgd_kernel(actions)

    # Flatten for matmul
    N, L, D = actions.shape
    actions_flat = actions.view(N, -1)
    scores_flat = scores.view(N, -1)
    dxkxy_flat = dxkxy.view(N, -1)

    grad_flat = (torch.matmul(Kxy, scores_flat) + dxkxy_flat) / N

    # RMS normalization per particle
    if normalize:
        eps = 1e-8
        grad_flat = grad_flat / (torch.sqrt(grad_flat.pow(2).mean(dim=-1, keepdim=True)) + eps)

    # Reshape back
    grad = grad_flat.view(N, L, D)
    return grad


def svgd_update(actions, obs_dict, policy, n_iter = 10, step_size = 1e-3, bandwidth = -1, alpha = 0.9):
    x = actions
    # adagrad with momentum
    historical_grad = 0
    for iter in range(n_iter):
        with torch.no_grad():
            score = policy.compute_score(x, obs_dict).float() 
        grad = svgd_gradient(x, score, normalize=False)
        # adagrad 
        if iter == 0:
            historical_grad = historical_grad + grad ** 2
        else:
            historical_grad = alpha * historical_grad + (1 - alpha) * (grad ** 2)
        adj_grad = grad / (torch.sqrt(historical_grad) + 1e-6)
        x += step_size * adj_grad 
        
    return x
    

# def svgd_kernel(x, h = -1):
#     """
#     Computes the RBF kernel and its gradient w.r.t. the first input.
    
#     Args:
#         x: Tensor of shape [N, D] (particles)
#         h: Bandwidth parameter. If < 0, uses the median heuristic.
        
#     Returns:
#         Kxy: Kernel matrix [N, N]
#         dxkxy: Gradient of Kernel [N, D] -> sum_y grad_x k(x, y)
#     """
#     N = x.shape[0]
#     if N == 1:
#         Kxy = torch.ones((N, N), device=x.device, dtype=x.dtype)
#         dxkxy = torch.zeros_like(x)
#         return Kxy, dxkxy
    
#     # 1. Compute Pairwise Squared Distances
#     diff = x.unsqueeze(1) - x.unsqueeze(0)
#     pairwise_dist = torch.sum(diff**2, dim=-1)  # [N, N]
#     # 2. Median Trick for Bandwidth (h)
#     if h < 0:
#         mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
#         median_dist = torch.median(pairwise_dist[mask])
#         eps = 1e-6
#         h = torch.sqrt(0.5 * median_dist / torch.log(torch.tensor(N + 1.0, device=x.device))) + eps
#     # 3. Compute RBF Kernel
#     Kxy = torch.exp(-pairwise_dist / (2 * h**2))
#     # 4. Compute Kernel Gradient (Vectorized)
#     # The original loop computes: sum_y [ k(x,y) * (x - y) / h^2 ]
#     # Which simplifies to: ( x * sum(k) - K * x ) / h^2
#     sum_kxy = torch.sum(Kxy, dim=1, keepdim=True)  # [N, 1]
#     # dxkxy = -Matmul(K, x) + x * Sum(K)
#     dxkxy = -torch.matmul(Kxy, x) + x * sum_kxy
#     dxkxy = dxkxy / (h**2)

#     return Kxy, dxkxy


# def svgd_gradient(actions, scores):
#     Kxy, dxkxy = svgd_kernel(actions)
#     grad = (torch.matmul(Kxy, scores) + dxkxy) / actions.shape[0]
#     # Normalize gradients (RMS normalization)
#     eps = 1e-8
#     grad = grad / (torch.sqrt(grad.pow(2).mean(dim=-1, keepdim=True)) + eps)
#     return grad


class ActionSampler:

    def __init__(
        self,
        policy: BasePolicy,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        action_dim: int,
        max_episode_steps: int,
        max_grad_norm: float = 0.05,
        step_size: float = 1.0
    ) -> None:
        self.global_step = 0
        self.policy = policy
        self.policy.eval()
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_grad_norm = max_grad_norm
        self.step_size = step_size
        # Action buffer
        H = self.horizon
        D = action_dim
        T = max_episode_steps
        self.all_time_action_preds = torch.zeros((T, T + H, D), device=self.policy.device)
    
    def reset(self):
        self.global_step = 0
        self.policy.reset()
        self.all_time_action_preds = torch.zeros_like(self.all_time_action_preds)

    def update(self, obs_dict):
        with torch.no_grad():
            result = self.policy.predict_action(obs_dict)
            action_pred = result['action_pred']  # [B, horizon, D]
            # Buffer actions
            t = self.global_step
            H = self.horizon
            self.all_time_action_preds[[t], t: t + H] = action_pred.float()
            # Get the latest K predictions staring from the current timestep
            K = 3
            # start = max(0, t - H + 1)
            start = max(0, t - K + 1)
            end = t + 1
            current_action_preds = self.all_time_action_preds[start: end, t: t + H]

            N = current_action_preds.shape[0]
            batched_obs_dict = dict_apply(obs_dict, lambda x: x.expand(N, *x.shape[1:]))
            scores = self.policy.compute_score(current_action_preds, batched_obs_dict)
            # Note for diffusion policy, valid actions start from the (self.n_obs_steps - 1)-th element of the predictions
            current_scores = scores[:, self.n_obs_steps - 1]
            current_actions = current_action_preds[:, self.n_obs_steps - 1]
            grads = svgd_gradient(current_actions, current_scores)
            
            # rms = torch.sqrt(grads.pow(2).mean(dim=-1)).mean()
            # scale = 0.01 / (rms + 1e-8)
            # current_actions += scale * grads

            # grads = F.normalize(grads, dim=-1)
            # grad_norms = torch.norm(grads, dim=-1)
            # scale_factor = torch.clamp(self.max_grad_norm / (grad_norms + 1e-6), max=1.0)
            # grads *= scale_factor[:, None]

            # Update current actions
            current_actions += self.step_size * grads
            
            # Aggregate actions
            exp_weights = torch.exp(-0.01 * torch.arange(len(current_actions), dtype=current_actions.dtype, device=current_actions.device).flip(dims=[0]))
            exp_weights = (exp_weights / exp_weights.sum())[..., None]
            final_action = (current_actions * exp_weights).sum(axis=0)
            # energy = torch.norm(current_scores, dim=-1)
            # best_idx = torch.argmin(energy)
            # leader = updated_actions[best_idx]
            # # 2. Find neighbors (Euclidean distance)
            # dists = torch.norm(updated_actions - leader, dim=-1)
            # # 3. Mask: Select particles close to the leader
            # threshold = 0.1
            # mask = dists < threshold
            # # 4. Robust Average
            # final_action = torch.mean(updated_actions[mask], dim=0)
            self.global_step += 1    
            return final_action