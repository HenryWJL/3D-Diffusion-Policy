# import torch
# import torch.fft
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.interpolate import make_interp_spline

# # --- 1. PUBLICATION STYLE SETTINGS ---
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'DejaVu Serif'],
#     'font.size': 14,              # Slightly larger for standalone figures
#     'axes.labelsize': 16,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 13,
#     'ytick.labelsize': 13,
#     'legend.fontsize': 13,
#     'lines.linewidth': 2.0,
#     'figure.figsize': (8, 6)      # Standard single-column ratio
# })

# # --- 2. HELPERS (Splitting & Smoothing) ---
# def freeu_frequency_split(x, bandwidth=0.15):
#     """Splits input (1, T, D) into Low and High freq components."""
#     B, T, D = x.shape
#     device = x.device
    
#     x_freq = torch.fft.fft(x, dim=1)
#     x_freq_shifted = torch.fft.fftshift(x_freq, dim=1)
    
#     mask = torch.zeros((1, T, 1), device=device, dtype=x_freq.dtype)
#     center_idx = T // 2
#     radius = int(bandwidth * T / 2)
#     radius = max(radius, 1)
    
#     mask[:, center_idx - radius : center_idx + radius + 1, :] = 1.0
    
#     x_freq_low = x_freq_shifted * mask
#     x_freq_low_unshifted = torch.fft.ifftshift(x_freq_low, dim=1)
#     x_low = torch.fft.ifft(x_freq_low_unshifted, dim=1).real
#     x_high = x - x_low
    
#     return x_low, x_high

# def smooth_line(x, y, num_points=300):
#     if len(x) < 4: return x, y
#     x_new = np.linspace(x.min(), x.max(), num_points)
#     spl = make_interp_spline(x, y, k=3) 
#     y_new = spl(x_new)
#     return x_new, y_new

# def style_axis(ax, title, ylabel):
#     """Applies clean paper-ready styling to an axis."""
#     ax.spines['left'].set_linewidth(1.2)
#     ax.spines['bottom'].set_linewidth(1.2)
#     ax.set_title(title, pad=12, fontweight='bold')
#     ax.set_xlabel("Time Step", labelpad=8)
#     ax.set_ylabel(ylabel, labelpad=8)

# # --- 3. MAIN PLOTTING FUNCTION (TWO FIGURES) ---
# def visualize_diffusion_frequency_evolution(trajectory_list, dim_idx=0, bandwidth=0.15, save_path = None):
#     num_steps = len(trajectory_list)
#     colors = plt.cm.viridis(np.linspace(0, 0.95, num_steps))
    
#     # Store data
#     all_low = []
#     all_high = []
    
#     for traj in trajectory_list:
#         traj = traj.cpu() if traj.is_cuda else traj
#         low, high = freeu_frequency_split(traj, bandwidth=bandwidth)
#         all_low.append(low[0, :, dim_idx].numpy())
#         all_high.append(high[0, :, dim_idx].numpy())

#     # ==========================================
#     # FIGURE 1: Low-Frequency (Structure)
#     # ==========================================
#     fig1, ax1 = plt.subplots(figsize=(8, 6))
    
#     t_axis = np.arange(len(all_low[0]))
    
#     for i, y_low in enumerate(all_low):
#         is_final = (i == num_steps - 1)
#         is_start = (i == 0)
        
#         # Style logic
#         alpha = 1.0 if is_final else 0.5
#         linewidth = 3.5 if is_final else 1.5
#         zorder = 10 if is_final else i
        
#         label = None
#         if is_start: label = r'$t=T-1$'
#         if is_final: label = r'$t=0$'
        
#         # Smooth the structure lines
#         t_smooth, y_low_smooth = smooth_line(t_axis, y_low)
#         ax1.plot(t_smooth, y_low_smooth, color=colors[i], label=label, 
#                  alpha=alpha, linewidth=linewidth, zorder=zorder)

#     style_axis(ax1, "", "Action")
#     # Legend
#     ax1.legend(loc='upper right', 
#               frameon=True,       # Turn the box on
#               framealpha=1.0,     # Opaque box background
#               edgecolor='gray',   # The requested gray line color
#               fancybox=False)     # False = sharp corners (standard for papers)
#     plt.tight_layout()
#     if save_path is not None:
#         plt.savefig(f"low_freq_{save_path}", dpi=300, bbox_inches='tight')
#     plt.show() # Shows Figure 1

#     # ==========================================
#     # FIGURE 2: High-Frequency (Detail)
#     # ==========================================
#     fig2, ax2 = plt.subplots(figsize=(8, 6))
    
#     for i, y_high in enumerate(all_high):
#         is_final = (i == num_steps - 1)
#         is_start = (i == 0)
        
#         alpha = 1.0 if is_final else 0.5
#         linewidth = 3.0 if is_final else 1.2
#         zorder = 10 if is_final else i
        
#         label = None
#         if is_start: label = r'$t=T-1$'
#         if is_final: label = r'$t=0$'

#         # Do NOT smooth high-freq (preserve the noise spikes)
#         ax2.plot(t_axis, y_high, color=colors[i], label=label, 
#                  alpha=alpha, linewidth=linewidth, zorder=zorder)

#     style_axis(ax2, "", "Action")
#     # Legend
#     ax2.legend(loc='upper right', 
#               frameon=True,       # Turn the box on
#               framealpha=1.0,     # Opaque box background
#               edgecolor='gray',   # The requested gray line color
#               fancybox=False)     # False = sharp corners (standard for papers)
#     plt.tight_layout()
#     if save_path is not None:
#         plt.savefig(f"high_freq_{save_path}", dpi=300, bbox_inches='tight')
#     plt.show() # Shows Figure 2


import copy
import dill
import zarr
import pywt
import hydra
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
from diffusion_policy_3d.common.pytorch_util import dict_apply


def load_policy(name: str, checkpoint: str, device: torch.device):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parent.parent.parent.joinpath('3D-Diffusion-Policy', 'diffusion_policy_3d', 'config')),
        version_base=None
    ):
        cfg = hydra.compose(config_name=name, overrides=["task=robosuite_square"])

    model = hydra.utils.instantiate(cfg.policy)
    ema_model = copy.deepcopy(model)

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    sd = payload.get("state_dicts", payload)
    
    # Prefer EMA weights (cfg.training.use_ema is true for dp3.yaml).
    which = "ema_model" if "ema_model" in sd else "model"
    policy = ema_model if which == "ema_model" else model
    policy.load_state_dict(sd[which], strict=False)
    policy.to(device)
    policy.eval()

    return policy, cfg


def main():
    policy_name = "fgdp"
    zarr_path = "/Users/wangjl/Desktop/Projects/3D-Diffusion-Policy/data/robosuite_square.zarr"
    ckpt_path = f"/Users/wangjl/Desktop/Projects/3D-Diffusion-Policy/checkpoints/{policy_name}.pth"
    device = torch.device('cpu')
    step_id = 1000
    action_dim = 1

    # load policy
    policy, cfg = load_policy(policy_name, ckpt_path, device)

    # load observations
    with zarr.open(zarr_path, 'r') as f:
        obs_dict = dict(
            frontview_pc = f['data/frontview_pc'][()],
            robot0_eye_in_hand_pc = f['data/robot0_eye_in_hand_pc'][()],
            robot0_eef_pos = f['data/robot0_eef_pos'][()],
            robot0_eef_quat = f['data/robot0_eef_quat'][()],
            robot0_gripper_qpos = f['data/robot0_gripper_qpos'][()],
        )
    obs = dict_apply(obs_dict, lambda x: torch.from_numpy(x[step_id: step_id + 2]).unsqueeze(0).float().to(device))

    # get trajectory history
    trajectory_history = []
    original_step = policy.noise_scheduler.step

    def hooked_step(*args, **kwargs):
        # Execute the original step function
        step_output = original_step(*args, **kwargs)
        
        # Extract the trajectory, clone it, and detach it from the compute graph
        # Moving to .cpu() is highly recommended to prevent GPU Out-Of-Memory errors
        current_trajectory = step_output.prev_sample.clone().detach().cpu()
        
        # Store it in our list
        trajectory_history.append(current_trajectory)
        
        # Return the exact original output so the policy loop continues normally
        return step_output
    
    policy.noise_scheduler.step = hooked_step
    with torch.no_grad():
        final_trajectory = policy.predict_action(obs)
    policy.noise_scheduler.step = original_step
    
    # decompose using wavelet transform
    low_freq_history = []
    high_freq_history = []
    for traj in trajectory_history:
        signal = traj[0, :, action_dim].detach().cpu().numpy()
        cA, cD = pywt.dwt(signal, 'haar')
        low_freq_component = pywt.idwt(cA, np.zeros_like(cD), 'haar')
        high_freq_component = pywt.idwt(np.zeros_like(cA), cD, 'haar')
        low_freq_component = low_freq_component[:16]
        high_freq_component = high_freq_component[:16]
        low_freq_history.append(low_freq_component)
        high_freq_history.append(high_freq_component)

    # plot
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 20,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 16,
        'lines.linewidth': 2.0,

        # --- MAKE ALL BORDERS AND TICKS BOLDER HERE ---
        'axes.linewidth': 1.5,       # Makes all 4 outer borders distinct and bold
        'xtick.major.width': 1.5,    
        'ytick.major.width': 1.5,    # Matches the y-tick thickness to the border
    })

    plt.figure(figsize=(8, 6))
    num_steps = len(trajectory_history)

    for i in range(num_steps):
        # Calculate an alpha value to fade older steps (early steps are faint, final step is solid)
        # Ranging from 0.2 (first step) to 1.0 (last step)
        alpha_val = 0.2 + 0.8 * (i / max(1, num_steps - 1))
        
        # Label only the final step to keep the legend clean
        label_low = 'Low Frequency' if i == num_steps - 1 else None
        label_high = 'High Frequency' if i == num_steps - 1 else None
        
        # Plot Low Frequency (Solid Blue Lines)
        plt.plot(low_freq_history[i], 
                color='tab:blue', 
                linestyle='-', 
                linewidth=2, 
                alpha=alpha_val, 
                label=label_low)
        
        # Plot High Frequency (Dashed Orange Lines)
        plt.plot(high_freq_history[i], 
                color='tab:orange', 
                linestyle='--', 
                linewidth=2, 
                alpha=alpha_val, 
                label=label_high)

    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')

    # Add a grid and the legend
    plt.grid(True, axis='y', color='gray', linestyle='--', alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('frequency_analysis.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    main()
