import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


def dct_reconstruct(trajectory, indices):
    B, H, D = trajectory.shape
    orig_dtype = trajectory.dtype
    
    # DCT
    # Transpose (B, H, D) -> (B, D, H) and cast to float64
    traj_reshaped = np.transpose(trajectory, (0, 2, 1)).astype(np.float64)
    dct_coeffs = dct(traj_reshaped, type=2, norm="ortho", axis=-1)
    
    # Masking
    # Ensure indices is at least a 1D array to handle integer/scalar inputs safely
    indices = np.atleast_1d(indices)
    if indices.size == 1:
        indices = np.full(B, indices[0])
        
    # Reshape indices to (B, 1, 1) for broadcasting
    indices = indices.reshape(B, 1, 1)
    freq_indices = np.arange(H).reshape(1, 1, H)
    
    # Create mask (NumPy automatically handles the broadcasting to dimension D)
    dct_mask = (freq_indices < indices).astype(np.float64)
    masked_coeffs = dct_coeffs * dct_mask
    
    # Inverse DCT
    traj_recons = idct(masked_coeffs, type=2, norm="ortho", axis=-1)
    
    # Transpose back (B, D, H) -> (B, H, D) and cast to original dtype
    traj_recons = np.transpose(traj_recons, (0, 2, 1)).astype(orig_dtype)
    
    return traj_recons


# Load action trajectories
with zarr.open("/Users/wangjl/Desktop/Projects/3D-Diffusion-Policy/data/robosuite_square.zarr", 'r') as f:
    actions = f['data/action'][()]
    actions = 2 * (actions - actions.min(axis=0, keepdims=True)) / (actions.max(axis=0, keepdims=True) - actions.min(axis=0, keepdims=True)) - 1

start = 0
end = 150
dim = 0
action_traj = actions[start: end, [dim]][np.newaxis, ...]
t = np.arange(end - start)

# Low-pass filter action trajectories
cut_off_freq_ratio = 0.5
action_traj = dct_reconstruct(action_traj, cut_off_freq_ratio * (end - start))
action_traj = np.squeeze(action_traj)

# Add noise
timestep = torch.tensor([5], dtype=torch.long)
noise_scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2')
action_traj = torch.from_numpy(action_traj)
noise = torch.rand_like(action_traj)
action_traj = noise_scheduler.add_noise(action_traj, noise, timestep)
action_traj = action_traj.cpu().detach().numpy()


# ==========================================
# Setup the Plot (Tighter Dimensions)
# ==========================================
fig, ax = plt.subplots(figsize=(3, 3))

# No background (transparent)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# No ticks
ax.set_xticks([])
ax.set_yticks([])

# Hide ALL standard spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Theme color (vibrant orange)
orange_color = '#F57C00'

# ==========================================
# Alignment and Bounds
# ==========================================
x_min, x_max = np.min(t), np.max(t)
y_min, y_max = np.min(action_traj), np.max(action_traj)

# Tiny visual margins
y_margin = (y_max - y_min) * 0.05
x_margin = (x_max - x_min) * 0.02

# Position the axes explicitly on the bounds
y_axis_pos = y_min - y_margin
x_axis_pos = x_min

# Extend the limits so the axes have room for the new arrowheads
ax.set_xlim(x_min, x_max + x_margin * 4)
ax.set_ylim(y_axis_pos, y_max + y_margin * 4)

# ==========================================
# Solid Arrows & Perfect Origin Corner
# ==========================================
# FIX: Draw a continuous 'L' shape first. This creates a perfect, sharp (mitered) 
# corner at the origin, preventing any gaps from line caps.
ax.plot([x_axis_pos, x_axis_pos, x_max + x_margin * 3.5], 
        [y_max + y_margin * 3.5, y_axis_pos, y_axis_pos], 
        color=orange_color, lw=3, solid_joinstyle='miter')

# Added shrinkA=0 and shrinkB=0 so the arrow tail doesn't auto-pad away from the origin
arrow_props = dict(arrowstyle="-|>", color=orange_color, lw=3, mutation_scale=20, shrinkA=0, shrinkB=0)

# Draw X-axis arrowhead
ax.annotate('', xy=(x_max + x_margin * 5, y_axis_pos), xytext=(x_axis_pos, y_axis_pos),
            arrowprops=arrow_props, annotation_clip=False)

# Draw Y-axis arrowhead
ax.annotate('', xy=(x_axis_pos, y_max + y_margin * 4.5), xytext=(x_axis_pos, y_axis_pos),
            arrowprops=arrow_props, annotation_clip=False)

# ==========================================
# Label Updates (Times New Roman Font)
# ==========================================
ax.text(x_max + x_margin * 3.5, y_axis_pos - y_margin * 1.5, 't', 
        color=orange_color, fontsize=25, fontweight='bold', 
        ha='center', va='top', fontname='Times New Roman', fontstyle='italic')

# ==========================================
# 2. Draw the clean trajectory
# ==========================================
ax.plot(t, action_traj, color=orange_color, linewidth=3.5)

# Adjust layout to remove extra whitespace
plt.tight_layout()

plt.savefig('action.svg', transparent=True, dpi=300)
plt.show()