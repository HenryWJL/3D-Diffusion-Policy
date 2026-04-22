import torch
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# --- 1. PUBLICATION STYLE SETTINGS ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,              # Slightly larger for standalone figures
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'lines.linewidth': 2.0,
    'figure.figsize': (8, 6)      # Standard single-column ratio
})

# --- 2. HELPERS (Splitting & Smoothing) ---
def freeu_frequency_split(x, bandwidth=0.15):
    """Splits input (1, T, D) into Low and High freq components."""
    B, T, D = x.shape
    device = x.device
    
    x_freq = torch.fft.fft(x, dim=1)
    x_freq_shifted = torch.fft.fftshift(x_freq, dim=1)
    
    mask = torch.zeros((1, T, 1), device=device, dtype=x_freq.dtype)
    center_idx = T // 2
    radius = int(bandwidth * T / 2)
    radius = max(radius, 1)
    
    mask[:, center_idx - radius : center_idx + radius + 1, :] = 1.0
    
    x_freq_low = x_freq_shifted * mask
    x_freq_low_unshifted = torch.fft.ifftshift(x_freq_low, dim=1)
    x_low = torch.fft.ifft(x_freq_low_unshifted, dim=1).real
    x_high = x - x_low
    
    return x_low, x_high

def smooth_line(x, y, num_points=300):
    if len(x) < 4: return x, y
    x_new = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3) 
    y_new = spl(x_new)
    return x_new, y_new

def style_axis(ax, title, ylabel):
    """Applies clean paper-ready styling to an axis."""
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.set_title(title, pad=12, fontweight='bold')
    ax.set_xlabel("Time Step", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)

# --- 3. MAIN PLOTTING FUNCTION (TWO FIGURES) ---
def visualize_diffusion_frequency_evolution(trajectory_list, dim_idx=0, bandwidth=0.15, save_path = None):
    num_steps = len(trajectory_list)
    colors = plt.cm.viridis(np.linspace(0, 0.95, num_steps))
    
    # Store data
    all_low = []
    all_high = []
    
    for traj in trajectory_list:
        traj = traj.cpu() if traj.is_cuda else traj
        low, high = freeu_frequency_split(traj, bandwidth=bandwidth)
        all_low.append(low[0, :, dim_idx].numpy())
        all_high.append(high[0, :, dim_idx].numpy())

    # ==========================================
    # FIGURE 1: Low-Frequency (Structure)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    t_axis = np.arange(len(all_low[0]))
    
    for i, y_low in enumerate(all_low):
        is_final = (i == num_steps - 1)
        is_start = (i == 0)
        
        # Style logic
        alpha = 1.0 if is_final else 0.5
        linewidth = 3.5 if is_final else 1.5
        zorder = 10 if is_final else i
        
        label = None
        if is_start: label = r'$t=T-1$'
        if is_final: label = r'$t=0$'
        
        # Smooth the structure lines
        t_smooth, y_low_smooth = smooth_line(t_axis, y_low)
        ax1.plot(t_smooth, y_low_smooth, color=colors[i], label=label, 
                 alpha=alpha, linewidth=linewidth, zorder=zorder)

    style_axis(ax1, "", "Action")
    # Legend
    ax1.legend(loc='upper right', 
              frameon=True,       # Turn the box on
              framealpha=1.0,     # Opaque box background
              edgecolor='gray',   # The requested gray line color
              fancybox=False)     # False = sharp corners (standard for papers)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"low_freq_{save_path}", dpi=300, bbox_inches='tight')
    plt.show() # Shows Figure 1

    # ==========================================
    # FIGURE 2: High-Frequency (Detail)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    for i, y_high in enumerate(all_high):
        is_final = (i == num_steps - 1)
        is_start = (i == 0)
        
        alpha = 1.0 if is_final else 0.5
        linewidth = 3.0 if is_final else 1.2
        zorder = 10 if is_final else i
        
        label = None
        if is_start: label = r'$t=T-1$'
        if is_final: label = r'$t=0$'

        # Do NOT smooth high-freq (preserve the noise spikes)
        ax2.plot(t_axis, y_high, color=colors[i], label=label, 
                 alpha=alpha, linewidth=linewidth, zorder=zorder)

    style_axis(ax2, "", "Action")
    # Legend
    ax2.legend(loc='upper right', 
              frameon=True,       # Turn the box on
              framealpha=1.0,     # Opaque box background
              edgecolor='gray',   # The requested gray line color
              fancybox=False)     # False = sharp corners (standard for papers)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"high_freq_{save_path}", dpi=300, bbox_inches='tight')
    plt.show() # Shows Figure 2