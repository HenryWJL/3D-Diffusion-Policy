import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import make_interp_spline
import scipy.ndimage

# --- 1. PLOT STYLE SETTINGS (Paper Quality) ---
# Use these settings to make fonts match LaTeX/Papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 5),
    'lines.linewidth': 2.5
})

def smooth_curve(x, y, num_points=300, sigma=1.0):
    """
    Smooths data using Gaussian filtering and Spline interpolation.
    """
    # 1. Interpolate to a dense grid (removes 'blockiness')
    x_new = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3)  # Cubic spline
    y_smooth = spl(x_new)
    
    # 2. Apply Gaussian filter to remove small jagged noise
    y_smooth = scipy.ndimage.gaussian_filter1d(y_smooth, sigma=sigma)
    
    return x_new, y_smooth

def visualize_spectral_analysis(bottleneck_feats, shallow_skip_feats, save_path=None):
    """
    Generates a publication-quality spectral density plot.
    
    Args:
        bottleneck_feats (after interpolation along time dimension): (Batch, Channels, T)
        shallow_skip_feats: (Batch, Channels, T)
    """
    
    def get_psd_and_std(feats):
        # Flatten batch and channels: (B*C, T)
        flat_feats = feats.reshape(-1, feats.shape[-1])
        
        # Normalize signal (optional, but good for relative comparison)
        flat_feats = flat_feats / (np.std(flat_feats, axis=1, keepdims=True) + 1e-6)

        # FFT
        yf = rfft(flat_feats, axis=1)
        power = np.abs(yf)**2
        
        # Mean and Std Dev across the Batch/Channels
        mean_power = np.mean(power, axis=0)
        std_power = np.std(power, axis=0) / np.sqrt(flat_feats.shape[0]) # Standard Error
        
        # Log scale handling (avoid log(0))
        mean_power = np.log10(mean_power + 1e-8)
        
        # For error bars in log space, we approximate
        lower_bound = np.log10(np.maximum(np.mean(power, axis=0) - std_power, 1e-8))
        upper_bound = np.log10(np.mean(power, axis=0) + std_power + 1e-8)
        
        return mean_power, lower_bound, upper_bound

    # --- PROCESS DATA ---
    # 1. Bottleneck Analysis
    y_bot, bot_low, bot_high = get_psd_and_std(bottleneck_feats)
    x_bot = rfftfreq(bottleneck_feats.shape[-1], 1/bottleneck_feats.shape[-1])
    
    # 2. Skip Analysis
    y_skip, skip_low, skip_high = get_psd_and_std(shallow_skip_feats)
    x_skip = rfftfreq(shallow_skip_feats.shape[-1], 1/shallow_skip_feats.shape[-1])

    # 3. Fusion Analysis
    fuse_feats = np.concatenate([bottleneck_feats, shallow_skip_feats], axis=1)
    y_fuse, fuse_low, fuse_high = get_psd_and_std(fuse_feats)
    x_fuse = rfftfreq(fuse_feats.shape[-1], 1/fuse_feats.shape[-1])

    # --- SMOOTHING ---
    # We smooth the curves to look like analytical functions
    x_bot_smooth, y_bot_smooth = smooth_curve(x_bot, y_bot, sigma=1.5)
    _, bot_low_smooth = smooth_curve(x_bot, bot_low, sigma=1.5)
    _, bot_high_smooth = smooth_curve(x_bot, bot_high, sigma=1.5)
    
    x_skip_smooth, y_skip_smooth = smooth_curve(x_skip, y_skip, sigma=1.5)
    _, skip_low_smooth = smooth_curve(x_skip, skip_low, sigma=1.5)
    _, skip_high_smooth = smooth_curve(x_skip, skip_high, sigma=1.5)

    x_fuse_smooth, y_fuse_smooth = smooth_curve(x_fuse, y_fuse, sigma=1.5)
    _, fuse_low_smooth = smooth_curve(x_fuse, fuse_low, sigma=1.5)
    _, fuse_high_smooth = smooth_curve(x_fuse, fuse_high, sigma=1.5)

    # --- PLOTTING ---
    fig, ax = plt.subplots()
    
    # Colors: Professional Palette (Deep Red vs Muted Blue)
    color_bot = '#D55E00'  # Matplotlib "Tab:Red"
    color_skip = '#0072B2' # Matplotlib "Tab:Blue"
    color_fuse = '#CC79A7'

    # Plot Bottleneck
    ax.plot(x_bot_smooth, y_bot_smooth, color=color_bot, label='Bottleneck')
    ax.fill_between(x_bot_smooth, bot_low_smooth, bot_high_smooth, color=color_bot, alpha=0.15, linewidth=0)
    
    # Plot Skip
    # Note: We limit the x-axis of skip to match bottleneck if they differ drastically, 
    # but usually plotting them against their own frequencies is correct.
    ax.plot(x_skip_smooth, y_skip_smooth, color=color_skip, label='Skip')
    ax.fill_between(x_skip_smooth, skip_low_smooth, skip_high_smooth, color=color_skip, alpha=0.15, linewidth=0)

    # Plot Fuse
    ax.plot(x_fuse_smooth, y_fuse_smooth, color=color_fuse, label='Fusion')
    ax.fill_between(x_fuse_smooth, fuse_low_smooth, fuse_high_smooth, color=color_fuse, alpha=0.15, linewidth=0)

    # Decorate
    # ax.set_title("Frequency Decomposition of Diffusion Policy", pad=15, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log Power Density")
    
    # # Custom Grid
    # ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(loc='upper right', 
              frameon=True,       # Turn the box on
              framealpha=1.0,     # Opaque box background
              edgecolor='gray',   # The requested gray line color
              fancybox=False)     # False = sharp corners (standard for papers)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()