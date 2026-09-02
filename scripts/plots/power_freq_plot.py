# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import rfft, rfftfreq
# from scipy.interpolate import make_interp_spline
# import scipy.ndimage

# # --- 1. PLOT STYLE SETTINGS (Paper Quality) ---
# # Use these settings to make fonts match LaTeX/Papers
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'DejaVu Serif'],
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 16,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'legend.fontsize': 12,
#     'figure.figsize': (8, 5),
#     'lines.linewidth': 2.5
# })

# def smooth_curve(x, y, num_points=300, sigma=1.0):
#     """
#     Smooths data using Gaussian filtering and Spline interpolation.
#     """
#     # 1. Interpolate to a dense grid (removes 'blockiness')
#     x_new = np.linspace(x.min(), x.max(), num_points)
#     spl = make_interp_spline(x, y, k=3)  # Cubic spline
#     y_smooth = spl(x_new)
    
#     # 2. Apply Gaussian filter to remove small jagged noise
#     y_smooth = scipy.ndimage.gaussian_filter1d(y_smooth, sigma=sigma)
    
#     return x_new, y_smooth

# def visualize_spectral_analysis(bottleneck_feats, shallow_skip_feats, save_path=None):
#     """
#     Generates a publication-quality spectral density plot.
    
#     Args:
#         bottleneck_feats (after interpolation along time dimension): (Batch, Channels, T)
#         shallow_skip_feats: (Batch, Channels, T)
#     """
    
#     def get_psd_and_std(feats):
#         # Flatten batch and channels: (B*C, T)
#         flat_feats = feats.reshape(-1, feats.shape[-1])
        
#         # Normalize signal (optional, but good for relative comparison)
#         flat_feats = flat_feats / (np.std(flat_feats, axis=1, keepdims=True) + 1e-6)

#         # FFT
#         yf = rfft(flat_feats, axis=1)
#         power = np.abs(yf)**2
        
#         # Mean and Std Dev across the Batch/Channels
#         mean_power = np.mean(power, axis=0)
#         std_power = np.std(power, axis=0) / np.sqrt(flat_feats.shape[0]) # Standard Error
        
#         # Log scale handling (avoid log(0))
#         mean_power = np.log10(mean_power + 1e-8)
        
#         # For error bars in log space, we approximate
#         lower_bound = np.log10(np.maximum(np.mean(power, axis=0) - std_power, 1e-8))
#         upper_bound = np.log10(np.mean(power, axis=0) + std_power + 1e-8)
        
#         return mean_power, lower_bound, upper_bound

#     # --- PROCESS DATA ---
#     # 1. Bottleneck Analysis
#     y_bot, bot_low, bot_high = get_psd_and_std(bottleneck_feats)
#     x_bot = rfftfreq(bottleneck_feats.shape[-1], 1/bottleneck_feats.shape[-1])
    
#     # 2. Skip Analysis
#     y_skip, skip_low, skip_high = get_psd_and_std(shallow_skip_feats)
#     x_skip = rfftfreq(shallow_skip_feats.shape[-1], 1/shallow_skip_feats.shape[-1])

#     # 3. Fusion Analysis
#     fuse_feats = np.concatenate([bottleneck_feats, shallow_skip_feats], axis=1)
#     y_fuse, fuse_low, fuse_high = get_psd_and_std(fuse_feats)
#     x_fuse = rfftfreq(fuse_feats.shape[-1], 1/fuse_feats.shape[-1])

#     # --- SMOOTHING ---
#     # We smooth the curves to look like analytical functions
#     x_bot_smooth, y_bot_smooth = smooth_curve(x_bot, y_bot, sigma=1.5)
#     _, bot_low_smooth = smooth_curve(x_bot, bot_low, sigma=1.5)
#     _, bot_high_smooth = smooth_curve(x_bot, bot_high, sigma=1.5)
    
#     x_skip_smooth, y_skip_smooth = smooth_curve(x_skip, y_skip, sigma=1.5)
#     _, skip_low_smooth = smooth_curve(x_skip, skip_low, sigma=1.5)
#     _, skip_high_smooth = smooth_curve(x_skip, skip_high, sigma=1.5)

#     x_fuse_smooth, y_fuse_smooth = smooth_curve(x_fuse, y_fuse, sigma=1.5)
#     _, fuse_low_smooth = smooth_curve(x_fuse, fuse_low, sigma=1.5)
#     _, fuse_high_smooth = smooth_curve(x_fuse, fuse_high, sigma=1.5)

#     # --- PLOTTING ---
#     fig, ax = plt.subplots()
    
#     # Colors: Professional Palette (Deep Red vs Muted Blue)
#     color_bot = '#D55E00'  # Matplotlib "Tab:Red"
#     color_skip = '#0072B2' # Matplotlib "Tab:Blue"
#     color_fuse = '#CC79A7'

#     # Plot Bottleneck
#     ax.plot(x_bot_smooth, y_bot_smooth, color=color_bot, label='Bottleneck')
#     ax.fill_between(x_bot_smooth, bot_low_smooth, bot_high_smooth, color=color_bot, alpha=0.15, linewidth=0)
    
#     # Plot Skip
#     # Note: We limit the x-axis of skip to match bottleneck if they differ drastically, 
#     # but usually plotting them against their own frequencies is correct.
#     ax.plot(x_skip_smooth, y_skip_smooth, color=color_skip, label='Skip')
#     ax.fill_between(x_skip_smooth, skip_low_smooth, skip_high_smooth, color=color_skip, alpha=0.15, linewidth=0)

#     # Plot Fuse
#     ax.plot(x_fuse_smooth, y_fuse_smooth, color=color_fuse, label='Fusion')
#     ax.fill_between(x_fuse_smooth, fuse_low_smooth, fuse_high_smooth, color=color_fuse, alpha=0.15, linewidth=0)

#     # Decorate
#     # ax.set_title("Frequency Decomposition of Diffusion Policy", pad=15, fontweight='bold')
#     ax.set_xlabel("Frequency (Hz)")
#     ax.set_ylabel("Log Power Density")
    
#     # # Custom Grid
#     # ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
#     # ax.spines['top'].set_visible(False)
#     # ax.spines['right'].set_visible(False)

#     # Legend
#     ax.legend(loc='upper right', 
#               frameon=True,       # Turn the box on
#               framealpha=1.0,     # Opaque box background
#               edgecolor='gray',   # The requested gray line color
#               fancybox=False)     # False = sharp corners (standard for papers)

#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Saved figure to {save_path}")
    
#     plt.show()



"""
Visualize an action trajectory's xyz path in 3D alongside the power
spectral density (PSD) of its motion, to inspect high-frequency
components (jitter, jerks) in expert demonstrations.

Usage:
    python visualize_trajectory_psd.py path/to/trajectory.npy --fs 30

The trajectory file should be a numpy array of shape [T, D], where the
first 3 columns are xyz. Any .npy, .npz (with a 'trajectory' key), or
.csv file works — see `load_trajectory` below.
"""
import zarr
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import welch
from scipy.interpolate import make_interp_spline
from matplotlib.patches import FancyArrowPatch

# --- Global styling ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Orange theme palette

def truncated_cmap(name: str, min_val: float = 0.35, max_val: float = 1.0, n: int = 256):
    """Return a colormap using only the [min_val, max_val] slice of `name`,
    so light/near-white low-end colors are excluded (i.e. a darker palette)."""
    base = plt.get_cmap(name)
    colors = base(np.linspace(min_val, max_val, n))
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_trunc", colors)
 
ORANGE_CMAP = truncated_cmap("Oranges", min_val=0.35, max_val=0.75)
ORANGE_LIGHT = mcolors.to_hex(ORANGE_CMAP(0.0))   # colormap's start color
ORANGE_DARK = mcolors.to_hex(ORANGE_CMAP(1.0))   # colormap's end color
ORANGE_MAIN = '#F57C00'


def smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving-average smoothing (odd window, edge-padded)."""
    if window < 3 or window >= len(y):
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_padded = np.pad(y, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(y_padded, kernel, mode="valid")


def compute_psd_dft(signal: np.ndarray, fs: float):
    """
    Compute the power spectral density of a 1-D signal via a direct
    discrete Fourier transform (single window over the whole signal,
    no segmentation/averaging as in Welch's method).

    Uses a Hann window to reduce spectral leakage, with the standard
    window-power normalization so the result is in power/Hz (density
    units) — i.e. sum(psd) * df approximates the signal's variance,
    the same convention scipy.signal.welch uses.

    Args:
        signal: 1-D array, the time-domain signal.
        fs: sampling frequency (Hz).

    Returns:
        freqs: 1-D array of frequency bins, 0 to fs/2 (Nyquist).
        psd: 1-D array of power spectral density values.
    """
    N = len(signal)
    signal = signal - np.mean(signal)  # remove DC component (like Welch's default detrend)

    window = np.hanning(N)
    windowed = signal * window

    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Density-scaling normalization: divide by fs * sum(window^2), and
    # double all but the DC/Nyquist bins since rfft only returns the
    # non-negative-frequency half of a real signal's spectrum.
    scale = 1.0 / (fs * np.sum(window ** 2))
    psd = (np.abs(fft_vals) ** 2) * scale
    if N % 2 == 0:
        psd[1:-1] *= 2  # double all bins except DC and Nyquist
    else:
        psd[1:] *= 2    # no Nyquist bin when N is odd

    return freqs, psd


def visualize_trajectory_and_psd(
    xyz: np.ndarray,
    method: str = 'dft',
    fs: float = 30.0,
    use_velocity: bool = True,
    save_path: str | None = None,
):
    """
    Plot the xyz trajectory in 3D next to the PSD of its motion magnitude.

    Args:
        xyz: [T, 3] position array.
        fs: sampling frequency (Hz) of the trajectory, for the PSD x-axis.
        use_velocity: if True, compute PSD on the finite-difference
            velocity of xyz (recommended — jitter/jerk shows up far more
            clearly in velocity than in raw position). If False, PSD is
            computed directly on the position signal.
        save_path: if given, saves the figure to this path instead of
            (or in addition to) showing it.
    """
    T = xyz.shape[0]
    t = np.arange(T) / fs

    # Signal used for the PSD: velocity (preferred) or raw position.
    if use_velocity:
        sig = np.diff(xyz, axis=0) * fs  # finite-difference velocity
    else:
        sig = xyz

    # Welch's method with generous overlap for a smoother, less jagged estimate.
    magnitude = np.linalg.norm(sig, axis=1)
    if method == 'dft':
        freqs, psd = compute_psd_dft(magnitude, fs=fs)
    elif method == 'welch':
        nperseg = min(256, magnitude.shape[0])
        noverlap = int(nperseg * 0.75)
        freqs, psd = welch(magnitude, fs=fs, nperseg=nperseg, noverlap=noverlap)
    else:
        raise ValueError
    log_psd = np.log10(psd + 1e-12)  # avoid log(0)
    log_psd = smooth(log_psd, window=7)  # light moving-average smoothing

    # Upsample onto a finer frequency grid via cubic spline interpolation,
    # then re-smooth — this gives a visibly smoother curve without altering
    # the underlying spectral shape.
    upsample_factor = 10
    freqs_fine = np.linspace(freqs.min(), freqs.max(), len(freqs) * upsample_factor)
    spline = make_interp_spline(freqs, log_psd, k=3)
    log_psd_fine = spline(freqs_fine)
    log_psd_fine = smooth(log_psd_fine, window=15)


    RIGHT_PANEL_WIDTH_IN = 10    # unchanged from the original layout
    RIGHT_PANEL_HEIGHT_IN = 6.0   # unchanged from the original layout
    LEFT_PANEL_WIDTH_IN = 10.0    # enlarged
    LEFT_PANEL_HEIGHT_IN = 10    # enlarged
 
    FIG_WIDTH_IN = LEFT_PANEL_WIDTH_IN + RIGHT_PANEL_WIDTH_IN + 6
    FIG_HEIGHT_IN = max(LEFT_PANEL_HEIGHT_IN, RIGHT_PANEL_HEIGHT_IN)
 
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(1, 2, width_ratios=[LEFT_PANEL_WIDTH_IN, RIGHT_PANEL_WIDTH_IN])

    # --- Left: 3D trajectory ---
    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=t, cmap=ORANGE_CMAP, s=50)
    ax1.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=ORANGE_MAIN, alpha=0.35, linewidth=3)
    # ax1.scatter(*xyz[0], color=ORANGE_LIGHT, s=50, edgecolor="orange", linewidth=0.6, label="Start")
    # ax1.scatter(*xyz[-1], color=ORANGE_DARK, s=50, edgecolor="orange", linewidth=0.6, label="End")

    # Keep only X / Y / Z axis labels close to the axes, with gridlines
    # retained but tick marks/numbers hidden.
    ax1.set_xlabel("X", fontsize=32, labelpad=-8)
    ax1.set_ylabel("Y", fontsize=32, labelpad=-8)
    ax1.set_zlabel("Z", fontsize=32, labelpad=-8)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.tick_params(axis="both", which="both", length=0)

    # ax1.set_title("End-Effector Trajectory", fontsize=13)
    # ax1.legend(loc="upper left", bbox_to_anchor=(0.3, 0.85), frameon=True, fontsize=16)

    # --- Right: Power spectral density (smooth, filled, log10 power) ---
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(freqs_fine, log_psd_fine, color=ORANGE_MAIN, linewidth=1.8)
    ax2.fill_between(freqs_fine, log_psd_fine, log_psd_fine.min() - 1, color=ORANGE_MAIN, alpha=0.4)
    ax2.set_xlabel("Frequency (Hz)", fontsize=32)
    ax2.set_ylabel("Power (log10)", fontsize=32)
    # ax2.set_title(f"{psd_label} Magnitude Power Spectral Density", fontsize=13)
    # Tight to the data on both axes — no padding around the curve/fill.
    ax2.set_xlim(0, fs / 2)
    ax2.set_ylim(int(log_psd_fine.min()) - 1, int(log_psd_fine.max()) + 1)
    ax2.margins(x=0, y=0)
    ax2.grid(False)
    
    # Ticks explicitly placed so the two extremes of each axis are labeled.
    xticks = np.linspace(0, fs / 2, int(fs / 4) + 1)
    yticks = np.linspace(int(log_psd_fine.min()) - 1, int(log_psd_fine.max()) + 1, int((int(log_psd_fine.max()) - int(log_psd_fine.min()) + 2) + 1))
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.set_xticklabels([f"{v:.1f}" for v in xticks])
    ax2.set_yticklabels([f"{v:.1f}" for v in yticks])

    # Bolder axis box and tick marks.
    for spine in ax2.spines.values():
        spine.set_linewidth(1.3)
    ax2.tick_params(axis="both", which="major", width=1.3, length=6, labelsize=32)
    # Annotate the Welch window length used for the PSD estimate.
    # ax2.text(
    #     0.95, 0.92, f"w = {nperseg}",
    #     transform=ax2.transAxes, ha="right", va="top",
    #     fontsize=13, fontweight="bold", color=ORANGE_DARK,
    #     bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
    # )

    # fig.suptitle("Trajectory Shape and Frequency Analysis", fontsize=14)
    fig.tight_layout()

    # Reclaim the default left-margin whitespace: tight_layout() reserves
    # room assuming a standard y-axis label on the left side, which the 3D
    # panel doesn't have. Pull ax1's left edge toward the figure boundary
    # and grow it into that space (right edge stays put).
    fig.canvas.draw()
    pos1 = ax1.get_position()
    new_x0 = 0.01
    ax1.set_position([new_x0, pos1.y0, pos1.x1 - new_x0, pos1.height])

    # Lock the right panel back to its original absolute height, vertically
    # centered — tight_layout stretches both panels to the full (now taller)
    # figure height by default, so this keeps only the left panel enlarged.
    fig.canvas.draw()
    pos_right_pre = ax2.get_position()
    target_h_frac = RIGHT_PANEL_HEIGHT_IN / FIG_HEIGHT_IN
    y_center = pos_right_pre.y0 + pos_right_pre.height / 2
    ax2.set_position([pos_right_pre.x0, y_center - target_h_frac / 2, pos_right_pre.width, target_h_frac])

    # --- Arrow + "DFT" block between the two panels ---
    fig.canvas.draw()  # finalize layout so axes positions are accurate
    pos_left = ax1.get_position()
    pos_right = ax2.get_position()
    arrow_y = (pos_left.y0 + pos_left.y1) / 2
 
    arrow = FancyArrowPatch(
        (pos_left.x1, arrow_y),
        (pos_right.x0 - 0.05, arrow_y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        # linestyle='--',
        mutation_scale=50,
        linewidth=10,
        color=ORANGE_MAIN,
        zorder=10,
    )
    fig.patches.append(arrow)
 
    mid_x = (pos_left.x1 + pos_right.x0 - 0.06) / 2
    fig.text(
        mid_x, arrow_y - 0.02, "DFT" if method == 'dft' else "WELCH",
        ha="center", va="bottom", fontsize=40 if method == 'dft' else 32, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=ORANGE_MAIN, edgecolor="none"),
        zorder=11,
    )

    # caption = "Proficient Human Expert"
    # fig.text(0.1, 0.1, caption, ha="left", va="bottom", fontsize=48, color="black")

    # Save figures
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", type=int, default=150, help="Demo id")
    parser.add_argument("--method", type=str, default='dft', help="Power spectral computing method (dft or welch)")
    parser.add_argument("--fs", type=float, default=20.0, help="Sampling frequency in Hz")
    parser.add_argument("--position", action="store_true", help="Compute PSD on position instead of velocity")
    parser.add_argument("--save", type=str, default="power_spectral.pdf", help="Path to save the figure instead of displaying it")
    args = parser.parse_args()

    # with h5py.File("data/robomimic_can/mh.hdf5", "r") as f:
    #     trajectory = f[f"/data/demo_{args.id}/actions"][()].astype(np.float32)
    #     delta_xyz = trajectory[:, :3]
    #     xyz = np.cumsum(delta_xyz, axis=0)  # reconstruct relative position path from deltas
    #     visualize_trajectory_and_psd(
    #         xyz, args.method, fs=args.fs, use_velocity=not args.position, save_path=args.save
    #     )

    with zarr.open("data/robosuite_can.zarr", 'r') as f:
        trajectory = f[f"/data/action"][()].astype(np.float32)[f[f"/meta/episode_ends"][()][20]:f[f"/meta/episode_ends"][()][21]]
        xyz = trajectory[:, :3]
        xyz[:, 0] += 3.32673304e-02
        xyz[:, 1] += 1.07853949e-01
        xyz[:, 2] -= 1.01142085
        xyz *= 50
        print(xyz)
        visualize_trajectory_and_psd(
            xyz, args.method, fs=args.fs, use_velocity=not args.position, save_path=args.save
        )
