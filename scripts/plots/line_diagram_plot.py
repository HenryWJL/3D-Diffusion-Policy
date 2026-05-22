import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. Theme and Aesthetics Setup
sns.set_theme(style='white', context='talk')
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False, 
    'axes.labelsize': 11,
    'axes.titlesize': 15,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10
})

# --- Data Setup ---
guidance_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
success_rates = np.array([86.0, 87.3, 87.3, 89.3, 87.3, 85.3, 84.7])
std_dev = np.array([5.3, 2.3, 3.1, 4.2, 4.2, 3.1, 1.2])

# --- Plotting ---
fig, ax = plt.subplots(figsize=(5, 4.5))

# Plot the shaded error band
ax.fill_between(
    guidance_weights, 
    success_rates - std_dev, 
    success_rates + std_dev, 
    color='#FFB74D', 
    alpha=0.3,
    linewidth=0
)

# Plot the main line with markers
ax.plot(
    guidance_weights, 
    success_rates, 
    marker='o', 
    markersize=7,
    color='#F57C00', 
    linewidth=2,
    linestyle='-'
)

# --- Aesthetics & Formatting ---
ax.set_xlabel(r'Guidance Weight $\omega$', fontsize=16)
ax.set_ylabel('Success Rate (%)', fontsize=16)

# Set the x-ticks to exactly match your requested data points
ax.set_xticks(guidance_weights)
ax.set_xticklabels([f"{x:.1f}" if x == 0 or x == 0.5 or x == 1 or x == 1.5 else f"{x:.2f}" for x in guidance_weights], fontsize=12)

# --- ADDED SPECIFIC Y-TICKS & VISIBLE TICK MARKS ---
ax.set_yticks([70, 75, 80, 85, 90, 95, 100])
ax.set_ylim(70, 100)

# Explicitly draw both x and y tick marks with matching border thickness (1.2)
ax.tick_params(axis='both', which='major', bottom=True, left=True, size=4, width=1.2, labelsize=14)

# Make the axes borders thicker to match the paper-style look
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()

# Optional: Save the figure
plt.savefig("constant_omega_ablations.pdf", bbox_inches='tight', pad_inches=0)
plt.show()