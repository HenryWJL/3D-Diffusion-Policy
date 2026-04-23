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

# 2. Global Data and Labels
legend_labels = ['FGO', r'$p_{\mathrm{base}} = 0$', 'W/o KFC Sampling', r'Cosine $f_k$ Schedule', r'Cosine $\omega_k$ Schedule']
# colors = ['#2d678e', '#b88b37', '#428769', '#af6231', '#b78bb5', '#b08b6f'] 
colors = ['#FFE0B2', '#FFB74D', '#FF9800', '#F57C00', '#E65100']


# Dictionary holding the specific configurations for each subplot
tasks = {
    'Robosuite Lift': {
        'data': [92.7, 88.0, 88.0, 89.3, 86.7],
        'std':  [3.1, 2.0, 2.0, 3.1, 5.0], 
        'ylabel': 'Success Rate (%)',
        'ylim': (75, 100), 
        'invert_y': False
    },
    'Adroit Door': {
        'data': [69.3, 68.7, 66.7, 66.0, 65.7], 
        'std':  [2.3, 3.1, 3.1, 2.0, 7.8],
        'ylabel': 'Success Rate (%)',
        'ylim': (50, 85), 
        'invert_y': False
    },
    'DexArt Toilet': {
        'data': [66.7, 62.7, 59.3, 62.0, 60.7],
        'std':  [1.2, 3.1, 5.0, 5.3, 9.2], 
        'ylabel': 'Success Rate (%)',
        'ylim': (45, 75), 
        'invert_y': False
    }
}

# 3. Create figure and subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(11, 4.5)) # Slightly taller figure to accommodate the legend

# 4. Plotting Loop
for ax, (title, info) in zip(axes, tasks.items()):
    data = info['data']
    stds = info['std']
    
    for i, (val, std_val, color) in enumerate(zip(data, stds, colors)):
        if val is None:
            continue
            
        # Draw the bar with error bars and black edge colors
        ax.bar(
            i, val, 
            yerr=std_val, capsize=4, 
            width=0.75, color=color, 
            edgecolor='black', linewidth=1.0, # Changed to black boundary
            error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': '#333333'}
        )
        
        # Determine text placement accounting for the error bar height
        if not info['invert_y']:
            # Normal axis
            if val < info['ylim'][0]:
                text_y = info['ylim'][0] + 1
            else:
                text_y = val + std_val + 1
        else:
            # Inverted axis
            text_y = val - std_val - 1
            
        label_text = f'{val:.1f}'
        # ax.text(i, text_y, label_text, ha='center', va='center', fontsize=8)

    # Subplot Formatting
    ax.set_title(title, pad=10)
    ax.set_ylabel(info['ylabel'], fontsize=16)
    ax.set_xticks([]) 
    ax.set_ylim(info['ylim']) 
        
    # Gridlines
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='#e0e0e0')
    ax.set_axisbelow(True)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

# 5. Shared Global Legend
legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, legend_labels)]

fig.legend(
    handles=legend_patches,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.25), # Adjusted higher relative to the bottom of the figure
    ncol=5, 
    frameon=True,
    borderpad=0.8,
    framealpha=1.0,
    edgecolor='#cccccc'
)

# 6. Layout Adjustment
# The rect argument [left, bottom, right, top] reserves the bottom 20% of the figure for the legend
plt.tight_layout(rect=[0, 0.2, 1, 1])
plt.savefig("full_ablations.pdf", bbox_inches='tight', pad_inches=0)
plt.show()