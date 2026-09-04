import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. Theme and Aesthetics Setup
sns.set_theme(style='white', context='talk')
plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'serif',
    'text.usetex': False, 
    'axes.labelsize': 11,
    'axes.titlesize': 15,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10
})

# 2. Global Data and Labels
legend_labels = ['DP3', 'DiT-Policy', 'DP3 + CFG', 'DiT-Policy + ACG', r'DP3 + FGO']
colors = ['#FFE0B2', '#FFB74D', '#FF9800', '#F57C00', '#E65100']

# Dictionary holding the specific configurations for each subplot
tasks = [
    {
        'data': [0.47, 0.42, 0.48, 0.42, 0.48],
        'std':  [0, 0, 0, 0, 0], 
        'ylabel': 'Training Time (GPU h)',
        'ylim': (0, 0.6), 
        'yticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], # Explicit y-ticks
        'invert_y': True
    },
    {
        'data': [39.5, 17.2, 40.1, 32.6, 44.2], 
        'std':  [0, 0, 0, 0, 0],
        'ylabel': 'Inference Speed (ms)',
        'ylim': (0, 60), 
        'yticks': [0, 10, 20, 30, 40, 50, 60], # Explicit y-ticks
        'invert_y': True
    },
]

# 3. Create figure and subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5)) 

# 4. Plotting Loop
for ax, info in zip(axes, tasks):
    data = info['data']
    stds = info['std']
    
    for i, (val, std_val, color) in enumerate(zip(data, stds, colors)):
        if val is None:
            continue
            
        # Draw the bar with error bars and black edge colors
        ax.bar(
            i, val, 
            capsize=4, 
            width=0.75, color=color, 
            edgecolor='black', linewidth=1.0, 
            error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': '#333333'}
        )
        
        # Determine text placement accounting for the error bar height
        if not info['invert_y']:
            if val < info['ylim'][0]:
                text_y = info['ylim'][0] + 1
            else:
                text_y = val + std_val + 1
        else:
            text_y = val - std_val - 1
            
            
        label_text = f'{val:.1f}'

    # Subplot Formatting
    ax.set_ylabel(info['ylabel'], fontsize=16)
    ax.set_xticks([]) 
    ax.set_ylim(info['ylim']) 
    
    # --- ADDED SPECIFIC Y-AXIS TICKS AND VISUAL TICK LINES HERE ---
    ax.set_yticks(info['yticks'])
    # Force the left y-ticks to display, length=4, width=1.2 to match spine width
    ax.tick_params(axis='y', which='major', left=True, size=4, width=1.2, labelsize=14)
        
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
    bbox_to_anchor=(0.5, 0.25), 
    ncol=5, 
    frameon=True,
    borderpad=0.8,
    framealpha=1.0,
    edgecolor='#cccccc'
)

# 6. Layout Adjustment
plt.tight_layout(rect=[0, 0.2, 1, 1])
plt.savefig("computational_cost.pdf", bbox_inches='tight', pad_inches=0)
plt.show()