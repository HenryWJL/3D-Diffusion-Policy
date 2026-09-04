import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    'ytick.labelsize': 10,
    
    # --- MAKE ALL BORDERS AND TICKS BOLDER HERE ---
    'axes.linewidth': 1.5,       # Makes all 4 outer borders distinct and bold
    'xtick.major.width': 1.5,    
    'ytick.major.width': 1.5,    # Matches the y-tick thickness to the border
})

# Data
tasks = ['Cup', 'Mouse']
dp3_success = [84, 48]
fgo_success = [96, 56]

# Setup label positions and bar width
x = np.arange(len(tasks))  
width = 0.3

fig, ax = plt.subplots(figsize=(5, 5))

# Colors sampled from your reference images
gray_color = '#d3d3d3'    
orange_color = '#f8b16a'  

# Plot bars
rects1 = ax.bar(x - width/2, dp3_success, width, label='DP3', 
                color=gray_color, linewidth=1.2, alpha=1)
rects2 = ax.bar(x + width/2, fgo_success, width, label='DP3 + FGO', 
                color=orange_color, linewidth=1.2, alpha=1)

# Labels and ticks
ax.set_ylabel('Success Rate (%)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=16)

# Force all four borders to be explicitly visible
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_visible(True)

# Set specific y-ticks
ax.set_yticks([0, 20, 40, 60, 80, 100])

# --- X TICK FIX: bottom=False removes the ticks, left=True keeps the y ticks ---
ax.tick_params(axis='both', which='major', bottom=False, left=True, size=5, labelsize=16)

ax.set_ylim(0, 110)  
leg = ax.legend(frameon=True, fontsize=12)
for patch in leg.get_patches():
    patch.set_edgecolor('none')

# Subtle gridlines behind the bars
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

sns.despine()  # Removes top and right borders instantly
plt.tight_layout()

# Save updated chart
plt.savefig('success_rates.pdf', bbox_inches='tight', pad_inches=0)
plt.show()