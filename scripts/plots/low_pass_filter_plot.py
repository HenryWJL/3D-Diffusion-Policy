import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Generate Ideal LPF Magnitude Response
# ==========================================
f_max = 1.0      # Max frequency to plot
fc = 0.1 * f_max            # Cutoff Frequency


x_min, x_max = 0, f_max
y_min, y_max = 0, 1.0

# Visual margins
y_margin = (y_max - y_min) * 0.05
x_margin = (x_max - x_min) * 0.02

# We extend the signal to the end of the axis stem so it doesn't stop early
x_stem_end = x_max + x_margin * 3.5

# Add a tiny visual offset so the stopband doesn't vanish into the x-axis
visual_offset = 0.004

# Define the coordinates of the perfect step function
f = [0, fc, fc, x_stem_end]
magnitude = [1.0, 1.0, visual_offset, visual_offset]

# ==========================================
# Setup the Plot
# ==========================================
fig, ax = plt.subplots(figsize=(3, 3))

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

# orange_color = '#63C3A0'
# orange_color = '#53A6B2'
orange_color = '#FBC02D'

# ==========================================
# Alignment and Bounds
# ==========================================
x_origin = 0
y_origin = 0

ax.set_xlim(x_origin, x_max + x_margin * 5)
ax.set_ylim(y_origin - y_margin * 0.5, y_max + y_margin * 5)

# ==========================================
# Solid Arrows & Perfect Origin Corner 
# ==========================================
# 1. The Stems
ax.plot([x_origin, x_origin, x_stem_end], 
        [y_max + y_margin * 3.5, y_origin, y_origin], 
        color=orange_color, lw=3, solid_joinstyle='miter', zorder=1)

# 2. The Arrowheads
arrow_props = dict(arrowstyle="-|>", color=orange_color, lw=3, mutation_scale=20, shrinkA=0, shrinkB=0)

ax.annotate('', xy=(x_max + x_margin * 5, y_origin), xytext=(x_origin, y_origin),
            arrowprops=arrow_props, annotation_clip=False, zorder=2)

ax.annotate('', xy=(x_origin, y_max + y_margin * 4.5), xytext=(x_origin, y_origin),
            arrowprops=arrow_props, annotation_clip=False, zorder=2)

# ==========================================
# Label: Italic 'f'
# ==========================================
ax.text(x_max + x_margin * 4, y_origin - y_margin * 1.5, 'f', 
        color=orange_color, fontsize=16, fontweight='bold', 
        ha='center', va='top', fontname='Times New Roman', fontstyle='italic')

# ==========================================
# 2. Draw the Ideal Filter Response
# ==========================================
# The signal now drops to 'visual_offset' instead of 0, so it rests distinctly above the axis
ax.plot(f, magnitude, color=orange_color, linewidth=3.5, solid_joinstyle='miter', zorder=3)

plt.tight_layout()
plt.savefig('lp_filter.svg', transparent=True, dpi=300)
plt.show()