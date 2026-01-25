import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data (cm) from your sweep-angle tables
# -----------------------------
angles = np.array([60, 90, 120, 150, 180], dtype=float)

# From Table: e_xy (cm) and e_s (cm)
e_xy = np.array([17.04, 1.29, 0.83, 0.87, 1.12], dtype=float)
e_s  = np.array([0.82, 0.17, 0.37, 0.51, 0.16], dtype=float)

# -----------------------------
# Plot: dual y-axes for readability
# -----------------------------
fig, ax1 = plt.subplots(figsize=(6.2, 3.2))

line1, = ax1.plot(angles, e_xy, marker='o', linewidth=1.6, label=r'$e_{xy}$ (cm)')
ax1.set_xlabel('Sweep angle (deg)')
ax1.set_ylabel(r'$e_{xy}$ (cm)')
ax1.set_xticks(angles)

# Make the big outlier still fit nicely
ax1.set_ylim(0.0, 20.0)

# Mark the default choice
ax1.axvline(120, linestyle='--', linewidth=1.0)
#ax1.text(120, 19.5, 'default', ha='center', va='top')

ax2 = ax1.twinx()
line2, = ax2.plot(angles, e_s, marker='s', linewidth=1.6, label=r'$e_s$ (cm)')
ax2.set_ylabel(r'$e_s$ (cm)')
ax2.set_ylim(0.0, 1.0)

# One combined legend
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right', frameon=True)

ax1.grid(True, linestyle=':', linewidth=0.8)
fig.tight_layout()

out_dir = "images"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "sweep_angle_errors.pdf"))
plt.savefig(os.path.join(out_dir, "sweep_angle_errors.png"), dpi=300)
print("Saved: images/sweep_angle_errors.pdf/png")
