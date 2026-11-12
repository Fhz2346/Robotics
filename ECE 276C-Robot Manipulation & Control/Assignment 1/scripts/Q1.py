
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# ========================================
# 1. Robot parameters
# ========================================

l1 = 2.4
l2 = 2.0
l3 = 1.55

# Joint angles in degrees
theta1 = 32
theta2 = -58
theta3 = 46

# Convert degrees to radians
angles = np.deg2rad([theta1, theta2, theta3])

# Absolute link orientations
a = np.cumsum(angles)

# ========================================
# 2. Forward kinematics
# ========================================

points = np.array([[0.0, 0.0]])

for L, t in zip([l1, l2, l3], a):
    next_point = points[-1] + L * np.array([
        np.cos(t),
        np.sin(t)
    ])
    points = np.vstack([points, next_point])

# ========================================
# 3. Create figure
# ========================================

fig, ax = plt.subplots(figsize=(11, 6.4))

colors = ["#2563eb", "#0e9f6e", "#d97706"]

# Draw robot links
for i in range(3):
    ax.plot(
        points[i:i+2, 0],
        points[i:i+2, 1],
        linewidth=7,
        color=colors[i],
        solid_capstyle="round",
        zorder=2
    )

    mid = (points[i] + points[i+1]) / 2

    ax.text(
        mid[0], mid[1] + 0.20,
        rf"$l_{i+1}$",
        fontsize=16,
        ha="center"
    )

# Draw joints and end effector
ax.scatter(
    points[:, 0],
    points[:, 1],
    s=[135, 110, 110, 120],
    c=["#1d4ed8"] * 3 + ["#b91c1c"],
    zorder=5,
    edgecolors="white",
    linewidths=1.5
)

labels = ["J1 / O0", "J2 / O2", "J3 / O3", "EE"]

for i, p in enumerate(points):
    ax.annotate(
        labels[i],
        p,
        xytext=(5, -21),
        textcoords="offset points",
        fontsize=10
    )

# ========================================
# 4. Draw body-fixed coordinate frames
# ========================================

for i, (p, t) in enumerate(zip(points[:3], a)):

    # Local x direction
    u = np.array([
        np.cos(t),
        np.sin(t)
    ])

    # Local y direction
    v = np.array([
        -np.sin(t),
        np.cos(t)
    ])

    scale = 0.53

    # x-axis (red)
    ax.annotate(
        "",
        p + u * scale,
        p,
        arrowprops=dict(
            arrowstyle="->",
            color="#b91c1c",
            lw=1.7
        )
    )

    # y-axis (purple)
    ax.annotate(
        "",
        p + v * scale,
        p,
        arrowprops=dict(
            arrowstyle="->",
            color="#7c3aed",
            lw=1.7
        )
    )

    ax.text(
        *(p + u * (scale + 0.10)),
        rf"$x_{i+1}$",
        color="#b91c1c",
        fontsize=11
    )

    ax.text(
        *(p + v * (scale + 0.10)),
        rf"$y_{i+1}$",
        color="#7c3aed",
        fontsize=11
    )

# ========================================
# 5. Draw world coordinate frame
# ========================================

ax.annotate(
    "", (0.95, 0), (0, 0),
    arrowprops=dict(
        arrowstyle="->",
        lw=1,
        linestyle="dashed",
        color="gray"
    )
)

ax.text(0.98, -0.10, "world x", color="gray")

ax.annotate(
    "", (0, 0.8), (0, 0),
    arrowprops=dict(
        arrowstyle="->",
        lw=1,
        linestyle="dashed",
        color="gray"
    )
)

ax.text(-0.1, 0.9, "world y", color="gray")

# ========================================
# 6. Draw joint angle arcs
# ========================================

for i, (p, ta) in enumerate(zip(points[:3], a)):

    start = 0 if i == 0 else np.rad2deg(a[i-1])
    end = np.rad2deg(ta)

    if abs(end - start) > 5:

        arc = Arc(
            p,
            1.0, 1.0,
            theta1=min(start, end),
            theta2=max(start, end),
            color="#333333",
            lw=1.25
        )

        ax.add_patch(arc)

        mid_angle = np.deg2rad((start + end) / 2)

        label_pos = p + 0.72 * np.array([
            np.cos(mid_angle),
            np.sin(mid_angle)
        ])

        ax.text(
            *label_pos,
            rf"$\theta_{i+1}$",
            fontsize=12
        )

# ========================================
# 7. Final plot settings
# ========================================

ax.set_aspect("equal")

ax.set_xlim(-0.85, 6.5)
ax.set_ylim(-1.2, 3)

ax.set_xlabel("X")
ax.set_ylabel("Y")

ax.set_title(
    "Three-link planar robot: body-fixed frames",
    fontsize=14
)

ax.grid(alpha=0.15)

fig.tight_layout()

# Save high-resolution image
plt.savefig(
    "three_link_robot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Print end-effector position
print("End effector position:", points[-1])
print("End effector orientation:", np.rad2deg(a[-1]))
