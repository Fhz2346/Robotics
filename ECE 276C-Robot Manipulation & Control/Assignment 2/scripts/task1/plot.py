import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
from init_task import *
from Pathes import P


def plot_results(t1, t2, grid):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    # plot the cartesian workspace and obstacles
    ax = axes[0]
    for x, y, w, h in RECTS:
        ax.add_patch(Rectangle((x, y), w, h, facecolor='#567bbd', edgecolor='black'))
    ax.scatter([0], [0], s=45, c='black', zorder=3)
    ax.text(1, 1, 'Base (0,0)', fontsize=10)
    W = F + B + E
    H = C
    ax.set_xlim(-W, W)
    ax.set_ylim(-H, H)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Cartesian workspace and obstacles')
    # plot the configuration space
    ax = axes[1]
    cmap = ListedColormap(['#d4edda', '#e66b6b', '#b4b4b4'])
    ax.imshow(
        grid,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=[-180.5, 179.5, -180.5, 180.5],
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
    )
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_xlabel(r'$\theta_1$ (degrees)')
    ax.set_ylabel(r'$\theta_2$ (degrees)')
    ax.set_title('Configuration space (1° resolution)')
    ax.legend(
        handles=[
            Patch(color="#d4edda", label="Collision free"),
            Patch(color="#e66b6b", label="Collision"),
            Patch(color="#b4b4b4", label="Outside joint limit"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.savefig(P.out_dir / "task1" / "configuration_space.png", dpi=200)
    print('Saved task1/configuration_space.png')
    plt.show()
