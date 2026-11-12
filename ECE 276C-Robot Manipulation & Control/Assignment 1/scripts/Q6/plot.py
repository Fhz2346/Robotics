import numpy as np
from init_task import *
import matplotlib.pyplot as plt


def plot_results(times, measured, path):
    desired = np.vstack([reference(time)[0] for time in times])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(desired[:, 0], desired[:, 1], color="#94a3b8", lw=4.0, label="reference", zorder=1)
    ax.plot(measured[:, 0], measured[:, 1], color="#b91c1c", lw=1.6, label="end effector", zorder=2)
    ax.scatter(WAYPOINTS[:-1, 0], WAYPOINTS[:-1, 1], s=60, c="#1d4ed8", zorder=3)
    for index, point in enumerate(WAYPOINTS[:-1], start=1):
        ax.annotate(str(index), (point[0], point[1]), xytext=(6, 6), textcoords="offset points", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Closed-loop end-effector path")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return np.max(np.linalg.norm(measured - desired, axis=1))
