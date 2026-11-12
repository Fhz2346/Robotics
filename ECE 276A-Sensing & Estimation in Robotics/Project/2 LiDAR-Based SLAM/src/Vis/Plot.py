import matplotlib.pyplot as plt
import numpy as np
from Config import cfgG


def save_traj(traj):
    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, 0], traj[:, 1])
    plt.axis("equal")
    plt.title("Robot Trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.savefig(f"{cfgG.out_dir}/traj.png", dpi=300)


def save_pc(P0, P1, title, dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(P0[:, 0], P0[:, 1], s=5, c="blue", label="P0")
    plt.scatter(P1[:, 0], P1[:, 1], s=5, c="red", label="P1")
    plt.axis("equal")
    plt.legend()
    plt.title(title)
    plt.savefig(f"{dir}/{title}.png", dpi=300)


def save_Theta(Theta, T):
    plt.figure()
    plt.plot(T, Theta)  # x轴用时间戳更直观
    plt.xlabel("time (s)")
    plt.ylabel("theta (rad)")
    plt.title("Theta over time")
    plt.tight_layout()
    plt.savefig(f"{cfgG.out_dir}/theta.png", dpi=300)


def save_map(prob):
    plt.figure(figsize=(6, 6))
    plt.imshow(prob.T, origin="lower", cmap="gray")
    plt.title("Occupancy Grid Map")
    plt.savefig(f"{cfgG.out_dir}/map.png", dpi=300)


def show_pc3D(pc, colors, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    idx = np.random.choice(len(pc), size=min(20000, len(pc)), replace=False)
    ax.scatter(pc[idx, 0], pc[idx, 1], pc[idx, 2], c=colors[idx] / 255.0, s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(title)
    plt.show()


def save_Cmap(Cmap):
    plt.figure(figsize=(6, 6))
    plt.imshow(Cmap, origin="lower")
    plt.title("Colored Grid Map")
    plt.savefig(f"{cfgG.out_dir}/colored_map.png", dpi=300)
