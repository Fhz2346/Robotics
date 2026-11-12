import matplotlib.pyplot as plt
from Config import cfgG


def plot_landmark_counts(counts):
    plt.figure(figsize=(6, 4))
    plt.hist(counts, bins=1000)
    plt.xlabel("Number of observations")
    plt.ylabel("Number of landmarks")
    plt.title("Landmark observation counts")
    plt.savefig(f"{cfgG.out_dir}/landmark_counts.png", dpi=300)


def plot_pts(pts, title="Points"):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=2)
    ax.set_title(title)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f"{cfgG.out_dir}/{title}.png", dpi=300)
    plt.show()


def mark_image(img, u, v, title="Image with marked points"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.scatter(u, v, s=8, c='r', marker='o')
    ax.set_title(title)
    plt.axis('on')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{cfgG.out_dir}/{title}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_map(traj, lm, title="Map"):
    plt.figure(figsize=(8, 8))
    plt.plot(traj[:, 0], traj[:, 1], 'b-', label='Trajectory')
    plt.scatter(lm[:, 0], lm[:, 1], s=1, c='r', label='Landmarks')
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.savefig(f"{cfgG.out_dir}/{title}.png", dpi=300)
