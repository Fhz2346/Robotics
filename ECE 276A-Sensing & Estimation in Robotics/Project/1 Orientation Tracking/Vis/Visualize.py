import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def plot_rpy(ts, rpy_gt, rpy_est, title_prefix=""):
    rpy_gt_u = np.column_stack([rpy_gt[:, i] for i in range(3)])
    rpy_est_u = np.column_stack([rpy_est[:, i] for i in range(3)])

    rpy_gt_deg = np.rad2deg(rpy_gt_u)
    rpy_est_deg = np.rad2deg(rpy_est_u)

    t = ts - ts[0]
    labels = ["Roll", "Pitch", "Yaw"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    for i in range(3):
        axes[i].plot(t, rpy_gt_deg[:, i], label="Ground Truth (VICON)")
        axes[i].plot(t, rpy_est_deg[:, i], label="Estimated (IMU gyro integration)")
        axes[i].set_ylabel(f"{labels[i]} (deg)")
        axes[i].grid(True)
        if i == 0:
            axes[i].set_title(f"{title_prefix}GT vs Estimated RPY")

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_RPY_v(ts_v, RPY_v, unwrap=True, deg=True, title="VICON Euler angles"):
    ts_v = ts_v.reshape(-1)
    E = RPY_v.copy()

    if unwrap:
        E = np.column_stack([np.unwrap(E[:, i]) for i in range(3)])

    if deg:
        E = np.rad2deg(E)

    t = ts_v - ts_v[0]
    labels = ["Roll", "Pitch", "Yaw"]
    yunit = "deg" if deg else "rad"

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i in range(3):
        ax[i].plot(t, E[:, i])
        ax[i].set_ylabel(f"{labels[i]} ({yunit})")
        ax[i].grid(True)

    ax[0].set_title(title)
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_mask_v(ts_v, mask_v, dist=None, title="Stationary Mask from VICON"):
    ts_v = ts_v.reshape(-1)
    mask_v = mask_v.astype(bool)
    t = ts_v - ts_v[0]

    if dist is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2.5))
        ax.plot(t, mask_v.astype(int))
        ax.set_yticks([0, 1])
        ax.set_ylabel("mask_v")
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return

    # two-panel plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # 1) dist curve
    ax[0].plot(t, dist)
    ax[0].set_ylabel("dist (rad)")
    ax[0].grid(True)
    ax[0].set_title(title)

    # 2) mask plot as 0/1
    ax[1].plot(t, mask_v.astype(int))
    ax[1].set_yticks([0, 1])
    ax[1].set_ylabel("mask_v")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True)

    # highlight True regions
    ax[1].fill_between(t, 0, 1, where=mask_v, alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_W(ts_i, W, mask_w=None, title="Gyro W", unit="rad/s"):
    ts_i = ts_i.reshape(-1)
    W = np.asarray(W)
    t = ts_i - ts_i[0]

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    labels = ["Wx", "Wy", "Wz"]
    for i in range(3):
        ax[i].plot(t, W[:, i])
        ax[i].set_ylabel(f"{labels[i]} ({unit})")
        ax[i].grid(True)
        if mask_w is not None:
            ax[i].fill_between(t, W[:, i].min(), W[:, i].max(), where=mask_w, alpha=0.15)

    ax[0].set_title(title)
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_A(ts_i, A, mask_a=None, title="Accel A", unit="g"):
    ts_i = ts_i.reshape(-1)
    A = np.asarray(A)
    t = ts_i - ts_i[0]

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    labels = ["Ax", "Ay", "Az"]
    for i in range(3):
        ax[i].plot(t, A[:, i])
        ax[i].set_ylabel(f"{labels[i]} ({unit})")
        ax[i].grid(True)
        if mask_a is not None:
            ax[i].fill_between(t, A[:, i].min(), A[:, i].max(), where=mask_a, alpha=0.15)

    ax[0].set_title(title)
    ax[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_A_norm(ts_i, A_norm, title="Accel A norm", unit="g"):
    plt.figure()
    plt.plot(ts_i, A_norm)
    plt.xlabel("t (s)")
    plt.ylabel(f"||A(t)|| ({unit})")
    plt.title(title)
    plt.show()


def plot_compare_A(ts_i, A_i, A_v, title="A (IMU) vs A (VICON)"):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    labels = ["Ax", "Ay", "Az"]
    for k in range(3):
        axs[k].plot(ts_i, A_i[:, k], label="A (IMU)", alpha=0.5)
        axs[k].plot(ts_i, A_v[:, k + 1], label="A (VICON)", alpha=0.5)
        axs[k].set_ylabel(labels[k] + " (g)")
        axs[k].grid(True)
        if k == 0:
            axs[k].legend()

    axs[2].set_xlabel("time")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
