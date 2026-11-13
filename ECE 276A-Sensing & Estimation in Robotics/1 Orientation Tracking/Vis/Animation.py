import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import art3d

from transforms3d.euler import euler2mat

# ========== 1) rotplot-style box, but returns artists so we can update ==========
def _make_box_vertices(scale=(3.0, 1.5, 1.0)):
    lx, ly, lz = scale
    x = 0.5 * np.array(
        [
            [+lx, -lx, +lx, -lx, +lx, -lx, +lx, -lx],
            [+ly, +ly, -ly, -ly, +ly, +ly, -ly, -ly],
            [+lz, +lz, +lz, +lz, -lz, -lz, -lz, -lz],
        ],
        dtype=float,
    )
    return x

def draw_box(ax, R, scale=(3.0, 1.5, 1.0), front_color="tab:blue", back_color="tab:red", alpha=0.35):
    x = _make_box_vertices(scale)
    xp = R @ x

    ifront = np.array([0, 2, 6, 4, 0])
    iback  = np.array([1, 3, 7, 5, 1])
    itop   = np.array([0, 1, 3, 2, 0])
    ibot   = np.array([4, 5, 7, 6, 4])

    # edges
    (line_top,) = ax.plot(xp[0, itop], xp[1, itop], xp[2, itop], "k-", lw=1)
    (line_bot,) = ax.plot(xp[0, ibot], xp[1, ibot], xp[2, ibot], "k-", lw=1)

    # faces
    front = art3d.Poly3DCollection(
        [list(zip(xp[0, ifront], xp[1, ifront], xp[2, ifront]))],
        facecolor=front_color,
        alpha=alpha,
    )
    back = art3d.Poly3DCollection(
        [list(zip(xp[0, iback], xp[1, iback], xp[2, iback]))],
        facecolor=back_color,
        alpha=alpha,
    )
    ax.add_collection3d(front)
    ax.add_collection3d(back)

    # store indices + scale in artists for update
    meta = dict(scale=scale, ifront=ifront, iback=iback, itop=itop, ibot=ibot)
    return (line_top, line_bot, front, back, meta)

def update_box(R, artists):
    line_top, line_bot, front, back, meta = artists
    x = _make_box_vertices(meta["scale"])
    xp = R @ x

    itop, ibot = meta["itop"], meta["ibot"]
    ifront, iback = meta["ifront"], meta["iback"]

    line_top.set_data(xp[0, itop], xp[1, itop])
    line_top.set_3d_properties(xp[2, itop])

    line_bot.set_data(xp[0, ibot], xp[1, ibot])
    line_bot.set_3d_properties(xp[2, ibot])

    front.set_verts([list(zip(xp[0, ifront], xp[1, ifront], xp[2, ifront]))])
    back.set_verts([list(zip(xp[0, iback],  xp[1, iback],  xp[2, iback]))])

# ========== 2) convert RPY -> rotation matrices ==========
def rpy_to_R_seq(RPY, axes="sxyz"):
    """
    RPY: (N,3) roll,pitch,yaw in radians
    returns: (N,3,3)
    """
    N = RPY.shape[0]
    R = np.zeros((N, 3, 3), dtype=float)
    for k in range(N):
        r, p, y = RPY[k]
        # transforms3d uses 'axes' to interpret order
        R[k] = euler2mat(r, p, y, axes=axes)
    return R

# ========== 3) main animation ==========
def animate_two_rot_paths(ts, RPY_gt, RPY_est, axes="sxyz", out_mp4="rot_compare.mp4",
                          stride=10, fps=30):
    """
    Draw two rotating boxes + their body-x-axis tip paths, then save mp4.
    stride: downsample frames to speed up rendering
    """
    # downsample
    ts = ts[::stride]
    RPY_gt = RPY_gt[::stride]
    RPY_est = RPY_est[::stride]

    R_gt = rpy_to_R_seq(RPY_gt, axes=axes)
    R_est = rpy_to_R_seq(RPY_est, axes=axes)

    N = ts.shape[0]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Rotation paths (GT vs EST)")
    ax.set_xlim3d(-2, 2); ax.set_ylim3d(-2, 2); ax.set_zlim3d(-2, 2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # draw two boxes
    artists_gt  = draw_box(ax, R_gt[0],  front_color="tab:blue",   back_color="tab:cyan", alpha=0.35)
    artists_est = draw_box(ax, R_est[0], front_color="tab:orange", back_color="tab:red",  alpha=0.35)

    # trajectory: use body x-axis tip (vector [1.5,0,0]) in world
    tip_len = 1.5
    gt_tip  = np.array([R_gt[k]  @ np.array([tip_len, 0, 0]) for k in range(N)])
    est_tip = np.array([R_est[k] @ np.array([tip_len, 0, 0]) for k in range(N)])

    (gt_traj,)  = ax.plot([gt_tip[0,0]],  [gt_tip[0,1]],  [gt_tip[0,2]],  lw=2, label="GT axis-tip path")
    (est_traj,) = ax.plot([est_tip[0,0]], [est_tip[0,1]], [est_tip[0,2]], lw=2, label="EST axis-tip path")
    ax.legend(loc="upper left")

    def _anim(k):
        update_box(R_gt[k], artists_gt)
        update_box(R_est[k], artists_est)
        gt_traj.set_data(gt_tip[:k+1, 0], gt_tip[:k+1, 1]);   gt_traj.set_3d_properties(gt_tip[:k+1, 2])
        est_traj.set_data(est_tip[:k+1, 0], est_tip[:k+1, 1]); est_traj.set_3d_properties(est_tip[:k+1, 2])
        return ()

    ani = animation.FuncAnimation(fig, _anim, frames=N, interval=1000/fps, blit=False)

    # save
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(out_mp4, writer=writer)
    plt.close(fig)
    print(f"[OK] saved: {out_mp4}")