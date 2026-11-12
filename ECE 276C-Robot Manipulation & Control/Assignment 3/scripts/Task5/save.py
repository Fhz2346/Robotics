import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from Pathes import P


out_dir = P.out_dir / "Task5"
def save_outputs(records, frames, events, reached, targets):
    # CSV
    header = [
        "time_s", "target_index",
        "x_m", "y_m", "z_m",
        "x_ref_m", "y_ref_m", "z_ref_m",
        "position_error_m",
        "q1_rad", "q2_rad", "q3_rad",
        "tau1_Nm", "tau2_Nm", "tau3_Nm",
    ]
    data = np.asarray(records, dtype=float)
    t = data[:, 0]
    xyz = data[:, 2:5]

    # 2D XY plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(xyz[:, 0], xyz[:, 1], label="Actual tip trajectory")
    ax.scatter(targets[:, 0], targets[:, 1], marker="x", s=80, label="Cartesian setpoints")
    for i, pt in enumerate(targets, start=1):
        ax.annotate(str(i), (pt[0], pt[1]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Task 5 - End-effector XY trajectory")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tip_xy.png", dpi=180)
    plt.close(fig)

    # XYZ vs time plot (three curves in one 2D time-series figure)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, xyz[:, 0], label="X")
    ax.plot(t, xyz[:, 1], label="Y")
    ax.plot(t, xyz[:, 2], label="Z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tip position (m)")
    ax.set_title("Task 5 - End-effector position vs time")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tip_xyz.png", dpi=180)
    plt.close(fig)

    # 3D robot GIF
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 0.65)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Task 5 - Cartesian stiffness control")
    ax.view_init(elev=25, azim=45)

    robot_line, = ax.plot([], [], [], "o-", linewidth=4, markersize=6)
    tip_trace, = ax.plot([], [], [], "--", linewidth=1.5)
    info = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    trace = []

    def update(i):
        fr = frames[i]
        pts = fr["points"]
        tip = fr["tip"]
        trace.append(tip)
        robot_line.set_data(pts[:, 0], pts[:, 1])
        robot_line.set_3d_properties(pts[:, 2])
        tr = np.asarray(trace)
        tip_trace.set_data(tr[:, 0], tr[:, 1])
        tip_trace.set_3d_properties(tr[:, 2])
        target = fr["target"]
        info.set_text(
            f"Target {fr['target_index']} | t={fr['time']:.2f}s\n"
            f"ref=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
        )
        return robot_line, tip_trace, info

    anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False)
    anim.save(out_dir / "robot.gif", writer=PillowWriter(fps=25))
    plt.close(fig)

    print("\nTarget arrival summary:")
    for e in events:
        status = "TIMEOUT" if e.get("timeout", False) else "SETTLED"
        print(
            f"Target {e['target']}: {status}, "
            f"time={e['time']:.3f}s, error={e['error']:.5f}m"
        )
    print("\nAll targets settled:", reached)
