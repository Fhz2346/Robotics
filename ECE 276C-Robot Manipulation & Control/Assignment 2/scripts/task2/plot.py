import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from init_task import *
from path_utils import densify_path
from Pathes import P

OUT_DIR = P.out_dir / "task2"
OUT_DIR.mkdir(exist_ok=True)


def plot_environment(ax):
    for x, y, w, h in RECTS:
        ax.add_patch(Rectangle((x, y), w, h, color="steelblue", alpha=0.8))
    ax.set_xlim(-22, 22); ax.set_ylim(-37, 37)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)


def wrap_angle(q):
    """Principal-angle representation in [-pi, pi)."""
    return (np.asarray(q, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def normalize_config(q):
    """Only theta1 wraps: theta2 has a hard +/-0.97*pi limit."""
    q = np.asarray(q, dtype=float).copy()
    q[..., 0] = wrap_angle(q[..., 0])
    return q


def short_delta(q0, q1):
    d = np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)
    d = d.copy()
    d[0] = wrap_angle(d[0])
    return d


def densify_wrapped_path(path, max_step_deg=2.0):
    """Interpolate every edge using the shortest theta1 angular displacement."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError('path must be an Nx2 array of joint angles in radians')
    if len(path) <= 1:
        return normalize_config(path)
    points = [normalize_config(path[0])]
    step = np.deg2rad(max_step_deg)
    for q0, q1 in zip(path[:-1], path[1:]):
        d = short_delta(q0, q1)
        n = max(1, int(np.ceil(np.max(np.abs(d)) / step)))
        for k in range(1, n + 1):
            points.append(normalize_config(q0 + (k / n) * d))
    return np.asarray(points)


def plot_wrapped_path(ax, path, color, label=None, lw=1.5, markersize=3, zorder=4):
    """Split every edge at theta1 = +/-180, not across the chart interior."""
    arr = normalize_config(np.asarray(path, dtype=float))
    if not len(arr):
        return
    degrees = np.rad2deg(arr)
    if label:
        ax.plot([], [], color=color, lw=lw, label=label, zorder=zorder)
    if len(degrees) == 1:
        ax.plot(degrees[:, 0], degrees[:, 1], '.', color=color, ms=markersize, zorder=zorder)
        return
    for p0, p1 in zip(degrees[:-1], degrees[1:]):
        dtheta = np.rad2deg(wrap_angle(np.deg2rad(p1[0] - p0[0])))
        end_x = p0[0] + dtheta
        # Show the geometrically short edge in the primary tile and, where
        # needed, its periodic continuation on the opposite side.
        for shift in (-360., 0., 360.):
            ax.plot([p0[0] + shift, end_x + shift],
                    [p0[1], p1[1]], color=color, lw=lw,
                    clip_on=True, zorder=zorder)
    ax.plot(degrees[:, 0], degrees[:, 1], '.', color=color, ms=markersize, zorder=zorder)


def plot_tree(ax, nodes, parents, color="#b5b5b5", lw=0.45, label=None, zorder=2):
    """Draw the RRT* parent tree. Edges use the short θ1 displacement."""
    nodes = np.asarray(nodes, dtype=float)
    parents = np.asarray(parents, dtype=int)
    child = np.flatnonzero(parents >= 0)
    if child.size == 0:
        return
    parent = parents[child]
    p0 = np.rad2deg(normalize_config(nodes[parent]))
    p1 = np.rad2deg(normalize_config(nodes[child]))
    end_x = p0[:, 0] + np.rad2deg(wrap_angle(np.deg2rad(p1[:, 0] - p0[:, 0])))
    chunks = []
    for shift in (-360.0, 0.0, 360.0):
        x0 = p0[:, 0] + shift
        x1 = end_x + shift
        visible = (np.maximum(x0, x1) >= -180.0) & (np.minimum(x0, x1) <= 180.0)
        if not np.any(visible):
            continue
        seg = np.empty((int(visible.sum()), 2, 2))
        seg[:, 0, 0] = x0[visible]
        seg[:, 0, 1] = p0[visible, 1]
        seg[:, 1, 0] = x1[visible]
        seg[:, 1, 1] = p1[visible, 1]
        chunks.append(seg)
    if not chunks:
        return
    ax.add_collection(LineCollection(
        np.concatenate(chunks, axis=0),
        colors=color, linewidths=lw, zorder=zorder,
    ))
    if label:
        ax.plot([], [], color=color, lw=1.4, label=label, zorder=zorder)


def save_segment_animation(start_point, goal_point, first_path, final_path, index,
                           tree_nodes=None, tree_parents=None):
    dense = densify_wrapped_path(final_path)
    if len(dense) == 0:
        return
    max_frames = 90
    if len(dense) > max_frames:
        picks = np.linspace(0, len(dense) - 1, max_frames).astype(int)
        dense = dense[picks]

    fig, (ax_xy, ax_q) = plt.subplots(1, 2, figsize=(12, 5))
    plot_environment(ax_xy)
    ax_xy.set_title(f'Cartesian space: waypoint {index} → {index + 1}')
    ax_xy.plot([start_point[0], goal_point[0]],
               [start_point[1], goal_point[1]], 'k:', alpha=0.5)
    arm_line, = ax_xy.plot([], [], 'o-', lw=3, color='darkorange')
    trace_line, = ax_xy.plot([], [], color='darkorange', lw=1.5, alpha=0.7)

    ax_q.set_xlim(-180, 180)
    ax_q.set_ylim(-180, 180)
    ax_q.set_xlabel(r'$\theta_1$ (degrees)')
    ax_q.set_ylabel(r'$\theta_2$ (degrees)')
    ax_q.set_title('Configuration space (wrapped θ₁)')
    ax_q.grid(True, alpha=0.25)
    if tree_nodes is not None and tree_parents is not None:
        plot_tree(ax_q, tree_nodes, tree_parents, label="RRT* tree")
    if first_path is not None:
        plot_wrapped_path(ax_q, first_path, 'blue', 'First feasible path', lw=1.3)
    plot_wrapped_path(ax_q, final_path, 'red', 'Optimized path', lw=1.8)
    ax_q.set_xlim(-180, 180)
    ax_q.set_ylim(-180, 180)
    q_dot, = ax_q.plot([], [], 'ko', ms=6, zorder=5)
    ax_q.legend(loc='best', fontsize=8)

    trace_x, trace_y = [], []

    def update(frame):
        q = dense[frame]
        p0, p1, p2 = forward_kinematics(q)
        arm_line.set_data([p0[0], p1[0], p2[0]],
                          [p0[1], p1[1], p2[1]])
        trace_x.append(p2[0])
        trace_y.append(p2[1])
        trace_line.set_data(trace_x, trace_y)
        q_dot.set_data([np.rad2deg(wrap_angle(q[0]))], [np.rad2deg(q[1])])
        return arm_line, trace_line, q_dot

    ani = FuncAnimation(fig, update, frames=len(dense), interval=40, blit=True)
    gif_path = OUT_DIR / "segment" / f"{index:02d}_to_{index+1:02d}.gif"
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(gif_path, writer=PillowWriter(fps=25))
    plt.close(fig)
