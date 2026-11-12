from itertools import product

from init_task import *
from plot import *
from RRTStar import RRTStar
from path_utils import shortcut_smooth

WAYPOINTS = np.array(
    [
        [0.0, 20.0],
        [0.0, -20.0],
        [-19.43, 4.62],
        [2.0, 2.0],
        [-8.0, 0.0],
        [-5.0, -5.0],
        [-9.0, 0.0],
    ]
)


def find_ik_candidates(points):
    ik_candidates = []
    for i, point in enumerate(points):
        candidates = [q for q in inverse_kinematics(point) if Cs.is_point_valid(q)]
        if not candidates:
            raise RuntimeError(
                f"Point {i+1} {tuple(map(float, point))} has no collision-free IK solution. "
                "Check URDF joint/link geometry and obstacle definitions."
            )
        ik_candidates.append(candidates)
    return ik_candidates


if __name__ == "__main__":
    ik_candidates = find_ik_candidates(WAYPOINTS)
    for i, candidates in enumerate(ik_candidates, 1):
        print(f"Waypoint {i}: {len(candidates)} collision-free IK solution(s)")
    all_paths, all_first_paths, all_trees, selected_q = [], [], [], []
    # For the first segment, compare both IK branches at the first waypoint and
    # both branches at the second waypoint, then keep the shortest feasible path.
    q_start_options = ik_candidates[0]
    for segment in range(len(WAYPOINTS) - 1):
        q_goals = ik_candidates[segment + 1]
        q_starts = q_start_options if segment == 0 else [current_q]
        best = None
        pair_count = len(q_starts) * len(q_goals)
        pair_number = 0
        for q_start, q_goal in product(q_starts, q_goals):
            pair_number += 1
            print(f"Planning segment {segment+1}/{len(WAYPOINTS)-1}, IK pair {pair_number}/{pair_count}...")
            planner_rng = np.random.default_rng(7000 + segment * 101 + pair_number)
            planner = RRTStar(q_start, q_goal, planner_rng)
            first_path, optimized_path, cost, iterations = planner.plan()
            if first_path is None or optimized_path is None:
                print(f"  No path found within {iterations} iterations.")
                continue
            if best is None or cost < best[0]:
                smooth = shortcut_smooth(optimized_path, attempts=120, rng=planner_rng)
                tree_nodes = np.asarray(planner.nodes, dtype=float).copy()
                tree_parents = np.asarray(planner.parents, dtype=int).copy()
                best = (cost, first_path, smooth, q_start.copy(), q_goal.copy(), iterations,
                        tree_nodes, tree_parents)
            print(f"  Feasible path cost={cost:.3f} rad, samples={iterations}")
        if best is None:
            raise RuntimeError(
                f"RRT* did not find a feasible path for waypoint {segment+1} -> {segment+2}. "
                "Try increasing MAX_ITERATIONS or STEP_SIZE."
            )
        _, first_path, final_path, start_q, goal_q, iterations, tree_nodes, tree_parents = best
        save_segment_animation(
            WAYPOINTS[segment], WAYPOINTS[segment + 1], first_path, final_path, segment + 1,
            tree_nodes, tree_parents,
        )
        # Store the first feasible path (blue) and optimized/smoothed path (red).
        all_first_paths.append(first_path)
        all_paths.append(final_path)
        all_trees.append((tree_nodes, tree_parents))
        selected_q.append((start_q, goal_q))
        current_q = goal_q
        start_options = [current_q]
        print(f"Selected segment {segment+1}: {len(first_path)} initial nodes -> {len(final_path)} smoothed nodes")

    # Save numeric trajectories for reproducibility and later plotting.
    np.savez_compressed(
        OUT_DIR / "paths.npz",
        waypoints=WAYPOINTS,
        first_paths=np.array(all_first_paths, dtype=object),
        final_paths=np.array(all_paths, dtype=object),
        tree_nodes=np.array([nodes for nodes, _ in all_trees], dtype=object),
        tree_parents=np.array([parents for _, parents in all_trees], dtype=object),
    )
    with (OUT_DIR / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("ECE276C Assignment 2 - Task 2 RRT* summary\n")
        f.write(f"Waypoints: {len(WAYPOINTS)}\nSegments planned: {len(all_paths)}\n")
        for i, path in enumerate(all_paths, 1):
            length = sum(As.distance(a, b) for a, b in zip(path[:-1], path[1:]))
            f.write(f"Segment {i}: nodes={len(path)}, joint-space path length={length:.6f} rad\n")

    # Final overview image: Cartesian waypoints and joint-space paths.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plot_environment(ax1)
    ax1.plot(WAYPOINTS[:, 0], WAYPOINTS[:, 1], "ko--", ms=4, label="Waypoints")
    for i, path in enumerate(all_paths):
        xy = np.array([forward_kinematics(q)[2] for q in densify_path(path)])
        ax1.plot(xy[:, 0], xy[:, 1], lw=1.8, label=f"Segment {i+1}")
    ax1.set_title("Cartesian paths"); ax1.legend(fontsize=7, ncol=2)
    ax2.set_xlim(-180, 180); ax2.set_ylim(-180, 180)
    ax2.set_xlabel(r"$\theta_1$ (degrees)"); ax2.set_ylabel(r"$\theta_2$ (degrees)")
    ax2.set_title("Configuration-space paths")
    ax2.grid(True, alpha=0.25)
    for i, ((nodes, parents), first, final) in enumerate(zip(all_trees, all_first_paths, all_paths)):
        first_label = "RRT* tree" if i == 0 else None
        plot_tree(ax2, nodes, parents, label=first_label)
        plot_wrapped_path(ax2, first, "blue", "First feasible path" if i == 0 else None, lw=1.0)
        plot_wrapped_path(ax2, final, "red", "Optimized path" if i == 0 else None, lw=1.6)
    ax2.set_xlim(-180, 180); ax2.set_ylim(-180, 180)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "path_overview.png", dpi=180)
    plt.close(fig)
    print(f"\nFinished. Results saved in: {OUT_DIR}")
    print("Files: summary.txt, paths.npz, path_overview.png, segment_*.gif")
