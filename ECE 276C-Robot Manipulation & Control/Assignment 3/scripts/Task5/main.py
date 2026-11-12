from omegaconf import OmegaConf
from robot.pybullet import *
from control import *
from save import save_outputs


def run_simulation(targets, cfg):
    records = []
    animation_frames = []
    target_events = []
    t = 0.0
    # Initial configuration for Location 1.
    q0 = np.array([np.pi / 4, np.pi / 4, -np.pi / 2], dtype=float)
    set_q(q0)
    x0 = get_p_tip()
    print("Initial IK q (rad):", q0)
    print("Initial tip position (m):", x0)
    print("Initial target (m):", targets[0])

    if np.linalg.norm(x0 - targets[0]) > 0.015:
        raise RuntimeError(
            "Initial target 1 is not reached by IK. "
            "Check the URDF tip offset / target reachability."
        )

    # No path interpolation: each target is a direct Cartesian setpoint.
    total_steps = int(cfg.max_total_time / DT)
    target_index = 0
    settle_count = 0
    segment_start_time = 0.0
    reached = False

    for step in range(total_steps):
        tau = get_torque(targets[target_index], jacobian_and_gravity_analytical)
        # tau = get_torque(targets[target_index], jacobian_and_gravity_fd)
        p.setJointMotorControlArray(robot, joint_ids, p.TORQUE_CONTROL, forces=tau.tolist())
        p.stepSimulation()
        t += DT
        # Read state after physics integration.
        q_new, dq_new = get_q_dq()
        x_new = get_p_tip()
        err_new = float(np.linalg.norm(targets[target_index] - x_new))
        records.append([
            t, target_index + 1,
            x_new[0], x_new[1], x_new[2],
            targets[target_index, 0],
            targets[target_index, 1],
            targets[target_index, 2],
            err_new,
            q_new[0], q_new[1], q_new[2],
            tau[0], tau[1], tau[2],
        ])

        if step % cfg.frame_every == 0:
            animation_frames.append(
                {
                    "points": get_polyline(),
                    "tip": x_new.copy(),
                    "target": targets[target_index].copy(),
                    "target_index": target_index + 1,
                    "time": t,
                }
            )

        if err_new < cfg.pos_tol:
            settle_count += 1
        else:
            settle_count = 0

        if settle_count >= cfg.settle_steps:
            target_events.append({
                "target": target_index + 1,
                "time": t,
                "error": err_new,
            })
            if target_index == len(targets) - 1:
                reached = True
                break
            target_index += 1
            settle_count = 0
            segment_start_time = t

        elif t - segment_start_time > cfg.max_segment_time:
            target_events.append({
                "target": target_index + 1,
                "time": t,
                "error": err_new,
                "timeout": True,
            })
            if target_index == len(targets) - 1:
                break
            target_index += 1
            settle_count = 0
            segment_start_time = t

    return records, animation_frames, target_events, reached

TARGETS = np.array(
    [
        [0.2, 0.2, 0.2],
        [0.1, 0.2, 0.2],
        [0.1, -0.2, 0.2],
        [0.2, -0.2, 0.2],
        [0.2, 0.2, 0.2],
    ],
    dtype=float,
)

if __name__ == "__main__":
    cfg = OmegaConf.load(P.cfg_dir / "task5.yaml")
    records, frames, events, reached = run_simulation(TARGETS, cfg)
    p.disconnect(cid)
    save_outputs(records, frames, events, reached, TARGETS)
