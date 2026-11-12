from omegaconf import OmegaConf
from robot.pybullet import *
from control_payload import *
from save import save_gif


def simulate_one(case_name, q_ref, gravity_torque_fn, cfg):
    set_q_dq(q_ref)
    frames = []
    max_q_error = 0.0
    max_dq = 0.0
    nsteps = int(round(cfg.duration / DT))
    for step in range(nsteps):
        q, dq = get_q_dq()
        tau = gravity_torque_fn(q)
        # Re-issue torque every step to make the applied command explicit.
        p.setJointMotorControlArray(robot, joint_ids, p.TORQUE_CONTROL, forces=tau.tolist())
        p.stepSimulation()
        states = [p.getJointState(robot, jid) for jid in joint_ids]
        q = np.array([s[0] for s in states])
        dq = np.array([s[1] for s in states])
        max_q_error = max(max_q_error, float(np.max(np.abs(q - q_ref))))
        max_dq = max(max_dq, float(np.max(np.abs(dq))))
        if step % cfg.record_every == 0:
            frames.append(fk_full(q))
    final_states = [p.getJointState(robot, jid) for jid in joint_ids]
    q_final = np.array([s[0] for s in final_states])
    dq_final = np.array([s[1] for s in final_states])
    return {
        "case": case_name,
        "q_ref": np.asarray(q_ref),
        "tau": tau,
        "q_final": q_final,
        "dq_final": dq_final,
        "max_q_error": max_q_error,
        "max_dq": max_dq,
        "frames": frames,
    }


if __name__ == "__main__":
    CASES = [
        ("Case 1", np.array([0.0, 0.0, np.pi / 2])),
        ("Case 2", np.array([np.pi / 2, np.pi / 2, np.pi / 4])),
        ("Case 3", np.array([np.pi / 3, np.pi / 2, 0.0])),
    ]
    methods = {
        "analytical": get_gravity_torque_analytical,
        "finite difference": get_gravity_torque_fd,
    }
    cfg = OmegaConf.load(P.cfg_dir / "task4.yaml")
    for method_name, gravity_torque_fn in methods.items():
        print(method_name)
        results = []
        for case_name, q_ref in CASES:
            result = simulate_one(case_name, q_ref, gravity_torque_fn, cfg)
            if result["max_q_error"] < 0.1:
                print(f"{case_name} passed with max_q_error = {result['max_q_error']}")
            else:
                print(f"{case_name} failed with max_q_error = {result['max_q_error']}")
            results.append(result)
        save_path = P.out_dir / "Task4" / "payload" / f"static_hold_{method_name}.gif"
        save_gif(results, save_path)
    p.disconnect(cid)
