from .robot import *
from .init_sim import *


def set_q(q):
    for jid, angle in zip(joint_ids, q):
        dq = p.getJointState(robot, jid)[1]
        p.resetJointState(robot, jid, float(angle), targetVelocity=dq)


def set_q_dq(q, dq=np.zeros(N)):
    for jid, qi, dqi in zip(joint_ids, q, dq):
        p.resetJointState(robot, jid, float(qi), targetVelocity=float(dqi))


def get_q_dq():
    states = [p.getJointState(robot, jid) for jid in joint_ids]
    q = np.array([s[0] for s in states], dtype=float)
    dq = np.array([s[1] for s in states], dtype=float)
    return q, dq


def get_links_p_CoM():
    p_CoMs = []
    for lid in link_ids:
        state = p.getLinkState(robot, lid, computeForwardKinematics=True)
        p_CoMs.append(np.asarray(state[0], dtype=float))
    return np.asarray(p_CoMs)


def get_potential_energy():
    p_CoMs = get_links_p_CoM()
    return np.sum(link_masses * G * p_CoMs[:, 2])


def get_p_tip():
    state = p.getLinkState(robot, link_ids[-1], computeForwardKinematics=True)
    link_pos = np.asarray(state[4], dtype=float)  # world link-frame origin
    link_orn = state[5]  # world link-frame quaternion
    rot = np.asarray(p.getMatrixFromQuaternion(link_orn)).reshape(3, 3)
    return link_pos + rot @ np.array([L, 0.0, 0.0])


def get_polyline():
    p_base = np.zeros(3)
    points = [p_base]
    for lid in link_ids:
        st = p.getLinkState(robot, lid, computeForwardKinematics=True)
        p_link_origin = np.asarray(st[4], dtype=float)
        points.append(p_link_origin)
    points.append(get_p_tip())
    return np.asarray(points)


def fd(fn, q, eps=1.0e-5):
    q = np.asarray(q, dtype=float)
    f = fn()
    if f.ndim == 0:
        grad = np.zeros(q.size) # scalar function
    else:
        grad = np.zeros((f.shape[0], q.size)) # vector function
    for k in range(q.size):
        q_plus = q.copy()
        q_minus = q.copy()
        q_plus[k] += eps
        q_minus[k] -= eps
        set_q(q_plus)
        f_plus = fn()
        set_q(q_minus)
        f_minus = fn()
        if f.ndim == 0:
            grad[k] = (f_plus - f_minus) / (2.0 * eps)
        else:
            grad[:, k] = (f_plus - f_minus) / (2.0 * eps)
    set_q(q)
    return grad
