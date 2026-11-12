import numpy as np

from .init_sim import *
from .euler_angle import *
from omegaconf import OmegaConf
from Pathes import P

cfgR = OmegaConf.load(P.cfg_dir / "robot.yaml")
JOINT_NAMES = ["baseHinge", "interArm", "interArm2"]
LINK_NAMES = ["arm1", "arm2", "arm3"]
N = 3
L = 0.2


if cfgR.payload == True:
    urdf_path = P.asset_dir / "three_link_payload.urdf"
else:
    urdf_path = P.asset_dir / "three_link_modified.urdf"
robot = p.loadURDF(str(urdf_path), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

jname2idx = {}
for j in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, j)
    joint_name = info[1].decode("utf-8")
    link_name = info[12].decode("utf-8")
    jname2idx[joint_name] = j

joint_ids = [jname2idx[n] for n in JOINT_NAMES]

for jid in joint_ids:
    p.setJointMotorControl2(robot, jid, p.VELOCITY_CONTROL, force=0)

lname2idx = {}
for jid in range(p.getNumJoints(robot)):
    child_name = p.getJointInfo(robot, jid)[12].decode("utf-8")
    if child_name in LINK_NAMES:
        lname2idx[child_name] = jid
link_ids = [lname2idx[n] for n in LINK_NAMES]
link_masses = np.array([p.getDynamicsInfo(robot, jid)[0] for jid in link_ids])


def fk_full(q):
    q1, q2, q3 = q
    R1 = rz(q1)
    R2 = R1 @ rx(np.pi / 2) @ rz(q2)
    R3 = R2 @ rz(q3)
    p0 = np.zeros(3)
    p1 = p0 + R1 @ np.array([0.0, 0.0, L])
    p2 = p1 + R2 @ np.array([L, 0.0, 0.0])
    p3 = p2 + R3 @ np.array([L, 0.0, 0.0])
    return np.vstack([p0, p1, p2, p3])
