from gtsam import Pose2
import numpy as np
import Utils.Transformation2D as D2

def T2pose2(T):
    dx = float(T[0, 2])
    dy = float(T[1, 2])
    dtheta = float(np.arctan2(T[1, 0], T[0, 0]))
    return Pose2(dx, dy, dtheta)


def pose22T(pose):
    return D2.make_T(pose.theta(), [pose.x(), pose.y()])
