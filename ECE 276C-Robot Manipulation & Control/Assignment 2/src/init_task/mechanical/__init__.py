from .structure import *
import roboticstoolbox as rtb
from Pathes import P

URDF_PATH = P.assets_dir / "planar_two_link.urdf"
robot = rtb.ERobot.URDF(str(URDF_PATH))
assert robot.n == 2, f"Expected 2 revolute joints in URDF, found {robot.n}."
