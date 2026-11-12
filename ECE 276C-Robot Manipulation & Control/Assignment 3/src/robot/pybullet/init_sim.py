import pybullet as p

G = 9.81
DT = 1.0 / 240.0

cid = p.connect(p.DIRECT)
if cid < 0:
    raise RuntimeError("PyBullet connection failed")
p.resetSimulation()
p.setGravity(0, 0, -G)
p.setTimeStep(DT)
p.setPhysicsEngineParameter(
    fixedTimeStep=DT,
    numSolverIterations=200,
    enableFileCaching=0,
)