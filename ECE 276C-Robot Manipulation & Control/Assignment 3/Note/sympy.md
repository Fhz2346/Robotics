# Modeling Robot Kinematics and Dynamics with SymPy

SymPy is useful not only because it performs symbolic calculations, but also because it lets me describe a robot as a collection of clear, manageable pieces. Instead of expanding the full dynamics equations by hand, I first define the links and coordinate frames, then derive the quantities needed from those definitions.

## Start with coordinate frames

For each frame, I explicitly define its position (`pos`) and orientation (`orn`) relative to the world frame. In this project, link orientations, joint origins, rotation axes, center-of-mass positions, and the end-effector position are defined in `src/robot/sympy/frames.py`. The order of the transforms matches the joint configuration in the URDF files under `asset/`, so the symbolic model corresponds to the robot simulated in PyBullet.

For example, the link orientations are built by composing rotation matrices:

$$
R_1=R_z(q_1),\qquad
R_2=R_1R_x\left(\frac{\pi}{2}\right)R_z(q_2),\qquad
R_3=R_2R_z(q_3).
$$

Once these frame poses are defined, I can compute joint positions, link center-of-mass positions, the end-effector position, and joint-axis directions. Correctly defining these basic quantities gives the Jacobian and dynamics derivations a clear foundation.

## Use symbolic differentiation

Differentiating position with respect to the joint variables gives the position Jacobian. For example:

$$
J_{tip}(q)=\frac{\partial p_{tip}}{\partial q}.
$$

Similarly, SymPy's `diff` function differentiates potential energy to obtain the generalized gravity force. Derivatives of the inertia-matrix elements can be used to construct the Coriolis and centrifugal terms. The main SymPy tools used here are `Matrix`, matrix multiplication, `jacobian()`, and `diff()`. They save me from doing most of the manual differentiation and expansion, and reduce the chance of mistakes when copying long equations by hand.

In `src/robot/sympy/__init__.py`, the inertia matrix is assembled from the links' center-of-mass linear and angular velocity Jacobians and their rotational inertias. The kinetic and potential energies are then computed as follows:

$$
D(q)=\sum_i\left(m_iJ_{v,i}^{T}J_{v,i}+J_{\omega,i}^{T}I_iJ_{\omega,i}\right),\qquad
G(q)=\frac{\partial P(q)}{\partial q}.
$$

The Coriolis/centrifugal vector $C(q,\dot q)$ is constructed from derivatives of $D(q)$ using the Christoffel symbols. The resulting standard dynamics form is

$$
D(q)\ddot q+C(q,\dot q)+G(q)=U.
$$

## Symbolic results need validation

Symbolic differentiation faithfully processes the model I provide, but it cannot guarantee that the model itself is correct. For example, if the order of coordinate transforms, a joint-axis direction, a center-of-mass position, or an inertial-frame definition is wrong, SymPy can still produce a mathematically consistent result for that incorrect model. The derivation therefore needs numerical validation.

For example, the gravity term $G(q)$ is the gradient of the potential energy $P(q)$ with respect to the joint positions:

$$
G_i(q)=\frac{\partial P(q)}{\partial q_i}.
$$

I compute and compare it in two ways:

1. **Analytical calculation:** SymPy differentiates the potential energy, and `lambdify` converts the result into a numerical function that can be called with NumPy.
2. **Finite differences:** In PyBullet, I perturb each joint angle by a small positive and negative amount and estimate the gradient from the change in potential energy:

$$
G_i(q)\approx\frac{P(q+\epsilon e_i)-P(q-\epsilon e_i)}{2\epsilon}.
$$

Comparing the analytical and finite-difference values at multiple configurations helps check whether the symbolic derivatives, coordinate definitions, and simulation model agree. Finite differences are affected by the step size and numerical error, so I use a reasonable tolerance and repeat the comparison at multiple configurations instead of relying on a single example.

I can also apply the analytically computed gravity-compensation torques to the PyBullet robot and check whether it remains stationary in the selected poses. When the analytical result, finite-difference result, and simulated behavior agree, I have greater confidence in the model. If they do not agree, I revisit the coordinate frames, URDF parameters, payload configuration, sign conventions, and finite-difference step size.

## Where to find this in the project

- `src/robot/sympy/frames.py`: defines rotation matrices, joint origins, joint axes, center-of-mass positions, and the end-effector position.
- `src/robot/sympy/basics.py`: defines robot parameters, joint variables, and link inertia tensors.
- `src/robot/sympy/__init__.py`: constructs the inertia matrix, kinetic energy, potential energy, gravity term, and Coriolis/centrifugal vector.
- `scripts/Task4/control/analytical.py`: converts the SymPy gravity term into a numerical function.
- `scripts/Task4/control/fd.py` and `src/robot/pybullet/__init__.py`: compute the generalized gravity force using central finite differences of PyBullet potential energy for cross-validation.
- `scripts/Task4/main.py`: applies gravity-compensation torques in simulation to check static holding behavior.

In short, SymPy turns the model definitions into analytical expressions, while finite differences and PyBullet simulation help me check whether those expressions match the intended robot model.
