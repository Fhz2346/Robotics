import sympy as sp

is_symbolic = False
if is_symbolic:
    rho = sp.symbols("rho", positive=True)
    r = sp.symbols("r", positive=True)
    l = sp.symbols("l", positive=True)
    g = sp.symbols("g", positive=True)
else: # numeric values
    rho = sp.Integer(300)
    r = sp.Rational(2, 100)
    l = sp.Rational(2, 10)
    g = sp.Rational(981, 100)


JOINT_NAMES = ["baseHinge", "interArm", "interArm2"]
LINK_NAMES = ["arm1", "arm2", "arm3"]
N = 3 # number of joints

q1, q2, q3 = sp.symbols("q1 q2 q3", real=True)
dq1, dq2, dq3 = sp.symbols("dq1 dq2 dq3", real=True)

q = sp.Matrix([q1, q2, q3])
dq = sp.Matrix([dq1, dq2, dq3])

m = rho * sp.pi * r**2 * l

I_ca = m * r**2 / 2  # central axis
I_dtc = m * r**2 / 2 + m * l**2 / 12  # diameter through center
I_body = sp.diag(I_dtc, I_dtc, I_ca)  # In the cylinder's own principal-axis coordinate system, with the reference point at the CoM
if is_symbolic:
    print("I_body=", I_body)
