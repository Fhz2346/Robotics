from .jvrc import JvrcWalkEnv, JvrcStepEnv
from .h1 import H1Env

env_dict = {
    "jvrc_walk": JvrcWalkEnv,
    "jvrc_step": JvrcStepEnv,
    "h1": H1Env,
}
