from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class Pathes:
    if os_name == "Windows":
        wsf = Path(r"E:\ProjectLocal\Robotics\Learning_Humanoid_Walking")
    elif os_name == "Linux":
        wsf = Path("/media/fhz/Learning/ProjectLocal/Learning_Humanoid_Walking")

    out_dir = wsf / "output"
    asset_dir = wsf / "assets"
    tmp_dir = wsf / "tmp"
    log_dir = tmp_dir / "logs"
    config_dir = wsf / "Config"


P = Pathes()
