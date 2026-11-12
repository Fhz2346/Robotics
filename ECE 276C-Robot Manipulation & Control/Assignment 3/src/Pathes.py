from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class Pathes:
    wsf = Path(r"E:\ProjectLocal\ECE 276C\Assignment 3")
    data_dir = Path("/media/fhz/Learning/Dataset/D2L")
    out_dir = wsf / "output"
    ckpt_dir = wsf / "ckpt"
    cfg_dir = wsf / "config"
    asset_dir = wsf / "asset"


P = Pathes()
