from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class Pathes:
    wsf = Path(r"E:\ProjectLocal\ECE 276C\Assignment 2")
    out_dir = wsf / "output"
    assets_dir = wsf / "asset"

P = Pathes()
