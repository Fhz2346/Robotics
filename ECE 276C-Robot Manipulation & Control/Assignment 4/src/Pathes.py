from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pathes:
    wsf = Path(__file__).resolve().parent.parent
    out_dir = wsf / "output"
    ckpt_dir = wsf / "ckpt"
    cfg_dir = wsf / "config"
    cache_dir = wsf / "cache"


P = Pathes()
