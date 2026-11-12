from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
from Config import cfgG


@dataclass(frozen=True)
class EncoderData:
    counts: np.ndarray
    time_stamps: np.ndarray


@dataclass(frozen=True)
class LidarData:
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray
    time_stamps: np.ndarray 


@dataclass(frozen=True)
class ImuData:
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray
    time_stamps: np.ndarray


@dataclass(frozen=True)
class KinectData:
    disparity_time_stamps: np.ndarray
    rgb_time_stamps: np.ndarray


@dataclass(frozen=True)
class Dataset:
    dataset_id: int
    encoders: EncoderData
    lidar: LidarData
    imu: ImuData
    kinect: KinectData

    # Optional payloads some pipelines may store later (e.g., loaded images)
    extra: Optional[Dict[str, Any]] = None


def _npz_path(data_dir: Path, prefix: str, dataset_id: int) -> Path:
    return data_dir / f"{prefix}{dataset_id}.npz"


def _require_keys(npz: Dict[str, Any], required: Dict[str, str], filename: str) -> None:
    missing = [k for k in required.keys() if k not in npz]
    if missing:
        pretty = ", ".join(missing)
        raise KeyError(f"{filename} is missing required keys: {pretty}")


def load_encoders(dataset_id=cfgG.dataset_id, data_dir=cfgG.data_dir) -> EncoderData:
    data_dir = Path(data_dir)
    path = _npz_path(data_dir, "Encoders", dataset_id)
    with np.load(path) as data:
        _require_keys(
            data,
            {"counts": "4 x n encoder counts", "time_stamps": "encoder time stamps"},
            str(path),
        )
        return EncoderData(counts=data["counts"].T, time_stamps=data["time_stamps"])


def load_lidar(dataset_id=cfgG.dataset_id, data_dir=cfgG.data_dir):
    data_dir = Path(data_dir)
    path = _npz_path(data_dir, "Hokuyo", dataset_id)
    with np.load(path) as data:
        _require_keys(
            data,
            {
                "angle_min": "start angle [rad]",
                "angle_max": "end angle [rad]",
                "angle_increment": "angular increment [rad]",
                "range_min": "min range [m]",
                "range_max": "max range [m]",
                "ranges": "range data [m]",
                "time_stamps": "lidar time stamps",
            },
            str(path),
        )
        return LidarData(
            angle_min=float(np.asarray(data["angle_min"])),
            angle_max=float(np.asarray(data["angle_max"])),
            angle_increment=float(np.asarray(data["angle_increment"])),
            range_min=float(np.asarray(data["range_min"])),
            range_max=float(np.asarray(data["range_max"])),
            ranges=data["ranges"].T,
            time_stamps=data["time_stamps"],
        )


def load_imu(dataset_id=cfgG.dataset_id, data_dir=cfgG.data_dir) -> ImuData:
    data_dir = Path(data_dir)
    path = _npz_path(data_dir, "Imu", dataset_id)
    with np.load(path) as data:
        _require_keys(
            data,
            {
                "angular_velocity": "angular velocity [rad/s]",
                "linear_acceleration": "linear acceleration [g]",
                "time_stamps": "imu time stamps",
            },
            str(path),
        )
        return ImuData(
            angular_velocity=data["angular_velocity"].T,
            linear_acceleration=data["linear_acceleration"].T,
            time_stamps=data["time_stamps"],
        )


def load_kinect(dataset_id=cfgG.dataset_id, data_dir=cfgG.data_dir) -> KinectData:
    data_dir = Path(data_dir)
    path = _npz_path(data_dir, "Kinect", dataset_id)
    with np.load(path) as data:
        _require_keys(
            data,
            {
                "disparity_time_stamps": "disparity acquisition times",
                "rgb_time_stamps": "rgb acquisition times",
            },
            str(path),
        )
        return KinectData(
            disparity_time_stamps=data["disparity_time_stamps"],
            rgb_time_stamps=data["rgb_time_stamps"],
        )


def load_dataset(dataset_id=cfgG.dataset_id, data_dir=cfgG.data_dir):
    enc = load_encoders(dataset_id, data_dir=data_dir)
    lid = load_lidar(dataset_id, data_dir=data_dir)
    imu = load_imu(dataset_id, data_dir=data_dir)
    kin = load_kinect(dataset_id, data_dir=data_dir)
    return Dataset(dataset_id=dataset_id, encoders=enc, lidar=lid, imu=imu, kinect=kin)


if __name__ == "__main__":
    # Small sanity check / demo
    ds_id = 20
    ds = load_dataset(ds_id, data_dir="../data")
    print(f"Loaded dataset {ds.dataset_id}")
    print(f"encoders.counts: {ds.encoders.counts.shape}, encoders.time_stamps: {ds.encoders.time_stamps.shape}")
    print(f"lidar.ranges: {ds.lidar.ranges.shape}, lidar.time_stamps: {ds.lidar.time_stamps.shape}")
    print(f"imu.angular_velocity: {ds.imu.angular_velocity.shape}, imu.time_stamps: {ds.imu.time_stamps.shape}")
    print(f"kinect.rgb_time_stamps: {ds.kinect.rgb_time_stamps.shape}")
