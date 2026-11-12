# ECE276A PR2 — LiDAR-Based SLAM

UCSD ECE276A *Sensing & Estimation in Robotics* Project 2.

A multi-sensor SLAM pipeline for a differential-drive robot equipped with wheel encoders, an IMU, a Hokuyo LiDAR, and a Kinect RGB-D camera. The system estimates the robot trajectory and builds 2D occupancy and texture maps of indoor environments. Key techniques include LiDAR scan matching for pose estimation and pose-graph optimization for loop-closure correction.

**Pipeline overview**
1. **Odometry** — dead reckoning from encoder distance and IMU yaw rate  
2. **Scan matching** — ICP alignment of consecutive LiDAR scans, initialized by relative odometry  
3. **Occupancy mapping** — Bresenham ray casting into a 2D log-odds grid  
4. **Texture mapping** — Kinect depth→RGB reconstruction projected onto a colored grid  
5. **Pose-graph SLAM** — odometry / ICP edges plus loop closures, optimized with GTSAM (Levenberg–Marquardt)

For implementation details and results, see `Report.pdf`.

## Pipeline

| Task | Script | Description | Required outputs |
|------|--------|-------------|------------------|
| 1 | `Scripts/Task/Task1.py` | Dead-reckoning odometry (Encoder + IMU) | — |
| 2 | `Scripts/Task/Task2.py` | LiDAR scan matching (odometry init + ICP) | — |
| 3.1 | `Scripts/Task/Task3_1.py` | 2D occupancy-grid mapping | `scan_matching.npz` |
| 3.2 | `Scripts/Task/Task3_2.py` | Kinect-colored texture map | `scan_matching.npz` |
| 4 | `Scripts/Task/Task4.py` | Pose-graph SLAM + loop closure (GTSAM) | — |

Suggested order: **Task1 → Task2 → Task3_1 / Task3_2 → Task4**.  
Task3 depends on `Output/scan_matching.npz` from Task2.

## Demo

### Trajectory
#### Dataset 20

| Odometry | Odometry+Lidar |
| --- | --- |
| ![](Pics/traj.png) | ![](Pics/20_traj.png) |
| Loop closure + pose graph optimization | |
| ![](Pics/traj%201.png) | |

| Odometry+Lidar | Loop closure + pose graph optimization |
| --- | --- |
| ![](Pics/20_map.png) | ![](Pics/map.png) |
| ![](Pics/colored_map.png) | ![](Pics/colored_map%2B.png) |

#### Dataset 21

| Odometry | Odometry+Lidar |
| --- | --- |
| ![](Pics/traj_task1.png) | ![](Pics/traj_task2.png) |
| Loop closure + pose graph optimization | |
| ![](Pics/traj_final.png) | |

| Odometry+Lidar | Loop closure + pose graph optimization |
| --- | --- |
| ![](Pics/map-.png) | ![](Pics/map%2B.png) |
| ![](Pics/colored_map-.png) | ![](Pics/colored_map%2B%201.png) |

Loop closure works well, robot successfully find it and there is crosspoint on the trajectory
The mapping performance is much better

### Validation of ICP

| 1 | 2 | 3 | 4 |
| --- | --- | --- | --- |
| ![](Pics/{0E8B45CC-6197-4547-A956-3F478227B9DA}.png) | ![](Pics/{45247E71-F5AA-4882-A302-EE0E173163FC}.png) | ![](Pics/{921FC743-1E45-4AC1-8AF1-3BAC30B647C0}.png) | ![](Pics/{07D64F37-5AAE-4F34-8B85-B61CED178FA8}.png) |
| ![](Pics/{F7D96929-F8B4-499B-9BFE-B96DF21B7B2F}.png) | ![](Pics/{0805D646-494A-4CD5-841E-6FCE47795F80}.png) | ![](Pics/{C8CF5827-926F-4959-8F62-D1D5104E4DE9}.png) | ![](Pics/{26482C06-4804-4845-83AA-83C3C093DA74}.png) |

ICP works relatively well.

| Before ICP | ICP Init | ICP End |
| --- | --- | --- |
| ![](Pics/scan%20Before%20ICP%20with%20stride%201.png) | ![](Pics/ICP%20Init.png) | ![](Pics/ICP%20End.png) |

Faulure cases

<img src="Pics/image.png" width="300">

Human can easily recognize this but ICP frequently mismatch this since the range is quite huge.
### Validation of Camera
Map RGB to disparity image

![](Pics/Texture_mapping.png)

Reconstructed Points in world frame

![](Pics/3D.png)
### Loop Closure Detection

| | |
| --- | --- |
| ![](Pics/0%20vs%20310.png) | ![](Pics/300%20vs%20660.png) |
| ![](Pics/770%20vs%202680.png) | ![](Pics/1500%20vs%202010.png) |
| ![](Pics/1570%20vs%201880.png) | ![](Pics/3290%20vs%203600.png) |
| ![](Pics/4580%20vs%204890.png) | ![](Pics/4630%20vs%204940.png) |
| ![](Pics/1240%20vs%202320.png) | ![](Pics/1430%20vs%202080.png) |

Faulure Cases

| | |
| --- | --- |
| ![](Pics/map%201.png) | ![](Pics/3170%20vs%203800.png) |

When robot met 2 parallel wall, it may recognize flipped orientation, and the localization may shift either.

## Project Structure

```
PR2/
├── src/                    # Core library (add to PYTHONPATH)
│   ├── Config.py           # Data paths and task hyperparameters
│   ├── load_data.py        # Load .npz datasets
│   ├── Robot.py            # Sensor management, time sync, relative odometry
│   ├── Sensor/             # Encoder / IMU / Lidar / Kinect
│   ├── Utils/              # ICP2D, maps, GTSAM helpers, 2D transforms
│   └── Vis/                # Trajectory / map / video visualization
├── Scripts/
│   ├── Task/               # Main assignment scripts
│   ├── Given/              # Course-provided tests (bresenham, GTSAM)
│   └── test/               # Debug / unit-test scripts
├── Output/                 # Run outputs (traj, map, npz, video)
└── Record/                 # Saved experiment results
```

## Dependencies

- Python 3
- `numpy`, `scipy`, `matplotlib`, `opencv-python` (`cv2`)
- [`gtsam`](https://github.com/borglab/gtsam) (Task4 / `Scripts/Given/test_gtsam.py`)

Verify GTSAM:

```bash
PYTHONPATH=src python Scripts/Given/test_gtsam.py
```

## Data Setup

Set paths in `src/Config.py`:

```python
data_dir = "/path/to/ECE276A/PR2"   # directory with encoder*.npz, lidar*.npz, etc.
dataset_id = 20                     # dataset ID (e.g. 20, 21)
out_dir = "Output"
```

Expected file naming (see `load_data.py`):

- `Encoders{id}.npz`
- `Hokuyo{id}.npz`
- `Imu{id}.npz`
- Kinect RGB / disparity images and their timestamps

## How to Run

Add `src` to `PYTHONPATH` (already set in `.vscode/launch.json` for VS Code / Cursor debugging):

```bash
cd "/path/to/PR2"
export PYTHONPATH="$(pwd)/src"

python Scripts/Task/Task1.py
python Scripts/Task/Task2.py
python Scripts/Task/Task3_1.py
python Scripts/Task/Task3_2.py
python Scripts/Task/Task4.py
```

### Task Details

**Task1 — Odometry**  
Dead reckoning from encoder distance and IMU angular velocity. Writes `Output/odom.npz`, `traj.png`, and `theta.png`.

**Task2 — Scan Matching**  
Align consecutive LiDAR scans with ICP, initialized by relative odometry (every N frames). Accumulates the trajectory into `Output/scan_matching.npz`.

**Task3.1 — Occupancy Mapping**  
Transform LiDAR points into the world frame and update an occupancy grid with Bresenham ray casting. Writes `map.png` and `mapping_process.mp4`.

**Task3.2 — Texture Mapping**  
Reconstruct colored 3D points from Kinect depth, project onto a 2D colored grid. Writes `colored_map.png`.

**Task4 — Pose-Graph SLAM**  
Odometry/ICP edges plus loop closures (distance gate + ICP validation), optimized with GTSAM Levenberg–Marquardt. Writes `posegraph_simple.npz`.

Main hyperparameters live in `Config4`: `stride`, `icp_every`, `loop_every`, `min_sep`, `dist_gate`, `th_err`, and noise models.

## Outputs

Default directory: `Output/`

| File | Source |
|------|--------|
| `odom.npz`, `traj.png`, `theta.png` | Task1 (later tasks may overwrite plots) |
| `scan_matching.npz` | Task2 |
| `map.png`, `mapping_process.mp4` | Task3.1 |
| `colored_map.png` | Task3.2 |
| `posegraph_simple.npz` | Task4 |

`Record/` keeps historical results organized by task / dataset ID for comparison.

## Notes

- Sensor timestamps are aligned via nearest-neighbor maps in `Robot.build_Tmap()`.
- Kinect→robot extrinsics are hard-coded in `Robot.activate(RGBD)` (translation and roll/pitch/yaw).
- ICP: `src/Utils/ICP2D.py`. Occupancy / color maps: `src/Utils/Map.py`.