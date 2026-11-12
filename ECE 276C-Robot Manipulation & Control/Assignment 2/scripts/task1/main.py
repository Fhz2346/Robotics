import numpy as np
from tqdm import tqdm

from Pathes import P
from plot import plot_results
from init_task import *


def scan_configuration_space():
    Theta1 = np.arange(-180, 180, 1, dtype=float)
    Theta2 = np.arange(-180, 181, 1, dtype=float)
    grid = np.zeros((len(Theta2), len(Theta1)), dtype=np.uint8)

    print(f'Scanning {len(Theta1)*len(Theta2):,} configurations (1 degree resolution)...')
    for j, theta2 in enumerate(tqdm(Theta2)):
        if not is_theta2_valid_deg([None, theta2]):
            grid[j, :] = 2  # joint constraint invalid
            continue
        for i, theta1 in enumerate(Theta1):
            q = np.deg2rad([theta1, theta2])
            if any(robot.iscollided(q, obs) for obs in obstacles):
                grid[j, i] = 1
    return Theta1, Theta2, grid


if __name__ == "__main__":
    scan_result_path = P.out_dir / "task1_cspace.npz"
    if not scan_result_path.exists():
        t1, t2, grid = scan_configuration_space()
        np.savez_compressed(P.out_dir / "task1" / "cspace.npz", theta1=t1, theta2=t2, occupancy=grid)
        print('Saved task1_cspace.npz (0=free, 1=collision, 2=invalid)')
    else:
        data = np.load(scan_result_path)
        t1 = data['theta1']
        t2 = data['theta2']
        grid = data['occupancy']
    plot_results(t1, t2, grid)
