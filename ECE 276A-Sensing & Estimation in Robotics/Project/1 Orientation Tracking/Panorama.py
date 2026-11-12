import numpy as np
import cv2
from Utils.Load import *
from Vis.Visualize import imshow


def precompute_camera_unit_sphere_dirs(HW, FoV = (60.0, 45.0)):
    FoVx, FoVy = FoV # degrees
    FoVx = np.deg2rad(FoVx)
    FoVy = np.deg2rad(FoVy)
    
    H, W = HW
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)  
    uu, vv = np.meshgrid(u, v)

    lam = (uu - cx) / (W - 1) * FoVx
    phi = -(vv - cy) / (H - 1) * FoVy
    x = np.cos(phi) * np.cos(lam)
    y = -np.cos(phi) * np.sin(lam)
    z = np.sin(phi)
    dirs_c = np.stack([x, y, z], axis=-1)
    return dirs_c


def world_dirs_to_equirect(dirs_w, Wp, Hp):
    x = dirs_w[..., 0]
    y = dirs_w[..., 1]
    z = dirs_w[..., 2]

    lam = np.arctan2(y, x)
    phi = np.arcsin(np.clip(z, -1, 1))

    xp = (lam + np.pi) / (2 * np.pi) * Wp
    yp = (np.pi / 2 - phi) / np.pi * Hp
    return xp, yp


def build_panorama_sphere_cylinder(R_cw, Fov=(60.0, 45.0), pShape=(2048, 1024), step=3):
    Wp, Hp = pShape
    N, H, W, _ = Imgs.shape

    dirs_c = precompute_camera_unit_sphere_dirs((H, W), Fov).reshape(-1, 3).T
    pano = np.zeros((Hp, Wp, 3), dtype=np.uint8)
    for k in range(0, N, step):
        I = Imgs[k]
        R = R_cw[k]
        dirs_w = (R @ dirs_c).T.reshape(H, W, 3)
        xp, yp = world_dirs_to_equirect(dirs_w, Wp, Hp)
        xi, yi = [np.round(coord).astype(np.int32) for coord in (xp, yp)]
        if (xi.min() < 0) or (xi.max() >= Wp) or (yi.min() < 0) or (yi.max() >= Hp):
            continue
        pano[yi, xi] = I
    return pano


def build_panorama_sphere_cylinder_to_video(
    R_cw,
    Fov=(60.0, 45.0),
    pShape=(2048, 1024),
    step=3,
    out_mp4="pano_build.mp4",
    fps=30,
    write_every=1,  # 每 write_every 次循环写一帧（可降视频体积）
):
    Wp, Hp = pShape
    N, H, W, _ = Imgs.shape

    # 预计算：相机像素 -> 单位球方向（camera frame）
    dirs_c = precompute_camera_unit_sphere_dirs((H, W), Fov)
    dirs_c_flat = dirs_c.reshape(-1, 3).T  # (3, H*W)

    pano = np.zeros((Hp, Wp, 3), dtype=np.uint8)

    # OpenCV VideoWriter (注意：OpenCV 写入需要 BGR)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (Wp, Hp))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {out_mp4}")

    frame_count = 0

    try:
        for it, k in enumerate(range(0, N, step)):
            if k > 300:
                pass
            I = Imgs[k]
            R = R_cw[k]
            dirs_w = (R @ dirs_c_flat).T.reshape(H, W, 3)
            xp, yp = world_dirs_to_equirect(dirs_w, Wp, Hp)
            xi = np.round(xp).astype(np.int32)
            yi = np.round(yp).astype(np.int32)
            if (xi.min() < 0) or (xi.max() >= Wp) or (yi.min() < 0) or (yi.max() >= Hp):
                continue
            pano[yi, xi] = I
            if (it % write_every) == 0:
                frame = pano.copy()
                # RGB -> BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                frame_count += 1
    finally:
        writer.release()

    return pano, out_mp4, frame_count


if __name__ == "__main__":
    R_est = np.load(f"R_est{trainset}.npy")
    pano = build_panorama_sphere_cylinder(R_v[idx_v2c])
    cv2.imwrite(f"Output/Panorama/{trainset}/panorama{trainset}.png", cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))
    pano, out_mp4, n_frames = build_panorama_sphere_cylinder_to_video(
        R_v[idx_v2c],
        out_mp4=f"Output/Panorama/{trainset}/panorama_video{trainset}.mp4",
        fps=30,
    )
