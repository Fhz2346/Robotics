import cv2
import numpy as np
import os
from src.load_data import load_dataset


def img2vid(dataset_id):
    dataset_id = 20
    ds = load_dataset(dataset_id, data_dir="../data")
    ts = ds.kinect.rgb_time_stamps

    dt = np.diff(ts)
    fps = 1.0 / np.mean(dt)
    print("Estimated FPS:", fps)

    img_dir = f"..\\data\\dataRGBD\\RGB{dataset_id}"
    N = len(ts)

    first_img = cv2.imread(os.path.join(img_dir, f"rgb{dataset_id}_1.png"))
    h, w, _ = first_img.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f"rgb{dataset_id}_video.mp4", fourcc, fps, (w, h))

    for i in range(N):
        path = os.path.join(img_dir, f"rgb{dataset_id}_{i+1}.png")
        img = cv2.imread(path)
        video.write(img)

    video.release()
    print(f"Video saved as rgb{dataset_id}_video.mp4")
