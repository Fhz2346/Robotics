import cv2
import numpy as np

def imgs_to_video(Imgs, out_path="output.mp4", fps=30):
    N, H, W, C = Imgs.shape
    assert C == 3

    # OpenCV 用 BGR，如果你图像是 RGB，需要转
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    for i in range(N):
        frame = Imgs[i]

        # 如果是 float，需要转 uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    print(f"Video saved to {out_path}")