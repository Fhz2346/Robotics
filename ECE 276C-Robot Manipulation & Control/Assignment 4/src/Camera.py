import numpy as np
import pybullet as p


class Camera:

    def __init__(self, cfg):
        self.H = cfg.image.height
        self.W = cfg.image.width
        self.aspect = self.W / self.H
        self.fov = cfg.fov
        self.near = cfg.clipping_plane.near
        self.far = cfg.clipping_plane.far
        projMat = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        # vfov + aspect = vfov + hfov
        self.projMat = np.array(projMat).reshape(4, 4).T

    def set_pose(self, pos, orn):
        self.pos = pos
        self.orn = orn
        self.viewMat = self.get_viewMat()
        self.capture()

    def get_viewMat(self):
        viewMat = p.computeViewMatrix(
            cameraEyePosition=self.pos,
            cameraTargetPosition=self.pos + self.orn[:, 2],
            cameraUpVector=-self.orn[:, 1],
        )
        return np.array(viewMat).reshape(4, 4).T

    def depth_buffer_linearize(self, depth):
        return (self.far * self.near) / (
            self.far + self.near - (self.far - self.near) * depth
        )

    def capture(self):
        _, _, rgb, depth, _ = p.getCameraImage(
            self.W, self.H,
            viewMatrix=self.viewMat.T.reshape(-1).tolist(),
            projectionMatrix=self.projMat.T.reshape(-1).tolist(),
            renderer=p.ER_TINY_RENDERER,
        )
        self.rgb = np.asarray(rgb, dtype=np.uint8).reshape(self.H, self.W, -1)[:, :, :3]
        # depth 得到的是到表面的距离, 并非到物体中心的距离
        depth = np.asarray(depth, dtype=np.float32).reshape(self.H, self.W)
        self.depth = self.depth_buffer_linearize(depth)

    def viewport_transform(self, NDC_pt):
        u = ((NDC_pt[0] + 1) / 2.0) * self.W
        v = ((1 - NDC_pt[1]) / 2.0) * self.H
        return [int(u), int(v)]

    def world2pixel(self, w_pt):
        _w_pt = np.append(w_pt, 1)
        _c_pt = self.viewMat @ _w_pt
        _NDC_pt = self.projMat @ _c_pt
        _NDC_pt = _NDC_pt / _NDC_pt[3]
        return self.viewport_transform(_NDC_pt)
