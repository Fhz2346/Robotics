from spatialgeometry import Cuboid
from spatialmath import SE3

A, B, C, D, E, F = 30.0, 1.0, 35.0, 5.0, 11.0, 28.0

RECTS = [
    (-E - B, D, B, A),  # upper-left
    (-E - B, -C, B, C),  # lower-left
    (E, 0.0, B, C),  # upper-right
    (E, -C, B, A),  # lower-right
]  # xmin, ymin, width, height


def build_obstacles():
    objects = []
    for x, y, w, h in RECTS:
        objects.append(
            Cuboid(
                scale=[w, h, 1.0],
                pose=SE3(x + w / 2, y + h / 2, 0.5),
                collision=True,
            )
        )
    return objects

obstacles = build_obstacles()
