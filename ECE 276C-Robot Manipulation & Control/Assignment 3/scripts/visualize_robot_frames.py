"""Visualize the three-link robot and its joint, link, and inertial frames.

The default joint angles reproduce the planar pose in fig1:
    q1 = 0 deg, q2 = 120 deg, q3 = 120 deg

Outputs:
    joint_frames.png    = joint zero frames (before the variable Rz(q))
    link_frames.png     = current link frames (after the variable Rz(q))
    inertial_frames.png = URDF inertial frames (origin at each center of mass)

Axis colors:
    red/green/blue = local x/y/z axes

Run:
    python scripts/visualize_robot_frames.py

Optionally choose another pose:
    python scripts/visualize_robot_frames.py --q-deg 0 90 -90
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LINK_LENGTH = 0.2
AXIS_COLORS = ("#d62728", "#2ca02c", "#1f77b4")
AXIS_NAMES = ("x", "y", "z")
FRAME_LABEL_OFFSETS = {
    "baseHinge": (12, 22),
    "interArm": (12, 18),
    "interArm2": (12, -34),
    "base": (12, 18),
    "arm1": (12, 40),
    "arm2": (12, 18),
    "arm3": (12, -34),
    "arm1/I": (16, -6),
    "arm2/I": (14, 16),
    "arm3/I": (14, 8),
}
# Inertial pose relative to the link frame, copied from <inertial><origin>.
# rpy is roll-pitch-yaw about fixed axes: R = Rz(yaw) Ry(pitch) Rx(roll).
INERTIAL_IN_LINK = {
    "arm1": (np.array([0.0, 0.0, LINK_LENGTH / 2.0]), (0.0, 0.0, 0.0)),
    "arm2": (np.array([LINK_LENGTH / 2.0, 0.0, 0.0]), (0.0, -math.pi / 2.0, 0.0)),
    "arm3": (np.array([LINK_LENGTH / 2.0, 0.0, 0.0]), (0.0, math.pi / 2.0, 0.0)),
}


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rpy_matrix(roll, pitch, yaw):
    """URDF fixed-axis rpy: first roll, then pitch, then yaw."""
    return rz(yaw) @ ry(pitch) @ rx(roll)


def forward_kinematics(q):
    """Return robot points plus joint, link, and inertial frames."""
    q1, q2, q3 = q

    origin1 = np.zeros(3)
    joint1_rotation = np.eye(3)
    link1_rotation = joint1_rotation @ rz(q1)

    origin2 = origin1 + link1_rotation @ np.array([0.0, 0.0, LINK_LENGTH])
    joint2_rotation = link1_rotation @ rx(math.pi / 2.0)
    link2_rotation = joint2_rotation @ rz(q2)

    origin3 = origin2 + link2_rotation @ np.array([LINK_LENGTH, 0.0, 0.0])
    joint3_rotation = link2_rotation
    link3_rotation = joint3_rotation @ rz(q3)

    tip = origin3 + link3_rotation @ np.array([LINK_LENGTH, 0.0, 0.0])

    joint_frames = [
        ("baseHinge", origin1, joint1_rotation),
        ("interArm", origin2, joint2_rotation),
        ("interArm2", origin3, joint3_rotation),
    ]
    link_frames = [
        ("base", origin1, np.eye(3)),
        ("arm1", origin1, link1_rotation),
        ("arm2", origin2, link2_rotation),
        ("arm3", origin3, link3_rotation),
    ]
    link_poses = {
        "arm1": (origin1, link1_rotation),
        "arm2": (origin2, link2_rotation),
        "arm3": (origin3, link3_rotation),
    }
    inertial_frames = []
    for name, (origin, rotation) in link_poses.items():
        offset, rpy = INERTIAL_IN_LINK[name]
        inertial_frames.append((
            f"{name}/I",
            origin + rotation @ offset,
            rotation @ rpy_matrix(*rpy),
        ))
    points = np.vstack([origin1, origin2, origin3, tip])
    return points, joint_frames, link_frames, inertial_frames


def make_projector(scene_points, width, height):
    """Create a fixed orthographic 3-D to 2-D camera projection."""
    elevation = math.radians(22.0)
    azimuth = math.radians(-62.0)
    view = np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ])
    right = np.cross(view, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    up /= np.linalg.norm(up)

    projected = np.column_stack((scene_points @ right, scene_points @ up))
    low = projected.min(axis=0)
    high = projected.max(axis=0)
    center = 0.5 * (low + high)
    span = max(*(high - low), 0.45)
    scale = min(width, height) * 0.70 / span

    def project(point):
        uv = np.array([np.dot(point, right), np.dot(point, up)])
        return (
            int(width / 2 + scale * (uv[0] - center[0])),
            int(height / 2 - scale * (uv[1] - center[1]) + 25),
        )

    return project


def draw_dashed_line(draw, start, end, color, width=4, dash=10, gap=7):
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(delta))
    if length == 0:
        return
    direction = delta / length
    distance = 0.0
    while distance < length:
        a = np.asarray(start) + direction * distance
        b = np.asarray(start) + direction * min(distance + dash, length)
        draw.line([tuple(a), tuple(b)], fill=color, width=width)
        distance += dash + gap


def draw_arrow(draw, start, end, color, dashed=False):
    if dashed:
        draw_dashed_line(draw, start, end, color)
    else:
        draw.line([start, end], fill=color, width=4)

    vector = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(vector))
    if length == 0:
        return
    unit = vector / length
    normal = np.array([-unit[1], unit[0]])
    tip = np.asarray(end, dtype=float)
    base = tip - 10.0 * unit
    triangle = [tuple(tip), tuple(base + 4.5 * normal), tuple(base - 4.5 * normal)]
    draw.polygon(triangle, fill=color)


def draw_frame(draw, project, frame, axis_length, dashed, font):
    name, origin, rotation = frame
    origin_2d = project(origin)
    for column, (axis_name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
        endpoint = origin + axis_length * rotation[:, column]
        endpoint_2d = project(endpoint)
        draw_arrow(draw, origin_2d, endpoint_2d, color, dashed=dashed)
        draw.text(
            (endpoint_2d[0] + 4, endpoint_2d[1] - 10),
            axis_name, fill=color, font=font,
            stroke_width=3, stroke_fill="white",
        )
    dx, dy = FRAME_LABEL_OFFSETS[name]
    draw.text(
        (origin_2d[0] + dx, origin_2d[1] + dy),
        f"{{{name}}}", fill="black", font=font,
        stroke_width=3, stroke_fill="white",
    )


def plot_robot(q, frames, frame_kind, output):
    points, _, _, _ = forward_kinematics(q)
    width, height = 1200, 900
    axis_length = 0.075 if frame_kind == "Joint" else 0.085

    scene_points = [*points]
    for _, origin, rotation in frames:
        scene_points.append(origin)
        for column in range(3):
            scene_points.append(origin + axis_length * rotation[:, column])
    project = make_projector(np.asarray(scene_points), width, height)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 25)
        label_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        title_font = label_font = ImageFont.load_default()

    title = f"Three-link robot: {frame_kind.lower()} frames"
    q_degrees = np.degrees(q)
    subtitle = (
        f"q = [{q_degrees[0]:.0f} deg, {q_degrees[1]:.0f} deg, "
        f"{q_degrees[2]:.0f} deg]"
    )
    draw.text((width // 2, 22), title, fill="black", font=title_font, anchor="ma")
    draw.text((width // 2, 53), subtitle, fill="black", font=label_font, anchor="ma")
    if frame_kind == "Inertial":
        draw.text(
            (width // 2, 78),
            "origin at center of mass; z is the cylinder axis",
            fill="black", font=label_font, anchor="ma",
        )

    link_colors = ("#333333", "#2864b4", "#333333")
    for index, color in enumerate(link_colors):
        draw.line(
            [project(points[index]), project(points[index + 1])],
            fill=color, width=13,
        )

    for point in points[:-1]:
        x, y = project(point)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="orange", outline="black")
    x, y = project(points[-1])
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="red", outline="black")

    if frame_kind == "Inertial":
        for _, origin, _ in frames:
            x, y = project(origin)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#6a3d9a", outline="black")

    for frame in frames:
        draw_frame(
            draw, project, frame, axis_length,
            dashed=(frame_kind == "Joint"),
            font=label_font,
        )

    legend_y = height - 70
    for index, (axis_name, color) in enumerate(zip(AXIS_NAMES, AXIS_COLORS)):
        x0 = 65 + index * 150
        draw.line((x0, legend_y, x0 + 35, legend_y), fill=color, width=4)
        draw.text(
            (x0 + 42, legend_y - 9), f"local {axis_name} axis",
            fill="black", font=label_font,
        )
    if frame_kind == "Inertial":
        draw.ellipse(
            (530, legend_y - 6, 542, legend_y + 6),
            fill="#6a3d9a", outline="black",
        )
        draw.text(
            (550, legend_y - 9), "center of mass",
            fill="black", font=label_font,
        )

    # Write first, then replace, so reruns are reliable on Windows.
    temporary_output = output.with_name(f"{output.stem}_new{output.suffix}")
    image.save(str(temporary_output.resolve()))
    temporary_output.replace(output)
    print(f"Saved: {output.resolve()}")


def save_frame_plots(q, output_dir):
    _, joint_frames, link_frames, inertial_frames = forward_kinematics(q)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_robot(q, joint_frames, "Joint", output_dir / "joint_frames.png")
    plot_robot(q, link_frames, "Link", output_dir / "link_frames.png")
    plot_robot(q, inertial_frames, "Inertial", output_dir / "inertial_frames.png")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the robot and its URDF joint, link, and inertial frames."
    )
    parser.add_argument(
        "--q-deg", nargs=3, type=float, default=[0.0, 120.0, 120.0],
        metavar=("Q1", "Q2", "Q3"),
        help="joint angles in degrees (default: 0 120 120)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/frames"),
        help="directory for the joint, link, and inertial frame figures",
    )
    args = parser.parse_args()

    save_frame_plots(np.radians(args.q_deg), args.output_dir)


if __name__ == "__main__":
    main()
