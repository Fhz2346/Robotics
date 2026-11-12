import numpy as np
import matplotlib.pyplot as plt
from init_task import *


def plot_path(times, measured, path):
    segment_duration = DURATION / (len(WAYPOINTS) - 1)
    speed = np.linalg.norm(np.gradient(measured, times, axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))

    ax = axes[0]
    ax.plot(measured[:, 0], measured[:, 1], color="#b91c1c", lw=2.0, label="end effector")
    ax.scatter(WAYPOINTS[:-1, 0], WAYPOINTS[:-1, 1], s=60, c="#1d4ed8", zorder=3)
    for index, point in enumerate(WAYPOINTS[:-1], start=1):
        ax.annotate(
            str(index),
            (point[0], point[1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=12,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("End-effector path")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(times, speed, color="#0f766e", lw=1.8)
    for index in range(len(WAYPOINTS)):
        ax.axvline(index * segment_duration, color="#94a3b8", lw=0.8, ls="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("Speed is zero at each setpoint")
    ax.set_xlim(0.0, DURATION)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_profile(path):
    tau = np.linspace(0.0, 1.0, 400)
    segment_duration = DURATION / (len(WAYPOINTS) - 1)
    fraction = s(tau)
    velocity = dot_s(tau) / segment_duration
    acceleration = np.gradient(velocity, tau * segment_duration)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(tau, fraction, label=r"$s$")
    ax.plot(tau, velocity / np.max(np.abs(velocity)), label=r"$\dot s$ (scaled)")
    ax.plot(tau, acceleration / np.max(np.abs(acceleration)), label=r"$\ddot s$ (scaled)")
    ax.set_xlabel(r"$\tau = t/T$")
    ax.set_title("2nd-order smooth rest-to-rest profile")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
