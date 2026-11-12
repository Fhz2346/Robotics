import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def save_gif(results, filename):
    frames = []
    for result in results:
        for points in result["frames"]:
            frames.append((result["case"], result["tau"], points))

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0.0, 0.65)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=25, azim=45)

    line, = ax.plot([], [], [], "o-", linewidth=5, markersize=8)
    label = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    def update(i):
        case, tau, pts = frames[i]
        line.set_data(pts[:, 0], pts[:, 1])
        line.set_3d_properties(pts[:, 2])
        label.set_text(f"{case}\n" f"U = [{tau[0]:+.5f}, {tau[1]:+.5f}, {tau[2]:+.5f}] N m")
        return line, label

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=40, blit=False
    )
    animation.save(filename, writer=PillowWriter(fps=25))
    plt.close(fig)
