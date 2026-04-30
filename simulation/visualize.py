import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


def animate(t, states, params, save_path=None, speedup=1.0):
    """
    Animate the cart-pendulum trajectory.

    Parameters
    ----------
    t         : 1-D time array
    states    : (4, N) array — [x, x_dot, theta, theta_dot]
    params    : dict with keys M, m, L (used for scaling visuals)
    save_path : if provided, save animation to this path (.mp4 or .gif)
    speedup   : playback speed multiplier (e.g. 2.0 = 2x faster)
    """
    x      = states[0]
    theta  = states[2]
    L      = params.get("L", 0.5)

    # Pendulum bob position
    bob_x = x + L * np.sin(theta)
    bob_y = L * np.cos(theta)

    # --- figure layout ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(np.min(x) - 1.5 * L, np.max(x) + 1.5 * L)
    ax.set_ylim(-1.5 * L, 1.5 * L)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")  # rail

    cart_w, cart_h = 0.3, 0.15

    cart_patch = patches.FancyBboxPatch(
        (x[0] - cart_w / 2, -cart_h / 2),
        cart_w, cart_h,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="steelblue",
        facecolor="lightsteelblue",
    )
    ax.add_patch(cart_patch)

    (rod_line,) = ax.plot([], [], "k-", linewidth=2.5)
    (bob_dot,)  = ax.plot([], [], "o", color="firebrick", markersize=10, zorder=5)
    time_text   = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)

    ax.set_title("Cart-Pendulum Simulation")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def init():
        rod_line.set_data([], [])
        bob_dot.set_data([], [])
        time_text.set_text("")
        return cart_patch, rod_line, bob_dot, time_text

    def update(i):
        cart_patch.set_x(x[i] - cart_w / 2)
        rod_line.set_data([x[i], bob_x[i]], [0, bob_y[i]])
        bob_dot.set_data([bob_x[i]], [bob_y[i]])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return cart_patch, rod_line, bob_dot, time_text

    dt_ms = (t[1] - t[0]) * 1000 / speedup
    anim = FuncAnimation(
        fig, update, frames=len(t),
        init_func=init, interval=dt_ms, blit=True
    )

    if save_path:
        anim.save(save_path, writer="ffmpeg", fps=int(1000 / dt_ms))
        print(f"Animation saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    return anim


def plot_states(t, states, forces=None, save_path=None):
    """
    Plot state trajectories over time.

    Parameters
    ----------
    t         : 1-D time array
    states    : (4, N) array — [x, x_dot, theta, theta_dot]
    forces    : optional 1-D array of applied forces
    save_path : if provided, save figure here and close; otherwise show
    """
    x, x_dot, theta, theta_dot = states

    n_plots = 5 if forces is not None else 4
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 2.5 * n_plots), sharex=True)

    axes[0].plot(t, x, color="steelblue")
    axes[0].set_ylabel("x (m)")
    axes[0].set_title("Cart position")

    axes[1].plot(t, x_dot, color="steelblue")
    axes[1].set_ylabel("ẋ (m/s)")
    axes[1].set_title("Cart velocity")

    axes[2].plot(t, np.degrees(theta), color="firebrick")
    axes[2].set_ylabel("θ (deg)")
    axes[2].set_title("Pendulum angle")
    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")

    axes[3].plot(t, np.degrees(theta_dot), color="firebrick")
    axes[3].set_ylabel("θ̇ (deg/s)")
    axes[3].set_title("Pendulum angular velocity")

    if forces is not None:
        axes[4].plot(t, forces, color="seagreen")
        axes[4].set_ylabel("F (N)")
        axes[4].set_title("Applied force")

    axes[-1].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show(block=True)
