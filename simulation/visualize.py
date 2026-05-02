import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


def animate(t, states, params, save_path=None, speedup=1.0, t_event=None, delta_m=None, title=None):
    """
    Animate the cart-pendulum trajectory.

    Parameters
    ----------
    t         : 1-D time array
    states    : (4, N) array — [x, x_dot, theta, theta_dot]
    params    : dict with keys M, m, L (used for scaling visuals)
    save_path : if provided, save animation to this path (.mp4 or .gif)
    speedup   : playback speed multiplier (e.g. 2.0 = 2x faster)
    t_event   : time at which mass is added (draws visual indicator)
    delta_m   : mass added at t_event (used in label)
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
    event_text  = ax.text(0.5, 0.92, "", transform=ax.transAxes, fontsize=11,
                          color="darkorange", fontweight="bold", ha="center")
    mode_text   = ax.text(0.98, 0.95, title if title else "", transform=ax.transAxes,
                          fontsize=9, ha="right", color="dimgray")

    ax.set_title("Cart-Pendulum Simulation")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    def init():
        rod_line.set_data([], [])
        bob_dot.set_data([], [])
        time_text.set_text("")
        event_text.set_text("")
        return cart_patch, rod_line, bob_dot, time_text, event_text, mode_text

    def update(i):
        cart_patch.set_x(x[i] - cart_w / 2)
        rod_line.set_data([x[i], bob_x[i]], [0, bob_y[i]])

        mass_dropped = t_event is not None and t[i] >= t_event
        bob_color  = "saddlebrown" if mass_dropped else "firebrick"
        bob_size   = 14            if mass_dropped else 10
        bob_dot.set_data([bob_x[i]], [bob_y[i]])
        bob_dot.set_color(bob_color)
        bob_dot.set_markersize(bob_size)

        time_text.set_text(f"t = {t[i]:.2f} s")

        if mass_dropped and delta_m is not None:
            event_text.set_text(f"mass +{delta_m} kg")
        else:
            event_text.set_text("")

        return cart_patch, rod_line, bob_dot, time_text, event_text, mode_text

    dt_sim = t[1] - t[0]
    dt_ms  = dt_sim * 1000 / speedup

    if save_path:
        # Subsample frames to hit 60fps at the given speedup
        export_fps = 60
        stride     = max(1, round(speedup / (dt_sim * export_fps)))
        frames     = range(0, len(t), stride)
        anim = FuncAnimation(
            fig, update, frames=frames,
            init_func=init, interval=dt_ms * stride, blit=True
        )
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=export_fps, codec="libx264",
                              extra_args=["-crf", "18", "-pix_fmt", "yuv420p"])
        anim.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
        plt.close(fig)
    else:
        anim = FuncAnimation(
            fig, update, frames=len(t),
            init_func=init, interval=dt_ms, blit=True
        )
        plt.tight_layout()
        plt.show()

    return anim


def plot_states(t, states, forces=None, t_event=None, save_path=None):
    """
    Plot state trajectories over time.

    Parameters
    ----------
    t         : 1-D time array
    states    : (4, N) array — [x, x_dot, theta, theta_dot]
    forces    : optional 1-D array of applied forces
    t_event   : optional time to draw a vertical dashed line (e.g. mass drop)
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

    if t_event is not None:
        for ax in axes:
            ax.axvline(t_event, color="red", linewidth=1.0, linestyle="--", alpha=0.7)

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


def plot_param_estimates(t, m_hat, b_hat, true_m=None, true_b=None, t_event=None, save_path=None):
    """
    Plot adaptive parameter estimates over time.

    Parameters
    ----------
    t       : 1-D time array
    m_hat   : estimated pendulum mass over time
    b_hat   : estimated cart friction over time
    true_m  : true pendulum mass (drawn as dashed reference line)
    true_b  : true cart friction (drawn as dashed reference line)
    t_event : optional time to draw a vertical dashed line (e.g. mass drop)
    save_path : if provided, save figure here and close; otherwise show
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    axes[0].plot(t, m_hat, color="steelblue", label="m_hat")
    if true_m is not None:
        axes[0].axhline(true_m, color="firebrick", linewidth=1,
                        linestyle="--", label=f"true m = {true_m} kg")
    axes[0].set_ylabel("m (kg)")
    axes[0].set_title("Pendulum mass estimate")
    axes[0].legend()

    axes[1].plot(t, b_hat, color="seagreen", label="b_hat")
    if true_b is not None:
        axes[1].axhline(true_b, color="firebrick", linewidth=1,
                        linestyle="--", label=f"true b = {true_b} N·s/m")
    axes[1].set_ylabel("b (N·s/m)")
    axes[1].set_title("Cart friction estimate")
    axes[1].legend()

    if t_event is not None:
        for ax in axes:
            ax.axvline(t_event, color="red", linewidth=1.0, linestyle="--", alpha=0.7)

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
