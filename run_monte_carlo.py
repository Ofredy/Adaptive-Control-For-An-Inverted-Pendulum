import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.dynamics import CartPendulum
from simulation.controller import lyapunov_control, ThetaDdotEstimator, XDdotEstimator, ParameterEstimator
from simulation.visualize import animate

# --- Args ---
parser = argparse.ArgumentParser(description="Monte Carlo cart-pendulum simulation")
parser.add_argument("-k",            type=float, default=50.0, help="Proportional gain on theta")
parser.add_argument("-p",            type=float, default=1.0,  help="Derivative gain on theta_dot")
parser.add_argument("--cart-weight", type=float, default=0.1,  help="Scale factor on cart position/velocity gains")
parser.add_argument("--runs",        type=int,   default=50,   help="Number of Monte Carlo runs")
parser.add_argument("--delta-m",     type=float, default=0.1,  help="Mass added to pendulum at t=40s (kg)")
parser.add_argument("--b",           type=float, default=0.1,  help="Controller's assumed friction coefficient (N·s/m)")
parser.add_argument("--adaptive",    action="store_true",      help="Enable online parameter adaptation for m and b")
parser.add_argument("--gamma-m",     type=float, default=1.0,  help="Learning rate for mass adaptation")
parser.add_argument("--gamma-b",     type=float, default=1.0,  help="Learning rate for friction adaptation")
parser.add_argument("--no-compare",  action="store_true",      help="Run single mode only (disables default comparison)")
parser.add_argument("--animate",     action="store_true",      help="Animate the first Monte Carlo run")
args = parser.parse_args()

# --- True system parameters ---
true_params = dict(M=1.0, m=0.3, L=1.0, g=9.81, b=0.2)

# --- Controller's initial belief ---
ctrl_params_init = dict(M=1.0, m=0.3, L=1.0, g=9.81, b=args.b)

# --- Simulation settings ---
t_end       = 200.0
t_mass_drop = 40.0
dt          = 0.01

k           = args.k
p           = args.p
cart_weight = args.cart_weight
n_runs      = args.runs
delta_m     = args.delta_m

theta0_values = np.linspace(np.radians(-25), np.radians(25), n_runs)

os.makedirs("results", exist_ok=True)


def run_batch(adaptive, oracle=False):
    """
    Run all Monte Carlo trials.
    adaptive : use ParameterEstimator to update m_hat/b_hat online
    oracle   : controller always knows the true params (perfect knowledge baseline)
    """
    all_states = []
    all_forces = []
    all_m_hats = []
    all_b_hats = []
    t_arr      = None

    if oracle:
        print(f"Running {n_runs} runs (oracle — perfect model knowledge, no mass drop, true b={true_params['b']})...")
    elif adaptive:
        print(f"Running {n_runs} runs (adaptive — delta_m={delta_m} at t={t_mass_drop}s, gamma_m={args.gamma_m}, gamma_b={args.gamma_b}, ctrl b={ctrl_params_init['b']})...")
    else:
        print(f"Running {n_runs} runs (non-adaptive — delta_m={delta_m} at t={t_mass_drop}s, ctrl b={ctrl_params_init['b']}, true b={true_params['b']})...")

    for i, theta0 in enumerate(theta0_values):
        x0    = [0.0, 0.0, theta0, 0.0]
        state = np.array(x0, dtype=float)

        system           = CartPendulum(**true_params)
        theta_ddot_e_est = ThetaDdotEstimator(theta_dot_0=0.0)
        x_ddot_est       = XDdotEstimator(x_dot_0=0.0)
        ctrl_params      = true_params.copy() if oracle else ctrl_params_init.copy()
        param_est        = ParameterEstimator(m_hat_0=ctrl_params['m'], b_hat_0=ctrl_params['b'], gamma_m=args.gamma_m, gamma_b=args.gamma_b)

        mass_dropped = False
        u            = 0.0
        t            = 0.0
        t_log        = [t]
        state_log    = [state.copy()]
        force_log    = []
        m_hat_log    = [ctrl_params['m']]
        b_hat_log    = [ctrl_params['b']]

        while t < t_end - 1e-10:
            if not oracle and not mass_dropped and t >= t_mass_drop:
                system.m    += delta_m
                mass_dropped = True

            x_dot        = state[1]
            theta_dot    = state[3]
            x_ddot_e     = x_ddot_est.update(x_dot, dt)
            theta_ddot_e = theta_ddot_e_est.update(theta_dot, dt)

            if adaptive:
                m_hat, b_hat     = param_est.update(state, x_ddot_e, theta_ddot_e, u, ctrl_params, dt)
                ctrl_params['m'] = m_hat
                ctrl_params['b'] = b_hat

            u = lyapunov_control(state, theta_ddot_e, ctrl_params, k=k, p=p, cart=cart_weight)
            force_log.append(u)

            state = system.step_rk4(state, t, dt, force_fn=lambda t, s: u)
            t    += dt
            t_log.append(t)
            state_log.append(state.copy())
            m_hat_log.append(ctrl_params['m'])
            b_hat_log.append(ctrl_params['b'])

        force_log.append(force_log[-1])

        all_states.append(np.array(state_log).T)
        all_forces.append(np.array(force_log))
        all_m_hats.append(np.array(m_hat_log))
        all_b_hats.append(np.array(b_hat_log))

        if t_arr is None:
            t_arr = np.array(t_log)

        print(f"  Run {i+1:3d}/{n_runs}  theta0={np.degrees(theta0):+.1f} deg")

    return t_arr, all_states, all_forces, all_m_hats, all_b_hats


def plot_states_batch(t_arr, all_states, all_forces, tag="", show_event=True, label=""):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    state_labels = ["x (m)", "ẋ (m/s)", "θ (deg)", "θ̇ (deg/s)"]
    titles = ["Cart position", "Cart velocity", "Pendulum angle", "Pendulum angular velocity"]

    for run_states in all_states:
        x, x_dot, theta, theta_dot = run_states
        data = [x, x_dot, np.degrees(theta), np.degrees(theta_dot)]
        for ax, d in zip(axes, data):
            ax.plot(t_arr, d, color="black", linewidth=0.6, alpha=0.4)

    for ax, lbl, ttl in zip(axes, state_labels, titles):
        ax.set_ylabel(lbl)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, linewidth=0.5, alpha=0.5)
        if show_event:
            ax.axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)

    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[-1].set_xlabel("Time (s)")
    if label:
        fig.suptitle(label, fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
    path = f"results/monte_carlo_states{tag}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mean_error_comparison(t_arr, states_nominal, states_adaptive):
    """Plot mean absolute error per state, adaptive vs non-adaptive."""
    labels  = ["x (m)", "ẋ (m/s)", "θ (deg)", "θ̇ (deg/s)"]
    scales  = [1.0, 1.0, 180/np.pi, 180/np.pi]   # convert angles to degrees

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for idx, (ax, lbl, sc) in enumerate(zip(axes, labels, scales)):
        nom_err  = np.mean([np.abs(s[idx] * sc) for s in states_nominal],  axis=0)
        adap_err = np.mean([np.abs(s[idx] * sc) for s in states_adaptive], axis=0)

        ax.plot(t_arr, nom_err,  color="firebrick",  linewidth=1.2, label="non-adaptive")
        ax.plot(t_arr, adap_err, color="steelblue",  linewidth=1.2, label="adaptive")
        ax.axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_ylabel(f"mean |{lbl}|")
        ax.set_title(f"Mean absolute error — {lbl}")
        ax.legend()
        ax.grid(True, linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("results/monte_carlo_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved: results/monte_carlo_comparison.png")


def plot_control_effort_comparison(t_arr, forces_nominal, forces_adaptive):
    """Plot mean absolute control force, adaptive vs non-adaptive."""
    nom_effort  = np.mean([np.abs(f) for f in forces_nominal],  axis=0)
    adap_effort = np.mean([np.abs(f) for f in forces_adaptive], axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_arr, nom_effort,  color="firebrick", linewidth=1.2, label="non-adaptive")
    ax.plot(t_arr, adap_effort, color="steelblue", linewidth=1.2, label="adaptive")
    ax.axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylabel("mean |F| (N)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Mean absolute control effort — adaptive vs non-adaptive")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/monte_carlo_control_effort.png", dpi=150)
    plt.close(fig)
    print("Saved: results/monte_carlo_control_effort.png")


# --- Run ---
if not args.no_compare:
    t_arr, states_nom,    forces_nom,    _, _           = run_batch(adaptive=False)
    t_arr, states_adap,   forces_adap,   m_hats, b_hats = run_batch(adaptive=True)
    t_arr, states_oracle, forces_oracle, _, _           = run_batch(adaptive=False, oracle=True)

    plot_states_batch(t_arr, states_nom,    forces_nom,    tag="_nominal",  label="Non-Adaptive Controller (fixed wrong params)")
    plot_states_batch(t_arr, states_adap,   forces_adap,   tag="_adaptive", label="Adaptive Controller (online m & b estimation)")
    plot_mean_error_comparison(t_arr, states_nom, states_adap)
    plot_control_effort_comparison(t_arr, forces_nom, forces_adap)

    # Oracle — separate plot showing perfect-knowledge controller performance (no mass drop)
    plot_states_batch(t_arr, states_oracle, forces_oracle, tag="_oracle", show_event=False, label="Oracle Controller (perfect model knowledge, no mass drop)")

    # Param estimates from adaptive run
    print("Saving parameter estimates plot...")
    true_m = true_params['m'] + delta_m
    true_b = true_params['b']
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for m_hat in m_hats:
        axes[0].plot(t_arr, m_hat, color="steelblue", linewidth=0.6, alpha=0.4)
    axes[0].axhline(true_m, color="firebrick", linewidth=1, linestyle="--", label=f"true m = {true_m} kg")
    axes[0].axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("m (kg)")
    axes[0].set_title("Pendulum mass estimate across runs")
    axes[0].legend()
    axes[0].grid(True, linewidth=0.5, alpha=0.5)
    for b_hat in b_hats:
        axes[1].plot(t_arr, b_hat, color="seagreen", linewidth=0.6, alpha=0.4)
    axes[1].axhline(true_b, color="firebrick", linewidth=1, linestyle="--", label=f"true b = {true_b} N·s/m")
    axes[1].axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("b (N·s/m)")
    axes[1].set_title("Cart friction estimate across runs")
    axes[1].legend()
    axes[1].grid(True, linewidth=0.5, alpha=0.5)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("results/monte_carlo_param_estimates.png", dpi=150)
    plt.close(fig)
    print("Saved: results/monte_carlo_param_estimates.png")

else:
    t_arr, all_states, all_forces, all_m_hats, all_b_hats = run_batch(adaptive=args.adaptive)

    plot_states_batch(t_arr, all_states, all_forces)

    # Forces
    fig, ax = plt.subplots(figsize=(12, 4))
    for run_forces in all_forces:
        ax.plot(t_arr, run_forces, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylabel("F (N)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Applied force")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig("results/monte_carlo_forces.png", dpi=150)
    plt.close(fig)
    print("Saved: results/monte_carlo_forces.png")

    # Param estimates
    if args.adaptive:
        print("Saving parameter estimates plot...")
        true_m = true_params['m'] + delta_m
        true_b = true_params['b']

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        for m_hat in all_m_hats:
            axes[0].plot(t_arr, m_hat, color="steelblue", linewidth=0.6, alpha=0.4)
        axes[0].axhline(true_m, color="firebrick", linewidth=1,
                        linestyle="--", label=f"true m = {true_m} kg")
        axes[0].axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        axes[0].set_ylabel("m (kg)")
        axes[0].set_title("Pendulum mass estimate across runs")
        axes[0].legend()

        for b_hat in all_b_hats:
            axes[1].plot(t_arr, b_hat, color="seagreen", linewidth=0.6, alpha=0.4)
        axes[1].axhline(true_b, color="firebrick", linewidth=1,
                        linestyle="--", label=f"true b = {true_b} N·s/m")
        axes[1].axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)
        axes[1].set_ylabel("b (N·s/m)")
        axes[1].set_title("Cart friction estimate across runs")
        axes[1].legend()

        axes[-1].set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(True, linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.savefig("results/monte_carlo_param_estimates.png", dpi=150)
        plt.close(fig)
        print("Saved: results/monte_carlo_param_estimates.png")

# --- Optional animation ---
if args.animate:
    import matplotlib
    matplotlib.use("TkAgg")
    print("Animating first run...")
    animate(t_arr, all_states[0], true_params)
