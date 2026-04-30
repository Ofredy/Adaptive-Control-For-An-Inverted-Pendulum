import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation.dynamics import CartPendulum
from simulation.controller import lyapunov_control, ThetaDdotEstimator, GainScheduler
from simulation.visualize import animate

# --- Args ---
parser = argparse.ArgumentParser(description="Monte Carlo cart-pendulum simulation")
parser.add_argument("-k",            type=float, default=50.0, help="Proportional gain on theta")
parser.add_argument("-p",            type=float, default=1.0,  help="Derivative gain on theta_dot")
parser.add_argument("--cart-weight", type=float, default=0.1,  help="Scale factor on cart position/velocity gains")
parser.add_argument("--runs",        type=int,   default=50,   help="Number of Monte Carlo runs")
parser.add_argument("--delta-m",     type=float, default=0.2,  help="Mass added to pendulum at t=40s (kg)")
parser.add_argument("--animate",     action="store_true",      help="Animate the first Monte Carlo run")
args = parser.parse_args()

# --- Nominal system parameters (what the controller knows) ---
params = dict(
    M=1.0,
    m=0.3,
    L=1.0,
    g=9.81,
    b=0.1,
)

# --- Simulation settings ---
t_end       = 100.0
t_mass_drop = 40.0
dt          = 0.01

k           = args.k
p           = args.p
cart_weight = args.cart_weight
n_runs      = args.runs
delta_m     = args.delta_m

# theta_0 linearly spaced from -25 to +25 deg; all other ICs zero (stagnant)
theta0_values = np.linspace(np.radians(-25), np.radians(25), n_runs)

os.makedirs("results", exist_ok=True)

# --- Storage across runs ---
all_states = []
all_forces = []
t_arr      = None

print(f"Running {n_runs} Monte Carlo simulations (mass drop of +{delta_m} kg at t={t_mass_drop}s)...")

for i, theta0 in enumerate(theta0_values):
    x0    = [0.0, 0.0, theta0, 0.0]
    state = np.array(x0, dtype=float)

    system    = CartPendulum(**params)   # starts with nominal mass
    estimator = ThetaDdotEstimator(theta_dot_0=0.0)
    scheduler = GainScheduler()

    mass_dropped = False

    t         = 0.0
    t_log     = [t]
    state_log = [state.copy()]
    force_log = []

    while t < t_end - 1e-10:
        # Drop extra mass onto pendulum at t=40s
        if not mass_dropped and t >= t_mass_drop:
            system.m  += delta_m
            mass_dropped = True

        theta_dot    = state[3]
        theta_ddot_e = estimator.update(theta_dot, dt)
        k_s, p_s     = scheduler.update(state[2])

        # Controller always uses nominal params — unaware of mass change
        u = lyapunov_control(state, theta_ddot_e, params, k=k_s, p=p_s, cart=cart_weight)
        force_log.append(u)

        state = system.step_rk4(state, t, dt, force_fn=lambda t, s: u)
        t    += dt
        t_log.append(t)
        state_log.append(state.copy())

    force_log.append(force_log[-1])

    run_t      = np.array(t_log)
    run_states = np.array(state_log).T
    run_forces = np.array(force_log)

    all_states.append(run_states)
    all_forces.append(run_forces)

    if t_arr is None:
        t_arr = run_t

    print(f"  Run {i+1:3d}/{n_runs}  theta0={np.degrees(theta0):+.1f} deg")

# --- Plot states ---
print("Saving states plot...")
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
labels = ["x (m)", "ẋ (m/s)", "θ (deg)", "θ̇ (deg/s)"]
titles = ["Cart position", "Cart velocity", "Pendulum angle", "Pendulum angular velocity"]

for run_states in all_states:
    x, x_dot, theta, theta_dot = run_states
    data = [x, x_dot, np.degrees(theta), np.degrees(theta_dot)]
    for ax, d in zip(axes, data):
        ax.plot(t_arr, d, color="black", linewidth=0.6, alpha=0.4)

for ax, lbl, ttl in zip(axes, labels, titles):
    ax.set_ylabel(lbl)
    ax.set_title(ttl)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.axvline(t_mass_drop, color="red", linewidth=1.0, linestyle="--", alpha=0.7)

axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[-1].set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("results/monte_carlo_states.png", dpi=150)
plt.close(fig)
print("Saved: results/monte_carlo_states.png")

# --- Plot forces ---
print("Saving forces plot...")
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

# --- Optional animation of first run ---
if args.animate:
    import matplotlib
    matplotlib.use("TkAgg")
    print("Animating first run (theta0 = -25 deg)...")
    animate(t_arr, all_states[0], params)
