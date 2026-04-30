import argparse
import os
import numpy as np
from simulation.dynamics import CartPendulum
from simulation.controller import lyapunov_control, ThetaDdotEstimator
from simulation.visualize import animate, plot_states

parser = argparse.ArgumentParser(description="Cart-pendulum Lyapunov controller")
parser.add_argument("-k",    type=float, default=50.0, help="Proportional gain on theta")
parser.add_argument("-p",    type=float, default=1.0,  help="Derivative gain on theta_dot")
parser.add_argument("--cart-weight", type=float, default=0.1,  help="Scale factor on cart position/velocity gains")
parser.add_argument("--animate", action="store_true", help="Show animation after simulation")
args = parser.parse_args()

k           = args.k
p           = args.p
cart_weight = args.cart_weight

# --- System parameters ---
params = dict(
    M=1.0,    # cart mass (kg)
    m=0.3,    # pendulum mass (kg)
    L=1.0,    # rod length (m)
    g=9.81,   # gravity (m/s^2)
    b=0.1,    # cart friction (N·s/m)
)

# --- Initial conditions [x, x_dot, theta, theta_dot] ---
# theta=0 is straight up; small angle offset to test stabilization
x0 = [0.0, 0.0, np.radians(25), 0.0]

# --- Simulation settings ---
t_span = (0.0, 50.0)   # seconds
dt     = 0.01          # fixed time step (s)

# --- Setup ---
system    = CartPendulum(**params)
estimator = ThetaDdotEstimator(theta_dot_0=x0[3])

state   = np.array(x0, dtype=float)
t       = 0.0
t_end   = t_span[1]

t_log      = [t]
state_log  = [state.copy()]
force_log  = []

# --- Simulation loop ---
while t < t_end - 1e-10:
    theta_dot    = state[3]
    theta_ddot_e = estimator.update(theta_dot, dt)
    u = lyapunov_control(state, theta_ddot_e, params, k=k, p=p, cart=cart_weight)
    force_log.append(u)

    state = system.step_rk4(state, t, dt, force_fn=lambda t, s: u)
    t    += dt

    t_log.append(t)
    state_log.append(state.copy())

# Final force entry to match length
force_log.append(force_log[-1])

t_arr      = np.array(t_log)
states_arr = np.array(state_log).T   # shape (4, N)
forces_arr = np.array(force_log)

# --- Visualize ---
os.makedirs("results", exist_ok=True)
print("Saving state trajectories...")
plot_states(t_arr, states_arr, forces=forces_arr, save_path="results/single_run_states.png")

if args.animate:
    print("Running animation...")
    animate(t_arr, states_arr, params)
