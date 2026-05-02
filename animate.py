import argparse
import numpy as np

from simulation.dynamics import CartPendulum
from simulation.controller import lyapunov_control, ThetaDdotEstimator, XDdotEstimator, ParameterEstimator
from simulation.visualize import animate

parser = argparse.ArgumentParser(description="Animate a single cart-pendulum run")
parser.add_argument("--mode",        choices=["nominal", "adaptive", "oracle"], default="adaptive",
                    help="nominal: fixed wrong params | adaptive: online adaptation | oracle: perfect knowledge, no mass drop")
parser.add_argument("--theta0",      type=float, default=-35.0, help="Initial pendulum angle (degrees)")
parser.add_argument("-k",            type=float, default=50.0,  help="Proportional gain on theta")
parser.add_argument("-p",            type=float, default=1.0,   help="Derivative gain on theta_dot")
parser.add_argument("--cart-weight", type=float, default=0.1,   help="Scale factor on cart position/velocity gains")
parser.add_argument("--delta-m",     type=float, default=0.1,   help="Mass added to pendulum mid-run (kg)")
parser.add_argument("--t-drop",      type=float, default=40.0,  help="Time of mass increase (s)")
parser.add_argument("--gamma-m",     type=float, default=1.0,   help="Learning rate for mass adaptation")
parser.add_argument("--gamma-b",     type=float, default=1.0,   help="Learning rate for friction adaptation")
parser.add_argument("--export",      action="store_true",       help="Save animation to results/ instead of rendering")
args = parser.parse_args()

# --- True system parameters ---
true_params = dict(M=1.0, m=0.3, L=1.0, g=9.81, b=0.2)

# --- Controller's initial belief ---
ctrl_params = dict(M=1.0, m=0.3, L=1.0, g=9.81, b=0.1)
if args.mode == "oracle":
    ctrl_params = true_params.copy()

# --- Initial conditions ---
x0 = [0.0, 0.0, np.radians(args.theta0), 0.0]

# --- Simulation settings ---
t_end   = 100.0
t_drop  = args.t_drop
delta_m = args.delta_m
dt      = 0.01
k, p    = args.k, args.p

# --- Setup ---
system           = CartPendulum(**true_params)
theta_ddot_est   = ThetaDdotEstimator(theta_dot_0=x0[3])
x_ddot_est       = XDdotEstimator(x_dot_0=x0[1])
param_est        = ParameterEstimator(m_hat_0=ctrl_params['m'], b_hat_0=ctrl_params['b'],
                                      gamma_m=args.gamma_m, gamma_b=args.gamma_b)

state        = np.array(x0, dtype=float)
t            = 0.0
u            = 0.0
mass_dropped = False
t_log        = [t]
state_log    = [state.copy()]

if args.mode == "oracle":
    print(f"Mode: oracle | theta0: {args.theta0}° | no mass drop | perfect model knowledge")
else:
    print(f"Mode: {args.mode} | theta0: {args.theta0}° | delta_m: {delta_m} kg at t={t_drop}s")

while t < t_end - 1e-10:
    if args.mode != "oracle" and not mass_dropped and t >= t_drop:
        system.m    += delta_m
        mass_dropped = True

    x_dot        = state[1]
    theta_dot    = state[3]
    x_ddot_e     = x_ddot_est.update(x_dot, dt)
    theta_ddot_e = theta_ddot_est.update(theta_dot, dt)

    if args.mode == "adaptive":
        m_hat, b_hat     = param_est.update(state, x_ddot_e, theta_ddot_e, u, ctrl_params, dt)
        ctrl_params['m'] = m_hat
        ctrl_params['b'] = b_hat

    u     = lyapunov_control(state, theta_ddot_e, ctrl_params, k=k, p=p, cart=args.cart_weight)
    state = system.step_rk4(state, t, dt, force_fn=lambda t, s: u)
    t    += dt

    t_log.append(t)
    state_log.append(state.copy())

t_arr      = np.array(t_log)
states_arr = np.array(state_log).T

import os
os.makedirs("results", exist_ok=True)

t_event   = t_drop if args.mode != "oracle" else None
save_path = f"results/animation_{args.mode}.mp4" if args.export else None

titles = {
    "nominal":  "Non-Adaptive Controller (fixed wrong params)",
    "adaptive": "Adaptive Controller (online m & b estimation)",
    "oracle":   "Oracle Controller (perfect model knowledge, no mass drop)",
}

animate(t_arr, states_arr, true_params,
        save_path=save_path,
        speedup=3.0,
        t_event=t_event,
        delta_m=delta_m if t_event else None,
        title=titles[args.mode])
