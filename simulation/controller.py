import numpy as np


def lyapunov_control(state, theta_ddot_e, params, k, p):
    """
    Lyapunov-based nonlinear feedback control law.

    u = [(M+m)g / cos(θ)] sin(θ)
      + [(M+m)L / cos(θ)] (k θ + p θ')
      + mL θ_e'' cos(θ)
      - mL θ'² sin(θ)
      + b x'

    Parameters
    ----------
    state       : [x, x_dot, theta, theta_dot]
    theta_ddot_e: estimated pendulum angular acceleration (finite difference)
    params      : dict with keys M, m, L, g, b
    k           : proportional gain on theta
    p           : derivative gain on theta_dot

    Returns
    -------
    u : float, force to apply to the cart (N)
    """
    _, x_dot, theta, theta_dot = state
    M, m, L, g, b = params['M'], params['m'], params['L'], params['g'], params['b']

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    u = ((M + m) * g / cos_th) * sin_th \
      + ((M + m) * L / cos_th) * (k * theta + p * theta_dot) \
      + m * L * theta_ddot_e * cos_th \
      - m * L * theta_dot**2 * sin_th \
      + b * x_dot

    return u


class ThetaDdotEstimator:
    """
    Finite-difference estimator for angular acceleration.

    Initializes to zero, then at each call:
        θ_e'' = (θ'_t - θ'_{t-1}) / dt
    """

    def __init__(self, theta_dot_0):
        self._theta_dot_prev = theta_dot_0

    def update(self, theta_dot, dt):
        theta_ddot_e = (theta_dot - self._theta_dot_prev) / dt
        self._theta_dot_prev = theta_dot
        return theta_ddot_e
