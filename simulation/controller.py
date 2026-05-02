import numpy as np


class GainScheduler:
    """
    Two-zone gain scheduler with linear interpolation on k and p only.

    Cart weight is constant throughout — only theta gains are scheduled.

    Far zone  (|theta| >= theta_far) : high k and p
    Near zone (|theta| <= theta_near): lower k and p
    Between   : linear interpolation
    """

    def __init__(self,
                 theta_near=np.radians(5),
                 theta_far=np.radians(15),
                 k_far=50.0, p_far=1.0,
                 k_near=25.0, p_near=0.5):
        self.theta_near = theta_near
        self.theta_far  = theta_far
        self.k_far      = k_far
        self.p_far      = p_far
        self.k_near     = k_near
        self.p_near     = p_near

    def update(self, theta):
        """Return (k, p) for the current theta."""
        abs_theta = abs(theta)

        if abs_theta >= self.theta_far:
            return self.k_far, self.p_far
        elif abs_theta <= self.theta_near:
            return self.k_near, self.p_near
        else:
            alpha = (abs_theta - self.theta_near) / (self.theta_far - self.theta_near)
            k = (1 - alpha) * self.k_near + alpha * self.k_far
            p = (1 - alpha) * self.p_near + alpha * self.p_far
            return k, p


def lyapunov_control(state, theta_ddot_e, params, k, p, cart=0.5):
    """
    Lyapunov-based nonlinear feedback control law.

    u = [(M+m)g / cos(θ)] sin(θ)
      + [(M+m)L / cos(θ)] (k θ + p θ')
      + mL θ_e'' cos(θ)
      - mL θ'² sin(θ)
      + b x'

    Parameters
    ----------
    state        : [x, x_dot, theta, theta_dot]
    theta_ddot_e : estimated pendulum angular acceleration (finite difference)
    params       : dict with keys M, m, L, g, b
    k            : proportional gain on theta
    p            : derivative gain on theta_dot

    Returns
    -------
    u : float, force to apply to the cart (N)
    """
    x, x_dot, theta, theta_dot = state
    M, m, L, g, b = params['M'], params['m'], params['L'], params['g'], params['b']

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    u = ((M + m) * g / cos_th) * sin_th \
      + ((M + m) * L / cos_th) * (k * theta + p * theta_dot + cart * k * x + cart * p * x_dot) \
      + m * L * theta_ddot_e * cos_th \
      - m * L * theta_dot**2 * sin_th \
      + b * x_dot

    return np.clip(u, -50.0, 50.0)


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


class XDdotEstimator:
    """Finite-difference estimator for cart acceleration."""

    def __init__(self, x_dot_0):
        self._x_dot_prev = x_dot_0

    def update(self, x_dot, dt):
        x_ddot_e = (x_dot - self._x_dot_prev) / dt
        self._x_dot_prev = x_dot
        return x_ddot_e


class ParameterEstimator:
    """
    Online gradient-descent estimator for pendulum mass m and cart friction b.

    Uses the x-equation regressor:
        m*φ₁ + b*x_dot = u - M*x_ddot    (= y)
        φ₁ = x_ddot + L*cos(θ)*θ_ddot - L*θ_dot²*sin(θ)

    Prediction error:
        ε = m_hat*φ₁ + b_hat*x_dot - y

    Normalized gradient update:
        m_hat -= γ_m * ε * φ₁ / norm * dt
        b_hat -= γ_b * ε * x_dot / norm * dt
        norm  = φ₁² + x_dot² + 1
    """

    def __init__(self, m_hat_0, b_hat_0, gamma_m=0.2, gamma_b=0.1):
        self.m_hat   = m_hat_0
        self.b_hat   = b_hat_0
        self.gamma_m = gamma_m
        self.gamma_b = gamma_b

    def update(self, state, x_ddot, theta_ddot, u, params, dt):
        x, x_dot, theta, theta_dot = state
        M, L = params['M'], params['L']

        phi1 = x_ddot + L * np.cos(theta) * theta_ddot - L * theta_dot**2 * np.sin(theta)
        phi2 = x_dot
        y    = u - M * x_ddot

        eps  = self.m_hat * phi1 + self.b_hat * phi2 - y
        norm = phi1**2 + phi2**2 + 1.0

        self.m_hat -= self.gamma_m * eps * phi1 / norm * dt
        self.b_hat -= self.gamma_b * eps * phi2 / norm * dt

        return self.m_hat, self.b_hat
