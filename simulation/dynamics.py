import numpy as np
from scipy.integrate import solve_ivp


class CartPendulum:
    """
    Full nonlinear cart-pendulum dynamics.

    State vector: [x, x_dot, theta, theta_dot]
      x     : cart position (m)
      theta : pendulum angle from upright vertical (rad)
              theta=0 -> pendulum pointing straight up

    Parameters
    ----------
    M : float
        Cart mass (kg)
    m : float
        Pendulum bob mass (kg)
    L : float
        Pendulum rod length (m)
    g : float
        Gravitational acceleration (m/s^2)
    b : float
        Cart friction coefficient (N·s/m)
    """

    def __init__(self, M=1.0, m=0.3, L=0.5, g=9.81, b=0.1):
        self.M = M
        self.m = m
        self.L = L
        self.g = g
        self.b = b

    def derivatives(self, t, state, force_fn):
        """
        Compute state derivatives using the full nonlinear EOM.

        EOM (derived from Lagrangian mechanics):
          (M+m)*x_ddot + m*L*theta_ddot*cos(theta) - m*L*theta_dot^2*sin(theta) = F - b*x_dot
          m*L^2*theta_ddot + m*L*x_ddot*cos(theta) = m*g*L*sin(theta)

        Solved in matrix form:
          [M+m,        m*L*cos(theta)] [x_ddot    ]   [F - b*x_dot + m*L*theta_dot^2*sin(theta)]
          [m*L*cos(theta), m*L^2     ] [theta_ddot] = [m*g*L*sin(theta)                        ]
        """
        x, x_dot, theta, theta_dot = state
        F = force_fn(t, state)

        M, m, L, g, b = self.M, self.m, self.L, self.g, self.b

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)

        # Mass matrix
        A = np.array([
            [M + m,          m * L * cos_th],
            [m * L * cos_th, m * L ** 2    ]
        ])

        # Right-hand side
        rhs = np.array([
            F - b * x_dot + m * L * theta_dot ** 2 * sin_th,
            m * g * L * sin_th
        ])

        x_ddot, theta_ddot = np.linalg.solve(A, rhs)

        return [x_dot, x_ddot, theta_dot, theta_ddot]

    def step_rk4(self, state, t, dt, force_fn):
        """Single fixed-step RK4 integration."""
        s = np.array(state, dtype=float)
        k1 = np.array(self.derivatives(t,          s,                force_fn))
        k2 = np.array(self.derivatives(t + dt/2,   s + dt/2 * k1,   force_fn))
        k3 = np.array(self.derivatives(t + dt/2,   s + dt/2 * k2,   force_fn))
        k4 = np.array(self.derivatives(t + dt,     s + dt   * k3,   force_fn))
        return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def simulate(self, t_span, x0, force_fn, dt=0.01):
        """
        Integrate the equations of motion over t_span.

        Parameters
        ----------
        t_span  : (t_start, t_end) in seconds
        x0      : initial state [x, x_dot, theta, theta_dot]
        force_fn: callable(t, state) -> float, horizontal force on cart (N)
        dt      : output time step (s)

        Returns
        -------
        t      : 1-D array of time points
        states : (4, N) array — rows are x, x_dot, theta, theta_dot
        """
        t_eval = np.arange(t_span[0], t_span[1], dt)

        sol = solve_ivp(
            fun=lambda t, s: self.derivatives(t, s, force_fn),
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        return sol.t, sol.y
