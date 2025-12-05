# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
g = 9.81  # m/s^2

def rk4_step(y, t, dt, derivs, *args):
    """Perform a single RK4 step.
    y: state vector
    t: current time
    dt: time step
    derivs: function returning dy/dt given (y, t, *args)
    *args: additional arguments passed to derivs
    """
    k1 = derivs(y, t, *args)
    k2 = derivs(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = derivs(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = derivs(y + dt * k3, t + dt, *args)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def pendulum_derivs(state, t, L):
    """Derivatives for the linearized pendulum.
    state = [theta, omega]
    """
    theta, omega = state
    dtheta_dt = omega
    domega_dt = -(g / L) * theta
    return np.array([dtheta_dt, domega_dt])

def simulate_pendulum(L, theta0, omega0=0.0, dt=0.001, t_max=10.0):
    """Return time array and theta(t) using RK4 for the linear pendulum."""
    t_vals = np.arange(0, t_max + dt, dt)
    theta_vals = np.empty_like(t_vals)
    state = np.array([theta0, omega0])
    for i, t in enumerate(t_vals):
        theta_vals[i] = state[0]
        state = rk4_step(state, t, dt, pendulum_derivs, L)
    return t_vals, theta_vals

def analytical_theta(t, theta0, L):
    omega0 = np.sqrt(g / L)
    return theta0 * np.cos(omega0 * t)

def compute_period_from_theta(t, theta):
    """Estimate period from the first two zero‑crossings (positive to negative).
    Returns period in seconds.
    """
    signs = np.sign(theta)
    zero_crossings = np.where(np.diff(signs) != 0)[0]
    if len(zero_crossings) < 2:
        return np.nan
    t1 = t[zero_crossings[0]]
    t2 = t[zero_crossings[1]]
    half_period = t2 - t1
    return 2 * half_period

def experiment1():
    L = 1.0  # meters
    theta0 = 0.1  # radians, small angle
    omega0 = np.sqrt(g / L)
    T = 2 * np.pi / omega0
    t_max = 5 * T
    dt = 0.001
    t_num, theta_num = simulate_pendulum(L, theta0, dt=dt, t_max=t_max)
    theta_ana = analytical_theta(t_num, theta0, L)
    plt.figure(figsize=(8, 4))
    plt.plot(t_num, theta_num, label='Numerical (RK4)')
    plt.plot(t_num, theta_ana, '--', label='Analytical')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Pendulum Angle vs Time (Small‑Angle Approximation)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('angle_vs_time.png')
    plt.close()

def experiment2():
    theta0 = 0.05  # keep angle small for all lengths
    dt = 0.001
    lengths = np.linspace(0.1, 2.0, 20)
    measured_periods = []
    sqrt_lengths = []
    for L in lengths:
        omega0 = np.sqrt(g / L)
        T_est = 2 * np.pi / omega0  # analytical period for reference
        t_max = 3 * T_est  # simulate a few periods
        t, theta = simulate_pendulum(L, theta0, dt=dt, t_max=t_max)
        period = compute_period_from_theta(t, theta)
        measured_periods.append(period)
        sqrt_lengths.append(np.sqrt(L))
    measured_periods = np.array(measured_periods)
    sqrt_lengths = np.array(sqrt_lengths)
    # Linear fit: period = a * sqrt(L) + b
    A = np.vstack([sqrt_lengths, np.ones_like(sqrt_lengths)]).T
    a, b = np.linalg.lstsq(A, measured_periods, rcond=None)[0]
    plt.figure(figsize=(8, 4))
    plt.plot(sqrt_lengths, measured_periods, 'o', label='Measured periods')
    plt.plot(sqrt_lengths, a * sqrt_lengths + b, '-', label='Linear fit')
    plt.xlabel('√Length (m^{1/2})')
    plt.ylabel('Period (s)')
    plt.title('Pendulum Period vs √Length (Small‑Angle)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('period_vs_sqrt_length.png')
    plt.close()
    return a  # slope, expected to be 2π/√g

def main():
    experiment1()
    slope = experiment2()
    print('Answer:', slope)

if __name__ == '__main__':
    main()

