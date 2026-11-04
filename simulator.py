import numpy as np


def simulate(dyn, x0, policy, t_final, dt=1e-3):
    x = x0.copy()
    times, states = [0.0], [x.copy()]
    for k in range(int(t_final / dt)):
        t = k * dt
        u = policy(t, x)
        k1 = dyn.f(x, u)
        k2 = dyn.f(x + dt / 2 * k1, u)
        k3 = dyn.f(x + dt / 2 * k2, u)
        k4 = dyn.f(x + dt * k3, u)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # sol
        on_ground = x[:, 2] <= 0
        x[on_ground, 2] = 0
        x[on_ground, 3:] = 0

        times.append(t + dt)
        states.append(x.copy())
    return np.array(times), np.array(states)
