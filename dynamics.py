import numpy as np
import torch
from scenario import G, MASS, THRUST_MIN, THRUST_MAX, MAX_TILT


class PointMass6D:
    # x = (p, v), u = accélération de poussée
    # coût L(u) = c |u - g e_z|^2 : l'énergie est comptée à partir du vol stationnaire
    dim = 6

    def f(self, x, u):
        dx = np.zeros_like(x)
        dx[:, 0:3] = x[:, 3:6]
        dx[:, 3:6] = u
        dx[:, 5] -= G
        return dx

    def hamiltonian(self, x, p, c):
        # H(x, p) = sup_u { -p.f(x, u) - L(u) }, p = grad phi
        return -(p[:, 0:3] * x[:, 3:6]).sum(1, keepdim=True) + (p[:, 3:6] ** 2).sum(1, keepdim=True) / (4 * c)

    def optimal_control(self, x, p, c):
        u = -p[:, 3:6] / (2 * c)
        u[:, 2] += G
        return u

    def clip_control(self, u):
        # limites Crazyflie : poussée et inclinaison
        uz = u[:, 2].clamp(THRUST_MIN / MASS, THRUST_MAX / MASS)
        uh = u[:, 0:2]
        k = (uz * np.tan(MAX_TILT) / uh.norm(dim=1).clamp(min=1e-9)).clamp(max=1.0)
        return torch.cat([uh * k[:, None], uz[:, None]], 1)


class Quadrotor12D:
    # x = (p, angles, v, omega), u = (poussée, 3 couples)
    dim = 12

    def f(self, x, u):
        raise NotImplementedError

    def hamiltonian(self, x, p):
        raise NotImplementedError

    def optimal_control(self, x, p):
        raise NotImplementedError
