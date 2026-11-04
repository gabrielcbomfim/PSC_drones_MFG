import numpy as np
from scenario import G


class PointMass6D:
    # x = (p, v), u = accélération de poussée
    dim = 6

    def f(self, x, u):
        dx = np.zeros_like(x)
        dx[:, 0:3] = x[:, 3:6]
        dx[:, 3:6] = u
        dx[:, 5] -= G
        return dx

    def hamiltonian(self, x, p):
        raise NotImplementedError

    def optimal_control(self, x, p):
        raise NotImplementedError


class Quadrotor12D:
    # x = (p, angles, v, omega), u = (poussée, 3 couples)
    dim = 12

    def f(self, x, u):
        raise NotImplementedError

    def hamiltonian(self, x, p):
        raise NotImplementedError

    def optimal_control(self, x, p):
        raise NotImplementedError
