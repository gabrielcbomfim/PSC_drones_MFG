import torch.nn as nn


# Coûts (Wang et al. 2022)
def collision_cost(x, x_same):
    raise NotImplementedError


def obstacle_cost(x):
    raise NotImplementedError


def capture_cost(x_attackers, x_defenders):
    raise NotImplementedError


def pursuit_cost(x_defenders, x_attackers):
    raise NotImplementedError


def terminal_cost(x_T):
    raise NotImplementedError


# Réseaux (APAC-Net)
class Phi(nn.Module):
    def __init__(self, dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, t):
        raise NotImplementedError


class Generator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        raise NotImplementedError

    def forward(self, z, t):
        raise NotImplementedError


def train(dyn):
    raise NotImplementedError


def load_policy(dyn, path):
    # retourne policy(t, x) -> u
    raise NotImplementedError
