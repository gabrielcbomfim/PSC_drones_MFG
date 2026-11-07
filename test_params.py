# Vérifications avant l'entraînement long + estimation du temps
# python test_params.py [--device cuda] [--rounds 6 --iters 30000]
import os
import tempfile
import time

import numpy as np
import torch

import mfg
from dynamics import PointMass6D
from scenario import T, G, ATTACKERS_START, DEFENDERS_START, sample_box
from simulator import simulate
from train import parser, run

a = parser.parse_args()
dyn = PointMass6D()

# 1. Hamiltonien : H(x, p) = sup_u {-p.f - c|u - g e_z|^2}, atteint en u*
x, p, c = torch.randn(100, 6), torch.randn(100, 6), 0.5
H = dyn.hamiltonian(x, p, c)
u_star = dyn.optimal_control(x, p, c)
val = lambda u: (-(p.numpy() * dyn.f(x.numpy(), u.numpy())).sum(1, keepdims=True)
                 - c * ((u - torch.tensor([0, 0, G])) ** 2).sum(1, keepdim=True).numpy())
assert np.allclose(val(u_star), H.numpy(), atol=1e-4)
assert all((val(u_star + 0.1 * torch.randn(100, 3)) <= H.numpy() + 1e-5).all() for _ in range(20))
print("hamiltonien et contrôle optimal : OK")

# 2. coûts finis aux positions initiales
for pop in (mfg.ATT, mfg.DEF):
    y = mfg.sample_rho0(pop, 500)
    F = mfg.interaction(pop, y, mfg.sample_rho0(pop, 500), mfg.sample_rho0(mfg.OTHER[pop], 500 * 8).view(500, 8, 6))
    assert torch.isfinite(F).all()
    print(f"coût d'interaction initial {pop} : {F.mean():.3f}")

# 3. mini-entraînement (2 phases) + chronométrage
full_rounds, full_iters = a.rounds, a.iters
a.rounds, a.iters, a.log_every = 1, 300, 100
a.out = os.path.join(tempfile.mkdtemp(), "smoke")
t0 = time.time()
run(a)
sec_it = (time.time() - t0) / (2 * a.iters)

# 4. simulation avec le modèle obtenu
n = 3
x0 = np.zeros((2 * n, 6))
x0[:n, 0:3] = sample_box(ATTACKERS_START, n)
x0[n:, 0:3] = sample_box(DEFENDERS_START, n)
for mode in ("gen", "phi"):
    _, states = simulate(dyn, x0, mfg.load_policy(dyn, os.path.join(a.out, "model.pt"), n, x0, mode), 0.5, dt=5e-3)
    assert np.isfinite(states).all()
print("simulation avec la politique : OK")

total = 2 * full_rounds * full_iters * sec_it
print(f"\n{sec_it * 1e3:.1f} ms/itération sur {a.device}")
print(f"estimation pour --rounds {full_rounds} --iters {full_iters} : {total / 3600:.1f} h")
