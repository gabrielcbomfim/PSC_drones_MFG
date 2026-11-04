import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scenario import ARENA, OBSTACLES, ATTACKERS_START, DEFENDERS_START, T, sample_box
from dynamics import PointMass6D
from simulator import simulate
import mfg

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5   # drones par camp

dyn = PointMass6D()
x0 = np.zeros((2 * n, dyn.dim))
x0[:n, 0:3] = sample_box(ATTACKERS_START, n)
x0[n:, 0:3] = sample_box(DEFENDERS_START, n)

try:
    policy = mfg.load_policy(dyn, "model.pt")
    t_final = T
except NotImplementedError:
    print("Pas de modèle entraîné : moteurs coupés (chute libre)")
    policy = lambda t, x: np.zeros((len(x), 3))
    t_final = 1.0

times, states = simulate(dyn, x0, policy, t_final)

# animation 3D
colors = ["tab:blue"] * n + ["tab:red"] * n
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.set_xlim(*ARENA[0]); ax.set_ylim(*ARENA[1]); ax.set_zlim(*ARENA[2])
for ob in OBSTACLES:
    (x0_, x1), (y0, y1), (z0, z1) = ob
    for x in (x0_, x1):
        for y in (y0, y1):
            ax.plot([x, x], [y, y], [z0, z1], c="0.5")
pts = ax.scatter(*x0[:, 0:3].T, c=colors)

def update(k):
    pts._offsets3d = states[k, :, 0:3].T
    ax.set_title(f"t = {times[k]:.2f} s")

anim = FuncAnimation(fig, update, frames=range(0, len(times), 20), interval=50)
plt.show()
