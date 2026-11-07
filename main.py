# python main.py [n] [model.pt] [--phi] [--save] [--episodes=N]
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scenario import (ARENA, OBSTACLES, ATTACKERS_START, DEFENDERS_START, T, sample_box,
                      CAPTURE_RADIUS, CAPTURE_COUNT, TARGET_RADIUS, TARGET_Y, TARGET_Z, SAFETY_RADIUS)
from dynamics import PointMass6D
from simulator import simulate
import mfg

args = [s for s in sys.argv[1:] if not s.startswith("--")]
n = int(args[0]) if len(args) > 0 else 5   # drones par camp
model = args[1] if len(args) > 1 else "model.pt"
save = "--save" in sys.argv
mode = "phi" if "--phi" in sys.argv else "gen"
episodes = int(next((s.split("=")[1] for s in sys.argv if s.startswith("--episodes=")), 1))

dyn = PointMass6D()


def episode():
    x0 = np.zeros((2 * n, dyn.dim))
    x0[:n, 0:3] = sample_box(ATTACKERS_START, n)
    x0[n:, 0:3] = sample_box(DEFENDERS_START, n)
    try:
        policy = mfg.load_policy(dyn, model, n, x0, mode)
        t_final = T
    except FileNotFoundError:
        print(f"Pas de modèle {model} : moteurs coupés (chute libre)")
        policy = lambda t, x: np.zeros((len(x), 3))
        t_final = 1.0
    return simulate(dyn, x0, policy, t_final, dt=5e-3)


def metrics(states):
    pa, pd = states[:, :n, 0:3], states[:, n:, 0:3]
    d = np.linalg.norm(pa[:, :, None] - pd[:, None], axis=-1)
    caught = (d < CAPTURE_RADIUS).sum(2) >= CAPTURE_COUNT
    k_caught = np.where(caught.any(0), caught.argmax(0), len(states))
    at_target = np.hypot(pa[..., 1] - TARGET_Y, pa[..., 2] - TARGET_Z) < TARGET_RADIUS
    k_target = np.where(at_target.any(0), at_target.argmax(0), len(states))
    in_obs = np.zeros(states.shape[:2], bool)
    for ob in OBSTACLES:
        in_obs |= np.all((states[..., 0:3] >= ob[:, 0]) & (states[..., 0:3] <= ob[:, 1]), axis=-1)
    pairs = np.linalg.norm(states[:, :, None, 0:3] - states[:, None, :, 0:3], axis=-1)
    pairs[:, np.arange(2 * n), np.arange(2 * n)] = np.inf
    pairs[:, :n, n:] = pairs[:, n:, :n] = np.inf  # collisions au sein d'une même population
    return dict(captures=np.sum(k_caught < k_target), arrivees=np.sum(k_target < k_caught),
                obstacles=in_obs.any(0).sum(), dmin=pairs.min())


if episodes > 1:
    res = [metrics(episode()[1]) for _ in range(episodes)]
    print(f"{episodes} épisodes, {n} contre {n}, mode {mode}")
    print(f"attaquants capturés          : {np.mean([r['captures'] for r in res]) / n:.0%}")
    print(f"attaquants arrivés à la cible : {np.mean([r['arrivees'] for r in res]) / n:.0%}")
    print(f"drones entrés dans un obstacle : {np.mean([r['obstacles'] for r in res]) / (2 * n):.0%}")
    print(f"distance min intra-population (médiane) : {np.median([r['dmin'] for r in res]):.3f} m (sécurité {SAFETY_RADIUS} m)")
    sys.exit()

times, states = episode()
m = metrics(states)
print(f"attaquants capturés          : {m['captures']}/{n}")
print(f"attaquants arrivés à la cible : {m['arrivees']}/{n}")
print(f"drones entrés dans un obstacle : {m['obstacles']}/{2 * n}")
print(f"distance min intra-population    : {m['dmin']:.3f} m (sécurité {SAFETY_RADIUS} m)")

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
ax.plot(ARENA[0], [TARGET_Y] * 2, [TARGET_Z] * 2, "k--")
pts = ax.scatter(*states[0, :, 0:3].T, c=colors)

def update(k):
    pts._offsets3d = states[k, :, 0:3].T
    ax.set_title(f"t = {times[k]:.2f} s")

anim = FuncAnimation(fig, update, frames=range(0, len(times), 4), interval=20)
if save:
    anim.save("simulation.gif", fps=50)
    print("simulation.gif")
else:
    plt.show()
