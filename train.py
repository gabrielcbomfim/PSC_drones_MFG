import argparse
import datetime
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import mfg
from dynamics import PointMass6D
from scenario import T, NU, OBSTACLES, FLIGHT, TARGET_Y, TARGET_Z

parser = argparse.ArgumentParser(description="APAC-Net attaque-défense (6D), fictitious play par population")
parser.add_argument("--rounds", type=int, default=6, help="tours de fictitious play (1 tour = attaquants puis défenseurs)")
parser.add_argument("--iters", type=int, default=30000, help="itérations APAC-Net par phase")
parser.add_argument("--opponent", default="average", choices=["average", "last"],
                    help="average = fictitious play, last = meilleure réponse itérée")
parser.add_argument("--batch", type=int, default=50)
parser.add_argument("--n_opp", type=int, default=32, help="échantillons adverses par agent pour la moyenne en champ moyen")
parser.add_argument("--ns", type=int, default=100, help="largeur des réseaux")
parser.add_argument("--hh", type=float, default=0.5, help="pas ResNet")
parser.add_argument("--disc_lr", type=float, default=4e-4)
parser.add_argument("--gen_lr", type=float, default=1e-4)
parser.add_argument("--lam", type=float, default=1.0, help="poids du résidu HJB")
parser.add_argument("--nu", type=float, default=NU)
parser.add_argument("--log_every", type=int, default=1000)
parser.add_argument("--device", default="cpu", help="cpu, cuda ou auto (petits réseaux : le CPU est souvent aussi rapide)")
parser.add_argument("--threads", type=int, default=1, help="threads torch sur CPU (1 suffit, lancer plutôt plusieurs runs)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None, help="dossier de sortie (défaut : experiments/run_<date>)")
parser.add_argument("--resume", action="store_true", help="reprendre depuis <out>/ckpt.pt")


def plot_densities(P, path, title, trained, n=200):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    times = np.linspace(0, T, 6)
    for pop, cmap in ((mfg.ATT, plt.cm.Blues), (mfg.DEF, plt.cm.Reds)):
        dev = next(P[pop]["gen"].parameters()).device
        gen = P[pop]["gen"] if pop in trained else mfg.hover  # pas encore entraîné : vol stationnaire
        y = mfg.sample_rho0(pop, n, dev)
        with torch.no_grad():
            for k, t in enumerate(times):
                x = gen(y, torch.full((n, 1), t, device=dev)).cpu().numpy()
                c = cmap(0.35 + 0.65 * k / (len(times) - 1))
                ax1.scatter(x[:, 0], x[:, 1], s=4, color=c)
                ax2.scatter(x[:, 1], x[:, 2], s=4, color=c)
    for ob in OBSTACLES:
        ax1.add_patch(plt.Rectangle((ob[0, 0], ob[1, 0]), *(ob[0:2, 1] - ob[0:2, 0]), color="0.5", alpha=0.6))
    ax1.axhline(TARGET_Y, c="k", ls="--")
    ax2.plot(TARGET_Y, TARGET_Z, "k*", ms=15)
    for ax, (i, j), lab in ((ax1, (0, 1), "xy (vue de dessus)"), (ax2, (1, 2), "yz (vue de côté)")):
        ax.add_patch(plt.Rectangle((FLIGHT[i, 0], FLIGHT[j, 0]), *(FLIGHT[[i, j], 1] - FLIGHT[[i, j], 0]),
                                   fill=False, ls=":"))
        ax.set_xlim(-1, 1 if i == 0 else 1)
        ax.set_ylim(-1 if j == 1 else 0, 1 if j == 1 else 2)
        ax.set_aspect("equal")
        ax.set_title(lab)
    fig.suptitle(title + "  (bleu = attaquants, rouge = défenseurs, clair -> foncé : t = 0 -> T)")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def run(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.out is None:
        a.out = os.path.join("experiments", "run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(os.path.join(a.out, "plots"), exist_ok=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    print(f"device : {a.device}   sortie : {a.out}")

    def on_phase_end(P, r, pop):
        trained = (mfg.ATT,) if (r, pop) == (1, mfg.ATT) else (mfg.ATT, mfg.DEF)
        for path in (os.path.join(a.out, "plots", f"r{r}_{pop}.png"), os.path.join(a.out, "latest.png")):
            plot_densities(P, path, f"tour {r}, après {pop}", trained)

    mfg.train(PointMass6D(), a, on_phase_end)
    print(f"\nterminé : {os.path.join(a.out, 'model.pt')}")


if __name__ == "__main__":
    run(parser.parse_args())
