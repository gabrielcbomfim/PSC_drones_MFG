import copy
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from scenario import (G, T, ATTACKERS_START, DEFENDERS_START, OBSTACLES, FLIGHT, TARGET_Y, TARGET_Z,
                      SAFETY_RADIUS, C1, C2, C3, C4, C5, L1, L2, L3, L4, GAMMA_OBS, C_ARENA)

ATT, DEF = "att", "def"
OTHER = {ATT: DEF, DEF: ATT}


_CONST = {}


def _t(name, a, like):
    # constantes du scénario, copiées une seule fois par device
    key = (name, str(like.device))
    if key not in _CONST:
        _CONST[key] = torch.as_tensor(np.asarray(a), dtype=torch.float32, device=like.device)
    return _CONST[key]


def sample_rho0(pop, n, device="cpu"):
    ref = torch.empty(0, device=device)
    box = _t(pop, ATTACKERS_START if pop == ATT else DEFENDERS_START, ref)
    p = box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(n, 3, device=device)
    v = math.sqrt(1e-3) * torch.randn(n, 3, device=device)  # petite perturbation (comme APAC-Net) : std > 0
    return torch.cat([p, v], 1)


# Coûts (Wang et al. 2022)
def _dist(a, b):
    return torch.sqrt(((a - b) ** 2).sum(-1, keepdim=True) + 1e-8)


def collision_cost(x, x_same):
    # éq. 4 (3) : indicatrice 1{d < e0} lissée par un noyau gaussien (congestion d'APAC-Net)
    d2 = ((x[:, 0:3] - x_same[:, 0:3]) ** 2).sum(1, keepdim=True)
    return torch.exp(-0.5 * d2 / SAFETY_RADIUS ** 2)


def obstacle_cost(x):
    # éq. 4 (4), 5-7 : répulsion 1/Q dans l'ellipsoïde Q <= 1.1 qui entoure chaque pavé
    c = _t("obs_c", [ob.mean(1) for ob in OBSTACLES], x)
    v = _t("obs_v", [(ob[:, 1] - ob[:, 0]) / 2 for ob in OBSTACLES], x)
    Q = ((x[:, None, 0:3] - c) ** 2 / (3 * v ** 2)).sum(2)
    return (GAMMA_OBS / Q.clamp(min=0.05) * (Q <= 1.1)).sum(1, keepdim=True)


def arena_cost(x):
    lo, hi = _t("lo", FLIGHT[:, 0], x), _t("hi", FLIGHT[:, 1], x)
    p = x[:, 0:3]
    return (torch.relu(lo - p) + torch.relu(p - hi)).sum(1, keepdim=True)


def capture_cost(x_att, x_def):
    # éq. 4 (2) : les attaquants s'éloignent des défenseurs ; x_def : (B, M, 6)
    return -_dist(x_att[:, None, 0:3], x_def[:, :, 0:3]).mean(1)


def pursuit_cost(x_def, x_att):
    # éq. 11 (2) : les défenseurs se rapprochent des attaquants
    return _dist(x_def[:, None, 0:3], x_att[:, :, 0:3]).mean(1)


def terminal_cost(x_T):
    # éq. 8 : distance à la ligne cible (., TARGET_Y, TARGET_Z), vitesse nulle
    e = torch.cat([x_T[:, 1:2] - TARGET_Y, x_T[:, 2:3] - TARGET_Z, x_T[:, 3:6]], 1)
    return C5 * torch.sqrt((e ** 2).sum(1, keepdim=True) + 1e-8)


def no_terminal_cost(x_T):
    return torch.zeros_like(x_T[:, 0:1])


POP = {
    ATT: dict(c_u=C1, c_opp=C2, opp=capture_cost, c_col=C3, c_obs=C4, psi=terminal_cost),
    DEF: dict(c_u=L1, c_opp=L2, opp=pursuit_cost, c_col=L3, c_obs=L4, psi=no_terminal_cost),
}


def interaction(pop, x, x_same, x_opp):
    c = POP[pop]
    return (c["c_opp"] * c["opp"](x, x_opp) + c["c_col"] * collision_cost(x, x_same)
            + c["c_obs"] * obstacle_cost(x) + C_ARENA * arena_cost(x))


# Réseaux (APAC-Net, neural_net_defs.py)
class Phi(nn.Module):
    def __init__(self, dim, psi, ns=100, hh=0.5):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(dim + 1, ns), nn.Linear(ns, ns), nn.Linear(ns, ns)
        self.out = nn.Linear(ns, 1)
        self.psi, self.hh = psi, hh

    def forward(self, x, t):
        h = torch.tanh(self.l1(torch.cat([t - T / 2, x], 1)))
        h = torch.tanh(h + self.hh * self.l2(h))
        h = torch.tanh(h + self.hh * self.l3(h))
        return (1 - t / T) * self.out(h) + t / T * self.psi(x)


class Generator(nn.Module):
    def __init__(self, dim, ns=100, hh=0.5, mu=None, std=None):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(dim + 1, ns), nn.Linear(ns, ns), nn.Linear(ns, ns)
        self.out = nn.Linear(ns, dim)
        self.hh = hh
        self.register_buffer("mu", torch.zeros(dim) if mu is None else mu.clone())
        self.register_buffer("std", torch.ones(dim) if std is None else std.clone())

    def forward(self, z, t):
        h = torch.relu(self.l1(torch.cat([t - T / 2, (z - self.mu) / self.std], 1)))
        h = torch.relu(h + self.hh * self.l2(h))
        h = torch.relu(h + self.hh * self.l3(h))
        return t / T * self.out(h) + (1 - t / T) * z


def hover(z, t):
    # politique initiale de l'adversaire : vol stationnaire
    return z


# Entraînement
def make_population(dyn, pop, a):
    y = sample_rho0(pop, 10000, a.device)
    phi = Phi(dyn.dim, POP[pop]["psi"], a.ns, a.hh).to(a.device)
    gen = Generator(dyn.dim, a.ns, a.hh, y.mean(0), y.std(0)).to(a.device)
    return dict(phi=phi, gen=gen,
                opt_phi=torch.optim.Adam(phi.parameters(), lr=a.disc_lr, betas=(0.5, 0.9), weight_decay=1e-4),
                opt_gen=torch.optim.Adam(gen.parameters(), lr=a.gen_lr, betas=(0.5, 0.9), weight_decay=1e-4))


def hjb_residual(dyn, phi, x, t, c_u, nu):
    # phi_t + nu Lap(phi) - H(x, grad phi)
    B, d = x.shape
    if nu > 0:  # comme APAC-Net : x répété d fois, Laplacien en une seule rétropropagation
        x, t = x.repeat(d, 1), t.repeat(d, 1)
    t = t.detach().requires_grad_(True)
    out = phi(x, t)
    phi_t, phi_x = torch.autograd.grad(out.sum(), (t, x), create_graph=True)
    res = phi_t - dyn.hamiltonian(x, phi_x, c_u)
    if nu > 0:
        eye = torch.eye(d, device=x.device).repeat_interleave(B, 0)
        hess = torch.autograd.grad(phi_x, x, grad_outputs=eye, create_graph=True)[0]
        lap = (hess * eye).sum(1, keepdim=True).view(d, B).sum(0)[:, None]
        res = res[:B] + nu * lap
    return res


def opponent_samples(opp_pop, opponents, t, m):
    # m échantillons de la densité adverse au temps t ; fictitious play : mélange uniforme des générateurs passés
    # (la ligne i est générée par opponents[i % K])
    B, K = t.shape[0], len(opponents)
    tt = t.repeat_interleave(m, 0)
    y = sample_rho0(opp_pop, B * m, t.device)
    x = torch.empty_like(y)
    with torch.no_grad():
        for j, g in enumerate(opponents):
            x[j::K] = g(y[j::K], tt[j::K])
    return x.view(B, m, -1)


def train_step(dyn, P, pop, opponents, a, disc):
    # une itération d'APAC-Net (utils/train_once.py), l'adversaire étant figé
    phi, gen = P["phi"], P["gen"]
    for p in phi.parameters():
        p.requires_grad_(disc)
    for p in gen.parameters():
        p.requires_grad_(not disc)
    opt = P["opt_phi"] if disc else P["opt_gen"]
    opt.zero_grad()

    B, dev = a.batch, a.device
    t = T * torch.rand(B, 1, device=dev)
    x = gen(sample_rho0(pop, B, dev), t)
    x_same = gen(sample_rho0(pop, B, dev), t)
    if disc:
        x, x_same = x.detach().requires_grad_(True), x_same.detach()
    x_opp = opponent_samples(OTHER[pop], opponents, t, a.n_opp)

    res = hjb_residual(dyn, phi, x, t, POP[pop]["c_u"], a.nu) * T
    F = interaction(pop, x, x_same, x_opp) * T
    err = (res + F).abs().mean() / T
    if disc:
        phi0 = phi(sample_rho0(pop, B, dev), torch.zeros(B, 1, device=dev)).mean()
        loss = -(phi0 + res.mean()) + a.lam * err
    else:
        loss = res.mean() + F.mean()
    loss.backward()
    opt.step()
    return loss.detach(), err.detach()


def _build_gen(dyn, sd, a):
    if sd is None:
        return hover
    g = Generator(dyn.dim, a.ns, a.hh).to(a.device)
    g.load_state_dict(sd)
    g.eval()
    return g


def _save_model(P, a, path):
    torch.save({"ns": a.ns, "hh": a.hh,
                **{f"{k}_{pop}": P[pop][k].state_dict() for pop in (ATT, DEF) for k in ("phi", "gen")}}, path)


def train(dyn, a, on_phase_end=None):
    # fictitious play par population : à chaque tour, APAC-Net pour les attaquants contre la moyenne
    # des défenseurs passés, puis pour les défenseurs contre la moyenne des attaquants passés
    os.makedirs(a.out, exist_ok=True)
    P = {pop: make_population(dyn, pop, a) for pop in (ATT, DEF)}
    hist = {ATT: [None], DEF: [None]}
    phases = [(r, pop) for r in range(1, a.rounds + 1) for pop in (ATT, DEF)]
    start = 0
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=a.device)
        for pop in (ATT, DEF):
            for k in ("phi", "gen", "opt_phi", "opt_gen"):
                P[pop][k].load_state_dict(ck["P"][pop][k])
        hist, start = ck["hist"], ck["phase"]
        print(f"reprise à la phase {start + 1}/{len(phases)}")

    log_path = os.path.join(a.out, "log.csv")
    if start == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["round", "pop", "iter", "loss_phi", "loss_gen", "hjb_err", "sec"])

    for i in range(start, len(phases)):
        r, pop = phases[i]
        opp_hist = hist[OTHER[pop]] if a.opponent == "average" else hist[OTHER[pop]][-1:]
        opponents = [_build_gen(dyn, sd, a) for sd in opp_hist]
        print(f"\n=== tour {r}/{a.rounds} : {pop} contre {len(opponents)} générateur(s) adverse(s) ===")
        t0, acc = time.time(), torch.zeros(3, device=a.device)
        for it in range(1, a.iters + 1):
            ld, err = train_step(dyn, P[pop], pop, opponents, a, disc=True)
            lg, _ = train_step(dyn, P[pop], pop, opponents, a, disc=False)
            acc += torch.stack([ld, lg, err])
            if it % a.log_every == 0:
                m = (acc / a.log_every).tolist()
                acc.zero_()
                sec = time.time() - t0
                print(f"[{pop}] it {it:6d}  loss_phi {m[0]:9.3f}  loss_gen {m[1]:9.3f}  hjb_err {m[2]:8.4f}  "
                      f"({sec / it * 1e3:.1f} ms/it)")
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([r, pop, it, *[f"{v:.5g}" for v in m], f"{sec:.1f}"])

        hist[pop].append({k: v.detach().cpu().clone() for k, v in P[pop]["gen"].state_dict().items()})
        torch.save({"phase": i + 1, "hist": hist,
                    "P": {p: {k: P[p][k].state_dict() for k in ("phi", "gen", "opt_phi", "opt_gen")}
                          for p in (ATT, DEF)}}, ckpt_path)
        _save_model(P, a, os.path.join(a.out, "model.pt"))
        _save_model(P, a, os.path.join(a.out, f"model_r{r}_{pop}.pt"))
        if on_phase_end:
            on_phase_end(P, r, pop)
    return P


def load_policy(dyn, path, n, x0=None, mode="gen", kp=4.0, kd=3.0, device="cpu"):
    # retourne policy(t, x) -> u ; les n premiers drones sont les attaquants
    # mode "gen" : chaque drone suit sa trajectoire APAC-Net t -> G(t, x0) (PD)
    # mode "phi" : retour d'état u = g e_z - grad_v Phi / (2c)
    m = torch.load(path, map_location=device)
    if mode == "gen":
        gens, y0 = {}, torch.tensor(x0, dtype=torch.float32)
        for pop in (ATT, DEF):
            gens[pop] = Generator(dyn.dim, m["ns"], m["hh"])
            gens[pop].load_state_dict(m[f"gen_{pop}"])
            gens[pop].eval()

        @torch.no_grad()
        def ref(t):
            out = []
            for pop, sl in ((ATT, slice(0, n)), (DEF, slice(n, None))):
                out.append(gens[pop](y0[sl], torch.full((len(y0[sl]), 1), float(np.clip(t, 0, T))))[:, 0:3])
            return torch.cat(out)

        def policy(t, x):
            x = torch.tensor(x, dtype=torch.float32)
            h = 0.02
            p_ref, v_ref = ref(t), (ref(t + h) - ref(t - h)) / (2 * h)
            u = kp * (p_ref - x[:, 0:3]) + kd * (v_ref - x[:, 3:6])
            u[:, 2] += G
            return dyn.clip_control(u).numpy()

        return policy

    phis = {}
    for pop in (ATT, DEF):
        phis[pop] = Phi(dyn.dim, POP[pop]["psi"], m["ns"], m["hh"])
        phis[pop].load_state_dict(m[f"phi_{pop}"])
        phis[pop].eval()

    def policy(t, x):
        u = []
        for pop, sl in ((ATT, slice(0, n)), (DEF, slice(n, None))):
            xs = torch.tensor(x[sl], dtype=torch.float32, requires_grad=True)
            tt = torch.full((len(xs), 1), min(t, T))
            p = torch.autograd.grad(phis[pop](xs, tt).sum(), xs)[0]
            u.append(dyn.optimal_control(xs.detach(), p, POP[pop]["c_u"]))
        return dyn.clip_control(torch.cat(u)).numpy()

    return policy
