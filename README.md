# PSC_drones_MFG

Attaque-défense par jeu à champ moyen (MFG) à deux populations, résolu avec APAC-Net
(Lin et al. 2021), sur le scénario de Wang et al. 2022 adapté à l'arène MBDA (2 × 2 × 2 m).
Modèle 6D (point matériel) : `x = (p, v)`, `u` = accélération de poussée.

| fichier | rôle |
|---|---|
| `scenario.py` | arène, obstacles, départs, cible, constantes Crazyflie, poids des coûts |
| `dynamics.py` | `PointMass6D` : dynamique, hamiltonien, contrôle optimal, saturation |
| `mfg.py` | coûts, réseaux Φ et G (APAC-Net), entraînement par fictitious play, `load_policy` |
| `train.py` | script d'entraînement |
| `test_params.py` | vérifications + mini-entraînement + estimation du temps |
| `main.py` | simulation des drones avec la politique apprise, métriques, animation 3D |
| `simulator.py` | intégration RK4 |

## Modèle

- Dynamique : `ṗ = v`, `v̇ = u − g e_z`.
- Coût d'énergie : `L(u) = c |u − g e_z|²`. Le vol stationnaire ne coûte rien.
  Hamiltonien : `H(x, ∇Φ) = −∇_pΦ·v + |∇_vΦ|²/(4c)`. Contrôle : `u* = g e_z − ∇_vΦ/(2c)`.
- Coûts d'interaction F (Wang et al. 2022, éq. 4 et 11), poids repris du §4.1 :

| terme | attaquants (bleu) | défenseurs (rouge) |
|---|---|---|
| énergie | c1 = 0.5 | l1 = 0.5 |
| distance à l'adversaire (moyenne en champ moyen) | −c2 · d, c2 = 1 | +l2 · d, l2 = 1 |
| collision intra-population | c3 = 20 | l3 = 20 |
| obstacles (répulsion 1/Q, ellipsoïde Q ≤ 1.1) | c4 = 5 | l4 = 5 |
| coût terminal : distance à la ligne cible (·, 0.7, 1.45), v = 0 | c5 = 5 | aucun |

Écarts par rapport à l'article, limités au minimum :
1. Le coût d'énergie est compté autour du vol stationnaire (`|u − g e_z|²` au lieu de `|u|²`) :
   sinon Φ doit apprendre un gradient constant de l'ordre de g.
2. L'indicatrice de collision `1{d < e0}` n'est pas dérivable. Elle est remplacée par le noyau
   gaussien d'APAC-Net (largeur `SAFETY_RADIUS`).
3. Ajout d'une pénalité `C_ARENA` hors de la zone `FLIGHT` (murs, sol, plafond de l'arène).
4. Entraînement séparé des deux populations (voir ci-dessous) au lieu de l'ECA-Net conjoint.

## Stratégie d'entraînement : fictitious play par population

Chaque tour comporte deux phases :
1. **Attaquants.** APAC-Net mono-population (Φ_A, G_A), exactement comme le code de référence
   `apac-net/`. Les défenseurs sont figés : leur densité est la **moyenne** des générateurs
   défenseurs obtenus aux tours précédents.
2. **Défenseurs.** Même chose, contre la moyenne des générateurs attaquants.

Au départ, l'adversaire est en vol stationnaire. Le premier tour donne donc « attaquants contre
défenseurs immobiles », un premier résultat facile à vérifier.

C'est le fictitious play pour les MFG (Cardaliaguet & Hadikhanloo 2017 ; Perrin et al. 2020 ;
version neuronale : Laurière et al. 2022). Chaque phase est un MFG classique à une population,
donc l'algorithme APAC-Net s'applique sans modification, et le terme de pénalité η de l'ECA-Net
(Wang et al., éq. 23) n'est plus nécessaire. La moyenne sur les adversaires passés stabilise les
oscillations « chat et souris » d'une simple meilleure réponse itérée. Pour comparer, on peut
utiliser `--opponent last` (meilleure réponse itérée, sans moyenne).

Garantie théorique : la convergence est démontrée pour les MFG potentiels ou monotones, et pour
les jeux à somme nulle à deux joueurs (Robinson 1951). Notre jeu n'est ni l'un ni l'autre, donc
la convergence se contrôle numériquement : `hjb_err` doit se stabiliser, et la politique ne doit
plus changer d'un tour à l'autre (`plots/`).

## Tutoriel

### 0. Installation

```bash
conda create -n drones python=3.11
conda activate drones
pip install -r requirements.txt
```

> Sous Windows, il faut **activer** l'environnement (`conda activate drones`). Lancer
> directement `envs\drones\python.exe` sans activation donne `DLL load failed while importing _C`.

### 1. Vérifications et estimation du temps (≈ 1 min)

```bash
python test_params.py                       # CPU
python test_params.py --device cuda         # pour comparer avec le GPU
python test_params.py --rounds 6 --iters 30000   # estimation pour une config donnée
```

Le script vérifie le hamiltonien et le contrôle optimal, fait 2 × 300 itérations, simule la
politique obtenue et affiche le temps par itération ainsi que la durée totale estimée.

### 2. Entraînement

```bash
python train.py                                  # défaut : 6 tours x 30000 itérations, CPU
python train.py --out experiments/base --seed 0
```

Sur un serveur, laisser tourner en arrière-plan :

```bash
nohup python train.py --out experiments/base > base.log 2>&1 &
```

Les réseaux sont petits (3 couches de 100), donc **une seule thread CPU** suffit et le GPU
n'accélère presque pas (il est même plus lent sous Windows). Sur une machine puissante, il vaut
mieux lancer **plusieurs runs en parallèle**, un par cœur (graines ou paramètres différents) :

```bash
for s in 0 1 2 3; do nohup python train.py --seed $s --out experiments/seed$s > seed$s.log 2>&1 & done
nohup python train.py --opponent last --out experiments/last > last.log 2>&1 &
```

Contenu de `experiments/<run>/` :
- `log.csv` : `loss_phi`, `loss_gen`, `hjb_err`, toutes les `--log_every` itérations ;
- `latest.png`, `plots/r<tour>_<pop>.png` : densités à t = 0 … T (vue de dessus, vue de côté) ;
- `model.pt` (dernier modèle) et `model_r<tour>_<pop>.pt` (après chaque phase) ;
- `ckpt.pt` : point de reprise.

Reprendre un run interrompu (il repart au début de la phase en cours) :

```bash
python train.py --out experiments/base --resume
```

### 3. Évaluer

```bash
python main.py 5 experiments/base/model.pt                # 5 contre 5, animation 3D
python main.py 5 experiments/base/model.pt --episodes=50  # statistiques sur 50 épisodes
python main.py 10 experiments/base/model.pt --save        # enregistre simulation.gif
python main.py 5 experiments/base/model.pt --phi          # commande par retour d'état ∇Φ
```

Deux façons d'exécuter la politique, saturées toutes deux aux limites du Crazyflie (poussée,
inclinaison de 20°) :
- **défaut, `gen`** : dans APAC-Net, la trajectoire d'un agent parti de x0 est t ↦ G(t, x0).
  Chaque drone suit cette référence avec un PD. C'est aussi ce qu'on enverra au Crazyflie
  (consignes de position) ;
- **`--phi`** : retour d'état `u = g e_z − ∇_vΦ(t, x)/(2c)`. Il n'est fiable que là où la
  densité apprise passe. Un drone qui en sort reçoit un gradient extrapolé et peut diverger.
  Ce mode demande un entraînement plus long.

Métriques : attaquants capturés (≥ 2 défenseurs à moins de `CAPTURE_RADIUS`) ou arrivés à la
cible, drones entrés dans un obstacle, distance minimale entre drones d'une même population.

### Ce qu'il faut regarder

- `hjb_err` diminue puis se stabilise dans chaque phase ;
- dans `plots/` : au tour 1, les attaquants contournent les piliers vers la cible, puis les
  défenseurs vont à leur rencontre ; aux tours suivants, les figures doivent se stabiliser ;
- `main.py` : pas de drone dans les obstacles, et le compte capturés / arrivés.

## Temps d'entraînement

Mesuré sur un portable (CPU 8 cœurs, 1 thread ; RTX 3060 Laptop) : **≈ 40–45 ms/itération
sur CPU**. Le GPU est plus lent (60–85 ms/it) : les réseaux sont trop petits pour en profiter.
Une itération = une mise à jour de Φ + une de G. Le coût augmente un peu avec le nombre de
générateurs adverses, et `test_params.py` donne une estimation un peu pessimiste.

| config | itérations | durée (≈ 45 ms/it) |
|---|---|---|
| test de viabilité `--rounds 2 --iters 8000` | 32 k | 25 min |
| **défaut `--rounds 6 --iters 30000`** | 360 k | **4–5 h** |
| long `--rounds 8 --iters 50000` | 800 k | 10 h |

Repère : Wang et al. obtiennent des résidus HJB stables après ≈ 2·10⁵ itérations (entraînement
conjoint, 12D). La config par défaut (1,8·10⁵ itérations par population) est du même ordre.
Un bon résultat est donc attendu en une nuit. Sur un CPU de bureau récent, compter ≈ 30 %
de moins. Lancer 4 à 8 runs en parallèle ne coûte presque rien de plus en temps.

### Résultat du test de viabilité (7 nov. 2026, `--rounds 2 --iters 8000`, 25 min)

`main.py 5 <model> --episodes=20`, mode `gen` :

| modèle | capturés | arrivés à la cible | dans un obstacle |
|---|---|---|---|
| après le tour 1 | 15 % | 34 % | 8 % |
| après le tour 2 | 24 % | 65 % | 3 % |

`hjb_err` passe de ≈ 2,7 à ≈ 1,1 et diminue encore. Les défenseurs passent entre les piliers
pour intercepter, les attaquants contournent et se regroupent sur la cible à t = T. Le jeu
est en place mais pas convergé. Avec si peu d'entraînement, le mode `--phi` diverge encore
(32 % des drones dans un obstacle), et la distance de sécurité entre drones d'un même camp
n'est pas encore respectée.

## Paramètres principaux (`train.py --help`)

| option | défaut | rôle |
|---|---|---|
| `--rounds` | 6 | tours de fictitious play |
| `--iters` | 30000 | itérations APAC-Net par phase |
| `--opponent` | average | `average` (fictitious play) ou `last` (meilleure réponse itérée) |
| `--batch` | 50 | taille de batch (APAC-Net, Wang) |
| `--n_opp` | 32 | échantillons adverses par agent pour la moyenne en champ moyen |
| `--disc_lr`, `--gen_lr` | 4e-4, 1e-4 | code APAC-Net (Wang : 2e-4, 5e-5) |
| `--lam` | 1 | poids du résidu HJB (Wang : 2) |
| `--nu` | `NU` | diffusion ν (0 : déterministe, ≈ 1,5× plus rapide) |

Les poids des coûts et la géométrie se trouvent dans `scenario.py`.

## Références

- A. T. Lin, S. W. Fung, W. Li, L. Nurbekyan, S. J. Osher, *APAC-Net: Alternating the population
  and agent control via two neural networks to solve high-dimensional stochastic mean field
  games*, PNAS 2021 (arXiv:2002.10113). Code de référence : `../apac-net/`.
- G. Wang, Z. Li, W. Yao, S. Xia, *A Multi-Population Mean-Field Game Approach for Large-Scale
  Agents Cooperative Attack-Defense Evolution in High-Dimensional Environments*, Mathematics 2022.
- P. Cardaliaguet, S. Hadikhanloo, *Learning in mean field games: the fictitious play*,
  ESAIM: COCV 2017.
- S. Perrin, J. Pérolat, M. Laurière, M. Geist, R. Elie, O. Pietquin, *Fictitious play for mean
  field games: continuous time analysis and applications*, NeurIPS 2020.
- M. Laurière et al., *Scalable deep reinforcement learning algorithms for mean field games*,
  ICML 2022.
- J. Robinson, *An iterative method of solving a game*, Annals of Mathematics 1951.
