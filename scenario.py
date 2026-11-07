import numpy as np

G = 9.81

# Crazyflie 2.1 Brushless
MASS = 0.044                                  # kg
INERTIA = np.diag([3.3e-5, 3.6e-5, 5.9e-5])   # kg m^2
THRUST_MIN = 4 * 0.0214                       # N
THRUST_MAX = 4 * 0.20                         # N
MAX_TILT = np.radians(20)
V_MAX = 1.0                                   # m/s

# Arène MBDA, boîtes [[xmin, xmax], [ymin, ymax], [zmin, zmax]] en m
ARENA = np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 2.0]])
FLIGHT = np.array([[-0.85, 0.85], [-0.85, 0.85], [0.3, 1.6]])

# Attaque-défense
ATTACKERS_START = np.array([[-0.5, 0.5], [-0.8, -0.6], [0.4, 0.7]])
DEFENDERS_START = np.array([[-0.5, 0.5], [0.6, 0.8], [0.4, 0.7]])
TARGET_Y, TARGET_Z = 0.7, 1.45
OBSTACLES = [
    np.array([[-0.55, -0.25], [-0.15, 0.15], [0.0, 2.0]]),
    np.array([[0.25, 0.55], [-0.15, 0.15], [0.0, 2.0]]),
]
CAPTURE_RADIUS = 0.25 #capture radius for defenders to capture attackers
SAFETY_RADIUS = 0.12
T = 8.0
NU = 2.5e-4


def sample_box(box, n, rng=np.random.default_rng()):
    return box[:, 0] + (box[:, 1] - box[:, 0]) * rng.random((n, 3))
