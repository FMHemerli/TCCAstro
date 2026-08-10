from __future__ import annotations

from dataclasses import dataclass

G = 6.67408e-11
PARTICLE_MASS = 1.0e9
N_PARTICLES = 1000
SPHERE_RADIUS = 6.2035049090
SOFTENING = 5.0e-2
SEED = 20190222

T_FF = 2.1007035
T_CONV = 1.05035175

TOTAL_MASS = 1.0e12
DENSITY = 1.0e9
V_CHAR = 3.2799885
L_SCALE = 2.0347400e13
OMEGA_PAIR_MAX = 32.678
P_EPS = 0.192276
OMEGA_MAX_DESIGN = 42.0
U_MIN_BOUND = -6.6674e14

DT_COLLAPSE = 5.0e-4
N_STEPS_COLLAPSE = 12600
OUT_DT = 1.0e-2

CONV_N_STEPS = (64, 128, 256, 512, 1024, 2048)
CONV_N_STEPS_EULER = CONV_N_STEPS + (4096,)
REF_N_STEPS = 16384
REF_CHECK_N_STEPS = 8192

DT_BENCH = 1.0e-2
N_STEPS_BENCH = 1

TWOBODY_MU = 0.1334816
KEPLER_A = 1.0
KEPLER_E = 0.5
KEPLER_PERIOD = 17.19765239
CIRC_SEPARATION = 1.0
CIRC_OMEGA = 0.3646677991
CIRC_PERIOD = 17.22988792

IC_R_HALF_0 = 4.881251
IC_R_MAX = 6.323302
IC_U_EPS005 = -6.4260397026e12
IC_A_MAX = 9.075891
IC_A_RMS = 1.544392
COLLAPSE_R_HALF_MIN = 0.3472
COLLAPSE_T_OVER_TFF = 1.0361

# Memory ceiling for tensor-backend tiling (Sec. "tiling e requisito, nao
# otimizacao" in integradores.md). tile_size=None resolves against this
# budget; see nbody._pairwise.default_tile_size. Not a physics constant, not
# part of Sec. 9 -- an implementation parameter the API contract requires
# config to expose ("teto de memoria configuravel em config").
TILE_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024

# --- mass spectrum (docs/simulacao-estocastica.md, Sec. 2 / Sec. 8) ---
MASS_ALPHA = 2.35
MASS_RATIO = 1000.0
MASS_G_FACTOR = 3.5136877959
MASS_MIN = 2.8460126741e8
MASS_MAX = 2.8460126741e11
MASS_BIG = 2.7509063196e10
MASS_TAIL_PROB = 2.0e-3
MASS_K_WEIGHTS = (0.37476530, 0.37514082, 0.25009388)
MASS_K_ACCEPT_PROB = 0.72223972
MASS_COND_MEAN_BIAS = -7.609250e-3
MASS_COND_CV = 9.2402e-2
MASS_TFF_CV = 4.6201e-2
MASS_SEED = 20190223

# --- velocities (docs/simulacao-estocastica.md, Sec. 3 / Sec. 8) ---
Q_DEFAULT = 0.25
F_CUT_DEFAULT = 0.5
Q_SUP_COEFF = 2.009056
Q_USABLE_COEFF = 1.532168
X_C_MIN = 2.0
VEL_SEED = 20190224

# values derived for equal masses (M_real = 1e12, |U_0| = 6.4260397026e12)
V_ESC_SPHERE = 4.6386556804
VEL_CUT_DEFAULT = 2.319328
VEL_SIGMA_DEFAULT = 0.7604389
VEL_XC_DEFAULT = 3.0500
VEL_RMS_DEFAULT = 1.267482
VEL_MODE_DEFAULT = 1.0754
VEL_REJECT_FRACTION = 2.5529e-2
VEL_LAMBDA_SD = 1.291e-2

# --- collisions, detection only (docs/simulacao-estocastica.md, Sec. 4 / Sec. 8) ---
CHI_DEFAULT = 0.1
R_REF_DEFAULT = 5.0e-3
DT_COLLISION = 1.25e-4
N_STEPS_COLLISION = 50400
COLLISION_SEED = 20190225

# --- collisions, resolution (docs/simulacao-estocastica.md, Sec. 4.6-4.10 / Sec. 8) ---
# NOTE: DT_COLLISION / N_STEPS_COLLISION above are the stage-2 (detection-only) sweep values
# and remain referenced by scripts/collision_rate.py, which is out of this change's scope.
# The 2026-08-07 revision of the spec REMOVES DT_COLLISION as a project symbol for stage-3
# (resolved) runs and fixes dt = DT_COLLAPSE, N_STEPS_COLLISION = 12600 instead; that stage-3
# campaign wiring (a config.CollisionParams / RUN_COLLISION analogue to RUN_COLLAPSE) does not
# exist yet and is not added here -- see the final report.
#
# Deterministic contact model, revision (f) 2026-08-09 (Sec. 4.6, 4.9, 4.10). Replaces the
# probabilistic regime map (MAP_X_CLAMP, MAP_ELASTIC_WEIGHT) and the drawn fragmentation split
# (FRAG_F_MIN, FRAG_ETA), both retired by this revision.
E_RESTITUTION = 0.8
K_BIND = 0.6
FRAG_CHIP_COEFF = 0.5
FRAG_CHIP_MAX = 0.5
FRAG_ENERGY_MAX = 0.9
COLLISION_DRAWS_PER_EVENT = 0


@dataclass(frozen=True)
class CollapseParams:
    n: int
    particle_mass: float
    sphere_radius: float
    softening: float
    dt: float
    n_steps: int
    t_end: float
    seed: int
    out_dt: float


@dataclass(frozen=True)
class ConvergenceParams:
    n: int
    particle_mass: float
    sphere_radius: float
    softening: float
    seed: int
    t_end: float
    n_steps_ladder: dict
    refinement_ratio: int
    ref_n_steps: int
    ref_check_n_steps: int


@dataclass(frozen=True)
class BenchParams:
    dt: float
    n_steps: int


@dataclass(frozen=True)
class TwoBodyEllipticParams:
    softening: float
    particle_mass: float
    a: float
    e: float
    mu: float
    period: float


@dataclass(frozen=True)
class TwoBodyCircularParams:
    softening: float
    particle_mass: float
    separation: float
    omega: float
    period: float


RUN_COLLAPSE = CollapseParams(
    n=N_PARTICLES,
    particle_mass=PARTICLE_MASS,
    sphere_radius=SPHERE_RADIUS,
    softening=SOFTENING,
    dt=DT_COLLAPSE,
    n_steps=N_STEPS_COLLAPSE,
    t_end=DT_COLLAPSE * N_STEPS_COLLAPSE,
    seed=SEED,
    out_dt=OUT_DT,
)

RUN_CONVERGENCE = ConvergenceParams(
    n=N_PARTICLES,
    particle_mass=PARTICLE_MASS,
    sphere_radius=SPHERE_RADIUS,
    softening=SOFTENING,
    seed=SEED,
    t_end=T_CONV,
    n_steps_ladder={
        "euler": CONV_N_STEPS_EULER,
        "symplectic_euler": CONV_N_STEPS,
        "velocity_verlet": CONV_N_STEPS,
        "rk4": CONV_N_STEPS,
    },
    refinement_ratio=2,
    ref_n_steps=REF_N_STEPS,
    ref_check_n_steps=REF_CHECK_N_STEPS,
)

RUN_BENCH = BenchParams(
    dt=DT_BENCH,
    n_steps=N_STEPS_BENCH,
)

TWOBODY_KEPLER = TwoBodyEllipticParams(
    softening=0.0,
    particle_mass=PARTICLE_MASS,
    a=KEPLER_A,
    e=KEPLER_E,
    mu=TWOBODY_MU,
    period=KEPLER_PERIOD,
)

TWOBODY_ECC = TwoBodyEllipticParams(
    softening=SOFTENING,
    particle_mass=PARTICLE_MASS,
    a=KEPLER_A,
    e=KEPLER_E,
    mu=TWOBODY_MU,
    period=KEPLER_PERIOD,
)

TWOBODY_CIRC = TwoBodyCircularParams(
    softening=SOFTENING,
    particle_mass=PARTICLE_MASS,
    separation=CIRC_SEPARATION,
    omega=CIRC_OMEGA,
    period=CIRC_PERIOD,
)
