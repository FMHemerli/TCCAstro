"""
Crude Solar System demonstration, driven entirely by the validated core in src/nbody/.

Nothing in src/nbody/ is touched, extended or reimplemented by this script. The point is
precisely that the core needs no changes: it already works in SI units with the physical
gravitational constant (config.G = 6.67408e-11), State accepts arbitrary heterogeneous
masses, and `softening` is a per-call argument of integrate() for which 0.0 is a supported
and tested value. The nominal scenario of this repository is a 1e12 kg cloud inside a
6.2 m sphere; the same kernels run the Solar System with no modification because no kernel
depends on that scale.

What this script is NOT:

  * It is not an ephemeris. Initial conditions come from the JPL "approximate positions of
    the major planets" mean Keplerian elements at J2000.0 (Standish, valid 1800-2050),
    converted to Cartesian state vectors here. Real ephemerides (Horizons, DE440) would be
    accurate to metres; these elements are accurate to arcminutes over the validity window.
  * It has no relativity, no J2, no moons (planets carry their system GM, Moon included in
    the Earth-Moon barycentre). Mercury's perihelion precession will NOT come out.
  * It uses a single global timestep, so the cost is set by Mercury even when only the
    outer planets matter.

Collisions are deliberately off. The contact-radius model in nbody.collisions scales from
R_REF_DEFAULT = 5e-3 m at m_bar = 1e9 kg; extrapolated to the Sun it predicts a body radius
of ~6e4 m against the real 7e8 m, and the v_coh scale has no meaning at these energies.
integrate() also rejects softening == 0.0 whenever collisions are enabled (INV-28).

Masses are derived as m = GM / config.G from the IAU/DE430 system GM values rather than
being quoted directly in kg. Orbital dynamics depends on GM, not on M and G separately, so
this makes the products exact under whichever G the repository happens to carry.

Usage:

    python scripts/solar_system.py                      # inner system, 12 years, writes a PNG
    python scripts/solar_system.py --bodies all --years 30
    python scripts/solar_system.py --live --bodies all --years 0 --speed 2

The default is headless: it runs to completion, prints a table and writes a figure. --live
instead opens an interactive vispy window and advances the simulation frame by frame, the
same way scripts/realtime.py drives the spherical collapse.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.integrators import integrate  # noqa: E402
from nbody.observables import (  # noqa: E402
    angular_momentum,
    center_of_mass,
    linear_momentum,
    total_energy,
)
from nbody.state import State  # noqa: E402

torch.set_default_dtype(torch.float64)

BACKEND = get_backend("torch_eager")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures", "solar_system")

AU = 1.495978707e11          # m, IAU 2012 definition
DAY = 86400.0                # s
YEAR = 365.25 * DAY          # s, Julian year

# GM in m^3 s^-2 (IAU / DE430 system values: planet + its satellites).
GM_SUN = 1.32712440018e20

# name, GM, a [au], e, i [deg], L [deg], varpi [deg], Omega [deg], sidereal period [days]
#
# Elements are the J2000.0 mean elements of JPL's "Keplerian Elements for Approximate
# Positions of the Major Planets" table, evaluated at T = 0 (the rates are not applied: a
# single epoch is all an initial condition needs). L is the mean longitude, varpi the
# longitude of perihelion, Omega the longitude of the ascending node -- longitudes, not
# angles measured in the orbital plane, so the argument of perihelion is varpi - Omega and
# the mean anomaly is L - varpi. The sidereal period is tabulated only to be printed
# alongside the measured one; it never enters the integration.
PLANETS = (
    ("Mercury", 2.20320e13,  0.38709927, 0.20563593, 7.00497902, 252.25032350,  77.45779628,  48.33076593,    87.969),
    ("Venus",   3.24859e14,  0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718,  76.67984255,   224.701),
    ("Earth",   4.03503236e14, 1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,          365.256),
    ("Mars",    4.282837e13, 1.52371034, 0.09339410, 1.84969142,  -4.55343205, -23.94362959,  49.55953891,   686.980),
    ("Jupiter", 1.26686534e17, 5.20288700, 0.04838624, 1.30439695, 34.39644051,  14.72847983, 100.47390909,  4332.589),
    ("Saturn",  3.7931187e16, 9.53667594, 0.05386179, 2.48599187,  49.95424423,  92.59887831, 113.66242448, 10759.220),
    ("Uranus",  5.793939e15, 19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630,  74.01692503, 30685.400),
    ("Neptune", 6.836529e15, 30.06992276, 0.00859048, 1.77004347, -55.12002969,  44.96476227, 131.78422574, 60189.000),
)

BODY_SETS = {
    "inner": ("Mercury", "Venus", "Earth", "Mars"),
    "all": tuple(p[0] for p in PLANETS),
}

# --- live viewer (--live) -----------------------------------------------------------------
# Same structure as scripts/realtime.py: a vispy SceneCanvas driven by an app.Timer whose
# callback advances the simulation by calling integrate() for a few steps and then updates the
# visuals. Unlike realtime.py this runs on the CPU and does not gate on a GPU being present:
# that viewer carries N=1000 and needs one, whereas nine bodies make any device transfer cost
# more than the force computation itself.
TARGET_FPS = 60.0
TRAIL_CAPACITY_MAX = 20000  # points per body; caps memory when --trail-years is large
BODY_COLORS = {
    "Sun": (1.00, 0.75, 0.15, 1.0),
    "Mercury": (0.65, 0.65, 0.68, 1.0),
    "Venus": (0.95, 0.80, 0.55, 1.0),
    "Earth": (0.35, 0.65, 1.00, 1.0),
    "Mars": (0.90, 0.40, 0.28, 1.0),
    "Jupiter": (0.90, 0.72, 0.52, 1.0),
    "Saturn": (0.88, 0.82, 0.60, 1.0),
    "Uranus": (0.55, 0.85, 0.88, 1.0),
    "Neptune": (0.35, 0.50, 0.95, 1.0),
}
HUD_N_LINES = 7


# ==============================================================================
# Keplerian elements -> Cartesian state, ecliptic J2000, Sun-centred.
# ==============================================================================
def solve_kepler(mean_anomaly: float, e: float, tol: float = 1e-14, max_iter: int = 100) -> float:
    """Eccentric anomaly E from M - e sin E = 0, Newton-Raphson.

    The starting guess M + e sin M is the standard first-order one; for e < 0.21 (Mercury
    is the worst case in this table) Newton converges in a handful of iterations and the
    loop below has never needed more than six. Failure to converge raises rather than
    returning a half-solved angle: a silently wrong E is a silently wrong initial
    condition, which would be indistinguishable from a physics bug downstream.
    """
    ecc_anomaly = mean_anomaly + e * math.sin(mean_anomaly)
    for _ in range(max_iter):
        residual = ecc_anomaly - e * math.sin(ecc_anomaly) - mean_anomaly
        derivative = 1.0 - e * math.cos(ecc_anomaly)
        delta = residual / derivative
        ecc_anomaly -= delta
        if abs(delta) < tol:
            return ecc_anomaly
    raise RuntimeError(f"solve_kepler: no convergence for M={mean_anomaly}, e={e}")


def elements_to_state(
    a_au: float,
    e: float,
    inc_deg: float,
    mean_long_deg: float,
    peri_long_deg: float,
    node_long_deg: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric position [m] and velocity [m/s] in the J2000 ecliptic frame.

    mu is G(M_sun + m_planet): the elements describe the relative orbit of the two-body
    problem, so the planet's own mass belongs in the gravitational parameter. Dropping it
    would misplace Jupiter by roughly its own mass ratio, ~1e-3 in the velocity.
    """
    a = a_au * AU
    inc = math.radians(inc_deg)
    node = math.radians(node_long_deg)
    arg_peri = math.radians(peri_long_deg - node_long_deg)
    mean_anomaly = math.radians((mean_long_deg - peri_long_deg + 180.0) % 360.0 - 180.0)

    ecc_anomaly = solve_kepler(mean_anomaly, e)
    cos_e = math.cos(ecc_anomaly)
    sin_e = math.sin(ecc_anomaly)
    root = math.sqrt(1.0 - e * e)

    # In the orbital plane, perifocal axes: x toward perihelion, y 90 deg ahead.
    x_orb = a * (cos_e - e)
    y_orb = a * root * sin_e

    # dE/dt = n / (1 - e cos E), with n the mean motion; differentiating the two lines above.
    mean_motion = math.sqrt(mu / (a * a * a))
    ecc_anomaly_dot = mean_motion / (1.0 - e * cos_e)
    vx_orb = -a * sin_e * ecc_anomaly_dot
    vy_orb = a * root * cos_e * ecc_anomaly_dot

    # Rz(node) Rx(inc) Rz(arg_peri), applied to the perifocal vectors.
    cos_w, sin_w = math.cos(arg_peri), math.sin(arg_peri)
    cos_o, sin_o = math.cos(node), math.sin(node)
    cos_i, sin_i = math.cos(inc), math.sin(inc)

    def rotate(px: float, py: float) -> np.ndarray:
        # Perifocal -> ecliptic, written out rather than assembled as a matrix product so
        # the three rotations stay visible.
        qx = px * cos_w - py * sin_w
        qy = px * sin_w + py * cos_w
        return np.array(
            [
                qx * cos_o - qy * cos_i * sin_o,
                qx * sin_o + qy * cos_i * cos_o,
                qy * sin_i,
            ]
        )

    return rotate(x_orb, y_orb), rotate(vx_orb, vy_orb)


def build_state(names: tuple[str, ...]) -> tuple[State, list[str]]:
    """Barycentric State for the Sun plus the named planets. Index 0 is always the Sun.

    The barycentric shift is what makes the run watchable rather than merely correct: a
    Sun-centred initial condition carries the whole system's momentum in the Sun, and the
    entire configuration then translates across the plot at ~12 m/s while the physics stays
    perfectly fine.
    """
    table = {p[0]: p for p in PLANETS}

    labels = ["Sun"]
    masses = [GM_SUN / config.G]
    positions = [np.zeros(3)]
    velocities = [np.zeros(3)]

    for name in names:
        _, gm, a_au, e, inc, mean_long, peri_long, node_long, _ = table[name]
        mass = gm / config.G
        r_hel, v_hel = elements_to_state(
            a_au, e, inc, mean_long, peri_long, node_long, mu=GM_SUN + gm
        )
        labels.append(name)
        masses.append(mass)
        positions.append(r_hel)
        velocities.append(v_hel)

    m = np.array(masses, dtype=np.float64)
    r = np.stack(positions).astype(np.float64)
    v = np.stack(velocities).astype(np.float64)

    total_mass = m.sum()
    r -= (m[:, None] * r).sum(axis=0) / total_mass
    v -= (m[:, None] * v).sum(axis=0) / total_mass

    state = State(
        r=torch.from_numpy(r),
        v=torch.from_numpy(v),
        m=torch.from_numpy(m),
    )
    return state, labels


# ==============================================================================
# Diagnostics computed from nbody.observables plus per-planet osculating elements.
# ==============================================================================
def velocity_rms(state: State) -> float:
    m = state.m.to(torch.float64)
    v = state.v.to(torch.float64)
    return math.sqrt((m * (v * v).sum(dim=-1)).sum().item() / m.sum().item())


def osculating_semi_major_axes(state: State) -> np.ndarray:
    """Heliocentric osculating a [au] for every planet, from the vis-viva relation.

    A symplectic integrator holds these bounded and oscillating; a secular trend in any of
    them is the signature that dt is too coarse for that body.
    """
    r_hel = (state.r[1:] - state.r[0]).to(torch.float64).numpy()
    v_hel = (state.v[1:] - state.v[0]).to(torch.float64).numpy()
    m = state.m.to(torch.float64).numpy()
    mu = config.G * (m[0] + m[1:])

    r_mag = np.linalg.norm(r_hel, axis=1)
    v_sq = (v_hel * v_hel).sum(axis=1)
    return 1.0 / (2.0 / r_mag - v_sq / mu) / AU


def heliocentric_longitudes(state: State) -> np.ndarray:
    """Ecliptic longitude [rad] of each planet as seen from the Sun."""
    r_hel = (state.r[1:] - state.r[0]).to(torch.float64).numpy()
    return np.arctan2(r_hel[:, 1], r_hel[:, 0])


def measured_periods(longitude_samples: np.ndarray, t_samples: np.ndarray):
    """Sidereal period [days] and revolutions covered, from the swept heliocentric longitude.

    Unwrapping the sampled longitude and dividing the elapsed time by the number of
    revolutions is deliberately cruder than a plane-crossing fit, and it is what this
    script wants: no interpolation, and an error dominated by the sampling cadence rather
    than by the integration.

    It has one failure mode that must not be read as a physics result, which is why the
    revolution count is returned alongside and printed: over a PARTIAL revolution the angular
    rate is not the orbit-averaged one, so an eccentric body sampled over less than a full
    orbit yields a period biased by up to ~2e/pi. Neptune over 40 years covers a quarter of
    its orbit and its number here means very little. Judge such a run by the osculating
    semi-major axis instead, which is meaningful at every instant.
    """
    unwrapped = np.unwrap(longitude_samples, axis=0)
    swept = unwrapped[-1] - unwrapped[0]
    elapsed = t_samples[-1] - t_samples[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(swept != 0.0, 2.0 * math.pi * elapsed / swept / DAY, np.nan)
    return periods, np.abs(swept) / (2.0 * math.pi)


# ==============================================================================
# Run
# ==============================================================================
def run(names: tuple[str, ...], years: float, steps_per_year: int, n_samples: int, tag: str):
    state0, labels = build_state(names)
    n_planets = len(labels) - 1

    dt = YEAR / steps_per_year
    n_steps = int(round(years * steps_per_year))
    callback_every = max(1, n_steps // n_samples)

    e0 = total_energy(state0, softening=0.0)
    l0 = angular_momentum(state0)
    l0_norm = torch.linalg.norm(l0).item()
    total_mass = state0.m.to(torch.float64).sum().item()

    t_samples: list[float] = []
    energy_rel: list[float] = []
    angular_rel: list[float] = []
    momentum_ratio: list[float] = []
    longitudes: list[np.ndarray] = []
    trajectory: list[np.ndarray] = []

    def callback(step_index, t, s):
        t_samples.append(t)
        energy_rel.append((total_energy(s, softening=0.0) - e0) / abs(e0))
        angular_rel.append(torch.linalg.norm(angular_momentum(s) - l0).item() / l0_norm)
        momentum_ratio.append(
            torch.linalg.norm(linear_momentum(s)).item() / (total_mass * velocity_rms(s))
        )
        longitudes.append(heliocentric_longitudes(s))
        trajectory.append(s.r.to(torch.float64).numpy().copy())

    print("=" * 92)
    print(f"Solar System, {n_planets} planets + Sun, velocity_verlet, softening = 0.0, fp64")
    print(
        f"  dt = {dt:.6e} s ({dt / DAY:.4f} d), {n_steps} steps, "
        f"t_end = {years:g} yr, {n_steps // callback_every + 1} samples"
    )
    print("=" * 92)

    final = integrate(
        state0,
        integrator="velocity_verlet",
        backend=BACKEND,
        dt=dt,
        n_steps=n_steps,
        softening=0.0,
        callback=callback,
        callback_every=callback_every,
    )

    t_arr = np.array(t_samples)
    lon_arr = np.stack(longitudes)
    periods, revolutions = measured_periods(lon_arr, t_arr)

    a_initial = osculating_semi_major_axes(state0)
    a_final = osculating_semi_major_axes(final)
    table = {p[0]: p for p in PLANETS}

    print()
    print(
        f"{'body':<10}{'a_0 [au]':>12}{'a_end [au]':>13}{'|da/a|':>12}"
        f"{'T_meas [d]':>14}{'T_tab [d]':>13}{'rel err':>11}{'revs':>9}"
    )
    partial = False
    for k, name in enumerate(labels[1:]):
        period_tab = table[name][8]
        rel_a = abs(a_final[k] - a_initial[k]) / a_initial[k]
        rel_t = abs(periods[k] - period_tab) / period_tab
        flag = ""
        if revolutions[k] < 1.0:
            flag = " *"
            partial = True
        print(
            f"{name:<10}{a_initial[k]:>12.6f}{a_final[k]:>13.6f}{rel_a:>12.2e}"
            f"{periods[k]:>14.3f}{period_tab:>13.3f}{rel_t:>11.2e}{revolutions[k]:>9.2f}{flag}"
        )
    if partial:
        print(
            "  * less than one revolution covered: T_meas is an extrapolation from a partial "
            "arc and\n    is biased by the eccentricity. Judge these bodies by |da/a|, not by "
            "the period."
        )

    com = center_of_mass(final)
    print()
    print("conserved quantities over the whole run:")
    print(f"  max |E - E0| / |E0|          = {max(abs(x) for x in energy_rel):.3e}")
    print(f"  max ||L - L0|| / ||L0||      = {max(angular_rel):.3e}")
    print(f"  max ||P|| / (M v_rms)        = {max(momentum_ratio):.3e}")
    print(f"  ||centre of mass|| final     = {torch.linalg.norm(com).item():.3e} m")
    print()

    plot(np.stack(trajectory), labels, t_arr, energy_rel, angular_rel, momentum_ratio, years, tag)
    return final


def plot(trajectory, labels, t_arr, energy_rel, angular_rel, momentum_ratio, years, tag):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, (ax_orbit, ax_cons) = plt.subplots(1, 2, figsize=(13, 6))

    for k, name in enumerate(labels):
        x = trajectory[:, k, 0] / AU
        y = trajectory[:, k, 1] / AU
        if k == 0:
            ax_orbit.plot(x, y, color="black", linewidth=0.8)
            ax_orbit.plot(x[-1], y[-1], marker="o", color="orange", markersize=9, label=name)
        else:
            ax_orbit.plot(x, y, linewidth=0.9, label=name)
            ax_orbit.plot(x[-1], y[-1], marker="o", markersize=4, color=ax_orbit.lines[-1].get_color())
    ax_orbit.set_aspect("equal")
    ax_orbit.set_xlabel("x [au]")
    ax_orbit.set_ylabel("y [au]")
    ax_orbit.set_title(f"Ecliptic plane, barycentric, {years:g} yr")
    ax_orbit.legend(fontsize=8, loc="upper right")

    t_years = t_arr / YEAR
    ax_cons.plot(t_years, np.abs(energy_rel), label="|E - E0| / |E0|")
    ax_cons.plot(t_years, angular_rel, label="||L - L0|| / ||L0||")
    ax_cons.plot(t_years, momentum_ratio, label="||P|| / (M v_rms)")
    ax_cons.set_yscale("log")
    ax_cons.set_xlabel("t [yr]")
    ax_cons.set_ylabel("relative deviation")
    ax_cons.set_title("Conserved quantities, velocity_verlet, fp64")
    ax_cons.legend(fontsize=9)

    fig.tight_layout()
    # Tagged by configuration: two runs with different body sets or spans are different
    # figures, and silently overwriting one with the other is how a README ends up showing a
    # picture that no longer matches the command printed above it.
    out = os.path.join(FIGURES_DIR, f"solar_system_{tag}.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure written to {os.path.normpath(out)}")


# ==============================================================================
# Live viewer (--live)
# ==============================================================================
def marker_sizes(m: np.ndarray) -> np.ndarray:
    """Screen size in px per body, on a log mass scale.

    Not a cube-root scale, which is the physically natural one for a radius: the masses here
    span seven decades, and under m^(1/3) the Sun and Jupiter both slam into whatever ceiling
    keeps the Sun from filling the frame, so the gas giants stop being distinguishable from
    the star. A log ramp keeps all nine separable, at the price of encoding nothing physical
    -- it is a legend, not a measurement.
    """
    return np.clip(4.0 + 2.2 * np.log10(m / m.min()), 3.0, 20.0)


class LiveViewer:
    def __init__(self, state, labels, dt, steps_per_frame, trail_capacity, t_end, view_radius):
        self.state = state
        self.labels = labels
        self.dt = dt
        self.steps_per_frame = steps_per_frame
        self.t_end = t_end
        self.view_radius = view_radius
        self.frozen = False

        self.step_count = 0
        self.sim_time = 0.0
        self.fps_ema = 0.0
        self._last_frame_time = time.perf_counter()

        self.e0 = total_energy(state, softening=0.0)
        self.l0 = angular_momentum(state)
        self.l0_norm = torch.linalg.norm(self.l0).item()
        self.total_mass = state.m.to(torch.float64).sum().item()
        self.a_initial = osculating_semi_major_axes(state)
        self.rel_energy = 0.0
        self.rel_angular = 0.0
        self.momentum_ratio = 0.0
        self.max_rel_a = 0.0

        m_np = state.m.to(torch.float64).numpy()
        self.sizes = marker_sizes(m_np)
        self.colors = np.array([BODY_COLORS.get(name, (1.0, 1.0, 1.0, 1.0)) for name in labels])

        # Trails are a plain FIFO of the last `trail_capacity` sampled positions per body. A
        # decimating history like realtime.py's energy panel would be wrong here: that panel
        # must always span the whole run, whereas a trail that keeps thinning would make an
        # orbit drawn early look different from the same orbit drawn later.
        n_bodies = len(labels)
        self._trail = np.zeros((n_bodies, trail_capacity, 3), dtype=np.float64)
        self._trail_n = 0
        self._trail_capacity = trail_capacity

        self._build_scene()

    # -- scene ---------------------------------------------------------------------------
    def _build_scene(self):
        n_planets = len(self.labels) - 1
        self.canvas = scene.SceneCanvas(
            keys="interactive",
            size=(1100, 850),
            show=True,
            bgcolor="black",
            title=f"Solar System, {n_planets} planets -- real time (CPU, fp64, velocity_verlet)",
        )
        self.view = self.canvas.central_widget.add_view()
        # distance = R / tan(fov/2) is the exact fit for a face-on disc of radius R; the small
        # margin above that keeps the outermost aphelion off the edge once the disc is tilted
        # by the elevation below.
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45,
            distance=self.view_radius / math.tan(math.radians(22.5)) * 1.05,
            elevation=25.0,
            azimuth=45.0,
        )

        # One Line per body rather than one Line with a connect mask: the trails must carry
        # different colours, and a single visual would need a per-vertex colour array rebuilt
        # every frame for no gain at nine bodies.
        self.trail_lines = [
            scene.visuals.Line(
                color=tuple(self.colors[k][:3]) + (0.55,), width=1.0, parent=self.view.scene
            )
            for k in range(len(self.labels))
        ]
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self._update_visuals()

        # Same one-Text-per-line workaround as realtime.py: vispy 0.16's Text does not lay out
        # embedded newlines against anchor_y="top".
        self._hud_positions = [(10, 14 + i * 14) for i in range(HUD_N_LINES)]
        self.hud = scene.visuals.Text(
            [""] * HUD_N_LINES, pos=self._hud_positions, color="white", font_size=9,
            anchor_x="left", anchor_y="top", parent=self.canvas.scene,
        )
        self.message = scene.visuals.Text(
            "", pos=(self.canvas.size[0] / 2, self.canvas.size[1] * 0.92), color="yellow",
            font_size=14, anchor_x="center", anchor_y="center", parent=self.canvas.scene,
        )
        self.timer = app.Timer(interval=1.0 / TARGET_FPS, connect=self.on_timer, start=True)

    def _push_trail(self, positions_au: np.ndarray) -> None:
        if self._trail_n == self._trail_capacity:
            self._trail[:, :-1] = self._trail[:, 1:]
            self._trail_n -= 1
        self._trail[:, self._trail_n] = positions_au
        self._trail_n += 1

    def _update_visuals(self) -> np.ndarray:
        positions_au = self.state.r.to(torch.float64).numpy() / AU
        self._push_trail(positions_au)
        self.markers.set_data(
            positions_au, face_color=self.colors, edge_color=None, size=self.sizes
        )
        if self._trail_n > 1:
            for k, line in enumerate(self.trail_lines):
                line.set_data(self._trail[k, : self._trail_n])
        return positions_au

    # -- loop ----------------------------------------------------------------------------
    def on_timer(self, event):
        if not self.frozen:
            self.state = integrate(
                self.state,
                integrator="velocity_verlet",
                backend=BACKEND,
                dt=self.dt,
                n_steps=self.steps_per_frame,
                softening=0.0,
            )
            self.step_count += self.steps_per_frame
            self.sim_time = self.step_count * self.dt

            e = total_energy(self.state, softening=0.0)
            self.rel_energy = abs((e - self.e0) / self.e0)
            self.rel_angular = (
                torch.linalg.norm(angular_momentum(self.state) - self.l0).item() / self.l0_norm
            )
            self.momentum_ratio = torch.linalg.norm(linear_momentum(self.state)).item() / (
                self.total_mass * velocity_rms(self.state)
            )
            a_now = osculating_semi_major_axes(self.state)
            self.max_rel_a = float(np.max(np.abs(a_now - self.a_initial) / self.a_initial))

            self._update_visuals()

            if self.t_end is not None and self.sim_time >= self.t_end:
                self._freeze(
                    f"t = {self.sim_time / YEAR:.2f} yr reached -- simulation stopped, "
                    f"camera still interactive"
                )

        self._update_hud()
        self.canvas.update()

    def _freeze(self, reason: str) -> None:
        self.frozen = True
        self.message.text = reason

    def _update_hud(self) -> None:
        now = time.perf_counter()
        dt_wall = now - self._last_frame_time
        self._last_frame_time = now
        inst_fps = 1.0 / dt_wall if dt_wall > 0 else 0.0
        self.fps_ema = inst_fps if self.fps_ema == 0.0 else 0.9 * self.fps_ema + 0.1 * inst_fps

        years_per_second = self.fps_ema * self.steps_per_frame * self.dt / YEAR
        self.hud.text = [
            f"backend: torch_eager (CPU)   dtype: float64   N: {self.state.n}   "
            f"softening: 0.0   collisions: OFF",
            f"integrator: velocity_verlet   dt: {self.dt:.4e} s ({self.dt / DAY:.4f} d)   "
            f"steps/frame: {self.steps_per_frame}",
            f"fps: {self.fps_ema:6.1f}   t = {self.sim_time / YEAR:9.3f} yr   "
            f"step {self.step_count}   ({years_per_second:.2f} yr per wall second)",
            f"|dE/E0| = {self.rel_energy:.4e}",
            f"||L - L0||/||L0|| = {self.rel_angular:.4e}",
            f"||P||/(M v_rms) = {self.momentum_ratio:.4e}   <- conserved by construction, "
            f"not by resolving the orbit",
            f"max |da/a| over planets = {self.max_rel_a:.4e}",
        ]
        self.message.text = self.message.text if self.frozen else ""


def _import_vispy() -> None:
    """Bind vispy's app and scene as module globals, on demand.

    Imported here rather than at module scope so the headless path keeps working on a machine
    with no display and no vispy: writing a PNG must not depend on a GUI toolkit being
    installable.
    """
    global app, scene
    from vispy import app, scene  # noqa: PLW0603


def run_live(names, years, steps_per_year, speed, trail_years):
    _import_vispy()
    state0, labels = build_state(names)
    table = {p[0]: p for p in PLANETS}

    dt = YEAR / steps_per_year
    steps_per_frame = max(1, int(round(speed * steps_per_year / TARGET_FPS)))

    if trail_years is None:
        # Long enough for the outermost body to close its orbit once, so every trail on screen
        # is a complete ellipse rather than an arc that silently depends on the frame rate.
        trail_years = max(table[n][8] for n in names) * DAY / YEAR
    trail_capacity = int(min(TRAIL_CAPACITY_MAX, max(2.0, trail_years * TARGET_FPS / speed)))

    view_radius = max(table[n][2] * (1.0 + table[n][3]) for n in names)
    t_end = years * YEAR if years > 0 else None

    print("=" * 92)
    print(f"Solar System live viewer, {len(names)} planets + Sun, velocity_verlet, fp64, CPU")
    print(
        f"  dt = {dt:.6e} s ({dt / DAY:.4f} d), {steps_per_frame} steps/frame at {TARGET_FPS:.0f} fps"
        f" -> {speed:g} yr per wall second"
    )
    print(f"  trail: {trail_years:.1f} yr ({trail_capacity} points/body), view radius {view_radius:.1f} au")
    print(f"  stop at: {'never (close the window)' if t_end is None else f'{years:g} yr'}")
    print("  drag to rotate, scroll to zoom")
    print("=" * 92)

    # The reference matters for the same reason realtime.py's main() keeps one: vispy holds the
    # canvas, but nothing holds the viewer, and collecting it takes its app.Timer along -- the
    # window would render one frame and never advance.
    viewer = LiveViewer(state0, labels, dt, steps_per_frame, trail_capacity, t_end, view_radius)
    app.run()
    return viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--bodies",
        choices=sorted(BODY_SETS),
        default="inner",
        help="which planets to include (default: inner, i.e. Mercury..Mars)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=12.0,
        help="integration span in Julian years; with --live, 0 means run until the window is "
        "closed",
    )
    parser.add_argument(
        "--steps-per-year",
        type=int,
        default=4000,
        help="fixed timestep, expressed as steps per Julian year; must resolve the shortest "
        "orbit in the set (Mercury needs a few hundred steps per 88 d)",
    )
    parser.add_argument(
        "--samples", type=int, default=2000, help="how many states to record along the run"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="open an interactive vispy window and advance the simulation in real time, "
        "instead of running headless and writing a PNG",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="--live only: simulated Julian years per wall-clock second (default: 1.0)",
    )
    parser.add_argument(
        "--trail-years",
        type=float,
        default=None,
        help="--live only: how much of each orbit stays drawn behind a body, in simulated "
        "years (default: one period of the outermost body in the set)",
    )
    args = parser.parse_args()

    if args.live:
        run_live(
            BODY_SETS[args.bodies],
            args.years,
            args.steps_per_year,
            args.speed,
            args.trail_years,
        )
        return

    run(
        BODY_SETS[args.bodies],
        args.years,
        args.steps_per_year,
        args.samples,
        tag=f"{args.bodies}_{args.years:g}yr",
    )


if __name__ == "__main__":
    main()
