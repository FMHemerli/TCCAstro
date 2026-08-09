"""
Real-time interactive visualisation of the spherical collapse.

A visual toy, not an instrument: it runs, shows the collapse happening live, and stops. It does
not write a report, a CSV, a figure, or any other file.

Physics comes only from the validated core in src/nbody/: initial_conditions.cold_sphere (with
--cold) or initial_conditions.random_sphere (the default, mass spectrum + virial-ratio velocities)
for the initial condition, and integrators.integrate with integrator="velocity_verlet" for every
step. No force law, integrator or sampling distribution is reimplemented here -- --alpha,
--mass-ratio, --virial-q, --f-cut and --seed are passed straight through to random_sphere.

Coupling between simulation and frames: a fixed simulation timestep dt (a fixed fraction of the
analytic free-fall time t_ff), a fixed number of simulation steps advanced per rendered frame
(STEPS_PER_FRAME), and a capped frame rate (TARGET_FPS). The picture on screen is the real
trajectory sampled every STEPS_PER_FRAME steps, not an interpolation. The coupling RULE (fixed
dt, a fixed step count per frame) is the same in both modes; dt is not -- DT_OVER_TFF without
--collisions, DT_OVER_TFF_COLLISION (roughly 8.4x finer) with it -- but the step count is the
same order in both, STEPS_PER_FRAME and STEPS_PER_FRAME_COLLISION (see that constant's docstring:
the collision RNG stream integrate() advances is now continuous across calls, so this no longer
needs to be coarser than the collisionless value to keep the trajectory correct). The on-screen
dt / t_ff / steps-per-frame readout always states which is active.

The cap exists so the collapse can be watched. Simulated time advances at
TARGET_FPS * STEPS_PER_FRAME * DT_OVER_TFF t_ff per wall-clock second without --collisions;
uncapped, this machine runs the whole collisionless collapse and first centre crossing in well
under a second. With --collisions, achieved fps replaces TARGET_FPS in that product, and dt is
finer, so simulated time advances more slowly per wall-clock second even at the same fps -- the
measured fps is reported live and in the final report, not assumed. The cap changes only the
pacing, never dt or the trajectory. The achieved rate is shown live as "fps".

State (positions, velocities, masses) lives on the GPU between steps. The only thing that crosses
to the host each frame is the (N, 3) position array needed to draw the markers -- plus, only with
--collisions on, since only then can masses or slot occupancy change frame to frame, the (N,) mass
array, needed to filter dead slots and size/color the markers by mass. Velocities and the
integrator's internal acceleration buffer never leave the device, in either mode.

Integrator: velocity_verlet only, fixed for the whole run. It is symplectic, so its energy error
oscillates in a bounded band instead of drifting away secularly -- the right property for a panel
meant to run indefinitely.

Precision: float32 without --collisions, fixed for that path exactly as before this flag existed.
On this project's GPU backends float32 is measured at roughly 20x the throughput of float64, and
the resulting round-off is far below the 5% band that matters physically here (see
docs/integradores.md for the measurement this decision is based on). With --collisions, float64
instead (COLLISION_DTYPE): docs/simulacao-estocastica.md validates every collision energy-
conservation invariant in fp64, and a probe run measured the difference is not academic at this
precision -- see COLLISION_DTYPE's docstring for the numbers.

GPU: required. If no CUDA/ROCm GPU is available this script raises GPUUnavailable and does not
fall back to the CPU backends -- CPU-speed dynamics would misrepresent what a real-time viewer is
supposed to show.

Initial condition parameters (particle count, mass, radius, mass-spectrum shape, virial ratio) are
read once from the command line at startup and never change during the run: there is no in-run
reconfiguration, no rebuilding of the state, and no control surface for it. Simplicity over a
feature nobody asked to keep live.

With --cold, the initial condition is exactly cold_sphere: equal masses, zero velocity, same marker
size and color as before this script gained a mass spectrum. Without --cold (the default), it is
random_sphere: a truncated Salpeter mass spectrum (dN/dm ~ m^-alpha, 1-3 massive bodies per
realization, docs/simulacao-estocastica.md Sec. 2) and, if --virial-q is nonzero, a truncated
Maxwellian velocity field at that virial ratio Q = 2K/|U| (Sec. 3). --mass is the exact per-particle
mass with --cold, but only the TARGET MEAN mass of the spectrum otherwise -- the realized mean and
total mass are read back from the sampled state, never assumed, because the k in {1,2,3} massive-
body conditioning biases the realized mean about -0.76% below the target (Sec. 2.8). Marker size
scales as (m_i / mean(m))^(1/3), clamped to a visible range, and marker color ramps from the
default blue at the mean mass towards gold at the heaviest body realized, so the 1-3 Salpeter-tail
bodies are visually distinguishable from the rest -- the whole point of turning the spectrum on.

Auto-stop, checked every frame: no particle left inside the viewed volume (the collapse has
ejected/dispersed everything worth looking at) always freezes the simulation (stops advancing
state, leaves the window/camera/HUD live, on-screen message naming what fired and when). Without
--collisions, |dE/E0| > 1 (accumulated mechanical energy error equals the total energy -- the
trajectory no longer means anything) freezes it too, exactly as before this flag existed. WITH
--collisions, that energy trigger does not apply: docs/simulacao-estocastica.md Sec. 4.10/4.13.6
retired the bound on the collisional ledger after three tightened formulations all failed against
correct, physically-accepted runs -- the documented runaway-merger phenomenon itself drives
max |E_int|/|E0| into [3, 40] (measured 7.8-10.8 across four seeds) by design, so a threshold on
the energy readout would be measuring the phenomenon the viewer exists to show, not a broken
trajectory. |dE_total/E0| and E_int/|E0| are REPORTED with --collisions, not policed by any
threshold here -- the HUD and energy panel say so explicitly, so a large number reads as the
documented signature it is, not as a defect.

Collisions (--collisions, off by default): wires src/nbody/collisions.py's already-validated
detection, pairing and three-outcome resolution (elastic, merger, fragmentation) into the same
integrate() call used for the collisionless path -- integrator="velocity_verlet",
collision=CollisionModel(...) -- so this script never reimplements any of that physics. The
contact radius R_i = chi * SOFTENING * (m_i / m_bar)^(1/3) is set by --chi (dimensionless,
default config.CHI_DEFAULT), read once at startup; m_bar is the run's target mean mass (--mass)
and v_coh is collisions.v_coh_from_state(state, radius), both computed once and held fixed for
the run. Off by default, so --cold and the plain random_sphere path are unaffected and stay
bit-for-bit identical to before this flag existed. Turning it on also switches precision to
float64 and dt to the finer DT_OVER_TFF_COLLISION (both described above) -- neither is a free
choice; both are what the source physics document's own validated collision campaign used.

With --collisions, four things change on screen, all only in that mode: (1) the HUD gains a
live particle count and running elastic/merger/fragmentation event counts; (2) marker size and
color are recomputed every frame instead of once at construction, because masses change under
merges and fragmentations, and dead slots (m == 0, left behind by a merger, carrying the
surviving body's own position -- nbody.observables.n_live's definition of "live" is m > 0) are
dropped before drawing so they cannot render as phantom markers on top of the body that absorbed
them; size and color are ABSOLUTE, anchored to the t=0 mean mass and never to the current
frame's range (see _mass_visuals_live), so a body that merges grows on screen and one that is
fragmented shrinks, with the ceiling raised to MARKER_SIZE_MAX_COLLISION to give the dominant
body the headroom the collisionless calibration was never given; (3) the energy panel plots
E_int/|E0| and |dE_total/E0| with E_total = K + U + E_int (E_int is the cumulative collisional
dissipation ledger, Sec. 4.10) instead of the collisionless |dE/E0| -- same panel, new label,
so the two quantities are never mistaken for one another on screen; (4) both are labelled
REPORTED, not watched -- see the Auto-stop paragraph above for why judging the trajectory by
them is wrong in this mode.

Usage:
    python scripts/realtime.py [--n N] [--mass MASS] [--radius RADIUS] [--cold]
        [--alpha ALPHA] [--mass-ratio RATIO] [--virial-q Q] [--f-cut F_CUT] [--seed SEED]
        [--collisions] [--chi CHI]

Defaults for every option match nbody.config. No upper-bound validation is applied to --n, --mass
or --radius by design: if a value is large enough to be slow or to exhaust GPU memory, it is
allowed to be slow or to fail loudly. --virial-q is the one exception: random_sphere rejects
Q > Q_USABLE_COEFF * f_cut**2 (the truncated Maxwellian degenerates past that point,
docs/simulacao-estocastica.md Sec. 3.4), and that combination is checked here at argument-parsing
time so it fails with a plain message before the GPU is touched, not as a traceback with the
window already open.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from vispy import app, scene

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbody import collisions, config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.backends.base import BackendUnavailable  # noqa: E402
from nbody.initial_conditions import cold_sphere, random_sphere  # noqa: E402
from nbody.integrators import integrate  # noqa: E402
from nbody.observables import total_energy  # noqa: E402

DTYPE = torch.float32
# Collision resolution runs in float64, not the collisionless DTYPE above. docs/simulacao-
# estocastica.md validates every collision energy-conservation invariant (INV-23's per-event and
# per-run residual bounds, TOL-EVENT-CONS, TOL-EINT-DRIFT) in fp64, not fp32, and a controlled
# probe (N=1000, chi=CHI_DEFAULT, --cold, both precisions run with identical 40-step-per-call
# chunking so the comparison isolates precision alone) measured float64 keeping |dE_total/E0| =
# 1.99e-2 at t = 2.0 t_ff against float32's 1.46e-1 at the same t -- roughly 7x smaller, and the
# gap was still widening at that point. This is a real, measured margin, not a nominal one.
# float32 stays the default (DTYPE above) for the collisionless path, where it is the validated,
# measured-adequate choice (see the Precision paragraph in the module docstring); this is a
# --collisions-only override, decided here because it is a simulation/precision coupling choice,
# not a change to collisions.py's physics.
COLLISION_DTYPE = torch.float64
STEPS_PER_FRAME = 4  # fixed simulation steps advanced per rendered frame
# --collisions advances the same number of steps per frame as the collisionless path. This was
# not always safe to do: integrate() used to reseed the collision RNG stream from collision.seed
# on every call, so calling it once per rendered frame with a small n_steps restarted that stream
# every frame instead of drawing one continuous sequence over the run -- measurably so (a probe at
# the time showed a ~20x difference in |dE_total/E0| between 4-step and 40-step chunking of the
# same span). integrate() now accepts a caller-owned collision_rng: np.random.Generator, advanced
# in place, so the stream is continuous across calls regardless of chunk size (fixed in
# src/nbody/integrators.py, confirmed there with a controlled probe across 1/4/40/600-step
# chunking, all bit-for-bit identical). This script owns that generator: self.collision_rng is
# created once, in Viewer.__init__ alongside self.collision_model, and the SAME object is passed
# to integrate() on every on_timer call for the run's whole duration -- creating a new generator
# per call, or per frame, would silently reintroduce the restart-every-call behaviour this
# depends on not having. STEPS_PER_FRAME_COLLISION is kept as its own name rather than reusing
# STEPS_PER_FRAME because dt still differs by mode (DT_OVER_TFF_COLLISION below), not because the
# step count needs to. With the continuous stream, chunk size no longer changes the trajectory
# (that was exactly what the fix's own controlled probe measured), so this is set comparably to
# the collisionless STEPS_PER_FRAME rather than the larger, now-unnecessary value chunking used to
# require -- see the final report for the fps measured at this value.
STEPS_PER_FRAME_COLLISION = 4
DT_OVER_TFF = 1.0 / 2000.0  # fixed simulation timestep, as a fraction of t_ff (matches sanity.py check 3)
# Frame rate is capped so the collapse is watchable. Uncapped, the loop runs at 600-880 fps
# (measured, N=1000), which advances 1.2-1.8 t_ff per wall-clock second: the whole collapse and
# first centre crossing would be over in under a second. Capping also stops the viewer rendering
# far faster than any display can show. At this cap the run advances
# TARGET_FPS * STEPS_PER_FRAME * DT_OVER_TFF = 0.12 t_ff per second, so the collapse takes ~8 s.
#
# --collisions uses a different, finer dt (DT_OVER_TFF_COLLISION below), not DT_OVER_TFF: the
# stage-3 collision-resolution campaign that produced the documented runaway-merger phenomenon
# (docs/simulacao-estocastica.md Sec. 4.13) was run at config.DT_COLLISION, roughly 8.4x finer
# than DT_OVER_TFF's t_ff/2000 -- coarser than that is not a validated regime for resolve(). The
# ratio is expressed as a fraction of a nominal t_ff (the default config's equal-mass, default-N
# free-fall time) rather than hardcoded as an absolute dt, so it generalizes to --n/--mass/--radius
# the same self-consistent way DT_OVER_TFF already does: self.dt = self.t_ff * (this ratio).
_NOMINAL_TOTAL_MASS = config.N_PARTICLES * config.PARTICLE_MASS
_NOMINAL_VOLUME = (4.0 / 3.0) * math.pi * config.SPHERE_RADIUS**3
_NOMINAL_DENSITY = _NOMINAL_TOTAL_MASS / _NOMINAL_VOLUME
_NOMINAL_T_FF = math.sqrt(3.0 * math.pi / (32.0 * config.G * _NOMINAL_DENSITY))
DT_OVER_TFF_COLLISION = config.DT_COLLISION / _NOMINAL_T_FF
TARGET_FPS = 60.0
VIEW_RADIUS_FACTOR = 4.0  # "viewed volume" and initial camera framing, as a multiple of sphere_radius
ENERGY_ERROR_LIMIT = 1.0  # |dE/E0| beyond this: the trajectory is no longer physical
# ENERGY_ERROR_LIMIT gates the auto-stop ONLY without --collisions. With --collisions it is not
# applied to |dE_total/E0| at all (docs/simulacao-estocastica.md Sec. 4.10/4.13.6: the bound on
# the collisional energy ledger was retired after three formulations all failed against correct,
# physically-accepted runs -- the documented runaway-merger phenomenon drives max |E_int|/|E0|
# into [3, 40] by design, measured 7.8-10.8 across four seeds, so a threshold here would flag the
# phenomenon itself, not a broken trajectory). Raising ENERGY_ERROR_LIMIT to a value the
# documented physics cannot reach would just be choosing a number to fit a measurement, which is
# exactly what that retraction forbids; removing the check in that mode is the honest option.
# Fixed HUD line count, sized for the larger of the two modes. Without --collisions this script
# still writes exactly the original 6-line layout (5 status lines + 1 freeze-message slot); with
# --collisions it writes 8 (the same 4 shared status lines, 3 collision lines -- live count and
# event counts, then E_int/|E0| and |dE_total/E0| -- and 1 freeze-message slot). A vispy Text
# with fewer text entries than positions simply ignores the extra positions (see
# vispy.visuals.text.text.TextVisual._prepare_draw), so the shorter, unchanged 6-line list keeps
# rendering at exactly its original 6 screen positions regardless of this constant.
HUD_N_LINES = 8

# Marker encoding of mass (--cold bypasses all of this and reuses the literal pre-spectrum values
# below directly, so its picture is bit-for-bit identical to before random_sphere was wired in).
MARKER_SIZE_BASE = 4.0  # the size every marker had before the mass spectrum existed
MARKER_SIZE_MIN = 2.0  # floor: the lightest sampled bodies (~0.28x mean) must stay visible
MARKER_SIZE_MAX = 20.0  # ceiling: the 1-3 Salpeter-tail bodies (up to ~285x mean) must not become blobs

# --collisions only. The collisional scale is ABSOLUTE: a marker's size is a function of that
# body's own mass and nothing else, so a body that gains mass grows on screen and one that is
# fragmented shrinks. That is the whole point of watching this mode, and a scale renormalized to
# the current frame cannot show it -- under that scheme the heaviest body IS the maximum by
# definition, so it sits pinned at the ceiling from the moment it takes the lead and its growth
# reads as everything else shrinking instead.
# The reference is the mean mass at t = 0, captured once (Viewer.mass_reference). It cannot be the
# running mean: total mass is conserved while N falls, so the running mean climbs, and every
# unchanged body would drift smaller for reasons having nothing to do with its own mass.
# Ceiling sized from the measured outcome: the dominant body reaches ~321x the mean
# (docs/simulacao-estocastica.md Sec. 4.13.7), and 321^(1/3) * MARKER_SIZE_BASE = 27.4 px.
MARKER_SIZE_MAX_COLLISION = 34.0
COLOR_SATURATION_RATIO = 100.0  # mass, in units of the t=0 mean, at which the ramp reaches HEAVY_COLOR
BASE_COLOR = (0.4, 0.7, 1.0, 0.9)  # the color every marker had before the mass spectrum existed
HEAVY_COLOR = (1.0, 0.85, 0.25, 1.0)  # ramp target at the heaviest body realized in this run

# Energy-error panel: fixed-capacity, adaptively decimated history (not a sliding window), so
# the trace covers the full run since t=0 in constant memory however long the run gets.
HIST_CAPACITY = 2000
ENERGY_BAND = 0.05  # arbitrary 5% tolerance from the 2019 work -- a reference marker, not the
ENERGY_BAND_LOG10 = math.log10(ENERGY_BAND)  # stopping criterion (that stays ENERGY_ERROR_LIMIT)
ERROR_FLOOR = 1e-16  # floor for log10(|dE/E0|) so a zero error does not produce -inf
PLOT_ROW_SPAN = 3  # 3D view spans this many grid rows; the plot spans one below it
PLOT_TOP_FRACTION = PLOT_ROW_SPAN / (PLOT_ROW_SPAN + 1)  # where the plot panel starts, in canvas height


class GPUUnavailable(RuntimeError):
    """No usable CUDA/ROCm GPU. This viewer does not run on the CPU backends."""


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time spherical collapse viewer.")
    parser.add_argument("--n", type=int, default=config.N_PARTICLES, help="particle count")
    parser.add_argument(
        "--mass", type=float, default=config.PARTICLE_MASS,
        help="particle mass (kg): exact per-particle value with --cold; otherwise the TARGET MEAN "
        "mass of the Salpeter spectrum -- the realized mean is read back from the sampled state "
        "and is not exactly this value (see docs/simulacao-estocastica.md Sec. 2.8)",
    )
    parser.add_argument("--radius", type=float, default=config.SPHERE_RADIUS, help="initial sphere radius (m)")
    parser.add_argument(
        "--alpha", type=float, default=config.MASS_ALPHA,
        help="Salpeter mass spectrum exponent, dN/dm ~ m^-alpha; ignored with --cold",
    )
    parser.add_argument(
        "--mass-ratio", type=float, default=config.MASS_RATIO,
        help="mass spectrum truncation ratio m_max / m_min; ignored with --cold",
    )
    parser.add_argument(
        "--virial-q", type=float, default=config.Q_DEFAULT,
        help="virial ratio Q = 2K/|U| for the initial velocity field; Q=0 is at rest (cold); "
        "ignored with --cold",
    )
    parser.add_argument(
        "--f-cut", type=float, default=config.F_CUT_DEFAULT,
        help="velocity truncation cutoff, as a fraction of the escape speed; ignored with --cold",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED,
        help="RNG seed for particle positions (mass and velocity sampling use their own seeds, "
        "fixed by config.MASS_SEED and config.VEL_SEED)",
    )
    parser.add_argument(
        "--cold", action="store_true",
        help="use the cold spherical collapse initial condition (equal masses, zero velocity, "
        "cold_sphere) instead of the default random_sphere; reproduces the reference run exactly",
    )
    parser.add_argument(
        "--collisions", action="store_true",
        help="enable collision resolution (elastic / merger / fragmentation, "
        "src/nbody/collisions.py) inside the velocity_verlet drift; OFF by default so --cold "
        "and the plain random_sphere path stay bit-for-bit identical to before this flag existed. "
        "Also switches precision to float64 and dt to a finer fraction of t_ff (both required for "
        "stability at the validated chi=CHI_DEFAULT, see COLLISION_DTYPE and "
        "DT_OVER_TFF_COLLISION), so a run advances noticeably slower in simulated time per "
        "wall-clock second than the collisionless default",
    )
    parser.add_argument(
        "--chi", type=float, default=config.CHI_DEFAULT,
        help="dimensionless contact-radius parameter, chi = R_ref / SOFTENING "
        "(R_i = chi * SOFTENING * (m_i / m_bar)^(1/3)); ignored unless --collisions is set",
    )
    args = parser.parse_args()

    if args.collisions and args.chi <= 0.0:
        parser.error(f"--chi must be positive, got {args.chi}")

    if not args.cold and args.virial_q != 0.0:
        q_max = config.Q_USABLE_COEFF * args.f_cut**2
        if args.virial_q > q_max:
            parser.error(
                f"--virial-q {args.virial_q} exceeds the non-degeneracy bound "
                f"Q_max = Q_USABLE_COEFF * f_cut**2 = {q_max:.6f} for --f-cut {args.f_cut} "
                "(the truncated Maxwellian degenerates into the uniform velocity ball past this "
                "point, docs/simulacao-estocastica.md Sec. 3.4); lower --virial-q or raise --f-cut"
            )

    return args


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        raise GPUUnavailable(
            "no CUDA/ROCm GPU detected. This viewer requires a GPU-enabled PyTorch build "
            "(install a CUDA or ROCm build matching your driver) and a supported GPU."
        )
    return torch.device("cuda")


def select_backend(device, probe_state):
    """
    Pick the fastest usable GPU backend and pay its one-time compile cost before the render
    loop starts, using the actual run's state as the warmup call so nothing recompiles later.

    Measured on this project's GPU backends at N in the low thousands: torch_compiled ran
    roughly 5x faster than torch_eager and roughly two orders of magnitude faster than the
    triton backend (JIT/codegen overhead not amortised well by this AMD/ROCm setup). Try
    torch_compiled first; fall back to torch_eager, with a printed reason, if no working C++
    compiler is available for torch.compile.
    """
    try:
        backend = get_backend("torch_compiled")
        backend.accelerations(probe_state.r, probe_state.m, config.SOFTENING)
        torch.cuda.synchronize()
        return "torch_compiled", backend
    except BackendUnavailable as exc:
        print(f"torch_compiled unavailable ({exc}); falling back to torch_eager", file=sys.stderr)
        backend = get_backend("torch_eager")
        backend.accelerations(probe_state.r, probe_state.m, config.SOFTENING)
        torch.cuda.synchronize()
        return "torch_eager", backend


class Viewer:
    def __init__(self, args):
        self.device = select_device()
        self.n = args.n
        self.mass = args.mass
        self.radius = args.radius
        self.cold = args.cold
        self.alpha = args.alpha
        self.mass_ratio = args.mass_ratio
        self.f_cut = args.f_cut
        self.seed = args.seed
        # Q=0.0 for --cold: the initial condition is at rest by construction, independent of
        # whatever --virial-q was left at its default.
        self.virial_q = 0.0 if self.cold else args.virial_q

        # --collisions and --chi are read here, before the state is built, because they decide
        # the working dtype (COLLISION_DTYPE vs DTYPE, see that constant's docstring) that
        # cold_sphere/random_sphere are constructed with below -- not just the eventual
        # CollisionModel, which is built further down once self.state exists.
        self.collisions_enabled = args.collisions
        self.chi = args.chi
        self.dtype = COLLISION_DTYPE if self.collisions_enabled else DTYPE

        if self.cold:
            self.state = cold_sphere(
                n=self.n, radius=self.radius, particle_mass=self.mass,
                seed=self.seed, dtype=self.dtype, device=self.device,
            )
        else:
            self.state = random_sphere(
                n=self.n, radius=self.radius, seed=self.seed,
                mass_spectrum={
                    "alpha": self.alpha, "ratio": self.mass_ratio, "particle_mass": self.mass,
                },
                virial_ratio=self.virial_q, f_cut=self.f_cut,
                dtype=self.dtype, device=self.device,
            )
        print(f"compiling GPU kernel for N={self.n} (torch.compile) ...", file=sys.stderr)
        # torch_compiled's one-time compile cost is measured (results/2026/sweep_n.csv) to grow
        # steeply with N: well under a second around N=1000, tens of seconds by N~16000, minutes
        # by N~65000. This happens once, here, before the window opens -- not as a frame stutter.
        self.backend_name, self.backend = select_backend(self.device, self.state)
        print(f"backend selected: {self.backend_name}", file=sys.stderr)

        # Total mass is read back from the realized state, never assumed as n * mass: with a mass
        # spectrum the k in {1,2,3} massive-body conditioning biases the realized mean away from
        # the target (docs/simulacao-estocastica.md Sec. 2.8), and t_ff must be computed from the
        # realized total, not the nominal one (Sec. 2.10) -- otherwise it, and dt, would be wrong.
        # Summed in float64: for --cold this is bit-identical to the previous n * mass, because
        # every fp32 mass here is exactly representable and their count is well within float64's
        # 53-bit mantissa, so the float64 sum recovers the exact total instead of the fp32-rounded
        # one a same-dtype reduction would give.
        self.total_mass = float(self.state.m.detach().to(torch.float64).sum().item())
        volume = (4.0 / 3.0) * math.pi * self.radius**3
        density = self.total_mass / volume
        self.t_ff = math.sqrt(3.0 * math.pi / (32.0 * config.G * density))
        self.dt = self.t_ff * (DT_OVER_TFF_COLLISION if self.collisions_enabled else DT_OVER_TFF)
        self.view_radius = VIEW_RADIUS_FACTOR * self.radius

        self.mass_min_real = float(self.state.m.min().item())
        self.mass_max_real = float(self.state.m.max().item())
        self.mass_ratio_real = self.mass_max_real / self.mass_min_real
        # Fixed reference for the absolute collisional marker scale, captured before any event
        # can change the mass distribution. See _mass_visuals_live for why it is not the running
        # mean. Unused when --collisions is off.
        self.mass_reference = float(self.state.m.detach().to(torch.float64).mean().item())

        self.marker_sizes, self.marker_colors = self._mass_visuals(self.state.m)

        # Collision resolution, OFF by default (--collisions and --chi were already read above,
        # before the state was built, because they decide the working dtype). config.SOFTENING is
        # fixed and positive throughout this script, so integrate()'s "collision requires
        # softening > 0.0" guard is always satisfied here; nothing more to validate at this point
        # (--chi <= 0.0 was already rejected in parse_args, before the GPU was touched). m_bar is
        # this run's target mean mass (--mass, the same scale random_sphere's spectrum was
        # generated around), not the hardcoded config.PARTICLE_MASS default, so the contact-radius
        # formula stays consistent with whatever mass scale the user chose. v_coh is computed once
        # here, from the realized initial state and the initial sphere radius (Sec. 4.6: v_coh =
        # V_CHAR = sqrt(G M_real / R_0)), and held fixed for the run -- it is an initial-condition-
        # derived scale, not a per-frame quantity, exactly like t_ff and dt above.
        self.n_live = self.n
        if self.collisions_enabled:
            r_ref = self.chi * config.SOFTENING
            v_coh = collisions.v_coh_from_state(self.state, self.radius)
            self.collision_model = collisions.CollisionModel(r_ref, self.mass, v_coh=v_coh)
            # integrate() draws exactly 2 uniform samples per accepted collision event from this
            # generator (Sec. 4.7.1), and that draw sequence must be continuous over the whole
            # run, not restarted at every integrate() call: on_timer calls integrate() once per
            # rendered frame, so the SAME np.random.Generator instance is created here, once, and
            # passed to every one of those calls (never rebuilt per frame -- rebuilding it would
            # restart the stream and bias the elastic/merger/fragment channel mix, as measured
            # before integrate() grew the collision_rng parameter that lets a caller own it).
            self.collision_rng = np.random.default_rng(self.collision_model.seed)
            self.n_elastic_total = 0
            self.n_merge_total = 0
            self.n_fragment_total = 0
            self.e_int_total = 0.0
            self.c_coll_max = 0.0
            self.f_reject_max = 0.0
        else:
            self.collision_model = None
            self.collision_rng = None

        self.e0 = total_energy(self.state, softening=config.SOFTENING)
        self.step_count = 0
        self.sim_time = 0.0
        self.rel_error = 0.0
        self.frozen = False

        self._hist_x = np.zeros(HIST_CAPACITY)
        self._hist_y = np.zeros(HIST_CAPACITY)
        self._hist_n = 0
        self._hist_stride = 1
        self._hist_skip = 0

        self.fps_ema = 0.0
        self._last_frame_time = time.perf_counter()

        self._build_scene()

    def _mass_visuals(self, m: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-particle marker size and color from mass, computed once (masses are fixed for the run).

        --cold uses the literal pre-spectrum constants directly rather than deriving them from a
        mass ratio that happens to be 1.0 everywhere: equal fp32 masses summed and divided back
        through .mean() do not round-trip to exactly 1.0, and --cold's picture must be bit-for-bit
        identical to before this script had a mass spectrum, not merely close.

        Otherwise: size ~ (m_i / mean(m))^(1/3), clamped to [MARKER_SIZE_MIN, MARKER_SIZE_MAX] so
        the 1-3 Salpeter-tail bodies (up to ~285x the mean, docs/simulacao-estocastica.md Sec. 2.5)
        read as visibly larger without filling the screen, and the lightest bodies (down to ~0.28x
        the mean) stay visible instead of shrinking to nothing. Color ramps linearly from
        BASE_COLOR at the mean mass to HEAVY_COLOR at the single heaviest body in this realization,
        so the ramp always spans its full range regardless of --alpha / --mass-ratio.
        """
        n = m.shape[0]
        if self.cold:
            sizes = np.full(n, MARKER_SIZE_BASE, dtype=np.float64)
            colors = np.tile(np.array(BASE_COLOR, dtype=np.float64), (n, 1))
            return sizes, colors

        m64 = m.detach().to("cpu", dtype=torch.float64).numpy()
        mean = m64.mean()
        ratio = np.cbrt(m64 / mean)
        sizes = np.clip(ratio * MARKER_SIZE_BASE, MARKER_SIZE_MIN, MARKER_SIZE_MAX)

        span = float(ratio.max() - 1.0)
        t = np.clip((ratio - 1.0) / span, 0.0, 1.0) if span > 1e-12 else np.zeros_like(ratio)
        base = np.array(BASE_COLOR, dtype=np.float64)
        heavy = np.array(HEAVY_COLOR, dtype=np.float64)
        colors = base[None, :] + t[:, None] * (heavy - base)[None, :]
        return sizes, colors

    def _mass_visuals_live(self, m_live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-frame marker size and color for the live subset, used only when --collisions is on.
        Unlike _mass_visuals (computed once, masses fixed for the whole run), this is called every
        frame, because merges and fragmentations change masses -- and slot occupancy -- as the run
        goes.

        Both size and color are ABSOLUTE: they depend on the body's own mass and on the t=0 mean
        (self.mass_reference), never on what else happens to be alive this frame. A body that
        merges grows on screen; a body that is fragmented shrinks. Two bodies of equal mass are
        always drawn the same, whatever else is going on.

        This is deliberately not a per-frame renormalization. Stretching [current_min, current_max]
        onto the marker range would keep the full palette in use at all times, but it makes the
        heaviest body the frame maximum BY DEFINITION -- pinned at the ceiling from the moment it
        takes the lead, so its growth from ~7x to ~321x the mean would render as the rest of the
        field shrinking around a marker that never changes. The growth of the dominant body is the
        phenomenon this mode exists to show, so it is the one thing the scale must not hide.

        Cube root, so marker AREA tracks mass more nearly than diameter does. Ceiling at
        MARKER_SIZE_MAX_COLLISION rather than MARKER_SIZE_MAX: the measured outcome needs headroom
        the collisionless calibration was never given.
        """
        m64 = m_live.astype(np.float64)
        ratio = np.cbrt(m64 / self.mass_reference)
        sizes = np.clip(ratio * MARKER_SIZE_BASE, MARKER_SIZE_MIN, MARKER_SIZE_MAX_COLLISION)

        saturation = np.cbrt(COLOR_SATURATION_RATIO) - 1.0
        tc = np.clip((ratio - 1.0) / saturation, 0.0, 1.0)
        base = np.array(BASE_COLOR, dtype=np.float64)
        heavy = np.array(HEAVY_COLOR, dtype=np.float64)
        colors = base[None, :] + tc[:, None] * (heavy - base)[None, :]
        return sizes, colors

    def _update_markers(self) -> np.ndarray:
        """
        Cross the current frame's state from device to host and redraw the marker set. Returns the
        positions actually drawn, for the caller's auto-stop dispersal check.

        Only what the renderer needs crosses per frame: the (N, 3) positions, and -- only with
        --collisions on, since only then can masses or slot occupancy change -- the (N,) masses.
        Velocities and the integrator's internal acceleration buffer never leave the device.

        With --collisions off this is exactly the original single set_data call (extracted into
        its own method, not changed), so --cold and the plain random_sphere path render
        bit-for-bit identically to before this method existed. With --collisions on, dead slots
        (m == 0, left behind by a merger, carrying the surviving body's own position -- Sec.
        4.9(0)) are dropped before drawing: left in, they would render as phantom markers exactly
        on top of the body that absorbed them.
        """
        if self.collisions_enabled:
            m_cpu = self.state.m.detach().to("cpu").numpy()
            live_mask = m_cpu > 0
            self.n_live = int(live_mask.sum())
            pos = self.state.r.detach().to("cpu").numpy()[live_mask]
            sizes, colors = self._mass_visuals_live(m_cpu[live_mask])
            self.markers.set_data(pos, face_color=colors, size=sizes, edge_width=0)
        else:
            pos = self.state.r.detach().to("cpu").numpy()
            self.markers.set_data(pos, face_color=self.marker_colors, size=self.marker_sizes, edge_width=0)
        return pos

    def _accumulate_collision_stats(self, frame_stats) -> None:
        """
        Fold one frame's CollisionRunStats into the run-long totals. integrate() builds a fresh
        CollisionRunStats per call (covering only that call's STEPS_PER_FRAME steps), so the
        running totals live here, in the viewer, not in the returned object. Cheap host-side
        scalar bookkeeping only -- everything folded in here was already computed by the
        collision pass inside integrate(); this adds no extra O(N) or O(N^2) work per frame.
        """
        self.n_elastic_total += frame_stats.n_elastic
        self.n_merge_total += frame_stats.n_merge
        self.n_fragment_total += frame_stats.n_fragment
        self.e_int_total += frame_stats.delta_e_int
        self.c_coll_max = max(self.c_coll_max, frame_stats.c_coll_max)
        self.f_reject_max = max(self.f_reject_max, frame_stats.f_reject_max)

    def ic_status_line(self) -> str:
        ic_label = "cold_sphere" if self.cold else "random_sphere"
        return (
            f"IC: {ic_label}   M_real: {self.total_mass:.4e} kg   "
            f"Q_target: {self.virial_q:.3f}   m_max/m_min: {self.mass_ratio_real:.3f}"
        )

    def _build_scene(self):
        ic_kind = "cold" if self.cold else "random"
        precision_label = "fp64, collisions" if self.collisions_enabled else "fp32"
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=(1000, 800), show=True, bgcolor="black",
            title=f"{ic_kind} spherical collapse -- real time (GPU, {precision_label}, velocity_verlet)",
        )
        # Two-row grid: 3D view keeps 75% of the height and stays mouse-interactive; the
        # diagnostic plot below is interactive=False so dragging it cannot be mistaken for
        # rotating the 3D scene.
        grid = self.canvas.central_widget.add_grid()
        self.view = grid.add_view(row=0, col=0, row_span=PLOT_ROW_SPAN)
        self.view.camera = scene.cameras.TurntableCamera(fov=45, distance=self.view_radius * 1.3)
        self.plot_view = grid.add_view(row=PLOT_ROW_SPAN, col=0, row_span=1)
        self.plot_view.camera = scene.cameras.PanZoomCamera(rect=(0, -8.5, 1, 9))
        self.plot_view.interactive = False
        self._build_energy_plot()

        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self._update_markers()

        # vispy 0.16 Text does not lay out embedded "\n" correctly against anchor_y="top"
        # (only the last line ends up on screen); use one Text instance per HUD line instead,
        # each with its own fixed screen position, and update all instances together.
        hud_line_height = 14
        self._hud_positions = [(10, 14 + i * hud_line_height) for i in range(HUD_N_LINES)]
        self.hud = scene.visuals.Text(
            [""] * HUD_N_LINES, pos=self._hud_positions, color="white", font_size=9,
            anchor_x="left", anchor_y="top", parent=self.canvas.scene,
        )
        # Placed in the lower part of the 3D view, but above the energy panel: the collapse
        # concentrates the particles in the middle of the frame, so a centred message becomes
        # unreadable, and anything below PLOT_TOP_FRACTION lands on top of the plot.
        self.message = scene.visuals.Text(
            "", pos=(self.canvas.size[0] / 2, self.canvas.size[1] * PLOT_TOP_FRACTION * 0.92),
            color="yellow", font_size=14, anchor_x="center", anchor_y="center",
            parent=self.canvas.scene,
        )

        self.timer = app.Timer(interval=1.0 / TARGET_FPS, connect=self.on_timer, start=True)

    def _build_energy_plot(self) -> None:
        """Static parts of the bottom panel: the empty trace, the 5% band, and axis labels."""
        self.energy_line = scene.visuals.Line(color=(0.4, 1.0, 0.6, 1.0), width=1.5, parent=self.plot_view.scene)
        self.band_line = scene.visuals.Line(
            np.array([[0.0, ENERGY_BAND_LOG10], [1.0, ENERGY_BAND_LOG10]]),
            color=(1.0, 0.5, 0.1, 0.9), width=1.0, parent=self.plot_view.scene,
        )
        scene.visuals.Text(
            "5% (2019 reference tolerance, not a stopping criterion)", pos=(0.0, ENERGY_BAND_LOG10),
            color=(1.0, 0.5, 0.1, 1.0), font_size=8, anchor_x="left", anchor_y="bottom",
            parent=self.plot_view.scene,
        )
        panel_top = self.canvas.size[1] * PLOT_TOP_FRACTION
        scene.visuals.Text(
            "t / t_ff", pos=(self.canvas.size[0] - 10, self.canvas.size[1] - 12), color="white",
            font_size=8, anchor_x="right", anchor_y="bottom", parent=self.canvas.scene,
        )
        self._y_top_label = scene.visuals.Text(
            "", pos=(10, panel_top + 12), color="white", font_size=8,
            anchor_x="left", anchor_y="top", parent=self.canvas.scene,
        )
        self._y_bot_label = scene.visuals.Text(
            "", pos=(10, self.canvas.size[1] - 12), color="white", font_size=8,
            anchor_x="left", anchor_y="bottom", parent=self.canvas.scene,
        )
        # Quantity label, --collisions only: the panel plots a different quantity in that mode
        # (|dE_total/E0|, E_total = K + U + E_int, instead of the collisionless |dE/E0|), and this
        # names it on screen so the two are never mistaken for one another. It also states plainly
        # that the curve is REPORTED, not a stopping criterion here (docs/simulacao-estocastica.md
        # Sec. 4.10/4.13.6: the documented runaway-merger phenomenon drives max |E_int|/|E0| into
        # [3, 40] by design, so judging the trajectory by this curve would flag the phenomenon
        # itself) -- without that label, a viewer seeing the curve climb past where the 5% band
        # sits could misread it as the integrator breaking down. Built once, here, and never
        # touched again -- parameters (including whether --collisions is on) are fixed for the
        # run. Added only in this mode so the collisionless / --cold canvas render is untouched --
        # no new visual element appears where none did before.
        if self.collisions_enabled:
            scene.visuals.Text(
                "log10 |dE_total/E0|  --  REPORTED, not a stopping criterion  "
                "(E_total = K + U + E_int)",
                pos=(self.canvas.size[0] / 2, panel_top + 12), color="white", font_size=8,
                anchor_x="center", anchor_y="top", parent=self.canvas.scene,
            )

    def _record_energy(self) -> None:
        """
        Append one (t/t_ff, log10|dE/E0|) sample with adaptive decimation: once the fixed-capacity
        buffer fills, keep every other point and double the stride. Index 0 (t=0) is never
        displaced, so the trace always spans the full run in O(HIST_CAPACITY) memory.
        """
        self._hist_skip += 1
        if self._hist_skip < self._hist_stride:
            return
        self._hist_skip = 0
        if self._hist_n == HIST_CAPACITY:
            self._hist_x[: HIST_CAPACITY // 2] = self._hist_x[0:HIST_CAPACITY:2]
            self._hist_y[: HIST_CAPACITY // 2] = self._hist_y[0:HIST_CAPACITY:2]
            self._hist_n = HIST_CAPACITY // 2
            self._hist_stride *= 2
        self._hist_x[self._hist_n] = self.sim_time / self.t_ff
        self._hist_y[self._hist_n] = math.log10(max(self.rel_error, ERROR_FLOOR))
        self._hist_n += 1

    def _update_plot(self) -> None:
        """Redraw the energy trace and autoscale the plot's PanZoomCamera to fit band + data."""
        x = self._hist_x[: self._hist_n]
        y = self._hist_y[: self._hist_n]
        self.energy_line.set_data(np.column_stack([x, y]))

        # The y range covers the data and the 5% band, and nothing more: forcing |dE/E0| = 1 (the
        # stopping criterion without --collisions; with --collisions it is not one -- see
        # ENERGY_ERROR_LIMIT's docstring) into view as well would waste over a decade of a panel
        # that is only a quarter of the canvas tall, and the trace is what needs the room.
        x_hi = max(x[-1], 1e-6) * 1.05 if self._hist_n else 1.0
        y_min = min(y.min(), ENERGY_BAND_LOG10) if self._hist_n else ENERGY_BAND_LOG10 - 1.0
        y_max = max(y.max(), ENERGY_BAND_LOG10) if self._hist_n else ENERGY_BAND_LOG10
        margin = 0.5
        self.band_line.set_data(np.array([[0.0, ENERGY_BAND_LOG10], [x_hi, ENERGY_BAND_LOG10]]))
        self.plot_view.camera.rect = (0.0, y_min - margin, x_hi, (y_max - y_min) + 2 * margin)

        self._y_top_label.text = f"1e{y_max:+.0f}"
        self._y_bot_label.text = f"1e{y_min:+.0f}"

    def on_timer(self, event):
        if not self.frozen:
            if self.collisions_enabled:
                # integrate() returns (State, CollisionRunStats) whenever collision is not None,
                # instead of the plain State the collisionless call below returns -- the two
                # branches exist because of that return-type difference, not for any other reason.
                self.state, frame_stats = integrate(
                    self.state, integrator="velocity_verlet", backend=self.backend,
                    dt=self.dt, n_steps=STEPS_PER_FRAME_COLLISION, softening=config.SOFTENING,
                    collision=self.collision_model, collision_rng=self.collision_rng,
                )
                self._accumulate_collision_stats(frame_stats)
                self.step_count += STEPS_PER_FRAME_COLLISION
            else:
                self.state = integrate(
                    self.state, integrator="velocity_verlet", backend=self.backend,
                    dt=self.dt, n_steps=STEPS_PER_FRAME, softening=config.SOFTENING,
                )
                self.step_count += STEPS_PER_FRAME
            self.sim_time = self.step_count * self.dt

            # e is always K + U (total_energy, unchanged). With --collisions, the honest
            # conserved-up-to-dissipation quantity is E_total = K + U + E_int (E_int is the
            # cumulative collisional ledger folded in by _accumulate_collision_stats above, not
            # recomputed here -- no extra O(N^2) pass), and self.rel_error becomes |dE_total/E0|
            # instead of |dE/E0|; every reader of self.rel_error below (HUD, energy panel,
            # auto-stop) is written to that quantity, whichever it currently is.
            e = total_energy(self.state, softening=config.SOFTENING)
            if self.collisions_enabled:
                e_total = e + self.e_int_total
                self.rel_error = abs((e_total - self.e0) / self.e0)
            else:
                self.rel_error = abs((e - self.e0) / self.e0)
            self._record_energy()

            pos = self._update_markers()
            self._update_plot()

            # The energy auto-stop applies ONLY without --collisions -- see ENERGY_ERROR_LIMIT's
            # docstring: with --collisions, |dE_total/E0| is a REPORTED quantity (the documented
            # runaway-merger phenomenon drives it far past 1 by design), not a stopping criterion,
            # so it must never freeze the simulation here. The dispersal check applies in both
            # modes unchanged.
            if not self.collisions_enabled and self.rel_error > ENERGY_ERROR_LIMIT:
                self._freeze(
                    f"stopped: |dE/E0| = {self.rel_error:.3f} exceeded {ENERGY_ERROR_LIMIT:.0f} "
                    f"at t = {self.sim_time / self.t_ff:.3f} t_ff -- trajectory no longer physical"
                )
            elif not np.any(np.linalg.norm(pos, axis=1) < self.view_radius):
                self._freeze(
                    f"stopped: no particles remain in the viewed volume "
                    f"at t = {self.sim_time / self.t_ff:.3f} t_ff -- system has dispersed"
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
        alpha = 0.1
        self.fps_ema = inst_fps if self.fps_ema == 0.0 else (1 - alpha) * self.fps_ema + alpha * inst_fps

        freeze_line = "SIMULATION FROZEN -- camera still interactive" if self.frozen else ""

        if self.collisions_enabled:
            lines = [
                f"backend: {self.backend_name}   dtype: float64   N: {self.n}",
                self.ic_status_line(),
                f"integrator: velocity_verlet   steps/frame: {STEPS_PER_FRAME_COLLISION}   "
                f"dt: {self.dt:.4e} s ({DT_OVER_TFF_COLLISION:.2e} t_ff)",
                f"fps: {self.fps_ema:6.1f}   t = {self.sim_time / self.t_ff:7.4f} t_ff   step {self.step_count}",
                f"collisions: ON   chi: {self.chi:.4g}   N_live: {self.n_live}/{self.n}",
                f"events -- elastic: {self.n_elastic_total}   merge: {self.n_merge_total}   "
                f"fragment: {self.n_fragment_total}",
                f"REPORTED, not a stopping criterion: E_int/|E0| = {self.e_int_total / abs(self.e0):.4e}   "
                f"|dE_total/E0| = {self.rel_error:.4e}",
                freeze_line,
            ]
        else:
            # Unchanged from before --collisions existed: same 6 lines, same content, same order.
            lines = [
                f"backend: {self.backend_name}   dtype: float32   N: {self.n}",
                self.ic_status_line(),
                f"integrator: velocity_verlet   steps/frame: {STEPS_PER_FRAME}   "
                f"dt: {self.dt:.4e} s ({DT_OVER_TFF:.2e} t_ff)",
                f"fps: {self.fps_ema:6.1f}   t = {self.sim_time / self.t_ff:7.4f} t_ff   step {self.step_count}",
                f"|dE/E0| = {self.rel_error:.4e}",
                freeze_line,
            ]
        self.hud.text = lines


def main():
    args = parse_args()
    try:
        # The reference matters: vispy keeps the canvas alive on its own, but nothing holds the
        # Viewer. Dropping it lets the garbage collector take the Viewer and its app.Timer with
        # it, leaving a window that renders the initial frame and never advances.
        viewer = Viewer(args)
    except GPUUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        # Defense in depth: parse_args() already rejects the known --virial-q/--f-cut degeneracy
        # before this point, but random_sphere (e.g. via a too-small --n) can still raise. Either
        # way this must fail here, before the window opens, not as a bare traceback.
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    app.run()
    return viewer


if __name__ == "__main__":
    main()
