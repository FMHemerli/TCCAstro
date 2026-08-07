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
trajectory sampled every STEPS_PER_FRAME steps, not an interpolation.

The cap exists so the collapse can be watched. Simulated time advances at
TARGET_FPS * STEPS_PER_FRAME * DT_OVER_TFF t_ff per wall-clock second; uncapped, this machine
runs the whole collapse and first centre crossing in well under a second. The cap changes only
the pacing, never dt or the trajectory. The achieved rate is shown live as "fps".

State (positions, velocities, masses) lives on the GPU between steps. The only thing that crosses
to the host each frame is the (N, 3) position array needed to draw the markers; velocities, masses
and the integrator's internal acceleration buffer never leave the device.

Integrator: velocity_verlet only, fixed for the whole run. It is symplectic, so its energy error
oscillates in a bounded band instead of drifting away secularly -- the right property for a panel
meant to run indefinitely.

Precision: float32, fixed. On this project's GPU backends float32 is measured at roughly 20x the
throughput of float64, and the resulting round-off is far below the 5% band that matters physically
here (see docs/integradores.md for the measurement this decision is based on).

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

Auto-stop (two criteria, checked every frame): |dE/E0| > 1 (accumulated energy error equals the
total energy -- the trajectory no longer means anything), or no particle left inside the viewed
volume (the collapse has ejected/dispersed everything worth looking at). Either one freezes the
simulation (stops advancing state) but leaves the window, camera and HUD live, with an on-screen
message naming which criterion fired and at what simulated time.

Usage:
    python scripts/realtime.py [--n N] [--mass MASS] [--radius RADIUS] [--cold]
        [--alpha ALPHA] [--mass-ratio RATIO] [--virial-q Q] [--f-cut F_CUT] [--seed SEED]

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

from nbody import config  # noqa: E402
from nbody.backends import get_backend  # noqa: E402
from nbody.backends.base import BackendUnavailable  # noqa: E402
from nbody.initial_conditions import cold_sphere, random_sphere  # noqa: E402
from nbody.integrators import integrate  # noqa: E402
from nbody.observables import total_energy  # noqa: E402

DTYPE = torch.float32
STEPS_PER_FRAME = 4  # fixed simulation steps advanced per rendered frame
DT_OVER_TFF = 1.0 / 2000.0  # fixed simulation timestep, as a fraction of t_ff (matches sanity.py check 3)
# Frame rate is capped so the collapse is watchable. Uncapped, the loop runs at 600-880 fps
# (measured, N=1000), which advances 1.2-1.8 t_ff per wall-clock second: the whole collapse and
# first centre crossing would be over in under a second. Capping also stops the viewer rendering
# far faster than any display can show. At this cap the run advances
# TARGET_FPS * STEPS_PER_FRAME * DT_OVER_TFF = 0.12 t_ff per second, so the collapse takes ~8 s.
TARGET_FPS = 60.0
VIEW_RADIUS_FACTOR = 4.0  # "viewed volume" and initial camera framing, as a multiple of sphere_radius
ENERGY_ERROR_LIMIT = 1.0  # |dE/E0| beyond this: the trajectory is no longer physical
HUD_N_LINES = 6  # fixed HUD line count (5 status lines + 1 freeze-message slot, blank when running)

# Marker encoding of mass (--cold bypasses all of this and reuses the literal pre-spectrum values
# below directly, so its picture is bit-for-bit identical to before random_sphere was wired in).
MARKER_SIZE_BASE = 4.0  # the size every marker had before the mass spectrum existed
MARKER_SIZE_MIN = 2.0  # floor: the lightest sampled bodies (~0.28x mean) must stay visible
MARKER_SIZE_MAX = 20.0  # ceiling: the 1-3 Salpeter-tail bodies (up to ~285x mean) must not become blobs
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
    args = parser.parse_args()

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

        if self.cold:
            self.state = cold_sphere(
                n=self.n, radius=self.radius, particle_mass=self.mass,
                seed=self.seed, dtype=DTYPE, device=self.device,
            )
        else:
            self.state = random_sphere(
                n=self.n, radius=self.radius, seed=self.seed,
                mass_spectrum={
                    "alpha": self.alpha, "ratio": self.mass_ratio, "particle_mass": self.mass,
                },
                virial_ratio=self.virial_q, f_cut=self.f_cut,
                dtype=DTYPE, device=self.device,
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
        self.dt = self.t_ff * DT_OVER_TFF
        self.view_radius = VIEW_RADIUS_FACTOR * self.radius

        self.mass_min_real = float(self.state.m.min().item())
        self.mass_max_real = float(self.state.m.max().item())
        self.mass_ratio_real = self.mass_max_real / self.mass_min_real
        self.marker_sizes, self.marker_colors = self._mass_visuals(self.state.m)

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

    def ic_status_line(self) -> str:
        ic_label = "cold_sphere" if self.cold else "random_sphere"
        return (
            f"IC: {ic_label}   M_real: {self.total_mass:.4e} kg   "
            f"Q_target: {self.virial_q:.3f}   m_max/m_min: {self.mass_ratio_real:.3f}"
        )

    def _build_scene(self):
        ic_kind = "cold" if self.cold else "random"
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=(1000, 800), show=True, bgcolor="black",
            title=f"{ic_kind} spherical collapse -- real time (GPU, fp32, velocity_verlet)",
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

        pos0 = self.state.r.detach().to("cpu").numpy()
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.markers.set_data(pos0, face_color=self.marker_colors, size=self.marker_sizes, edge_width=0)

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

        # The y range covers the data and the 5% band, and nothing more: forcing |dE/E0| = 1
        # (the stopping criterion) into view as well would waste over a decade of a panel that
        # is only a quarter of the canvas tall, and the trace is what needs the room.
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
            self.state = integrate(
                self.state, integrator="velocity_verlet", backend=self.backend,
                dt=self.dt, n_steps=STEPS_PER_FRAME, softening=config.SOFTENING,
            )
            self.step_count += STEPS_PER_FRAME
            self.sim_time = self.step_count * self.dt

            e = total_energy(self.state, softening=config.SOFTENING)
            self.rel_error = abs((e - self.e0) / self.e0)
            self._record_energy()

            pos = self.state.r.detach().to("cpu").numpy()
            self.markers.set_data(pos, face_color=self.marker_colors, size=self.marker_sizes, edge_width=0)
            self._update_plot()

            if self.rel_error > ENERGY_ERROR_LIMIT:
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

        lines = [
            f"backend: {self.backend_name}   dtype: float32   N: {self.n}",
            self.ic_status_line(),
            f"integrator: velocity_verlet   steps/frame: {STEPS_PER_FRAME}   "
            f"dt: {self.dt:.4e} s ({DT_OVER_TFF:.2e} t_ff)",
            f"fps: {self.fps_ema:6.1f}   t = {self.sim_time / self.t_ff:7.4f} t_ff   step {self.step_count}",
            f"|dE/E0| = {self.rel_error:.4e}",
            "SIMULATION FROZEN -- camera still interactive" if self.frozen else "",
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
