"""Capture reference simulation runs as animated GIFs for the README.

These are separate, deliberate artifacts. scripts/realtime.py itself writes nothing to disk;
recording is done here so the viewer stays a pure display and this stays reproducible.

Three modes, selected by the same --cold / --collisions flags realtime.py already accepts
(realtime.parse_args() is reused as-is; no new CLI surface is added here):

    python scripts/capture_collapse_gif.py --cold
        Cold reference collapse -> figures/collapse.gif. UNCHANGED since this script's original
        single purpose: this recording, and every number quoted for it in README.md and the 2019
        work, are pinned to cold_sphere run to t = 1.40 t_ff. The code path and every constant
        under "Mode: cold reference collapse" below are exactly what this script has always used
        -- do not edit them to accommodate the other two modes.

    python scripts/capture_collapse_gif.py
        Heterogeneous population, no collisions -> figures/collapse_heterogeneo.gif. Default
        random_sphere (Salpeter mass spectrum + virial-ratio velocities, docs/simulacao-
        estocastica.md Sec. 2-3): shows stages 1-2 of the initial-condition construction without
        stage 3 (collisions).

    python scripts/capture_collapse_gif.py --collisions
        Bound collisions -> figures/collapse_colisoes.gif. The repository's main phenomenon: a
        dominant body grows by absorbing the core. --cold is forced on here regardless of the
        command line (see that branch below for why), so this reproduces the exact configuration
        behind docs/simulacao-estocastica.md Sec. 4.13.7's accepted seed 20190225 row.

Every other flag realtime.py accepts (--n, --mass, --radius, --alpha, --mass-ratio, --virial-q,
--f-cut, --seed, --chi) still passes through unchanged; only --cold and --collisions select which
of the three recordings above gets made and where it is written.

The wall-clock frame rate shown live by the viewer is deliberately removed from the HUD in every
mode here. In a recording it measures how fast frames were captured, not how fast anything runs,
so leaving it in would put a meaningless number in a published figure.
"""
from __future__ import annotations

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import realtime  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
FIGURES_DIR = os.path.join(REPO, "figures")

# --- Mode: cold reference collapse (--cold) -------------------------------------------------
# UNCHANGED since this script's original purpose (see module docstring). The recording runs
# from t = 0 to shortly past the first centre crossing (measured at t/t_ff = 1.033 in
# results/2026/longrun_energy.csv), which is the interesting part: the sphere contracts, passes
# through itself, and starts to disperse.
COLD_GIF_PATH = os.path.join(FIGURES_DIR, "collapse.gif")
COLD_END_TFF = 1.40  # first centre crossing is at 1.033 t_ff; stop a little after it
COLD_STEPS_PER_GIF_FRAME = 20  # 20 * DT_OVER_TFF = 0.01 t_ff per captured frame
COLD_PLAYBACK_FPS = 20
COLD_OUTPUT_WIDTH = 500  # rendered at the viewer's native size, then downscaled for the README
COLD_GIF_COLORS = 64  # the scene is a dark field with two accent colours; 64 is plenty
COLD_GIF_METHOD = Image.MEDIANCUT  # PIL's default; unchanged since this script's original purpose

# --- Mode: heterogeneous population, no collisions (default, no flags) ----------------------
# Shows stages 1-2 of the initial-condition construction (mass spectrum + virial-ratio
# velocities, docs/simulacao-estocastica.md Sec. 2-3) without stage 3 (collisions): bodies of
# different sizes, 1-3 massive ones visible, a warmer (later, gentler) collapse than the cold
# reference. First compression (r_half minimum) measured by a standalone probe at this exact
# configuration (default seeds, Q=0.25, alpha=2.35, mass_ratio=1000, f_cut=0.5) at t = 1.19
# t_ff -- later than the cold reference's 1.03-1.04 t_ff, consistent with the Lagrange-Jacobi
# scaling t_collapse ~ (1-Q)^(-1/2) (docs/simulacao-estocastica.md Sec. 3, item 9: predicts a
# factor 1.1964 at Q = 0.25 relative to Q = 0).
HETERO_GIF_PATH = os.path.join(FIGURES_DIR, "collapse_heterogeneo.gif")
HETERO_END_TFF = 1.60  # a little past the measured first crossing at 1.19 t_ff
HETERO_STEPS_PER_GIF_FRAME = 20  # same 0.01 t_ff/frame cadence as the cold reference
HETERO_PLAYBACK_FPS = 20
HETERO_OUTPUT_WIDTH = 500
HETERO_GIF_COLORS = 64
# MAXCOVERAGE, not MEDIANCUT (the cold reference's method). Measured empirically, per captured
# frame, across a full 160-frame run of this exact mode: MEDIANCUT drops the gold marker cluster
# (the heaviest realized body's colour, the one thing this GIF exists to make visible -- "1-3
# massive bodies visible") in the great majority of frames once the scene gets busier past the
# first ~15 frames (22/160 frames kept it, even at colors=128) -- it optimises the palette for
# the most common colours (black background, blue/green field), and a small, rare, but visually
# essential accent cluster loses that competition. MAXCOVERAGE, which explicitly prioritises
# colours covering the largest contiguous area rather than the most frequent ones, kept the gold
# cluster in 158/160 frames at this SAME 64-colour budget as the cold reference (the 2 misses
# both fall at t = 1.18-1.22 t_ff, within the crossing itself, and were confirmed by checking the
# unquantized render: the raw frame has only 1-17 gold pixels there, i.e. the body really is
# almost fully occluded by the core at that instant, not a quantizer failure).
HETERO_GIF_METHOD = Image.MAXCOVERAGE

# --- Mode: bound collisions (--collisions) ---------------------------------------------------
# The repository's main phenomenon: a dominant body grows by absorbing the core. Measured
# (docs/simulacao-estocastica.md Sec. 4.13.7, ensemble of 4 collision seeds) the transition to
# runaway growth falls in t_runaway in [1.95, 1.99] t_ff, lasts [0.54, 0.70] t_ff, and the
# outcome (dominant body at ~30% of the total mass) is consolidated by t = 3-3.5 t_ff -- a
# recording stopping anywhere near the cold reference's 1.4 t_ff would show none of this, so
# this mode's END_TFF is set well past the consolidation point instead.
COLLISION_GIF_PATH = os.path.join(FIGURES_DIR, "collapse_colisoes.gif")
COLLISION_END_TFF = 3.30
# 420 steps/frame * DT_OVER_TFF_COLLISION =~ 0.025 t_ff per captured frame: coarser than the
# cold/heterogeneous 0.01 t_ff/frame cadence, chosen so the frame count (~132) stays close to
# the cold reference's 140 despite this mode covering 2.4x more simulated time -- otherwise the
# file would be several times the reference's size for no gain in what is visible per frame.
# Must be a multiple of realtime.STEPS_PER_FRAME (4) for calls_per_frame below to divide evenly.
COLLISION_STEPS_PER_GIF_FRAME = 420
COLLISION_PLAYBACK_FPS = 20
COLLISION_OUTPUT_WIDTH = 500
COLLISION_GIF_COLORS = 64
# See HETERO_GIF_METHOD: same mass-based colour ramp, same MAXCOVERAGE fix for the same measured
# gold-marker palette starvation under MEDIANCUT.
COLLISION_GIF_METHOD = Image.MAXCOVERAGE


def _cold_hud_lines(viewer, t_tff: float) -> list[str]:
    """Unchanged from this script's original single-purpose HUD block."""
    return [
        f"backend: {viewer.backend_name}   dtype: float32   N: {viewer.n}",
        viewer.ic_status_line(),
        f"integrator: velocity_verlet   dt: {viewer.dt:.4e} s "
        f"({realtime.DT_OVER_TFF:.2e} t_ff)",
        f"t = {t_tff:7.4f} t_ff   step {viewer.step_count}",
        f"|dE/E0| = {viewer.rel_error:.4e}",
        "",
    ]


def _collision_hud_lines(viewer, t_tff: float) -> list[str]:
    """
    Mirrors realtime.Viewer._update_hud's --collisions branch, minus the wall-clock fps line
    (see module docstring: a recording's fps measures capture speed, not run speed, so it does
    not belong in a published figure -- true in this mode exactly as in the cold one). Reports,
    rather than gates on, E_int/|E0| and |dE_total/E0| for the same reason realtime.py does not
    apply ENERGY_ERROR_LIMIT in this mode: the documented runaway-merger phenomenon drives these
    numbers far past what would be a defect in the collisionless path (docs/simulacao-
    estocastica.md Sec. 4.10/4.13.6).
    """
    return [
        f"backend: {viewer.backend_name}   dtype: float64   N: {viewer.n}",
        viewer.ic_status_line(),
        f"integrator: velocity_verlet   steps/frame: {realtime.STEPS_PER_FRAME_COLLISION}   "
        f"dt: {viewer.dt:.4e} s ({realtime.DT_OVER_TFF_COLLISION:.2e} t_ff)",
        f"t = {t_tff:7.4f} t_ff   step {viewer.step_count}",
        f"collisions: ON   chi: {viewer.chi:.4g}   N_live: {viewer.n_live}/{viewer.n}",
        f"events -- ricochet: {viewer.n_ricochet_total}   merge: {viewer.n_merge_total}   "
        f"erosion: {viewer.n_erosion_total}",
        f"REPORTED, not a stopping criterion: E_int/|E0| = "
        f"{viewer.e_int_total / abs(viewer.e0):.4e}   |dE_total/E0| = {viewer.rel_error:.4e}",
        "",
    ]


def _capture(
    viewer, *, end_tff: float, steps_per_gif_frame: int, dt_over_tff: float, playback_fps: int,
    output_width: int, gif_colors: int, gif_method: int, gif_path: str, hud_fn,
) -> None:
    n_frames = int(round(end_tff / (steps_per_gif_frame * dt_over_tff)))
    calls_per_frame = steps_per_gif_frame // realtime.STEPS_PER_FRAME
    print(f"capturing {n_frames} frames to t = {end_tff} t_ff", file=sys.stderr)

    frames: list[Image.Image] = []
    for i in range(n_frames):
        for _ in range(calls_per_frame):
            viewer.on_timer(None)

        t_tff = viewer.sim_time / viewer.t_ff
        viewer.hud.text = hud_fn(viewer, t_tff)

        rgb = viewer.canvas.render()[..., :3]
        img = Image.fromarray(rgb)
        height = round(output_width * img.height / img.width)
        img = img.resize((output_width, height), Image.LANCZOS)
        frames.append(img.quantize(colors=gif_colors, method=gif_method))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_frames}  t={t_tff:.3f} t_ff", file=sys.stderr)

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], optimize=True,
        duration=round(1000 / playback_fps), loop=0,
    )
    size_mb = os.path.getsize(gif_path) / 1024**2
    print(f"wrote {os.path.abspath(gif_path)}  "
          f"({len(frames)} frames, {frames[0].size[0]}x{frames[0].size[1]}, {size_mb:.2f} MB)")


def main() -> None:
    args = realtime.parse_args()

    if args.collisions:
        # The bound-collisions campaign behind every number quoted for this GIF in README.md
        # (docs/simulacao-estocastica.md Sec. 4.13.7) used equal masses (cold_sphere), not the
        # mass spectrum. Force --cold here regardless of what was passed on this script's own
        # command line, for the same reason the cold branch below always has: the recording
        # must never silently drift to a different initial condition than the one the quoted
        # numbers came from. --chi and the collision-model seed are left at realtime.py's
        # defaults (config.CHI_DEFAULT, config.COLLISION_SEED), which is exactly Sec. 4.13.7's
        # seed 20190225 row.
        args.cold = True
        viewer = realtime.Viewer(args)
        _capture(
            viewer, end_tff=COLLISION_END_TFF, steps_per_gif_frame=COLLISION_STEPS_PER_GIF_FRAME,
            dt_over_tff=realtime.DT_OVER_TFF_COLLISION, playback_fps=COLLISION_PLAYBACK_FPS,
            output_width=COLLISION_OUTPUT_WIDTH, gif_colors=COLLISION_GIF_COLORS,
            gif_method=COLLISION_GIF_METHOD, gif_path=COLLISION_GIF_PATH,
            hud_fn=_collision_hud_lines,
        )
    elif args.cold:
        viewer = realtime.Viewer(args)
        _capture(
            viewer, end_tff=COLD_END_TFF, steps_per_gif_frame=COLD_STEPS_PER_GIF_FRAME,
            dt_over_tff=realtime.DT_OVER_TFF, playback_fps=COLD_PLAYBACK_FPS,
            output_width=COLD_OUTPUT_WIDTH, gif_colors=COLD_GIF_COLORS,
            gif_method=COLD_GIF_METHOD, gif_path=COLD_GIF_PATH, hud_fn=_cold_hud_lines,
        )
    else:
        viewer = realtime.Viewer(args)
        _capture(
            viewer, end_tff=HETERO_END_TFF, steps_per_gif_frame=HETERO_STEPS_PER_GIF_FRAME,
            dt_over_tff=realtime.DT_OVER_TFF, playback_fps=HETERO_PLAYBACK_FPS,
            output_width=HETERO_OUTPUT_WIDTH, gif_colors=HETERO_GIF_COLORS,
            gif_method=HETERO_GIF_METHOD, gif_path=HETERO_GIF_PATH, hud_fn=_cold_hud_lines,
        )


if __name__ == "__main__":
    main()
