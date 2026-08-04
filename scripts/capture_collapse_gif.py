"""Capture the cold collapse as an animated GIF for the README.

This is a separate, deliberate artifact. scripts/realtime.py itself writes nothing to disk;
recording is done here so the viewer stays a pure display and this stays reproducible.

The recording runs from t = 0 to shortly past the first centre crossing (measured at
t/t_ff = 1.033 in results/2026/longrun_energy.csv), which is the interesting part: the sphere
contracts, passes through itself, and starts to disperse.

The wall-clock frame rate shown live by the viewer is deliberately removed from the HUD here.
In a recording it measures how fast frames were captured, not how fast anything runs, so
leaving it in would put a meaningless number in a published figure.
"""
from __future__ import annotations

import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import realtime  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")
GIF_PATH = os.path.join(REPO, "figures", "collapse.gif")

END_TFF = 1.40  # first centre crossing is at 1.033 t_ff; stop a little after it
STEPS_PER_GIF_FRAME = 20  # 20 * DT_OVER_TFF = 0.01 t_ff per captured frame
PLAYBACK_FPS = 20
OUTPUT_WIDTH = 500  # rendered at the viewer's native size, then downscaled for the README
GIF_COLORS = 64  # the scene is a dark field with two accent colours; 64 is plenty


def main() -> None:
    args = realtime.parse_args()
    viewer = realtime.Viewer(args)

    n_frames = int(round(END_TFF / (STEPS_PER_GIF_FRAME * realtime.DT_OVER_TFF)))
    calls_per_frame = STEPS_PER_GIF_FRAME // realtime.STEPS_PER_FRAME
    print(f"capturing {n_frames} frames to t = {END_TFF} t_ff", file=sys.stderr)

    frames: list[Image.Image] = []
    for i in range(n_frames):
        for _ in range(calls_per_frame):
            viewer.on_timer(None)

        t_tff = viewer.sim_time / viewer.t_ff
        viewer.hud.text = [
            f"backend: {viewer.backend_name}   dtype: float32   N: {viewer.n}",
            f"integrator: velocity_verlet   dt: {viewer.dt:.4e} s "
            f"({realtime.DT_OVER_TFF:.2e} t_ff)",
            f"t = {t_tff:7.4f} t_ff   step {viewer.step_count}",
            f"|dE/E0| = {viewer.rel_error:.4e}",
            "",
        ]

        rgb = viewer.canvas.render()[..., :3]
        img = Image.fromarray(rgb)
        height = round(OUTPUT_WIDTH * img.height / img.width)
        img = img.resize((OUTPUT_WIDTH, height), Image.LANCZOS)
        frames.append(img.quantize(colors=GIF_COLORS, method=Image.MEDIANCUT))

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_frames}  t={t_tff:.3f} t_ff", file=sys.stderr)

    os.makedirs(os.path.dirname(GIF_PATH), exist_ok=True)
    frames[0].save(
        GIF_PATH, save_all=True, append_images=frames[1:], optimize=True,
        duration=round(1000 / PLAYBACK_FPS), loop=0,
    )
    size_mb = os.path.getsize(GIF_PATH) / 1024**2
    print(f"wrote {os.path.abspath(GIF_PATH)}  "
          f"({len(frames)} frames, {frames[0].size[0]}x{frames[0].size[1]}, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
