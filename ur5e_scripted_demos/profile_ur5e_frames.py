"""Measure MuJoCo image rendering and image-file save cost."""

import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image

from ur5e_auto_grasp import make_scene

model, data, _ = make_scene()
for size in (256, 128):
    renderer = mujoco.Renderer(model, height=size, width=size)
    try:
        t = time.monotonic()
        for i in range(10):
            renderer.update_scene(data)
            pixels = renderer.render()
        render_time = time.monotonic() - t
        for directory in (Path("/tmp/ur5e_render_profile"), Path("ur5e_render_profile")):
            directory.mkdir(exist_ok=True)
            t = time.monotonic()
            for i in range(10):
                Image.fromarray(pixels).save(directory / f"{size}_{i}.jpg", quality=88)
            print(size, directory, "render_10_s", round(render_time, 3),
                  "save_10_s", round(time.monotonic() - t, 3), flush=True)
    finally:
        renderer.close()
