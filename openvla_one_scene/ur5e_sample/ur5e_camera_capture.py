"""Save one virtual-camera image of the UR5e block scene."""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image

from ur5e_auto_grasp import make_scene


model, data, _ = make_scene()

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0.2, 0.0, 0.28]
camera.distance = 1.8
camera.azimuth = 130
camera.elevation = -35

renderer = mujoco.Renderer(model, height=480, width=640)
try:
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    output = Path.home() / "ur5e_camera.png"
    Image.fromarray(pixels).save(output)
    print(f"Saved: {output}")
    print(f"Image size: {pixels.shape}")
finally:
    renderer.close()
