"""Run the existing evaluation worker and add read-only grasp measurements."""

import contextlib
from pathlib import Path
import sys

import numpy as np
from libero.libero import envs

sys.path.insert(0, '/home/jovyan/projects/lerobot-openvla/examples')
import libero_eval_worker as worker


class MeasuredEnv(envs.OffScreenRenderEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = self.env.objects_dict['akita_black_bowl_1']
        self.plate = self.env.objects_dict['plate_1']
        self.target_id = self.sim.model.body_name2id(self.target.root_body)
        self.plate_id = self.sim.model.body_name2id(self.plate.root_body)
        self.gripper = self.robots[0].gripper

    def measure(self, step):
        bowl = self.sim.data.body_xpos[self.target_id].copy()
        plate = self.sim.data.body_xpos[self.plate_id]
        if step == 0:
            self.start_bowl = bowl
        return {
            'bowl_xyz_m': bowl.tolist(),
            'plate_xyz_m': plate.tolist(),
            'bowl_rise_m': float(bowl[2] - self.start_bowl[2]),
            'bowl_plate_xy_m': float(np.linalg.norm(bowl[:2] - plate[:2])),
            'both_fingerpads': bool(self.env._check_grasp(self.gripper, self.target.contact_geoms)),
        }


def main():
    emit = worker.emit

    def make_env(**kwargs):
        nonlocal active_env
        active_env = MeasuredEnv(**kwargs)
        return active_env

    def emit_with_measurements(payload):
        payload.update(active_env.measure(payload['step']))
        emit(payload)

    active_env = None
    envs.OffScreenRenderEnv = make_env
    worker.emit = emit_with_measurements
    worker.main()


if __name__ == '__main__':
    with contextlib.redirect_stdout(sys.stderr):
        main()
