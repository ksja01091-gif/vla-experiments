"""Expose an ACT checkpoint to XenoArm's existing local simulator client."""
import argparse
import asyncio

import numpy as np
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from xenoarm.serving import serve


class ACTAdapter:
    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    @torch.inference_mode()
    def infer(self, element):
        obs = {
            "observation.images.image": np.ascontiguousarray(element["observation/image"]),
            "observation.images.wrist_image": np.ascontiguousarray(element["observation/wrist_image"]),
            "observation.state": np.asarray(element["observation/state"], dtype=np.float32),
        }
        batch = self.pre(prepare_observation_for_inference(obs, self.device))
        chunk = self.post(self.policy.predict_action_chunk(batch))[0].float().cpu().numpy()
        if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
            raise ValueError("ACT must output a finite sequence of seven-dimensional actions")
        return {"actions": chunk}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    adapter = ACTAdapter(args.checkpoint, args.device)
    asyncio.run(serve(adapter, "127.0.0.1", args.port, {"action_space": "ee_delta_si", "model": "ACT"}))


if __name__ == "__main__":
    main()
