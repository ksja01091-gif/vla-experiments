"""Run one saved LIBERO image through the LeRobot OpenVLA policy.

This checks the inference connection only. No simulator, training, or W&B.
Image processing is resize-only, not the official LIBERO benchmark pipeline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.openvla.configuration_openvla import OpenVLAConfig
from lerobot.policies.openvla.modeling_openvla import OpenVLAPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path.home() / "projects/openvla/libero_test.png")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/openvla_select_action")
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    config = OpenVLAConfig(device=args.device, model_id=args.model_id)
    assert issubclass(OpenVLAPolicy, PreTrainedPolicy)
    assert not OpenVLAPolicy.__abstractmethods__
    eager_modules = [name for name in sys.modules if name.startswith((
        "lerobot.policies.groot.modeling_", "lerobot.policies.pi0.modeling_",
        "lerobot.policies.pi05.modeling_", "lerobot.policies.pi05.processor_",
    ))]
    if eager_modules:
        raise RuntimeError(f"Unexpected unrelated model imports: {eager_modules}")
    print("CLASS_IMPORT_OK: OpenVLAPolicy is a LeRobot PreTrainedPolicy", flush=True)
    print("LAZY_IMPORT_OK: unrelated GR00T/Pi0/Pi0.5 implementations not loaded", flush=True)
    if args.check_imports:
        return

    image_path = args.image.expanduser().resolve(strict=True)
    with Image.open(image_path) as source:
        original_size = source.size
        image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    batch = {
        config.image_key: pil_to_tensor(image).to(torch.float32).div(255).unsqueeze(0),
        "task": [args.task],
    }
    torch.manual_seed(7)
    print("INPUT_OK:", tuple(batch[config.image_key].shape), flush=True)
    print("Loading cached checkpoint; no package changes or downloads.", flush=True)
    policy = OpenVLAPolicy(config)
    policy.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    action = policy.select_action(batch)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    if action.shape != (1, 7) or not bool(torch.isfinite(action).all()):
        raise RuntimeError("Expected finite action with shape (1, 7).")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir.expanduser().resolve() / stamp
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "passed", "scope": "inference_smoke_test_only",
        "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
        "model_id": args.model_id, "checkpoint_path": policy.checkpoint_path,
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "original_image_size": list(original_size),
        "preprocessing": "RGB; PIL bilinear resize to 224x224; CHW float32 /255; batch dimension",
        "official_benchmark_preprocessing": False,
        "task": args.task, "unnorm_key": config.unnorm_key,
        "input_shape": list(batch[config.image_key].shape),
        "action_shape": list(action.shape), "all_finite": True,
        "action": action.detach().cpu().tolist(),
        "action_dtype": str(action.dtype), "device": str(action.device),
        "gpu": torch.cuda.get_device_name(), "seed": 7,
        "inference_seconds": inference_seconds,
        "versions": {name: version(name) for name in [
            "lerobot", "torch", "transformers", "timm", "peft", "diffusers", "numpy",
        ]},
        "training_run": False, "simulator_evaluation_run": False, "wandb_used": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (output / "packages.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True,
    ))
    print("SELECT_ACTION_OK:", action.detach().cpu().tolist(), flush=True)
    print("SHAPE:", tuple(action.shape), "FINITE:", bool(torch.isfinite(action).all()), flush=True)
    print("RESULT:", output / "result.json", flush=True)


if __name__ == "__main__":
    main()
