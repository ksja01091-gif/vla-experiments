"""Inference-only OpenVLA adapter for LeRobot 0.4.0.

Input: one RGB float32 image [1, 3, 224, 224] in [0, 1], and task=[text].
Output: one unnormalized action [1, 7]. This is not a simulator command.
"""
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForVision2Seq, AutoProcessor

from lerobot.policies.pretrained import PreTrainedPolicy
from .configuration_openvla import OpenVLAConfig


class OpenVLAPolicy(PreTrainedPolicy):
    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(self, config: OpenVLAConfig):
        super().__init__(config)
        config.validate_features()
        if torch.device(config.device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("This smoke test requires the existing CUDA environment.")
        source = Path(config.model_id).expanduser()
        if not source.is_dir():
            source = Path(snapshot_download(config.model_id, local_files_only=True))
        self.checkpoint_path = str(source.resolve())
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint_path, trust_remote_code=True, local_files_only=True,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        ).to(config.device)
        if config.unnorm_key not in self.model.norm_stats:
            raise ValueError(f"Missing action statistics: {config.unnorm_key}")
        self.eval()

    def reset(self):
        """This single-observation adapter keeps no rollout history."""
        return None

    @torch.inference_mode()
    def select_action(self, batch, **kwargs):
        """Return a finite float32 tensor with shape (1, 7)."""
        images = batch[self.config.image_key]
        tasks = batch["task"]
        if not isinstance(images, torch.Tensor) or images.shape != (1, 3, 224, 224):
            raise ValueError("Expected image tensor with shape (1, 3, 224, 224).")
        if not images.is_floating_point():
            raise ValueError("Expected a floating-point RGB image in [0, 1].")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("Image contains NaN or infinity.")
        if bool((images < 0).any()) or bool((images > 1).any()):
            raise ValueError("Image values must be in [0, 1].")
        if (not isinstance(tasks, list) or len(tasks) != 1
                or not isinstance(tasks[0], str) or not tasks[0].strip()):
            raise ValueError("Expected task=[one nonempty instruction string].")

        image = to_pil_image(images[0].detach().to(device="cpu", dtype=torch.float32))
        task = tasks[0].strip().lower()
        prompt = f"In: What action should the robot take to {task}?\nOut:"
        inputs = self.processor(prompt, image).to(
            self.config.device, dtype=torch.bfloat16,
        )
        self.eval()
        raw_action = self.model.predict_action(
            **inputs, unnorm_key=self.config.unnorm_key, do_sample=False,
        )
        action = torch.tensor(
            raw_action, dtype=torch.float32, device=self.config.device,
        ).unsqueeze(0)
        if action.shape != (1, 7) or not bool(torch.isfinite(action).all()):
            raise RuntimeError("Model returned an invalid action; expected finite (1, 7).")
        return action

    def predict_action_chunk(self, batch, **kwargs):
        """A one-step chunk, with shape (1, 1, 7)."""
        return self.select_action(batch, **kwargs).unsqueeze(1)

    def forward(self, batch):
        raise NotImplementedError("Training is outside this inference-only adapter.")

    def get_optim_params(self):
        raise NotImplementedError("Optimizer setup is outside this inference-only adapter.")
