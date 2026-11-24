"""Opt-in LoRA training for the existing LeRobot OpenVLA policy.
The original inference adapter and cached base weights are never overwritten.
"""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from .modeling_openvla import OpenVLAPolicy

OFFICIAL = Path("/home/jovyan/projects/openvla")
NORM_KEY = "libero_bowl_train20"

def load_reference(relative, name):
    path = OFFICIAL / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}

class TrainableOpenVLAPolicy(OpenVLAPolicy):
    def __init__(self, config, action_statistics, rank=8):
        super().__init__(config)
        action_module, hashes1 = load_reference("prismatic/vla/action_tokenizer.py", "openvla_reference_action_tokens")
        prompt_module, hashes2 = load_reference("prismatic/models/backbones/llm/prompting/base_prompter.py", "openvla_reference_prompts")
        self.reference_hashes = {**hashes1, **hashes2}
        self.action_tokenizer = action_module.ActionTokenizer(self.processor.tokenizer)
        self.prompt_builder_class = prompt_module.PurePromptBuilder
        self.action_statistics = dict(action_statistics)
        self.action_statistics["mask"] = [True] * 6 + [False]
        self.model.norm_stats[NORM_KEY] = {"action": self.action_statistics}
        self.config.unnorm_key = NORM_KEY
        lora = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0,
                          target_modules=["q_proj", "v_proj"], init_lora_weights="gaussian")
        self.model = get_peft_model(self.model, lora)
        trainable = [name for name, param in self.model.named_parameters() if param.requires_grad]
        if not trainable or not all("lora_" in n and "language_model" in n for n in trainable):
            raise RuntimeError(f"Unexpected trainable parameters: {trainable}")
        self.model.language_model.config.use_cache = False
        self.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.print_trainable_parameters()

    def normalize_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        low = np.asarray(self.action_statistics["q01"], dtype=np.float32)
        high = np.asarray(self.action_statistics["q99"], dtype=np.float32)
        mask = np.asarray(self.action_statistics["mask"], dtype=bool)
        scaled = np.clip(2 * (action - low) / (high - low + 1e-8) - 1, -1, 1)
        result = np.where(mask, scaled, action)
        zeros = np.asarray(self.action_statistics["min"]) == np.asarray(self.action_statistics["max"])
        result = np.where(zeros, 0, result).astype(np.float32)
        assert result.shape == (7,) and np.isfinite(result).all()
        return result

    def prepare_training_batch(self, sample):
        pixels = sample[self.config.image_key].permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
        action = self.normalize_action(sample["action"].cpu().numpy())
        builder = self.prompt_builder_class("openvla")
        builder.add_turn("human", f"What action should the robot take to {sample['task'].strip().lower()}?")
        builder.add_turn("gpt", self.action_tokenizer(action))
        encoded = self.processor(builder.get_prompt(), image, return_tensors="pt")
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        labels[:, :-8] = -100
        expected = self.processor.tokenizer.vocab_size - np.digitize(action, self.action_tokenizer.bins)
        assert np.array_equal(labels[0, -8:-1].numpy(), expected), "Action token roundtrip changed labels"
        assert labels[0, -1].item() == self.processor.tokenizer.eos_token_id
        assert int((labels != -100).sum()) == 8
        return {"input_ids": input_ids, "attention_mask": encoded["attention_mask"],
                "pixel_values": encoded["pixel_values"], "labels": labels}

    def forward(self, batch):
        device = self.config.device
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16),
                                labels=batch["labels"].to(device), use_cache=False)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        with torch.no_grad():
            patches = output.logits.shape[1] - batch["input_ids"].shape[1]
            predicted = output.logits[:, patches:-1].argmax(-1)
            target = batch["labels"][:, 1:].to(device)
            mask = target > self.action_tokenizer.action_token_begin_idx
            assert int(mask.sum()) == 7
            accuracy = ((predicted == target) & mask).sum().float() / mask.sum()
            pred_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted[mask].cpu().numpy())
            target_actions = self.action_tokenizer.decode_token_ids_to_actions(target[mask].cpu().numpy())
        return loss, {"loss": float(loss.detach()), "action_token_accuracy": float(accuracy),
                      "normalized_action_l1": float(np.abs(pred_actions - target_actions).mean())}

    def get_optim_params(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    def save_training_adapter(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(output, safe_serialization=True)
        (output / "action_normalization.json").write_text(json.dumps({NORM_KEY: {"action": self.action_statistics}}, indent=2) + "\n")
        (output / "base_model.json").write_text(json.dumps({"checkpoint": self.checkpoint_path,
            "model_id": self.config.model_id, "reference_hashes": self.reference_hashes}, indent=2) + "\n")
