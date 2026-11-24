"""Batched, float32-LoRA policy for the full bowl experiment.
Adds functionality without changing the earlier inference or connection-test classes.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import load_file
from peft import set_peft_model_state_dict, get_peft_model_state_dict
from .training_openvla import TrainableOpenVLAPolicy, NORM_KEY

class FullExperimentOpenVLAPolicy(TrainableOpenVLAPolicy):
    def __init__(self, config, action_statistics, rank=8):
        super().__init__(config, action_statistics, rank)
        # Keep the pretrained backbone in bf16; optimize LoRA in float32.
        for parameter in self.get_optim_params():
            parameter.data = parameter.data.float()

    def collate_samples(self, samples):
        prepared = [self.prepare_training_batch(sample) for sample in samples]
        pad = self.processor.tokenizer.pad_token_id
        if pad is None:
            pad = 0  # padding positions are excluded by attention_mask and labels
        batch = {
            "input_ids": pad_sequence([b["input_ids"][0] for b in prepared], batch_first=True, padding_value=pad),
            "attention_mask": pad_sequence([b["attention_mask"][0] for b in prepared], batch_first=True, padding_value=0),
            "labels": pad_sequence([b["labels"][0] for b in prepared], batch_first=True, padding_value=-100),
            "pixel_values": torch.cat([b["pixel_values"] for b in prepared], dim=0),
            "normalized_actions": torch.tensor(np.stack([self.normalize_action(s["action"].cpu().numpy()) for s in samples])),
        }
        assert int((batch["labels"] != -100).sum()) == 8 * len(samples)
        return batch

    def forward(self, batch):
        device = self.config.device
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model(input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                pixel_values=batch["pixel_values"].to(device, dtype=torch.bfloat16),
                                labels=batch["labels"].to(device), use_cache=False)
        loss = output.loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        with torch.no_grad():
            patches = output.logits.shape[1] - batch["input_ids"].shape[1]
            predicted = output.logits[:, patches:-1].argmax(-1)
            target = batch["labels"][:, 1:].to(device)
            mask = target > self.action_tokenizer.action_token_begin_idx
            count = len(batch["input_ids"])
            assert int(mask.sum()) == 7 * count
            accuracy = ((predicted == target) & mask).sum().float() / mask.sum()
            ids = predicted[mask].cpu().numpy()
            decoded = self.action_tokenizer.decode_token_ids_to_actions(ids).reshape(count, 7)
            truth = batch["normalized_actions"].numpy()
            l1 = float(np.abs(decoded - truth).mean())
            action_fraction = float(np.mean((ids > self.action_tokenizer.action_token_begin_idx) & (ids < self.processor.tokenizer.vocab_size)))
        return loss, {"loss": float(loss.detach()), "action_token_accuracy": float(accuracy),
                      "normalized_action_l1": l1, "predicted_action_token_fraction": action_fraction}

    def adapter_hash(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(get_peft_model_state_dict(self.model).items()):
            digest.update(name.encode()); digest.update(tensor.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()

    def save_adapter(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
        (output/"action_normalization.json").write_text(json.dumps({NORM_KEY:{"action":self.action_statistics}},indent=2))
        (output/"base_model.json").write_text(json.dumps({"checkpoint":self.checkpoint_path,"model_id":self.config.model_id,
            "reference_hashes":self.reference_hashes,"rank":self.model.peft_config["default"].r},indent=2))
        (output/"adapter_sha256.txt").write_text(self.adapter_hash())

    def load_saved_adapter(self, output):
        output = Path(output)
        weights = load_file(str(output/"adapter_model.safetensors"), device="cpu")
        set_peft_model_state_dict(self.model, weights, adapter_name="default")
        loaded = get_peft_model_state_dict(self.model)
        assert set(loaded) == set(weights)
        assert all(torch.equal(loaded[n].detach().cpu(),weights[n]) for n in weights), "LoRA reload mismatch"
        assert self.adapter_hash() == (output/"adapter_sha256.txt").read_text().strip()
        self.eval()
