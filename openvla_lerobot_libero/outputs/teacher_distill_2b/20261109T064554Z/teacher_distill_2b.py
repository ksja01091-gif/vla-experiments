"""Isolated OpenVLA teacher-rollout distillation for the LIBERO bowl task.

Run on the remote Linux host with lerobot312.  Existing checkpoints and code are read-only.
"""
import argparse
import bisect
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np
import torch
from PIL import Image

ROOT = Path("/home/jovyan/projects/lerobot-openvla")
SOURCE = ROOT / "outputs/rlrc_matched_3h/20261106T090120_404808Z"
STUDENT_START = SOURCE / "checkpoints/pruned/after_ppo"
TEACHER_START = ROOT / "outputs/five_epoch_bowl/20261103T064207_586519Z/checkpoints/step_005035"
ROLLOUT = ROOT / "outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py"
RUN = Path(os.environ["OPENVLA_KD_RUN"]).resolve()
sys.path.insert(0, str(SOURCE))
import policy as recovery

common = recovery.common
exp = recovery.exp
TASK = exp.TASK
IMAGE_KEY = "observation.images.image"


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_actor(role, stats, checkpoint=None):
    if role == "teacher":
        actor = exp.make_policy(stats, rank=8)
        actor.load_saved_adapter(TEACHER_START)
        common.prepare_inference(actor)
    elif role == "student":
        actor = recovery.load_checkpoint(checkpoint or STUDENT_START, stats)
        common.prepare_inference(actor)
    else:
        raise ValueError(role)
    actor.eval()
    return actor


def rollout_function():
    namespace = dict(vars(exp))
    exec(compile(ROLLOUT.read_text(), str(ROLLOUT), "exec"), namespace)
    return namespace["simulate_trial"]


def export_success(trial):
    state = trial["init_state_index"]
    assert 5 <= state <= 44 and trial["success"] and trial["trial_executed"]
    src = RUN / "simulation/teacher" / f"state_{state:02d}" / "steps.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    assert len(rows) == trial["policy_steps"] and rows
    images = []
    actions = []
    for row in rows:
        with Image.open(row["before"]["frame"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        action = np.asarray(row["raw_action"], dtype=np.float32)
        assert pixels.shape == (256, 256, 3) and action.shape == (7,)
        assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1.00001
        images.append(pixels)
        actions.append(action)
    out = RUN / "teacher_data" / f"state_{state:02d}"
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "images.npy", np.stack(images), allow_pickle=False)
    np.save(out / "actions.npy", np.stack(actions), allow_pickle=False)
    saved_images = np.load(out / "images.npy", mmap_mode="r", allow_pickle=False)
    saved_actions = np.load(out / "actions.npy", mmap_mode="r", allow_pickle=False)
    assert np.array_equal(saved_images[0], images[0])
    assert np.array_equal(saved_images[-1], images[-1])
    assert np.array_equal(saved_actions, actions)
    record = {"state": state, "frames": len(rows), "task": TASK,
              "teacher_success": True, "source_result": str(src.parent / "result.json"),
              "image_sha256": sha(out / "images.npy"), "action_sha256": sha(out / "actions.npy"),
              "teacher_checkpoint": str(TEACHER_START), "student_training_only": True}
    write(out / "metadata.json", record)
    return record


def collect():
    assert TEACHER_START.is_dir() and STUDENT_START.is_dir()
    train, val, stats = exp.get_data()
    assert len(train) == 4025 and len(val) == 462
    actor = make_actor("teacher", stats)
    actor_hash = actor.adapter_hash()
    simulate = rollout_function()
    trials = []
    exports = []
    for state in range(50):
        path = RUN / "simulation/teacher" / f"state_{state:02d}" / "result.json"
        if path.exists():
            trial = json.loads(path.read_text())
        else:
            trial = simulate(actor, "teacher", state, types.SimpleNamespace(run=RUN))
        if not trial["trial_executed"]:
            raise RuntimeError(f"Technical rollout error at state {state}: {trial.get('error')}")
        assert trial["init_state_index"] == state and trial["model_tag"] == "teacher"
        trials.append(trial)
        if 5 <= state <= 44 and trial["success"]:
            meta = RUN / "teacher_data" / f"state_{state:02d}" / "metadata.json"
            exports.append(json.loads(meta.read_text()) if meta.exists() else export_success(trial))
        write(RUN / "collect_progress.json", {"complete": False, "trials": len(trials),
              "successes": sum(x["success"] for x in trials),
              "successful_training_episodes": len(exports),
              "successful_training_frames": sum(x["frames"] for x in exports)})
        print("COLLECT", state, "success", trial["success"], "steps", trial["policy_steps"], flush=True)
    assert actor.adapter_hash() == actor_hash
    assert len({x["initial_image_sha256"] for x in trials}) == 50
    write(RUN / "teacher_collection.json", {"complete": True, "teacher_checkpoint": str(TEACHER_START),
          "states": list(range(50)), "training_states": list(range(5,45)),
          "evaluation_only_states": list(range(5)) + list(range(45,50)),
          "overall_successes": sum(x["success"] for x in trials),
          "training_state_successes": sum(x["success"] for x in trials[5:45]),
          "evaluation_only_successes": sum(x["success"] for x in trials[:5] + trials[45:]),
          "new_training_episodes": len(exports), "new_training_frames": sum(x["frames"] for x in exports),
          "teacher_unchanged": True, "trials": trials})


class TeacherFrames:
    def __init__(self):
        self.items = []
        self.ends = []
        total = 0
        for meta in sorted((RUN / "teacher_data").glob("state_*/metadata.json")):
            row = json.loads(meta.read_text())
            assert 5 <= row["state"] <= 44 and row["teacher_success"]
            folder = meta.parent
            assert sha(folder / "images.npy") == row["image_sha256"]
            assert sha(folder / "actions.npy") == row["action_sha256"]
            images = np.load(folder / "images.npy", mmap_mode="r", allow_pickle=False)
            actions = np.load(folder / "actions.npy", mmap_mode="r", allow_pickle=False)
            assert images.shape == (row["frames"], 256, 256, 3)
            assert actions.shape == (row["frames"], 7)
            self.items.append((images, actions))
            total += row["frames"]
            self.ends.append(total)
        self.total = total

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        episode = bisect.bisect_right(self.ends, index)
        first = 0 if episode == 0 else self.ends[episode - 1]
        images, actions = self.items[episode]
        pixels = np.array(images[index - first], copy=True)
        action = np.array(actions[index - first], copy=True)
        return {IMAGE_KEY: torch.from_numpy(pixels).permute(2,0,1).float().div(255),
                "action": torch.from_numpy(action), "task": TASK}


def train():
    report = json.loads((RUN / "teacher_collection.json").read_text())
    assert report["complete"] and report["new_training_frames"] >= 100
    human, validation, stats = exp.get_data()
    synthetic = TeacherFrames()
    assert len(human) == 4025 and len(synthetic) == report["new_training_frames"]
    actor = recovery.load_checkpoint(STUDENT_START, stats)
    actor.configure_trainables(False)
    frozen_before = exp.frozen_sample_hash(actor)
    actor_before = actor.actor_hash()
    before_val = exp.evaluate(actor, validation, 4, RUN / "validation_before.jsonl", 0)
    optimizer = torch.optim.AdamW(actor.get_optim_params(), lr=3e-5, weight_decay=0.0)
    rng = np.random.default_rng(20261109)
    order = [("human", int(i)) for i in rng.permutation(len(human))]
    order += [("teacher", int(i)) for i in rng.permutation(len(synthetic))]
    rng.shuffle(order)
    counts = {"human": 0, "teacher": 0}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (RUN / "training_metrics.jsonl").open("x") as log:
        for offset in range(0, len(order), 4):
            batch_items = order[offset:offset+4]
            samples = [human[i] if source == "human" else synthetic[i]
                       for source, i in batch_items]
            for source, _ in batch_items:
                counts[source] += 1
            actor.train()
            actor.model.vision_backbone.eval()
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = actor(actor.collate_samples(samples))
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(actor.get_optim_params(), 1.0)
            assert bool(torch.isfinite(grad))
            optimizer.step()
            row = {"step": offset//4 + 1, "frames_seen": offset+len(batch_items),
                   "human_frames_seen": counts["human"], "teacher_frames_seen": counts["teacher"],
                   **metrics, "gradient_norm": float(grad), "gpu_peak_gib": recovery.memory()["peak_gib"]}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if row["step"] % 25 == 0 or row["step"] == 1:
                log.flush()
                write(RUN / "train_progress.json", row)
                print("TRAIN", row["step"], row["frames_seen"], len(order), "loss", row["loss"], flush=True)
    assert counts == {"human": 4025, "teacher": len(synthetic)}
    assert exp.frozen_sample_hash(actor) == frozen_before
    assert actor.actor_hash() != actor_before
    actor.eval()
    after_val = exp.evaluate(actor, validation, 4, RUN / "validation_after.jsonl", 0)
    dest = RUN / "checkpoints/student_after_distill"
    bytes_written = recovery.save_checkpoint(actor, dest)
    write(RUN / "training_result.json", {"complete": True, "student_start": str(STUDENT_START),
          "checkpoint": str(dest), "checkpoint_weight_bytes": bytes_written,
          "human_frames": 4025, "teacher_frames": len(synthetic), "batch_size": 4,
          "epochs": 1, "lr": 3e-5, "optimizer": "AdamW", "gradient_clip": 1.0,
          "trainable_parameters": sum(p.numel() for p in actor.get_optim_params()),
          "frozen_backbone_unchanged": True, "before_validation": before_val,
          "after_validation": after_val, "training_seconds": time.perf_counter()-started,
          "gpu_peak_gib": recovery.memory()["peak_gib"]})


def evaluate(role, checkpoint=None):
    _, val, stats = exp.get_data()
    actor = make_actor(role, stats, checkpoint)
    if role == "student":
        tag = "student_after" if checkpoint else "student_before"
        actor_hash = actor.actor_hash()
    else:
        tag = "teacher"
        actor_hash = actor.adapter_hash()
    simulate = rollout_function()
    trials = []
    for state in range(50):
        path = RUN / "simulation" / tag / f"state_{state:02d}" / "result.json"
        if path.exists():
            trial = json.loads(path.read_text())
        else:
            trial = simulate(actor, tag, state, types.SimpleNamespace(run=RUN))
        if not trial["trial_executed"]:
            raise RuntimeError(f"Technical rollout error at state {state}: {trial.get('error')}")
        trials.append(trial)
        write(RUN / f"{tag}_progress.json", {"completed": len(trials),
              "successes": sum(x["success"] for x in trials)})
        print("EVALUATE", tag, state, trial["success"], trial["policy_steps"], flush=True)
    assert (actor.actor_hash() if role == "student" else actor.adapter_hash()) == actor_hash
    groups = {"training_states_5_44": trials[5:45],
              "evaluation_only_0_4_45_49": trials[:5] + trials[45:]}
    write(RUN / f"{tag}_evaluation.json", {"complete": True,
          "checkpoint": str(checkpoint or (TEACHER_START if role == "teacher" else STUDENT_START)),
          "all_50": {"successes": sum(x["success"] for x in trials), "attempts": 50},
          "groups": {name: {"successes": sum(x["success"] for x in subset),
                    "attempts": len(subset), "mean_steps": sum(x["policy_steps"] for x in subset)/len(subset)}
                    for name, subset in groups.items()},
          "note": "40 states are used for new teacher data; only 10 states are evaluation-only. No independent-50 generalization claim.",
          "trials": trials})


def smoke():
    human, validation, stats = exp.get_data()
    assert len(human) == 4025 and len(validation) == 462
    actor = recovery.load_checkpoint(STUDENT_START, stats)
    with torch.no_grad():
        loss, metrics = actor(actor.collate_samples([human[0]]))
        synthetic = TeacherFrames()
        assert len(synthetic) > 0
        synthetic_loss, synthetic_metrics = actor(actor.collate_samples([synthetic[0]]))
    common.prepare_inference(actor)
    action = common.sample_action(actor, validation, 0)
    assert action.shape == (1, 7) and bool(torch.isfinite(action).all())
    write(RUN / "smoke_result.json", {"complete": True, "student_checkpoint": str(STUDENT_START),
          "student_parameters": actor.parameter_report()["model_parameters"],
          "one_human_frame_loss": float(loss), "one_human_frame_metrics": metrics,
          "available_teacher_frames": len(synthetic),
          "one_teacher_frame_loss": float(synthetic_loss),
          "one_teacher_frame_metrics": synthetic_metrics,
          "action_shape": list(action.shape), "action_finite": True,
          "gpu_peak_gib": recovery.memory()["peak_gib"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "collect", "train", "evaluate_before", "evaluate_after"])
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.phase == "smoke":
        smoke()
    elif args.phase == "collect":
        collect()
    elif args.phase == "train":
        train()
    elif args.phase == "evaluate_before":
        evaluate("student")
    else:
        checkpoint = json.loads((RUN / "training_result.json").read_text())["checkpoint"]
        evaluate("student", Path(checkpoint))


if __name__ == "__main__":
    main()
