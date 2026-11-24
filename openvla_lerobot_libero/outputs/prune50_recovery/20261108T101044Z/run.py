"""Continue matched dense/pruned SFT, then matched on-policy PPO and rollouts.

This file is staged on Windows and executed only on the remote Linux server.
Existing checkpoints and source code are read-only.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import previous_worker as w
from policy import *

RUN = Path(__file__).resolve().parent
CFG = json.loads((RUN / "experiment_config.json").read_text())
w.R = RUN
w.CFG = CFG


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def sft(branch):
    policy, train, val, source = w.load(branch, "initial")
    assert len(train) == 4025 and len(val) == 462
    old_state = torch.load(source / "trainer_state.pt", map_location="cpu", weights_only=False)
    policy.configure_trainables(False)
    frozen_before = exp.frozen_sample_hash(policy)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4, weight_decay=0)
    assert old_state["optimizer"] is not None
    optimizer.load_state_dict(old_state["optimizer"])
    previous_hash = policy.actor_hash()
    for epoch in (2, 3):
        w.check_budget(reserve=600)
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(700 + epoch)).tolist()
        coverage = []
        totals = {k: 0.0 for k in exp.METRICS}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for offset in range(0, len(order), 4):
            w.check_budget(reserve=600)
            indices = order[offset:offset + 4]
            m = common.step(policy, optimizer, train, indices)
            coverage.extend(indices)
            for key in totals:
                totals[key] += m[key] * len(indices)
            if offset == 0 or (offset // 4 + 1) % 25 == 0:
                save_json(RUN / "progress.json", {"phase": "sft", "branch": branch,
                    "epoch": epoch, "frames": len(coverage), "total": 4025, "loss": m["loss"]})
                print("SFT", branch, epoch, len(coverage), "/4025", m["loss"], flush=True)
        assert sorted(coverage) == list(range(4025))
        assert policy.actor_hash() != previous_hash
        assert exp.frozen_sample_hash(policy) == frozen_before
        previous_hash = policy.actor_hash()
        metrics = w.validation(policy, val, branch, "epoch" + str(epoch))
        stage = "after_sft" if epoch == 3 else "epoch2"
        cp, size = w.save(policy, branch, stage, optimizer, total_epochs=epoch,
            optimizer_updates=epoch * math.ceil(4025 / 4), order_seed=700 + epoch)
        save_json(RUN / branch / f"epoch{epoch}_result.json", {
            "complete": True, "branch": branch, "total_epochs": epoch,
            "additional_epoch": epoch - 1, "frames": 4025,
            "order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(),
            "training_metrics": {k: v / 4025 for k, v in totals.items()},
            "validation": metrics, "seconds": time.perf_counter() - started,
            "memory": memory(), "checkpoint": str(cp), "checkpoint_bytes": size,
            "frozen_samples_unchanged": True})
    print("SFT_COMPLETE", branch, flush=True)


def simulate_sft(branch):
    policy, train, val, cp = w.load(branch, "after_sft")
    policy.eval()
    before = policy.actor_hash()
    action = common.sample_action(policy, val, 0).cpu()
    assert action.shape == (1, 7) and bool(torch.isfinite(action).all())
    namespace = dict(vars(exp))
    source = Path("/home/jovyan/projects/lerobot-openvla/outputs/input_pipeline_check/20261103T055533_073692Z/matched_rollout_function.py")
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    tag = branch + "_sft3"
    base = Path("/home/jovyan/projects/lerobot-openvla/outputs/five_epoch_bowl/20261103T064207_586519Z/simulation/epoch5")
    trials = []
    for state in CFG["evaluation_states"]:
        w.check_budget(reserve=600)
        print("EVAL_SFT", branch, state, flush=True)
        trial = namespace["simulate_trial"](policy, tag, state, types.SimpleNamespace(run=RUN))
        reference = json.loads((base / f"state_{state:02d}" / "result.json").read_text())
        for key in ("initial_image_sha256", "selected_init_state_sha256", "init_state_file_sha256",
                    "reference_sha256", "simulator_versions", "max_policy_steps", "warmup_steps"):
            assert trial[key] == reference[key], key
        trials.append(trial)
        save_json(RUN / branch / "simulation_sft3_progress.json", {"trials": trials})
    assert before == policy.actor_hash()
    save_json(RUN / branch / "simulation_sft3_result.json", {
        "complete": True, "tag": tag, "checkpoint": str(cp),
        "states": CFG["evaluation_states"], "attempts": len(trials),
        "successes": sum(x["success"] for x in trials), "trials": trials,
        "mean_episode_steps": float(np.mean([x["policy_steps"] for x in trials])),
        "all_actions_finite": all(x["all_actions_finite"] for x in trials)})


def collect(branch):
    policy, train, val, cp = w.load(branch, "after_sft")
    before = policy.actor_hash()
    rows, trials = [], []
    for state in CFG["rl_training_states"]:
        w.check_budget(reserve=600)
        print("RL_COLLECT_BEGIN", branch, state, flush=True)
        part, result = w.collect_episode(policy, branch, state)
        rows.extend(part)
        trials.append(result)
        torch.save(rows, RUN / branch / "on_policy_transitions.pt")
        save_json(RUN / branch / "collection_progress.json", {"trials": trials, "transitions": len(rows)})
    assert before == policy.actor_hash()
    save_json(RUN / branch / "collection_result.json", {"complete": True,
        "episodes": len(trials), "transitions": len(rows),
        "successes": sum(x["success"] for x in trials), "trials": trials})


def matched_budget():
    counts = {b: json.loads((RUN / b / "collection_result.json").read_text())["transitions"]
              for b in ("dense", "pruned")}
    n = min(counts.values()) // 4 * 4
    assert n >= 4
    save_json(RUN / "matched_ppo_budget.json", {"transitions_each": n,
        "source_counts": counts, "passes": 2,
        "optimizer_updates_each": 2 * (n // 4), "selection": "same-size deterministic subset"})
    print("MATCHED_RL_BUDGET", n, flush=True)


def report():
    rows = []
    for branch in ("dense", "pruned"):
        for stage, file in (("after_sft3", "simulation_sft3_result.json"),
                            ("after_ppo", "simulation_result.json")):
            result = json.loads((RUN / branch / file).read_text())
            rows.append({"branch": branch, "stage": stage, "successes": result["successes"],
                         "attempts": result["attempts"], "mean_steps": result["mean_episode_steps"],
                         "checkpoint": result.get("checkpoint")})
    save_json(RUN / "summary.json", {"complete": True, "rows": rows,
        "limitations": "One task; five previously inspected evaluation states; no generalization claim."})
    print("SUMMARY", json.dumps(rows), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["sft", "simulate_sft", "collect", "budget", "ppo", "evaluate_rl", "report"])
    parser.add_argument("--branch", choices=["dense", "pruned"])
    args = parser.parse_args()
    if args.phase in {"sft", "simulate_sft", "collect", "ppo", "evaluate_rl"}:
        assert args.branch
    if args.phase == "sft": sft(args.branch)
    elif args.phase == "simulate_sft": simulate_sft(args.branch)
    elif args.phase == "collect": collect(args.branch)
    elif args.phase == "budget": matched_budget()
    elif args.phase == "ppo": w.ppo(args.branch)
    elif args.phase == "evaluate_rl": w.evaluate(args.branch)
    else: report()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        save_json(RUN / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise
