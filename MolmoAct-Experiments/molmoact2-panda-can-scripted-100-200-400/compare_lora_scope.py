"""Compare two Panda 400-demo runs on the same serialized MuJoCo reset states."""

import argparse
import json
import math
from pathlib import Path

from collect_manifest import write_json

BASE = "allenai/MolmoAct2-LIBERO-LeRobot"
ADAPTER = Path("molmoact2__teacher__panda_k400_r32/seed777")
EXPERT = ("expert_attn", "expert_ffn")
VARIANTS = {
    "both": (Path("training") / ADAPTER, Path("evaluation/k400"),
             ("vlm_attn", "vlm_ffn", *EXPERT)),
    "expert_only": (Path("training_action_expert_only") / ADAPTER,
                    Path("evaluation/k400_action_expert_only"), EXPERT),
}


def read(path):
    return json.loads(path.read_text())


def verified_result(variant, layouts):
    adapter, evaluation, scopes = VARIANTS[variant]
    run = read(adapter / "xenoarm_run.json")
    lora = read(adapter / "xenoarm_lora.json")
    report = run["parameters"]
    settings = run["optim"]
    expected_settings = {"seed": 777, "steps": 4000, "lr": 5e-5, "warmup_steps": 200}
    if (run["base"], run["robot"], run["shots"], run["dataset"], run["seed"]) != (
        BASE, "panda", 400, "xenoarm_panda_scripted/can_k400", 777
    ) or Path(run["dataset_root"]).resolve() != Path("converted/k400").resolve():
        raise ValueError(f"{variant} used a different model or dataset.")
    if any(settings[key] != value for key, value in expected_settings.items()):
        raise ValueError(f"{variant} used different optimizer settings.")
    if (lora["lora"]["rank"], tuple(lora["lora"]["scopes"]), lora["step"],
        lora["fully_trained_params"]) != (32, scopes, 4000, 0):
        raise ValueError(f"{variant} did not finish the expected LoRA training.")
    if set(report["lora_params_by_scope"]) != set(scopes):
        raise ValueError(f"{variant} has unexpected trainable scopes.")

    training = [json.loads(line) for line in (adapter / "train_log.jsonl").read_text().splitlines() if line]
    if not training or training[-1]["step"] != 4000:
        raise ValueError(f"{variant} is missing the final training log entry.")
    peak = max(row["max_memory_gb"] for row in training)
    seconds = int((evaluation / "training_seconds.txt").read_text())
    if not math.isfinite(peak) or min(peak, seconds) <= 0:
        raise ValueError(f"{variant} has invalid time or memory measurements.")

    result = read(evaluation / "summary.json")
    rows = result["episodes"]
    starts = [(row["seed"], row["initial_state_hash"]) for row in rows]
    if (result["trials"], result["max_steps"], result["replan_steps"]) != (50, 1000, 5):
        raise ValueError(f"{variant} used different rollout settings.")
    if starts != layouts or any(row["placed"] and not row["lifted"] for row in rows):
        raise ValueError(f"{variant} evaluation starts or outcomes do not match the protocol.")
    for metric in ("lifted", "placed"):
        if result[metric] != sum(row[metric] for row in rows):
            raise ValueError(f"{variant} summary disagrees with the per-episode {metric} results.")

    return {
        "scopes": list(scopes),
        "trials": result["trials"],
        "lifted": result["lifted"],
        "placed": result["placed"],
        "trainable_parameters": report["trainable_params"],
        "training_wall_seconds": seconds,
        "peak_training_allocated_vram_gib": peak,
    }, rows, settings


def report_comparison(both, expert):
    lines = [
        "# Panda 400 demonstrations · MolmoAct2 LoRA scope comparison",
        "",
        "Both runs use the same Panda can demonstration data and the same starting checkpoint. "
        "Training uses seed 777, LoRA rank 32, batch size 4, and 4,000 optimizer updates. "
        "The evaluations use the same 50 MuJoCo initial states, seeds 30000–30049.",
        "",
        "| LoRA placement | Lift | Place | Training time | Peak training GPU memory | Trainable parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in (("Language backbone + action expert", both), ("Action expert only", expert)):
        seconds = row["training_wall_seconds"]
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        lines.append(
            f"| {label} | {row['lifted']}/50 ({2 * row['lifted']}%) | "
            f"{row['placed']}/50 ({2 * row['placed']}%) | {hours}h {minutes}m {secs}s | "
            f"{row['peak_training_allocated_vram_gib']:.2f} GiB | {row['trainable_parameters']:,} |"
        )
    lines += [
        "",
        "![Pick, place, training time, and peak GPU memory](comparison.png)",
        "",
        "Training time runs from launching the trainer through saving the final adapter. "
        "Peak GPU memory is PyTorch's peak allocated CUDA memory during training in GiB.",
        "",
        "- [Both scopes rollout data](../k400/summary.json)",
        "- [Action expert only rollout data](summary.json)",
        "- [Numeric comparison and paired outcomes](comparison.json)",
    ]
    return "\n".join(lines) + "\n"


def plot_comparison(path, both, expert):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    metrics = (
        ("Lift success (%)", [2 * row["lifted"] for row in (both, expert)], "%.0f%%"),
        ("Placement success (%)", [2 * row["placed"] for row in (both, expert)], "%.0f%%"),
        ("Training time (min)", [row["training_wall_seconds"] / 60 for row in (both, expert)], "%.1f"),
        ("Peak training GPU memory (GiB)", [row["peak_training_allocated_vram_gib"]
                                             for row in (both, expert)], "%.2f"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    figure.suptitle("Panda can pick-and-place · MolmoAct2 LoRA", fontsize=16, fontweight="bold")
    for axis, (title, values, label) in zip(axes.flat, metrics, strict=True):
        bars = axis.bar(["Language + action", "Action only"], values,
                        color=["#284766", "#137B6D"], width=0.52, zorder=2)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylim(0, 112 if label == "%.0f%%" else max(values) * 1.2)
        axis.bar_label(bars, fmt=label, padding=4)
        axis.grid(axis="y", alpha=0.22, zorder=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-baseline", action="store_true")
    args = parser.parse_args()
    starts = read(Path("raw/evaluation_layouts.json"))
    layouts = [(row["seed"], row["initial_state_hash"]) for row in starts]
    if [seed for seed, _ in layouts] != list(range(30000, 30050)) or len({h for _, h in layouts}) != 50:
        raise ValueError("The evaluation list must contain 50 unique, reserved starts.")
    both, both_rows, both_optim = verified_result("both", layouts)
    if args.check_baseline:
        print("Verified the completed 400-demo baseline and all 50 evaluation starts.", flush=True)
        return
    expert, expert_rows, expert_optim = verified_result("expert_only", layouts)
    if both_optim != expert_optim:
        raise ValueError("Training optimizer settings differ between the two runs.")
    paired = [
        {"seed": base["seed"], "both_lifted": base["lifted"], "both_placed": base["placed"],
         "expert_lifted": scoped["lifted"], "expert_placed": scoped["placed"]}
        for base, scoped in zip(both_rows, expert_rows, strict=True)
    ]
    _, folder, _ = VARIANTS["expert_only"]
    plot_comparison(folder / "comparison.png", both, expert)
    write_json(folder / "comparison.json", {"both": both, "expert_only": expert, "paired": paired})
    draft = folder / "COMPARISON.md.tmp"
    draft.write_text(report_comparison(both, expert), encoding="utf-8")
    draft.replace(folder / "COMPARISON.md")


if __name__ == "__main__":
    main()
