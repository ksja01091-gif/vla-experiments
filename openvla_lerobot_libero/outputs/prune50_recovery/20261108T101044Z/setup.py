"""Prepare an isolated remote experiment directory; never edit prior runs."""
import hashlib
import json
import shutil
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = Path("/home/jovyan/projects/lerobot-openvla")
OLD90 = ROOT / "outputs/rlrc_matched_3h/20261106T090120_404808Z"
OLD50 = ROOT / "outputs/prune50_sft1/20261106T122012_474442Z"
DIAG = ROOT / "outputs/prune50_diagnosis/20261106T132021_191937Z"

assert RUN.is_relative_to(ROOT / "outputs/prune50_recovery")
for name, source in {
    "policy.py": OLD50 / "policy.py",
    "rl_probe.py": OLD90 / "rl_probe.py",
    "previous_worker.py": OLD90 / "worker.py",
    "replay.py": DIAG / "replay.py",
}.items():
    target = RUN / name
    assert source.is_file() and not target.exists()
    shutil.copy2(source, target)

target = RUN / "replay.py"
text = target.read_text()
needle = "CFG=json.loads((R/'diagnostic_config.json').read_text());SOURCE=Path(CFG['source_run'])"
assert text.count(needle) == 1
text = text.replace(needle, needle + "\nimport sys\nif '--tags' in sys.argv: CFG['tags']=sys.argv[sys.argv.index('--tags')+1:]", 1)
target.write_text(text)

config = json.loads((OLD90 / "experiment_config.json").read_text())
config.update({
    "run_description": "50% structural pruning recovery: matched total3 SFT epochs then matched PPO",
    "starting_dense_checkpoint": str(OLD90 / "checkpoints/dense/after_sft"),
    "starting_pruned_checkpoint": str(OLD50 / "checkpoints/p50/after_sft"),
    "additional_sft_epochs_each": 2,
    "total_sft_epochs_each": 3,
    "sft_order_seeds": [702, 703],
    "rl_training_states": [5, 6, 7, 8, 9],
    "rl_episodes_each": 5,
    "evaluation_states": [0, 1, 2, 3, 4],
    "ppo_passes": 2,
    "runtime_cap_seconds": 10800,
    "prior_runs_read_only": True,
    "no_package_installation": True,
    "known_limitation": "Five previously inspected evaluation states; no new-state generalization claim",
    "reward_limitation": "Success-only sparse RL reward; a zero-success collection may not teach grasping",
})
(RUN / "experiment_config.json").write_text(json.dumps(config, indent=2))
for branch, source in {
    "dense": OLD90 / "checkpoints/dense/after_sft",
    "pruned": OLD50 / "checkpoints/p50/after_sft",
}.items():
    assert (source / "model.safetensors").is_file()
    stage = RUN / "checkpoints" / branch
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "initial").symlink_to(source, target_is_directory=True)
    (RUN / branch).mkdir()

diagnosis = json.loads((DIAG / "diagnostic_config.json").read_text())
diagnosis["source_run"] = str(RUN)
diagnosis["tags"] = ["pruned_sft3", "dense_sft3", "pruned", "dense"]
diagnosis["states"] = [0, 1, 2, 3, 4]
(RUN / "diagnostic_config.json").write_text(json.dumps(diagnosis, indent=2))
(RUN / "setup_complete.json").write_text(json.dumps({"complete": True,
    "source_checkpoints": {b: str(s) for b, s in {
        "dense": OLD90 / "checkpoints/dense/after_sft",
        "pruned": OLD50 / "checkpoints/p50/after_sft"}.items()},
    "source_code_hashes": {name: hashlib.sha256((RUN/name).read_bytes()).hexdigest()
                           for name in ["policy.py", "rl_probe.py", "previous_worker.py", "replay.py"]}}, indent=2))
print("SETUP_COMPLETE", RUN)
