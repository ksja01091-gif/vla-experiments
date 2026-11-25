from pathlib import Path
import shutil

base = Path("source/src")
target = Path("full_train_src")
shutil.copytree(base, target)
file = target / "xenoarm/backbones/molmoact2.py"
overlay = Path(__file__).with_name("xenoarm_molmoact2_adapter.py")
code = (overlay if overlay.exists() else file).read_text(encoding="utf-8")
changes = {
    "from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig":
        "from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig",
    "delta_timestamps=delta, image_transforms=transforms)":
        'delta_timestamps=delta, image_transforms=transforms, video_backend="pyav")',
    "num_workers=num_workers, drop_last=True,": "num_workers=num_workers, drop_last=False,",
}
for old, new in changes.items():
    if code.count(old) != 1:
        raise ValueError(f"Expected one reference to {old!r}")
    code = code.replace(old, new)
file.write_text(code, encoding="utf-8")
trainer = target / "xenoarm/teacher/train.py"
code = trainer.read_text(encoding="utf-8")
old = "warmup_steps=args.warmup, seed=args.seed)"
if code.count(old) != 1:
    raise ValueError("Could not locate the optimizer configuration")
trainer.write_text(code.replace(old, "warmup_steps=args.warmup, seed=args.seed, save_every=1000)"), encoding="utf-8")
