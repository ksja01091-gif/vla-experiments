"""Build a table from the converted datasets and the three verified evaluation summaries."""

import json
from pathlib import Path

from collect_manifest import write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    rows = []
    for count in (100, 200, 400):
        directory = Path("evaluation") / f"k{count}"
        result = read(directory / "summary.json")
        dataset = read(f"converted/k{count}/conversion_verified.json")
        url = directory / "wandb_url.txt"
        rows.append({"demonstrations": count, "action_frames": dataset["frames"],
                     "lifted": result["lifted"], "placed": result["placed"], "trials": result["trials"],
                     "training_seconds": int((directory / "training_seconds.txt").read_text()),
                     "evaluation_seconds": result["elapsed_seconds"],
                     "wandb_url": url.read_text().strip() if url.exists() else None})
    write_json(Path("results.json"), rows)
    header = "# Panda can pick-and-place with MolmoAct2\n\n"
    header += "| Successful demos | Action frames | Lift | Place | Training time | W&B |\n"
    header += "| ---: | ---: | ---: | ---: | ---: | --- |\n"
    lines = []
    for row in rows:
        url = f"[Run]({row['wandb_url']})" if row["wandb_url"] else "Pending"
        lines.append(f"| {row['demonstrations']} | {row['action_frames']:,} | {row['lifted']}/50 | "
                     f"{row['placed']}/50 | {row['training_seconds']} s | {url} |")
    Path("RESULTS.md").write_text(header + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
