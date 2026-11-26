"""Resume the public MolmoAct checkpoint download on the Linux HPC workspace."""

import json
import subprocess
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
REPO = "allenai/MolmoAct-7B-D-0812"
DEST = ROOT / "model" / "MolmoAct-7B-D-0812"
CDN_RESOLVE = "us.aws.cdn.hf.co:443:18.136.219.244"
STATUS = ROOT / "model_download_status.json"


def main():
    response = requests.get(
        f"https://huggingface.co/api/models/{REPO}/tree/main",
        params={"recursive": "true", "expand": "true"},
        timeout=30,
    )
    response.raise_for_status()
    files = [x for x in response.json() if x["type"] == "file"]
    status = {
        "phase": "downloading",
        "repo": REPO,
        "started_unix": time.time(),
        "expected_files": len(files),
        "expected_bytes": sum(x.get("size") or 0 for x in files),
        "completed_files": 0,
        "completed_bytes": 0,
    }
    for entry in files:
        path = entry["path"]
        size = entry.get("size") or 0
        target = DEST / path
        target.parent.mkdir(parents=True, exist_ok=True)
        status["current_file"] = path
        STATUS.write_text(json.dumps(status, indent=2))
        if not target.exists() or target.stat().st_size != size:
            part = target.with_name(target.name + ".part")
            url = f"https://huggingface.co/{REPO}/resolve/main/{path}"
            subprocess.run(
                ["curl", "--fail", "--location", "--silent", "--show-error",
                 "--retry", "5", "--retry-delay", "5", "--connect-timeout", "30",
                 "--resolve", CDN_RESOLVE, "--continue-at", "-",
                 "--output", str(part), url],
                check=True,
            )
            if size and part.stat().st_size != size:
                raise RuntimeError(f"Size mismatch: {path}")
            part.replace(target)
        status["completed_files"] += 1
        status["completed_bytes"] += size
        STATUS.write_text(json.dumps(status, indent=2))
    status["phase"] = "complete"
    status["finished_unix"] = time.time()
    STATUS.write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (ROOT / "model_download_error.txt").write_text(repr(exc))
        raise
