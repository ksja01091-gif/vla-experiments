"""Download the public Panda/UR5e Pick Place D0 pair on the Linux server."""

import json
import subprocess
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "datasets"
ROBOTS = {
    "panda": "OliverHausdoerfer/demo_src_pick_place_task_D0_robot_Panda_gripper_PandaGripper",
    "ur5e": "OliverHausdoerfer/demo_src_pick_place_task_D0_robot_UR5e_gripper_Robotiq85Gripper",
}
CDN_RESOLVE = "us.aws.cdn.hf.co:443:18.136.219.244"


def save_status(status):
    (ROOT / "download_status.json").write_text(json.dumps(status, indent=2))


def main():
    status = {"phase": "inventory", "started_unix": time.time(), "robots": {}}
    save_status(status)
    for robot, repo in ROBOTS.items():
        response = requests.get(
            f"https://huggingface.co/api/datasets/{repo}/tree/main",
            params={"recursive": "true", "expand": "true"},
            timeout=30,
        )
        response.raise_for_status()
        files = [entry for entry in response.json() if entry["type"] == "file"]
        status["robots"][robot] = {
            "repo": repo,
            "file_count": len(files),
            "expected_bytes": sum(entry.get("size") or 0 for entry in files),
            "completed_files": 0,
            "completed_bytes": 0,
        }
        save_status(status)
        for entry in files:
            path = entry["path"]
            target = DEST / robot / path
            target.parent.mkdir(parents=True, exist_ok=True)
            size = entry.get("size") or 0
            if not target.exists() or target.stat().st_size != size:
                partial = target.with_name(target.name + ".part")
                url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
                subprocess.run(
                    [
                        "curl", "--fail", "--location", "--silent", "--show-error",
                        "--retry", "4", "--retry-delay", "3", "--connect-timeout", "30",
                        "--resolve", CDN_RESOLVE, "--continue-at", "-",
                        "--output", str(partial), url,
                    ],
                    check=True,
                )
                if size and partial.stat().st_size != size:
                    raise RuntimeError(f"Size mismatch for {robot}/{path}")
                partial.replace(target)
            status["phase"] = "downloading"
            status["robots"][robot]["completed_files"] += 1
            status["robots"][robot]["completed_bytes"] += size
            status["current_file"] = f"{robot}/{path}"
            save_status(status)
    status["phase"] = "complete"
    status["finished_unix"] = time.time()
    save_status(status)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (ROOT / "download_error.txt").write_text(repr(exc))
        raise
