"""Share manifests between the simulator and LeRobot environments."""

import json


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_episodes(raw):
    return json.loads((raw / "manifest.json").read_text())["episodes"]
