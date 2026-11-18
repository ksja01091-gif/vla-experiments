"""Build a VS Code notebook whose cells execute on the user's Colab server."""

import base64
import json
from pathlib import Path


ROOT = Path(__file__).parent
OUTPUT = ROOT / "OpenVLA_base_LIBERO_one_scene.ipynb"


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


prepare_code = (ROOT / "prepare_libero_target_scene.py").read_text(encoding="utf-8")
predict_code = (ROOT / "base_openvla_one_scene.py").read_text(encoding="utf-8")
convert_code_b64 = base64.b64encode((ROOT / "convert_openvla_action.py").read_bytes()).decode("ascii")
train20_stats_b64 = base64.b64encode((ROOT / "action_stats.json").read_bytes()).decode("ascii")

cells = [
    markdown_cell(
        "# Basic OpenVLA on one LIBERO scene\n"
        "This notebook runs in Colab from VS Code. It uses **openvla/openvla-7b**, "
        "not the LIBERO-fine-tuned checkpoint. It extracts the exact bowl task "
        "from the public OpenVLA LIBERO Spatial RLDS data, predicts seven "
        "**normalized** action values, and saves a JSON result. "
        "The demonstration actions are raw robot values, so the two arrays "
        "must not be directly subtracted. No training or robot rollout occurs.\n"
    ),
    code_cell(
        "import subprocess\n"
        "from pathlib import Path\n"
        "subprocess.run(['nvidia-smi'], check=True)\n"
        "subprocess.run(['uv', 'python', 'install', '3.10'], check=True)\n"
        "venv = Path('/content/openvla310')\n"
        "if not venv.exists():\n"
        "    subprocess.run(['uv', 'venv', str(venv), '--python', '3.10', '--seed'], check=True)\n"
        "packages = ['numpy==1.26.4', 'torch==2.2.0', 'torchvision==0.17.0', "
        "'transformers==4.40.1', 'tokenizers==0.19.1', 'timm==0.9.10', "
        "'accelerate==1.15.0', 'huggingface-hub==0.36.2', 'pillow', 'sentencepiece']\n"
        "subprocess.run(['uv', 'pip', 'install', '--python', str(venv / 'bin/python'), *packages], check=True)\n"
        "subprocess.run([str(venv / 'bin/python'), '-m', 'pip', 'check'], check=True)\n"
        "print('OpenVLA Python environment ready')\n"
    ),
    code_cell(
        "from pathlib import Path\n"
        f"Path('/content/prepare_libero_target_scene.py').write_text({prepare_code!r}, encoding='utf-8')\n"
        "result = subprocess.run([str(venv / 'bin/python'), '/content/prepare_libero_target_scene.py'], "
        "capture_output=True, text=True)\n"
        "print(result.stdout[-3000:])\n"
        "print(result.stderr[-500:])\n"
        "result.check_returncode()\n"
    ),
    code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "from PIL import Image\n"
        "from IPython.display import display\n"
        "sample = json.loads(Path('/content/openvla_base_one_scene/sample.json').read_text())\n"
        "print('Instruction:', sample['instruction'])\n"
        "print('Demonstration action (raw robot values):', sample['demonstration_action_raw'])\n"
        "display(Image.open(sample['image_path']))\n"
    ),
    code_cell(
        "from pathlib import Path\n"
        f"Path('/content/base_openvla_one_scene.py').write_text({predict_code!r}, encoding='utf-8')\n"
        "result = subprocess.run([str(venv / 'bin/python'), '/content/base_openvla_one_scene.py', "
        "'--image', sample['image_path'], '--instruction', sample['instruction'], "
        "'--output', '/content/openvla_base_one_scene/prediction.json'], "
        "capture_output=True, text=True)\n"
        "print(result.stdout[-3000:])\n"
        "print(result.stderr[-1000:])\n"
        "result.check_returncode()\n"
    ),
    code_cell(
        "# Convert the normalized prediction using the prior training-only statistics.\n"
        "import base64, subprocess\n"
        "from pathlib import Path\n"
        f"Path('/content/convert_openvla_action.py').write_bytes(base64.b64decode('{convert_code_b64}'))\n"
        f"Path('/content/openvla_base_one_scene/train20_action_stats.json').write_bytes(base64.b64decode('{train20_stats_b64}'))\n"
        "result = subprocess.run([str(venv / 'bin/python'), '/content/convert_openvla_action.py', "
        "'--prediction', '/content/openvla_base_one_scene/prediction.json', "
        "'--stats', '/content/openvla_base_one_scene/train20_action_stats.json', "
        "'--output', '/content/openvla_base_one_scene/converted_action.json'], "
        "capture_output=True, text=True)\n"
        "print(result.stdout)\n"
        "print(result.stderr[-500:])\n"
        "result.check_returncode()\n"
    ),
    code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "root = Path('/content/openvla_base_one_scene')\n"
        "converted = json.loads((root / 'converted_action.json').read_text())\n"
        "demo = json.loads((root / 'sample.json').read_text())\n"
        "command = converted['libero_controller_action_candidate']\n"
        "print('Arm command (6):', [round(x, 4) for x in command[:6]])\n"
        "print('Gripper command (1):', command[6], '(demo first step:', demo['demonstration_action_raw'][6], ')')\n"
        "print('Shape:', converted['shape'], 'Finite:', converted['all_values_finite'])\n"
        "print('Validation used for scale:', converted['validation_used_for_statistics'])\n"
        "print('Executed in simulator:', converted['controller_executed'])\n"
    ),
]

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
