"""Exact image operations from the local official OpenVLA LIBERO evaluator.
Run in the isolated TensorFlow CPU / simulator environment, not lerobot312.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
import torch

REFERENCE_ROOT = Path("/home/jovyan/projects/openvla")
LIBERO_UTILS = REFERENCE_ROOT / "experiments/robot/libero/libero_utils.py"
OPENVLA_UTILS = REFERENCE_ROOT / "experiments/robot/openvla_utils.py"
REFERENCE_NAMESPACE = {"np": np, "tf": tf, "Image": Image, "torch": torch, "DEVICE": "cpu"}


def load_reference(path, names):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError(f"Missing reference functions in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), REFERENCE_NAMESPACE)


load_reference(LIBERO_UTILS, ["resize_image", "get_libero_image"])
load_reference(OPENVLA_UTILS, ["crop_and_resize", "get_vla_action"])


def prepare_image(raw_agentview_image):
    """Rotate, JPEG roundtrip, Lanczos3 resize, then 90%-area center crop."""
    resized = REFERENCE_NAMESPACE["get_libero_image"]({"agentview_image": raw_agentview_image}, 224)
    image = Image.fromarray(resized).convert("RGB")
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = REFERENCE_NAMESPACE["crop_and_resize"](image, 0.9, 1)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return np.asarray(Image.fromarray(image.numpy()).convert("RGB"))


class CapturedInputs(dict):
    def to(self, *args, **kwargs):
        return self


class CaptureProcessor:
    def __call__(self, prompt, image):
        return CapturedInputs(image=np.asarray(image))


class CaptureModel:
    def predict_action(self, image, **kwargs):
        return image


def check_against_reference(raw):
    """Compare pixels with the complete official get_vla_action preprocessing."""
    resized = REFERENCE_NAMESPACE["get_libero_image"]({"agentview_image": raw}, 224)
    expected = REFERENCE_NAMESPACE["get_vla_action"](
        CaptureModel(), CaptureProcessor(), "openvla/openvla-7b-finetuned-libero-spatial",
        {"full_image": resized}, "test instruction", "libero_spatial", center_crop=True,
    )
    actual = prepare_image(raw)
    diff = int(np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max())
    if diff != 0 or actual.shape != (224, 224, 3):
        raise RuntimeError(f"Preprocessing differs from the reference: max pixel error {diff}")
    return {"shape": list(actual.shape), "max_pixel_error": diff,
            "processed_sha256": hashlib.sha256(actual.tobytes()).hexdigest()}


def self_test():
    rng = np.random.default_rng(7)
    images = [np.zeros((256,256,3),dtype=np.uint8),
              rng.integers(0,256,size=(256,256,3),dtype=np.uint8)]
    return [check_against_reference(image) for image in images]


def reference_hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [LIBERO_UTILS, OPENVLA_UTILS]}


if __name__ == "__main__":
    import json
    print(json.dumps({"tests": self_test(), "reference_sha256": reference_hashes()}, indent=2))
