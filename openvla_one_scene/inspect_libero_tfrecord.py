"""Inspect the first TFDS RLDS episode without installing TensorFlow."""

from pathlib import Path
import struct
from io import BytesIO

from PIL import Image


def varint(data, pos):
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
        shift += 7


def fields(data):
    pos = 0
    while pos < len(data):
        tag, pos = varint(data, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(data, pos)
        elif wire == 1:
            value, pos = data[pos : pos + 8], pos + 8
        elif wire == 2:
            length, pos = varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire == 5:
            value, pos = data[pos : pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, value


def records(path):
    with path.open("rb") as f:
        while raw_length := f.read(8):
            length = struct.unpack("<Q", raw_length)[0]
            f.read(4)  # TFRecord length CRC
            data = f.read(length)
            f.read(4)  # TFRecord data CRC
            yield data


def example_features(data):
    outer = dict(fields(data))
    for _, entry in fields(outer[1]):
        parts = dict(fields(entry))
        yield parts[1].decode("utf-8"), parts[2]


def feature_summary(feature):
    kind, content = next(iter(fields(feature)))
    values = [value for _, value in fields(content)]
    return kind, values


if __name__ == "__main__":
    downloads = Path.home() / "Downloads"
    paths = sorted(downloads.glob("libero_spatial-train.tfrecord-*-of-00016"))
    selected = None
    for path in paths:
        for episode_index, data in enumerate(records(path)):
            features = dict(example_features(data))
            instructions = feature_summary(features["steps/language_instruction"])[1]
            instruction = instructions[0].decode("utf-8")
            print(path.name, "episode", episode_index, "steps", len(instructions), "instruction", instruction)
            if "black bowl between the plate and the ramekin" in instruction.lower():
                selected = (path, episode_index, features)
                break
        if selected:
            break
    if selected is None:
        raise SystemExit("The target task is not in the locally downloaded shards.")

    path, episode_index, features = selected
    print("\nSELECTED:", path.name, "episode", episode_index)
    instructions = feature_summary(features["steps/language_instruction"])[1]
    images = feature_summary(features["steps/observation/image"])[1]
    action_bytes = feature_summary(features["steps/action"])[1][0]
    action_values = struct.unpack("<" + "f" * (len(action_bytes) // 4), action_bytes)
    output = Path(__file__).parent / "libero_sample"
    output.mkdir(exist_ok=True)
    for step in range(min(2, len(images))):
        action = action_values[7 * step : 7 * (step + 1)]
        image_path = output / f"scene_{step + 1}.png"
        Image.open(BytesIO(images[step])).save(image_path)
        print("STEP:", step, "instruction:", instructions[step].decode("utf-8"))
        print("ACTION:", [round(x, 5) for x in action])
        print("IMAGE:", image_path)
