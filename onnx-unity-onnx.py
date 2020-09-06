###NOtes
# Cannot have names of inputs and ouputsts that are numbers (the degault of pytorch)
# cannot have more than one output
##https://github.com/onnx/onnx/issues/654

import numpy as np
from pathlib import Path
import json

import onnxruntime as rt

HOST = 'thready1'

PROJECT = "unity"
EXPERIMENT = 'unity-68-thready1-gpu2'
EPOCH = '100'


if HOST == "thready1":
    OUTPUT_DIR = Path("/") / "mnt" / "Storage" / "matt-output"
    ONNX_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / "onnx"
elif HOST == "matt-laptop":
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Data" / "matt-output"
    ONNX_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / "onnx"
else:
    raise Exception


def get_keypoint_names_and_colors_from_json(json_path):
    with open(json_path, "r") as read_file:
        data = json.load(read_file)

    keypoint_names = list(data.keys())
    keypoint_cols = []

    for keypoint in keypoint_names:
        r, g, b = data[keypoint]['rgb'].split()
        keypoint_cols.append([float(r), float(g), float(b)])

    return keypoint_names, keypoint_cols


def main():

    ONNX_PATH = ONNX_DIR / f"{PROJECT}-{EXPERIMENT}-{EPOCH}.onnx"
    ONNX_KEYS_PATH = ONNX_DIR / f"{PROJECT}-{EXPERIMENT}-{EPOCH}-keys.json"

    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(ONNX_KEYS_PATH)

    print(keypoint_names)
    print(keypoint_cols)
    print(f"number of keypoints: {len(list(keypoint_names))}")
    print(f"Please do not refer to layers as [:, layer, :, :] as may change")
    print(f"But as [:, keypoint_names.index('curve-lv-endo'), :, :]")
    print(f"So that we can add new points etc")

    x = np.random.randint(0, 255+1, (2, 9, 608, 608)).astype(np.uint8)

    x = (x / 255.0) - 0.5

    x = x.astype(np.float32)

    sess = rt.InferenceSession(str(ONNX_PATH))  # wont take Path objects just str

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    temp = sess.run([output_name], {input_name: x})

    temp = temp[0]

    print(temp.shape)

if __name__ == "__main__":
    main()


