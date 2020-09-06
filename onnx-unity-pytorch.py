import os
import shutil

from pathlib import Path

import numpy as np

import torch
import torch.nn

from Scantensus.utils.json import get_keypoint_names_and_colors_from_json

from ScantensusPT.utils import load_and_fix_state_dict

from Scantensus.nets.HRNet_CFG_I_Sigmoid import get_net_cfg
from ScantensusPT.nets.HRNetV2M7 import get_seg_model


HOST = 'thready1'
PROJECT = "unity"
EXPERIMENT = 'unity-68-thready1-gpu2'
EPOCH = '100'

BATCH_SIZE = 16

DEVICE = "cpu"

###############
if HOST == "thready1":
    DATA_DIR = Path("/") / "mnt" / "Storage" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "mnt" / "Storage" / "matt-output"
elif HOST == "matt-laptop":
    DATA_DIR = Path("/") / "Users" / "matthew" / "Box" / "scantensus-data"
    OUTPUT_DIR = Path("/") / "Volumes" / "Matt-Data" / "matt-output"
else:
    raise Exception
################

########
JSON_KEYS_PATH = DATA_DIR / "labels" / "unity" / "keys.json"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / PROJECT / EXPERIMENT
CHECKPOINT_PATH = CHECKPOINT_DIR / f'weights-{EPOCH}.pt'

ONNX_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / "onnx"
ONNX_PATH = ONNX_DIR / f"{PROJECT}-{EXPERIMENT}-{EPOCH}.onnx"
ONNX_KEYS_PATH = ONNX_DIR / f"{PROJECT}-{EXPERIMENT}-{EPOCH}-keys.json"
os.makedirs(ONNX_DIR, exist_ok=True)
#########


class MyEnsemble(torch.nn.Module):
    def __init__(self, model_a):
        super().__init__()
        self.model_a = model_a
        self.a = torch.FloatTensor([2.2])

    def forward(self, inputs):
        out = self.model_a(inputs)
        return out[1] * self.a


if __name__ == '__main__':

    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(JSON_KEYS_PATH)

    net_cfg = get_net_cfg()

    net_cfg['DATASET'] = {}
    net_cfg['DATASET']['NUM_CLASSES'] = len(keypoint_names)
    net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 9
    net_cfg['MODEL']['PRETRAINED'] = False
    net_cfg['ONNX_OUTPUT'] = True

    single_model = get_seg_model(cfg=net_cfg)

    single_model.init_weights()
    state_dict = load_and_fix_state_dict(CHECKPOINT_PATH)
    single_model.load_state_dict(state_dict)
    single_model.eval()

    #final_model = MyEnsemble(model_a=single_model)
    #final_model.eval()

    x = torch.randn(1, 3, 608, 608, requires_grad=True)
    input_names = ["images"]
    output_names = ["scores"]
    dynamic_axes = {'images': {0: 'batch'}, 'scores': {0: 'batch'}}

    print(f"{ONNX_PATH} Starting Export")
    torch.onnx.export(single_model,
                      x,
                      ONNX_PATH,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11,
                      do_constant_folding=True,
                      export_params=True,
                      dynamic_axes=dynamic_axes)
    print(f"{ONNX_PATH} Finishing Export")

    shutil.copy(JSON_KEYS_PATH, ONNX_KEYS_PATH)
