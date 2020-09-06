from collections import OrderedDict

import torch


def load_and_fix_state_dict(checkpoint_path, device="cpu"):
    if device == "cpu":
        print(f'Loading weights onto CPU')
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        print(f'Loading weights onto GPU')
        state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v

    return new_state_dict