import torch
import torch.nn
import torch.nn.functional


def get_point(logits: torch.Tensor, period: float=None, temporal_smooth=False):

    if logits.dim() == 2:
        batch_size = 1
        channels = 1
        height = logits.shape[0]
        width = logits.shape[1]
    elif logits.dim() == 4:
        batch_size = logits.shape[0]
        channels = logits.shape[1]
        height = logits.shape[2]
        width = logits.shape[3]
    else:
        raise Exception

    logits = logits.view(batch_size, channels, height * width)
    
    if temporal_smooth:

        if batch_size > 1:
            new_logits = torch.empty_like(logits)
            new_logits[0, ...] = logits[0, ...] * 0.75 + logits[1, ...] * 0.25
            new_logits[-1, ...] = logits[-1, ...] * 0.75 + logits[-2, ...] * 0.25

            if batch_size > 2:
                new_logits[1:-1, ...] = logits[1:-1, ...] * 0.5 + logits[0:-2, ...] * 0.25 + logits[2:, ...] * 0.25
            logits = new_logits

    val, index = torch.topk(logits, k=2, dim=2)

    y = (index[..., 0] // height).float()
    x = (index[..., 0] % height).float()

    y2 = (index[..., 1] // height).float()
    x2 = (index[..., 1] % height).float()

    lens = torch.sqrt(torch.square(y2-y) + torch.square(x2-x))

    y = y + 0.25 * (y2-y)/lens
    x = x + 0.25 * (x2-x)/lens

    out = torch.stack((y, x, val[..., 0]), dim=2)

    return out

