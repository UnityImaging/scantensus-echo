import torch
import torch.nn

class MSEClampLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_weights: torch.Tensor):

        out = ((y_pred - y_true) ** 2) * torch.clamp(y_weights, 0, 1)

        return out

class MSESumLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_weights: torch.Tensor):

        # The weights are 1 to 255, with 0 used for those that are missing. So this turns it into a check.
        y_weights_mask = torch.clamp(y_weights, 0, 1)

        out = (y_pred - y_true) ** 2.
        out = out * torch.max(y_weights, dim=1, keepdim=True)[0] * y_weights_mask

        return out




###############
###############

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class MattFocalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neg_loss = _neg_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_weights: torch.Tensor):

        pos_inds = y_true.eq(1).float()
        neg_inds = y_true.lt(1).float()

        y_weights_mask = torch.clamp(y_weights, 0, 1)

        neg_weights = torch.pow(1 - y_true, 4)

        lossA = 0
        lossB = 0

        pos_loss = torch.log(y_pred) * torch.pow(1 - y_pred, 2) * pos_inds * y_weights_mask
        neg_loss = torch.log(1 - y_pred) * torch.pow(y_pred, 2) * neg_weights * neg_inds * y_weights_mask

        loss = -(pos_loss + neg_loss)
        return torch.mean(loss)


