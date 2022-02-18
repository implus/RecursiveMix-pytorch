import torch
import torch.nn as nn
from utils import *


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        bs = float(pred.size(0))
        pred = pred.log_softmax(dim=1)
        if len(target.shape) == 2:
            true_dist = target
        else:
            true_dist = smooth_one_hot(target, self.num_classes, self.smoothing)
        loss = (-pred * true_dist).sum() / bs
        return loss


if __name__ == '__main__':
    criterion = LabelSmoothingLoss(5)
    pred = torch.randn(2, 5)
    target = torch.tensor([3, 1])
    loss = criterion(pred, target)
    print(loss)
