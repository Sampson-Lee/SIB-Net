import torch.nn as nn
import torch.nn.functional as F

class EntropyMinimizationLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(EntropyMinimizationLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x):
        entropy = -1.0 * F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

        if self.reduction == 'mean':
            return entropy.mean()
        if self.reduction == 'sum':
            return entropy.sum()
        return entropy