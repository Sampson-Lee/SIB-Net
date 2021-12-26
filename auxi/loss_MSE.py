import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math

from torch.autograd import Variable
from torch.autograd.function import Function
# from IPython import embed

class MSE_disc(nn.Module):
    def __init__(self, weight_list=None):
        super(MSE_disc, self).__init__()
        self.weight_list = weight_list

    def forward(self, x, labels):
        loss = (x-labels)**2
        if self.weight_list is not None:
            loss = loss * self.weight_list
        return loss.mean(dim=0)

class MSE_cont(nn.Module):
    def __init__(self, theta=1/10):
        super(MSE_cont, self).__init__()
        self.theta = theta

    def forward(self, x, labels):
        loss = (x-labels)**2

        mask = loss.gt(self.theta).float()
        loss = loss*mask
        return loss.mean(dim=0)