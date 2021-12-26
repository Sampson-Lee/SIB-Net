import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from torch.autograd import Variable
from torch.autograd.function import Function
from IPython import embed
import pdb

class ABS_disc(nn.Module):
    def __init__(self, weight_list=None):
        super(ABS_disc, self).__init__()
        self.weight_list = weight_list

    def forward(self, x, labels):
        loss = torch.abs(x-labels)
        if self.weight_list is not None:
            loss = loss * self.weight_list
        return loss.mean(dim=0)

class ABS_cont(nn.Module):
    def __init__(self, theta=1/10):
        super(ABS_cont, self).__init__()
        self.theta = theta

    def forward(self, x, labels):
        loss = torch.abs(x-labels)

        mask = loss.gt(self.theta).float()
        loss = loss*mask
        return loss.mean(dim=0)

class ABS_disc_sm_v3(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(ABS_disc_sm_v3, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        labels = (1-self.lb_sm) * labels
        loss = torch.abs(x-labels)
        return loss.mean(dim=0)

class ABS_disc_sm_v3_uniform(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(ABS_disc_sm_v3_uniform, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        mul = torch.Tensor(np.random.uniform(0, 2*self.lb_sm, labels.size())).cuda()
        labels = (1-mul) * labels
        loss = torch.abs(x-labels)
        return loss.mean(dim=0)

class ABS_disc_sm_v3_beta(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(ABS_disc_sm_v3_beta, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        mul = torch.Tensor(np.random.beta(self.lb_sm, (1-self.lb_sm), labels.size())).cuda()
        labels = (1-mul) * labels
        loss = torch.abs(x-labels)
        return loss.mean(dim=0)

class ABS_disc_sm_v3_normal(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(ABS_disc_sm_v3_normal, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        mul = torch.Tensor(np.random.normal(self.lb_sm, 0.1, labels.size())).cuda()
        labels = (1-mul) * labels
        loss = torch.abs(x-labels)
        return loss.mean(dim=0)
