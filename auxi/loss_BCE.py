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

class BCE_disc(nn.Module):
    def __init__(self, weight_list=None):
        super(BCE_disc, self).__init__()
        self.weight_list = weight_list

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        div_fac = labels.sum(dim=1, keepdim=True)
        assert (div_fac>=1).all()
        # embed()
        labels = (1-self.lb_sm)/div_fac*labels + self.lb_sm/(labels.size(1)-div_fac)*(1-labels)
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v2(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v2, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'
        
        labels = (1-self.lb_sm) * labels + self.lb_sm * (1-labels)
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v3(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v3, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        labels = (1-self.lb_sm) * labels
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v4(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v4, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        labels = labels + self.lb_sm * (1-labels)
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v5(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v5, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        # assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'
 
        labels = labels + self.lb_sm * (1-labels)
        labels = labels/labels.sum(dim=1, keepdim=True)
        # embed()
        loss = -F.log_softmax(x, dim=1)*labels
        return loss.mean(dim=0)

class BCE_disc_sm_v6(nn.Module):
    def __init__(self, weight_list=None, lb_sm1=0.5, lb_sm0=0.1):
        super(BCE_disc_sm_v6, self).__init__()
        self.weight_list = weight_list
        self.lb_sm1 = lb_sm1
        self.lb_sm0 = lb_sm0

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        labels = self.lb_sm1 * labels + self.lb_sm0 * (1-labels)
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v7(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v7, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        labels = labels/3
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v8(nn.Module):
    def __init__(self, lb_sm=0.2):
        super(BCE_disc_sm_v8, self).__init__()
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        labels = torch.ones_like(labels).cuda() * self.lb_sm
        loss = F.binary_cross_entropy(x, labels, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v3_uniform(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v3_uniform, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        mul = torch.Tensor(np.random.uniform(0, 2*self.lb_sm, labels.size())).cuda()
        labels = (1-mul) * labels
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v3_beta(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v3_beta, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        mul = torch.Tensor(np.random.beta(self.lb_sm, (1-self.lb_sm), labels.size())).cuda()
        labels = (1-mul) * labels
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_v3_normal(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_v3_normal, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        mul = torch.Tensor(np.random.normal(self.lb_sm, 0.1, labels.size())).cuda()
        labels = (1-mul) * labels
        loss = F.binary_cross_entropy(x, labels, weight=self.weight_list, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_freq(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_freq, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = torch.Tensor(weight_list).cuda()*lb_sm + lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        labels = labels + self.lb_sm * (1-labels)
        loss = F.binary_cross_entropy(x, labels, reduction='none')
        return loss.mean(dim=0)

class BCE_disc_sm_focol(nn.Module):
    def __init__(self, weight_list=None, lb_sm=0.2):
        super(BCE_disc_sm_focal, self).__init__()
        self.weight_list = weight_list
        self.lb_sm = lb_sm

    def forward(self, x, labels):
        assert (x>=0).all() and (x<=1).all(), 'x is wrong'
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        x = x*labels + (1-x)*(1-labels)
        preds_log = torch.log(x)
        loss = -((1-x) * preds_log)

        return loss.mean(dim=0)
