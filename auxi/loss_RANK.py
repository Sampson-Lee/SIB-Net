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

class rank_loss(nn.Module):
    def __init__(self, alpha=0.1):
        super(rank_loss, self).__init__()
        self.alpha = alpha

    def forward(self, out_list, labels):
        assert (labels>=0).all() and (labels<=1).all(), 'labels is wrong'

        true_score_whole = F.sigmoid(out_list[0][labels.to(torch.bool)])
        true_score_head = F.sigmoid(out_list[1][labels.to(torch.bool)])
        true_score_body = F.sigmoid(out_list[2][labels.to(torch.bool)])
        true_score_scene = F.sigmoid(out_list[3][labels.to(torch.bool)])

        true_rank_head, _ = torch.max(self.alpha - (true_score_whole - true_score_head), 0)
        true_rank_body, _ = torch.max(self.alpha - (true_score_whole - true_score_body), 0)
        true_rank_scene, _ = torch.max(self.alpha - (true_score_whole - true_score_scene), 0)

        false_score_whole = F.sigmoid(out_list[0][(1-labels).to(torch.bool)])
        false_score_head = F.sigmoid(out_list[1][(1-labels).to(torch.bool)])
        false_score_body = F.sigmoid(out_list[2][(1-labels).to(torch.bool)])
        false_score_scene = F.sigmoid(out_list[3][(1-labels).to(torch.bool)])

        false_rank_head, _ = torch.max(self.alpha - (false_score_head - false_score_whole), 0)
        false_rank_body, _ = torch.max(self.alpha - (false_score_body - false_score_whole), 0)
        false_rank_scene, _ = torch.max(self.alpha - (false_score_scene - false_score_whole), 0)

        loss = true_rank_head + true_rank_body + true_rank_scene \
                + false_rank_head + false_rank_body + false_rank_scene
        return loss.mean()
