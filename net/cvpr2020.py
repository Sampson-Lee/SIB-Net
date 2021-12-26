import torch
import torch.nn as nn
import torch.nn.functional as F
from net.resnet import resnet34, resnet18
import copy
from torch.nn import init
import pdb

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class net(nn.Module):
    def __init__(self, config):
        super(net, self).__init__()
        class_num=26

        self.net_head = resnet18(pretrained=True, num_classes=26)
        self.net_body = resnet18(pretrained=True, num_classes=26)
        self.net_scene = resnet18(pretrained=True, num_classes=26)

        self.mlp = nn.Sequential(
            nn.Linear(26*3, 52),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(52, class_num)
        )

    def forward(self, data, mode):
        out_head, fea_head = self.net_head(data['image_head'])
        out_body, fea_body = self.net_body(data['image_body'])
        out_scene, fea_scene = self.net_scene(data['image_scene'])

        fea_cat = torch.cat((out_head, out_body, out_scene), dim=1)

        out = self.mlp(fea_cat)

        return [out,], \
                {'ly4': {'head':fea_head, 'body':fea_body, 'scene':fea_scene}}
