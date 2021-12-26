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

        conv_nd = nn.Conv2d

        self.att_H = nn.Sequential(
            conv_nd(512, 128, kernel_size = 1, stride=1, padding=0),
            conv_nd(128, 512, kernel_size = 1, stride=1, padding=0)
        )

        self.att_B = nn.Sequential(
            conv_nd(512, 128, kernel_size = 1, stride=1, padding=0),
            conv_nd(128, 512, kernel_size = 1, stride=1, padding=0)
        )

        self.att_S = nn.Sequential(
            conv_nd(512, 128, kernel_size = 1, stride=1, padding=0),
            conv_nd(128, 512, kernel_size = 1, stride=1, padding=0)
        )

        self.mlp = nn.Sequential(
            nn.Linear(512*3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, class_num)
        )

    def forward(self, data, mode):
        out_head, fea_head_ = self.net_head(data['image_head'])
        out_body, fea_body_ = self.net_body(data['image_body'])
        out_scene, fea_scene_ = self.net_scene(data['image_scene'])

        fea_head = F.adaptive_avg_pool2d(fea_head_, (1,1))
        fea_body = F.adaptive_avg_pool2d(fea_body_, (1,1))
        fea_scene = F.adaptive_avg_pool2d(fea_scene_, (1,1))
        
        att_H = self.att_H(fea_head)
        att_B = self.att_B(fea_body)
        att_S = self.att_S(fea_scene)
        att = F.softmax(torch.cat((att_H, att_B, att_S), dim=2), dim=2)

        # pdb.set_trace()
        fea_head = att[:,:,0,:].unsqueeze(-1)*fea_head
        fea_body = att[:,:,1,:].unsqueeze(-1)*fea_body
        fea_scene = att[:,:,2,:].unsqueeze(-1)*fea_scene

        fea_cat = torch.cat((fea_head, fea_body, fea_scene), dim=1)
        fea_cat = fea_cat.view(fea_cat.size(0), -1)

        out = self.mlp(fea_cat)

        return [out,], \
                {'ly4': {'head':fea_head_, 'body':fea_body_, 'scene':fea_scene_}}
