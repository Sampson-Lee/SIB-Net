import torch
import torch.nn as nn
import torch.nn.functional as F

from net.resnet import resnet34, resnet18
import copy
from torch.nn import init
from IPython import embed
from auxi.module import FFM_v4, RNNModule

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

        self.net_head = resnet18(pretrained=True, num_classes=26)
        self.head_layer1 = nn.Sequential(*list(self.net_head.children())[:5])
        self.head_layer2 = list(self.net_head.children())[5]
        self.head_layer3 = list(self.net_head.children())[6]
        self.head_layer4 = list(self.net_head.children())[7]
        self.head_cls = list(self.net_head.children())[-1]

        self.net_body = resnet18(pretrained=True, num_classes=26)
        self.body_layer1 = nn.Sequential(*list(self.net_body.children())[:5])
        self.body_layer2 = list(self.net_body.children())[5]
        self.body_layer3 = list(self.net_body.children())[6]
        self.body_layer4 = list(self.net_body.children())[7]
        self.body_cls = list(self.net_body.children())[-1]

        self.net_scene = resnet18(pretrained=True, num_classes=26)
        self.scene_layer1 = nn.Sequential(*list(self.net_scene.children())[:5])
        self.scene_layer2 = list(self.net_scene.children())[5]
        self.scene_layer3 = list(self.net_scene.children())[6]
        self.scene_layer4 = list(self.net_scene.children())[7]
        self.scene_cls = list(self.net_scene.children())[-1]
        self.fc3 = nn.Sequential(nn.Linear(512*3 , 128), nn.Linear(128, 26))

        self.csr_ly1 = RNNModule(input_size=64, hidden_size=64, num_layers=1)
        self.csr_ly2 = RNNModule(input_size=128, hidden_size=128, num_layers=1)
        self.csr_ly3 = RNNModule(input_size=256, hidden_size=256, num_layers=1)
        self.csr_ly4 = RNNModule(input_size=512, hidden_size=512, num_layers=1)

        self.ffm_head_ly2 = FFM_v4(dimension=2, in_channel=128, inter_channel=128*2)
        self.ffm_body_ly2 = FFM_v4(dimension=2, in_channel=128, inter_channel=128*2)
        self.ffm_scene_ly2 = FFM_v4(dimension=2, in_channel=128, inter_channel=128*2)

        self.ffm_head_ly4 = FFM_v4(dimension=2, in_channel=512, inter_channel=512*2)
        self.ffm_body_ly4 = FFM_v4(dimension=2, in_channel=512, inter_channel=512*2)
        self.ffm_scene_ly4 = FFM_v4(dimension=2, in_channel=512, inter_channel=512*2)

    def forward(self, data, mode):
        head_ly1 = self.head_layer1(data['image_head'])
        body_ly1 = self.body_layer1(data['image_body'])
        scene_ly1 = self.scene_layer1(data['image_scene'])

        head_ly1_ = F.adaptive_avg_pool2d(head_ly1, (1,1)).view(head_ly1.size(0), -1)
        body_ly1_ = F.adaptive_avg_pool2d(body_ly1, (1,1)).view(body_ly1.size(0), -1)
        scene_ly1_ = F.adaptive_avg_pool2d(scene_ly1, (1,1)).view(scene_ly1.size(0), -1)

        feature = self.csr_ly1(torch.stack((head_ly1_, body_ly1_, scene_ly1_), dim=1))
        head_ly1 = head_ly1+feature[:,0,:].view(head_ly1.size(0),head_ly1.size(1),1,1).expand_as(head_ly1)
        body_ly1 = body_ly1+feature[:,1,:].view(body_ly1.size(0),body_ly1.size(1),1,1).expand_as(body_ly1)
        scene_ly1 = scene_ly1+feature[:,2,:].view(scene_ly1.size(0),scene_ly1.size(1),1,1).expand_as(scene_ly1)

        head_ly2 = self.head_layer2(head_ly1)
        body_ly2 = self.body_layer2(body_ly1)
        scene_ly2 = self.scene_layer2(scene_ly1)

        head_ly2_ = F.adaptive_avg_pool2d(head_ly2, (1,1)).view(head_ly2.size(0), -1)
        body_ly2_ = F.adaptive_avg_pool2d(body_ly2, (1,1)).view(body_ly2.size(0), -1)
        scene_ly2_ = F.adaptive_avg_pool2d(scene_ly2, (1,1)).view(scene_ly2.size(0), -1)

        feature = self.csr_ly2(torch.stack((head_ly2_, body_ly2_, scene_ly2_), dim=1))
        head_ly2 = head_ly2+feature[:,0,:].view(head_ly2.size(0),head_ly2.size(1),1,1).expand_as(head_ly2)
        body_ly2 = body_ly2+feature[:,1,:].view(body_ly2.size(0),body_ly2.size(1),1,1).expand_as(body_ly2)
        scene_ly2 = scene_ly2+feature[:,2,:].view(scene_ly2.size(0),scene_ly2.size(1),1,1).expand_as(scene_ly2)

        head_ly2 = self.ffm_head_ly2(head_ly2, body_ly2, scene_ly2) + head_ly2
        body_ly2 = self.ffm_body_ly2(body_ly2, head_ly2, scene_ly2) + body_ly2
        scene_ly2 = self.ffm_scene_ly2(scene_ly2, head_ly2, body_ly2) + scene_ly2

        head_ly3 = self.head_layer3(head_ly2)
        body_ly3 = self.body_layer3(body_ly2)
        scene_ly3 = self.scene_layer3(scene_ly2)

        head_ly3_ = F.adaptive_avg_pool2d(head_ly3, (1,1)).view(head_ly3.size(0), -1)
        body_ly3_ = F.adaptive_avg_pool2d(body_ly3, (1,1)).view(body_ly3.size(0), -1)
        scene_ly3_ = F.adaptive_avg_pool2d(scene_ly3, (1,1)).view(scene_ly3.size(0), -1)

        feature = self.csr_ly3(torch.stack((head_ly3_, body_ly3_, scene_ly3_), dim=1))
        head_ly3 = head_ly3+feature[:,0,:].view(head_ly3.size(0),head_ly3.size(1),1,1).expand_as(head_ly3)
        body_ly3 = body_ly3+feature[:,1,:].view(body_ly3.size(0),body_ly3.size(1),1,1).expand_as(body_ly3)
        scene_ly3 = scene_ly3+feature[:,2,:].view(scene_ly3.size(0),scene_ly3.size(1),1,1).expand_as(scene_ly3)

        head_ly4 = self.head_layer4(head_ly3)
        body_ly4 = self.body_layer4(body_ly3)
        scene_ly4 = self.scene_layer4(scene_ly3)

        head_ly4_ = F.adaptive_avg_pool2d(head_ly4, (1,1)).view(head_ly4.size(0), -1)
        body_ly4_ = F.adaptive_avg_pool2d(body_ly4, (1,1)).view(body_ly4.size(0), -1)
        scene_ly4_ = F.adaptive_avg_pool2d(scene_ly4, (1,1)).view(scene_ly4.size(0), -1)

        feature = self.csr_ly4(torch.stack((head_ly4_, body_ly4_, scene_ly4_), dim=1))
        head_ly4 = head_ly4+feature[:,0,:].view(head_ly4.size(0),head_ly4.size(1),1,1).expand_as(head_ly4)
        body_ly4 = body_ly4+feature[:,1,:].view(body_ly4.size(0),body_ly4.size(1),1,1).expand_as(body_ly4)
        scene_ly4 = scene_ly4+feature[:,2,:].view(scene_ly4.size(0),scene_ly4.size(1),1,1).expand_as(scene_ly4)

        head_ly4 = self.ffm_head_ly4(head_ly4, body_ly4, scene_ly4) + head_ly4
        body_ly4 = self.ffm_body_ly4(body_ly4, head_ly4, scene_ly4) + body_ly4
        scene_ly4 = self.ffm_scene_ly4(scene_ly4, head_ly4, body_ly4) + scene_ly4

        head_ly4 = F.adaptive_avg_pool2d(head_ly4, (1,1)).view(head_ly4.size(0), -1)
        body_ly4 = F.adaptive_avg_pool2d(body_ly4, (1,1)).view(body_ly4.size(0), -1)
        scene_ly4 = F.adaptive_avg_pool2d(scene_ly4, (1,1)).view(scene_ly4.size(0), -1)

        out_head = self.head_cls(head_ly4)
        out_body = self.body_cls(body_ly4)
        out_scene = self.scene_cls(scene_ly4)
        out = self.fc3(torch.cat((head_ly4, body_ly4, scene_ly4), dim=1))

        return [out, out_head, out_body, out_scene]
