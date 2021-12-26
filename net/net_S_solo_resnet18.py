import torch
import torch.nn as nn

from net.resnet import resnet34, resnet18, resnet50

class net(nn.Module):
    def __init__(self, config):
        super(net, self).__init__()

        self.net = resnet18(pretrained=True, num_classes=26)

    def forward(self, data, mode):
        image = data['image_scene']
        output, _ = self.net(image)

        return output
