import torch
import torch.nn as nn

from net.resnet import resnet34, resnet18

class net(nn.Module):
    def __init__(self, config):
        super(net, self).__init__()

        self.net_head = resnet18(pretrained=True, num_classes=26)
        self.net_body = resnet18(pretrained=True, num_classes=26)
        self.net_scene = resnet18(pretrained=True, num_classes=26)
        
        self.cls = nn.Sequential(
            nn.Linear(512 * 3, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 26),
        )
        
        for m in self.cls.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data, mode):
        out_head, fea_head = self.net_head(data['image_head'])
        out_body, fea_body = self.net_body(data['image_body'])
        out_scene, fea_scene = self.net_scene(data['image_scene'])

        fea_cat = torch.cat([fea_head, fea_body, fea_scene], dim=1)
        output  = self.cls(fea_cat)
        return [output, out_head, out_body, out_scene]
