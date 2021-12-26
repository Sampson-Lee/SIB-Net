import torch.nn as nn
import torch
import pdb
import torch.nn.functional as F
from torch.nn import init
import math
from IPython import embed
import numpy as np

class FFM(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()

        x = self.conv_x(x).view(x.size(0), x.size(1), -1)
        x = x.permute(0,2,1) # hw x c

        y = self.conv_y(y).view(y.size(0), y.size(1), -1)
        y = y.permute(0,2,1) # hw x c

        z = self.conv_z(z).view(z.size(0), z.size(1), -1) # c x hw
        yz = torch.matmul(y, z) # hw x hw
        yz = yz/yz.size(-1)

        xyz = torch.matmul(yz, x) # hw x c
        xyz = xyz.permute(0,2,1).contiguous() # c x hw 
        xyz = xyz.view(b, self.inter_channel, h, w)

        xyz = self.bn(self.conv_xyz(xyz))

        return xyz

class FFM_v2(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v2, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()
        
        x = self.conv_x(x).view(x.size(0), x.size(1), -1) # c x hw
        
        y = self.conv_y(y).view(y.size(0), y.size(1), -1)
        y = y.permute(0,2,1) # hw x c
        
        z = self.conv_z(z).view(z.size(0), z.size(1), -1) # c x hw
        yz = torch.matmul(z, y) # c x c
        yz = yz/yz.size(-1)
        
        xyz = torch.matmul(yz, x)
        xyz = xyz.view(b, self.inter_channel, h, w)
        
        xyz = self.bn(self.conv_xyz(xyz))
        
        return xyz

class FFM_v3(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v3, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x1 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_yz = conv_nd(in_channel*2, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()
        
        x1 = self.conv_x1(x).view(x.size(0), x.size(1), -1) # c x hw
        
        x2 = self.conv_x2(x).view(x.size(0), x.size(1), -1) 
        x2 = x2.permute(0,2,1) # hw x c
        
        yz = torch.cat((y, z),dim=1)
        yz = self.conv_yz(yz).view(x.size(0), x.size(1), -1) # c x hw
        
        xyz = torch.matmul(yz, x2) # c x c
        xyz = xyz/xyz.size(-1)
        
        xyz = torch.matmul(xyz, x1) # c x hw
        xyz = xyz.view(b, self.inter_channel, h, w)
        
        xyz = self.bn(self.conv_xyz(xyz))
        
        return xyz

class FFM_v4(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v4, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x1 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x3 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()

        xyz = torch.cat((x, y, z),dim=1)

        # pdb.set_trace()
        xyz1 = self.conv_x1(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz2 = self.conv_x2(xyz).view(b, self.inter_channel, -1)
        xyz2 = xyz2.permute(0,2,1) # hw x c

        xyz3 = self.conv_x3(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz23 = torch.matmul(xyz3, xyz2) # c x c
        xyz23 = xyz23/math.sqrt(xyz23.size(-1))
        xyz23 = F.softmax(xyz23, dim=-1)

        xyz = torch.matmul(xyz23, xyz1) # c x hw
        xyz = xyz.view(b, self.inter_channel, h, w)

        xyz = self.bn(self.conv_xyz(xyz))

        return xyz

class FFM_v5(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v5, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_1 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_2 = conv_nd(inter_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_3 = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()

        xyz = torch.cat((x, y, z),dim=1)

        # pdb.set_trace()
        xyz = self.conv_1(xyz)
        xyz = self.conv_2(xyz)
        xyz = self.conv_3(xyz)

        xyz = self.bn(xyz)

        return xyz

class FFM_v6(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v6, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x1 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x3 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()

        xyz = torch.cat((x, y, z), dim=1)

        # pdb.set_trace()
        xyz1 = self.conv_x1(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz2 = self.conv_x2(xyz).view(b, self.inter_channel, -1)
        xyz2 = xyz2.permute(0,2,1) # hw x c

        xyz3 = self.conv_x3(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz23 = torch.matmul(xyz3, xyz2) # c x c
        xyz23 = xyz23/xyz23.size(-1)

        xyz = torch.matmul(xyz23, xyz1) # c x hw
        xyz = xyz.view(b, self.inter_channel, h, w)

        xyz = self.bn(self.conv_xyz(xyz))

        return xyz

class FFM_v7(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v7, self).__init__()
        self.ffm1 = FFM_v6(dimension, in_channel, inter_channel)
        self.ffm2 = FFM_v6(dimension, in_channel, inter_channel)
        self.ffm3 = FFM_v6(dimension, in_channel, inter_channel)
        
    def forward(self, x, y, z):
        xyz1 = self.ffm1(x,y,z)
        xyz2 = self.ffm2(x,y,z)
        xyz3 = self.ffm3(x,y,z)
        
        return xyz1*xyz2*xyz3

class FFM_v8(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v8, self).__init__()
        self.ffm1 = FFM_v6(dimension, in_channel, inter_channel)
        self.ffm2 = FFM_v6(dimension, in_channel, inter_channel)
        self.ffm3 = FFM_v6(dimension, in_channel, inter_channel)

    def forward(self, x, y, z):
        xyz1 = self.ffm1(x,y,z)
        xyz2 = self.ffm2(x,y,z)
        xyz3 = self.ffm3(x,y,z)

        return xyz1+xyz2+xyz3

class FFM_v9(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v9, self).__init__()
        self.ffm1 = FFM_v6(dimension, in_channel, inter_channel)
        self.ffm2 = FFM_v6(dimension, in_channel, inter_channel)
        
    def forward(self, x, y, z):
        xyz1 = self.ffm1(x,y,z)
        xyz2 = self.ffm2(x,y,z)
        
        return xyz1*xyz2

class FFM_v10(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v10, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x1 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x3 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()

        xyz = torch.cat((x, y, z),dim=1)

        xyz1 = self.conv_x1(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz2 = self.conv_x2(xyz).view(b, self.inter_channel, -1)
        xyz2 = xyz2.permute(0,2,1) # hw x c

        xyz3 = self.conv_x3(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz23 = torch.matmul(xyz2, xyz3) # hw x hw
        xyz23 = F.softmax(xyz23, dim=1)
        
        xyz = torch.matmul(xyz1, xyz23) # c x hw
        xyz = xyz.view(b, self.inter_channel, h, w)

        xyz = self.bn(self.conv_xyz(xyz))

        return xyz

class FFM_v11(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v11, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_xyz = conv_nd(in_channel*3, in_channel, kernel_size=1, stride=1, padding=0)
        self.ffm1 = FFM_v4(dimension, in_channel, inter_channel)
        self.ffm2 = FFM_v4(dimension, in_channel, inter_channel)
        self.ffm3 = FFM_v4(dimension, in_channel, inter_channel)

    def forward(self, x, y, z):
        xyz1 = self.ffm1(x,y,z)
        xyz2 = self.ffm2(x,y,z)
        xyz3 = self.ffm3(x,y,z)
        xyz = self.conv_xyz(torch.cat((xyz1,xyz2,xyz3), dim=1))
        return xyz1

class FFM_v13(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v13, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x1 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x3 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel*3, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel*3)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()

        xyz = torch.cat((x, y, z),dim=1)

        # pdb.set_trace()
        xyz1 = self.conv_x1(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz2 = self.conv_x2(xyz).view(b, self.inter_channel, -1)
        xyz2 = xyz2.permute(0,2,1) # hw x c

        xyz3 = self.conv_x3(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz23 = torch.matmul(xyz3, xyz2) # c x c
        xyz23 = xyz23/math.sqrt(xyz23.size(-1))
        xyz23 = F.softmax(xyz23, dim=-1)

        xyz = torch.matmul(xyz23, xyz1) # c x hw
        xyz = xyz.view(b, self.inter_channel, h, w)

        xyz = self.bn(self.conv_xyz(xyz))

        return xyz[:,:c,:,:], xyz[:,c:2*c,:,:], xyz[:,2*c:,:,:]

class IMM(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=512):
        super(IMM, self).__init__()
        self.dimension = dimension
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.mlp = nn.Sequential(
            conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0),
            bn(inter_channel),
            nn.ReLU(),
            conv_nd(inter_channel, inter_channel, kernel_size = 1, stride=1, padding=0),
            bn(inter_channel),
            nn.ReLU(),
            conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0),
            bn(in_channel),
            nn.ReLU())

    def forward(self, x, y, z):
        
        x = self.mlp(x)
        yz = self.mlp(y+z)
        xyz = self.mlp(x+y+z)
        
        return F.sigmoid(torch.mean(xyz-yz-x))

class IMM_v2(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=512):
        super(IMM_v2, self).__init__()
        self.dimension = dimension
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.mlp = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)

    def forward(self, x, y, z):
        
        x = self.mlp(x)
        y = self.mlp(y)
        z = self.mlp(z)
        
        xyz = self.mlp(x+y+z)
        
        return F.sigmoid(xyz-y-z-x)

class FFM_v12(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=128):
        super(FFM_v12, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x1 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x3 = conv_nd(in_channel*3, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel, in_channel, kernel_size = 1, stride=1, padding=0)
        self.bn = bn(in_channel)

        init.constant_(self.conv_xyz.weight, 0)
        init.constant_(self.conv_xyz.bias, 0)

    def forward(self, x, y, z):
        x_ori = x
        b,c,h,w = x.size()

        xyz = torch.cat((x, y, z),dim=1)

        # pdb.set_trace()
        xyz1 = self.conv_x1(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz2 = self.conv_x2(xyz).view(b, self.inter_channel, -1)
        xyz2 = xyz2.permute(0,2,1) # hw x c

        xyz3 = self.conv_x3(xyz).view(b, self.inter_channel, -1) # c x hw

        xyz23 = torch.matmul(xyz3, xyz2) # c x c
        xyz23 = xyz23/math.sqrt(xyz23.size(-1))
        xyz23 = F.softmax(xyz23, dim=-1)

        xyz = torch.matmul(xyz23, xyz1) # c x hw
        xyz = xyz.view(b, self.inter_channel, h, w)

        xyz = self.bn(self.conv_xyz(xyz))

        return xyz

class LSTMModule(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1):
        super(LSTMModule, self).__init__()

        # self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=False, bias=False)

    def forward(self, embedding):
        ## embedding: N*3*D
        # embedding = self.dropout(embedding)
        feature = self.calculate_different_length_lstm(embedding)

        return feature

    def calculate_different_length_lstm(self, embedding):

        text_length = torch.from_numpy(np.array(3)).repeat(embedding.size(0))
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(embedding,
                                                                    text_length,
                                                                    batch_first=True)

        self.lstm.flatten_parameters()
        packed_feature, [hn, _] = self.lstm(packed_text_embedding)  # [hn, cn]
        fea_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        feature = fea_unpacked
        # unsort_feature = feature[0]
        # unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
        #     + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2

        # feature, _ = unsort_feature.max(dim=1)
        
        return feature

class RNNModule(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1):
        super(RNNModule, self).__init__()

        # self.dropout = nn.Dropout(0.3)
        self.lstm = nn.RNN(input_size, hidden_size, num_layers=num_layers, bidirectional=False, bias=False)

    def forward(self, embedding):
        ## embedding: N*3*D
        # embedding = self.dropout(embedding)
        feature = self.calculate_different_length_lstm(embedding)

        return feature

    def calculate_different_length_lstm(self, embedding):

        text_length = torch.from_numpy(np.array(3)).repeat(embedding.size(0))
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(embedding,
                                                                    text_length,
                                                                    batch_first=True)

        self.lstm.flatten_parameters()
        packed_feature, hn = self.lstm(packed_text_embedding)  # [hn, cn]
        fea_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        feature = fea_unpacked
        # unsort_feature = feature[0]
        # unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
        #     + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2

        # feature, _ = unsort_feature.max(dim=1)
        
        return feature

class GRUModule(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1):
        super(GRUModule, self).__init__()

        # self.dropout = nn.Dropout(0.3)
        self.lstm = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=False, bias=False)

    def forward(self, embedding):
        ## embedding: N*3*D
        # embedding = self.dropout(embedding)
        feature = self.calculate_different_length_lstm(embedding)

        return feature

    def calculate_different_length_lstm(self, embedding):

        text_length = torch.from_numpy(np.array(3)).repeat(embedding.size(0))
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(embedding, text_length, batch_first=True)
        # embed()
        self.lstm.flatten_parameters()
        packed_feature, hn = self.lstm(packed_text_embedding)  # [hn, cn]
        fea_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        feature = fea_unpacked
        # unsort_feature = feature[0]
        # unsort_feature = (unsort_feature[:, :, :int(unsort_feature.size(2) / 2)]
        #     + unsort_feature[:, :, int(unsort_feature.size(2) / 2):]) / 2

        # feature, _ = unsort_feature.max(dim=1)
        
        return feature

class LSTMModule_Bi(nn.Module):
    def __init__(self, input_size=512, hidden_size=512, num_layers=1):
        super(LSTMModule_Bi, self).__init__()

        # self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True, bias=False)

    def forward(self, embedding):
        ## embedding: N*3*D
        # embedding = self.dropout(embedding)
        feature = self.calculate_different_length_lstm(embedding)

        return feature

    def calculate_different_length_lstm(self, embedding):

        text_length = torch.from_numpy(np.array(3)).repeat(embedding.size(0))
        packed_text_embedding = nn.utils.rnn.pack_padded_sequence(embedding,
                                                                    text_length,
                                                                    batch_first=True)

        self.lstm.flatten_parameters()
        packed_feature, [hn, _] = self.lstm(packed_text_embedding)  # [hn, cn]
        fea_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(packed_feature, batch_first=True)  # including[feature, length]
        feature = fea_unpacked
        feature = (feature[:, :, :int(feature.size(2) / 2)]
            + feature[:, :, int(feature.size(2) / 2):]) / 2

        # feature, _ = unsort_feature.max(dim=1)
        
        return feature

# to do list
# class MGU(nn.Module):
#     def __init__(self, input_size, hidden_size=1, num_layers=1):
#         """Initialize params."""
#         super(MGU, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers

#         self.gate_weights = nn.Linear(input_size+hidden_size, hidden_size)
#         self.hidden_weights = nn.Linear(hidden_size, 2 * hidden_size)

#     def forward(self, input, hidden):
#         """Propogate input through the network."""
        
#         def recurrence(input, hidden):
#             """Recurrence helper."""
#             hx = hidden  # n_b x hidden_dim
#             forgetgate =  F.sigmoid(self.gate_weights(torch.cat((input, hx), dim=1)))
#             hd = F.tanh(self.hidden_weights(torch.cat((input, forgetgate*hx), dim=1)))  # o_t
#             hy = (1-forgetgate) * hd + forgetgate * hx
#             return hy

#         input = input.transpose(0, 1)
#         if hidden==None: hidden = 
#         output = []
#         steps = range(input.size(0))
#         for i in steps:
#             hidden = recurrence(input[i], hidden)
#             output.append(hidden)

#         output = torch.cat(output, 0).view(input.size(0), *output[0].size())
#         output = output.transpose(0, 1)
#         return output, hidden

class SFM(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=12, out_channel=26):
        super(SFM, self).__init__()
        self.dimension = dimension
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        self.conv_x = conv_nd(in_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y = conv_nd(in_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z = conv_nd(in_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(in_channel*3, out_channel, kernel_size = 1, stride=1, padding=0)

    def forward(self, x, y, z):
        x = x.view(x.size(0), x.size(1), 1)
        y = y.view(y.size(0), y.size(1), 1)
        z = z.view(z.size(0), z.size(1), 1)
        xyz = torch.cat((x,y,z), dim=1)

        x = self.conv_x(x)
        y = self.conv_y(y)
        z = self.conv_z(z)
        xyz = self.conv_xyz(xyz)

        wei = F.softmax(torch.cat((x,y,z,xyz), dim=2), dim=2)
        return wei[:,:,0], wei[:,:,1], wei[:,:,2], wei[:,:,3]

class SFM_v2(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=12, out_channel=26):
        super(SFM_v2, self).__init__()
        self.dimension = dimension
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        self.conv_x = conv_nd(in_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y = conv_nd(in_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z = conv_nd(in_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(in_channel*3, out_channel, kernel_size = 1, stride=1, padding=0)
        self.lstm = LSTMModule(input_size=in_channel, hidden_size=in_channel, num_layers=1)
        
    def forward(self, x, y, z):
        
        feature = self.lstm(torch.stack((x, y, z), dim=1))
        x, y, z = feature[:,0,:], feature[:,1,:], feature[:,2,:]
        xyz = torch.cat((x, y, z), dim=1)
        
        x = x.view(x.size(0), x.size(1), 1)
        y = y.view(y.size(0), y.size(1), 1)
        z = z.view(z.size(0), z.size(1), 1)
        xyz = xyz.view(xyz.size(0), xyz.size(1), 1)

        x = self.conv_x(x)
        y = self.conv_y(y)
        z = self.conv_z(z)
        xyz = self.conv_xyz(xyz)

        wei = F.softmax(torch.cat((x,y,z,xyz), dim=2), dim=2)
        return wei[:,:,0], wei[:,:,1], wei[:,:,2], wei[:,:,3]

class SFM_v3(nn.Module):
    def __init__(self, dimension=None, in_channel=26, inter_channel=26, out_channel=26):
        super(SFM_v3, self).__init__()
        self.dimension = dimension
        self.in_channel = in_channel
        self.inter_channel = inter_channel
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d
        self.conv_x1 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x3 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x4 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)

    def forward(self, x1, x2, x3, x4):
        b,c = x1.size()

        x1 = x1.view(b, c, 1)
        x2 = x2.view(b, c, 1)
        x3 = x3.view(b, c, 1)
        x4 = x4.view(b, c, 1)

        # pdb.set_trace()
        x1 = self.conv_x1(x1) # c x hw
        x2 = self.conv_x2(x2)
        x3 = self.conv_x3(x3)
        x4 = self.conv_x4(x4)

        wei = F.softmax(torch.cat(
                                (torch.mean(x1*x2, 1, keepdim=True),
                                torch.mean(x1*x3, 1, keepdim=True),
                                torch.mean(x1*x4, 1, keepdim=True),
                                torch.mean(x1*x1, 1, keepdim=True)),
                                dim=2), dim=2)
        return wei[:,:,0], wei[:,:,1], wei[:,:,2], wei[:,:,3]

class SFM_v4(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=12, out_channel=26):
        super(SFM, self).__init__()
        self.dimension = dimension
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        self.conv_xyz1 = conv_nd(in_channel*3, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz2 = conv_nd(in_channel*3, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz3 = conv_nd(in_channel*3, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz4 = conv_nd(in_channel*3, out_channel, kernel_size = 1, stride=1, padding=0)

    def forward(self, x, y, z):
        x = x.view(x.size(0), x.size(1), 1)
        y = y.view(y.size(0), y.size(1), 1)
        z = z.view(z.size(0), z.size(1), 1)
        xyz = torch.cat((x,y,z), dim=1)

        x = self.conv_xyz1(xyz)
        y = self.conv_xyz2(xyz)
        z = self.conv_xyz3(xyz)
        xyz = self.conv_xyz4(xyz)

        wei = F.softmax(torch.cat((x,y,z,xyz), dim=2), dim=2)
        return wei[:,:,0], wei[:,:,1], wei[:,:,2], wei[:,:,3]

class SFM_v5(nn.Module):
    def __init__(self, dimension=None, in_channel=512, inter_channel=12, out_channel=26):
        super(SFM_v5, self).__init__()
        self.dimension = dimension
        if dimension==3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
        if dimension==2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
        if dimension==1:
            conv_nd = nn.Conv1d
            bn = nn.BatchNorm1d

        self.conv_x1 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x2 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_x3 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y1 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y2 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y3 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z1 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z2 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z3 = conv_nd(in_channel, inter_channel, kernel_size = 1, stride=1, padding=0)
        
        self.conv_x = conv_nd(inter_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_y = conv_nd(inter_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_z = conv_nd(inter_channel, out_channel, kernel_size = 1, stride=1, padding=0)
        self.conv_xyz = conv_nd(inter_channel*3, out_channel, kernel_size = 1, stride=1, padding=0)
        self.lstm = LSTMModule(input_size=inter_channel, hidden_size=inter_channel, num_layers=1)
        
    def forward(self, x, y, z):
        x = x.view(x.size(0), x.size(1), 1)
        y = y.view(y.size(0), y.size(1), 1)
        z = z.view(z.size(0), z.size(1), 1)
        
        x_ = self.conv_x1(x.clone())
        x_y = self.conv_x2(y.clone())
        x_z = self.conv_x3(z.clone())
        
        x += torch.matmul(torch.matmul(x_, x_y.permute(0,2,1).contiguous()), x_)/x_y.size(1) \
            + torch.matmul(torch.matmul(x_, x_z.permute(0,2,1).contiguous()), x_)/x_z.size(1)
        
        y_ = self.conv_y1(y.clone())
        y_z = self.conv_y2(z.clone())
        y_x = self.conv_y3(x.clone())

        y += torch.matmul(torch.matmul(y_, y_z.permute(0,2,1).contiguous()), y_)/y_z.size(1) \
            + torch.matmul(torch.matmul(y_, y_x.permute(0,2,1).contiguous()), y_)/y_x.size(1)

        z_ = self.conv_z1(z.clone())
        z_y = self.conv_z2(y.clone())
        z_x = self.conv_z3(x.clone())

        z += torch.matmul(torch.matmul(z_, z_y.permute(0,2,1).contiguous()), z_)/z_y.size(1) \
            + torch.matmul(torch.matmul(z_, z_x.permute(0,2,1).contiguous()), z_)/z_x.size(1)

        feature = self.lstm(torch.cat((x, y, z), dim=-1).permute(0,2,1).contiguous())
        x, y, z = feature[:,0,:], feature[:,1,:], feature[:,2,:]
        xyz = torch.cat((x, y, z), dim=1)

        x = x.view(x.size(0), x.size(1), 1)
        y = y.view(y.size(0), y.size(1), 1)
        z = z.view(z.size(0), z.size(1), 1)
        xyz = xyz.view(xyz.size(0), xyz.size(1), 1)

        x = self.conv_x(x)
        y = self.conv_y(y)
        z = self.conv_z(z)
        xyz = self.conv_xyz(xyz)

        wei = F.softmax(torch.cat((x,y,z,xyz), dim=2), dim=2)
        return wei[:,:,0], wei[:,:,1], wei[:,:,2], wei[:,:,3]
