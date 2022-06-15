### Code adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/

import sys
sys.path.append('/local/crv/danich/thesis/src/backbones/pointnet2')

from regex import X
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, embedding_depth=1024, input_feats=8):
        super(get_model, self).__init__()
        in_channel = input_feats

        self.embedding_depth = embedding_depth
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=16, in_channel=in_channel, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, self.embedding_depth], group_all=True)

    def forward(self, xyz):
        B, S1, S2 = xyz.shape

        if S1 > S2:
            xyz = xyz.permute(0, 2, 1)

        norm = xyz[:, 3:, :]
        xyz = xyz[:, :3, :]

        
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        _, l4_points = self.sa4(l3_xyz, l3_points)

        x = l4_points.view(B, self.embedding_depth)

        return x
