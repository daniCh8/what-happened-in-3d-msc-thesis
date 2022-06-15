from typing import OrderedDict
from omegaconf import OmegaConf
from torch_points3d.applications.kpconv import KPConv

from backbones.pointnet2.pointnet2 import get_model as pointnet2_model
from backbones.minkowski.minkowski import MinkowskiFCNN

import torch.nn as nn
from torchvision import models


def get_kpconv_encoder_conf(o_size):
    return OmegaConf.load(f'./backbones/kpconv/kpconv_encoder_{o_size}.yaml')


def get_backbone_list():
    return ['kpconv', 'pointnet2', 'minkowski']


def get_assertion_error_backbone_list():
    return f'network type must be one of {get_backbone_list()}'


def get_backbone(net_type, input_feats, o_size):
    assert net_type in get_backbone_list(), get_assertion_error_backbone_list()

    if net_type == 'kpconv':
        kp_params = get_kpconv_encoder_conf(o_size=o_size)
        encoder = KPConv('encoder', input_nc=input_feats-3, num_layers=4, config=kp_params)
    elif net_type == 'pointnet2':
        encoder = pointnet2_model(input_feats=input_feats)
    elif net_type == 'minkowski':
        encoder = MinkowskiFCNN(in_channel=input_feats-3)

    return encoder


class BaselineFeatureExtractor(nn.Module):
    def __init__(self, backbone, siamese):
        super(BaselineFeatureExtractor, self).__init__()

        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.in_c = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            self.in_c = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.in_c = 2048
        elif backbone == 'resnext':
            self.backbone = models.resnext50_32x4d(pretrained=True)
            self.in_c = 2048
        
        self.mid_output_size = max(256, 1024//siamese)
        self.backbone.fc = nn.Sequential(
            nn.Linear(self.in_c, self.mid_output_size),
            nn.LeakyReLU()
        )
    
    def forward(self, x):
        return self.backbone(x)


class FeatureExtractor(nn.Module):
    def __init__(self, net_type, input_feats, process, o_size):
        super(FeatureExtractor, self).__init__()

        self.encoder = net_type
        self.backbone = get_backbone(net_type, input_feats, o_size)
        self.sizes = [1, 32, 128, 256, 512]
        self.kernels = [3, 3, 3, 5]
        self.output_size = self.sizes[-1]
        
        self.process = process
        if self.process:
            self.post_process = nn.Sequential(
                OrderedDict(
                    [
                        (
                            f'conv_block_{i}', 
                            self.get_conv_block(
                                in_c=self.sizes[i],
                                out_c=self.sizes[i+1],
                                k=self.kernels[i]
                            )
                        ) for i in range(len(self.kernels))
                    ]
                )
            )
    
    def get_conv_block(self, in_c, out_c, k):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=k,
                padding='same'
            ),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.encoder == 'kpconv':
            x = x.x
        
        if not self.process:
            return x
        
        x = x.unsqueeze(1)
        x = self.post_process(x)
        return x


def get_feature_extractor(net_type, input_feats, post_process=False, siamese=1, o_size=1024):
    if net_type in ['resnet18', 'resnet34', 'resnet50', 'resnext']:
        return BaselineFeatureExtractor(net_type, siamese)
    
    return FeatureExtractor(net_type, input_feats, post_process, o_size)
