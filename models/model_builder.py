"""BaseModel class that builds other models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import resnet as models_resnet
from models.dale_rnn import DaleRNNLayer
from models.utils import get_gabor_conv
from models.hgru import hConvGRU


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.in_res = cfg.in_res
        self.name = cfg.name
        self.backbone = cfg.backbone[0]
        self.nlayers = cfg.nlayers
        self.num_classes = cfg.num_classes

        if self.name.startswith("dalernn"):
            self.rnn = DaleRNNLayer(self.backbone['out_channels'],
                                    self.backbone['out_channels'],
                                    timesteps=self.backbone['timesteps'],
                                    exc_fsize=self.backbone['fsize'],
                                    )
        elif self.name.startswith("hgru"):
            self.rnn = hConvGRU(filt_size=11, hidden_dim=self.backbone['out_channels'])

        self.num_units = self.backbone['out_channels']

        if self.name == 'ff':
            backbone = []
            in_channels = 3
            dilation = self.backbone['dilation']
            for l_i in range(self.nlayers):
                if l_i == 0:
                    backbone.append(get_gabor_conv(3, self.backbone['out_channels'],
                                                    f_size=11, stride=2))
                else:
                    backbone.append(nn.Conv2d(in_channels,
                                            self.backbone['out_channels'],
                                            self.backbone['fsize'],
                                            padding=1,
                                            stride=self.backbone['stride'],
                                            dilation=dilation
                                            ))
                backbone.append(nn.BatchNorm2d(self.backbone['out_channels']))
                backbone.append(nn.ReLU())
                if l_i == 0:
                    backbone.append(nn.MaxPool2d(kernel_size=3, 
                                                stride=2, padding=1))
                in_channels = self.backbone['out_channels']
            self.backbone = nn.Sequential(*backbone)

        elif self.name == 'dalernn' or self.name =='hgru':
            self.backbone = nn.Sequential(
                get_gabor_conv(3, self.backbone['out_channels'],
                               f_size=11, stride=2),
                nn.BatchNorm2d(self.backbone['out_channels']),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=3, stride=2, padding=1),
                self.rnn,
            )
        elif self.name == 'bn_ff':
            pass
        elif self.name == 'gn_ff':
            pass
        elif self.name == 'ln_ff':
            pass
        elif self.name == 'resnet':
            self.backbone = models_resnet.__dict__['resnet%s' % self.nlayers](pretrained=False,
                                                                              num_classes=self.num_classes)
            self.num_units = backbone.out_dim
        elif self.name == 'r_resnet':
            pass
        self.final_conv = nn.Conv2d(self.num_units, 2, 1)
        self.final_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.readout = nn.Linear(2, self.num_classes)

    def forward(self, x):
        if self.name.startswith('resnet'):
            x = self.backbone.get_intermediate_layers(
                x, n=1)  # retrieve post-avg-pool output
        else:
            x = self.backbone(x)
        self.rnn_out = x
        x = self.final_conv(x)
        x = self.final_pool(x)
        x = torch.flatten(x, 1)
        x = self.readout(x)
        return x
