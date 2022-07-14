"""BaseModel class that builds other models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.in_res = cfg.in_res
        self.name = cfg.name
        self.backbone = cfg.backbone[0]
        self.nlayers = cfg.nlayers
        self.num_classes = cfg.num_classes
        self.features = self.get_backbone()
        self.readout = self.get_readout()

    def get_backbone(self):
        backbone = []
        in_channels = 3
        if self.name == 'ff':
            for l_i in range(self.nlayers):
                backbone.append(nn.Conv2d(in_channels, 
                                          self.backbone['out_channels'], 
                                          self.backbone['fsize'],
                                          stride=self.backbone['stride'],
                                          ))
                backbone.append(nn.BatchNorm2d(self.backbone['out_channels']))
                backbone.append(nn.ReLU())
                in_channels = self.backbone['out_channels']
            backbone = nn.Sequential(*backbone)
            self.num_units = self.in_res * self.in_res * self.backbone['out_channels']
        elif self.name == 'r_ff':
            pass
        elif self.name == 'bn_ff':
            pass
        elif self.name == 'gn_ff':
            pass
        elif self.name == 'ln_ff':
            pass
        elif self.name == 'resnet':
            backbone = models_resnet.__dict__['resnet%s' % self.nlayers](pretrained=False, 
                                                                        num_classes=self.num_classes)
            self.backbone['num_channels'] = backbone.num_channels
            self.num_units = 10 * 10 * self.backbone['out_channels']
        elif self.name == 'r_resnet':
            pass
        return backbone
    
    def get_readout(self):
        readout = nn.Linear(self.num_units, self.num_classes)
        return readout

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x