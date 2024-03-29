"""BaseModel class that builds other models"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import resnet as models_resnet
from models.resnet_pathfinder import *
from models.dale_rnn import DaleRNNLayer
from models.hgru import hConvGRU
from models.convgru import ConvGRU
from models.ext_rnn import ExtRNNLayer
from models.convrnn import ConvRNNLayer
from models.cornets import CORblock_S
from models.utils import get_gabor_conv

class BaseModel(nn.Module):
    def __init__(self, cfg, args):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.in_res = cfg.in_res
        self.name = cfg.name
        self.backbone = cfg.backbone[0]
        self.nlayers = cfg.nlayers
        self.num_classes = cfg.num_classes
        self.timesteps = args.timesteps if args.timesteps else self.backbone['timesteps']
        self.num_ori = 32  # max(32, self.backbone['out_channels'])
        if self.name.startswith("dalernn"):
            interneuron = True
            if "abl_interneuron" in self.name:
                interneuron = False
            self.rnn = DaleRNNLayer(  # self.backbone['out_channels'],
                self.num_ori,
                self.backbone['out_channels'],
                # timesteps=self.backbone['timesteps'],
                timesteps=self.timesteps,
                exc_fsize=self.backbone['fsize'],
                init_=self.backbone['init_'],
                return_all_states=False,
                use_ln=self.backbone['use_ln']=='True',
                use_gates=self.backbone['use_gates']=='True',
                nl=self.backbone['nl'],
                interneuron=interneuron,
            )
            # if "abl_interneuron" in self.name:
            #     self.num_units = self.backbone['out_channels'] * 2
            # else:
            self.num_units = self.backbone['out_channels']

        elif self.name.startswith("hgru"):
            self.rnn = hConvGRU(
                filt_size=self.backbone['fsize'],
                hidden_dim=self.backbone['out_channels'],
                timesteps=self.backbone['timesteps'])
            self.num_units = self.backbone['out_channels']

        elif self.name.startswith("gru"):
            self.rnn = ConvGRU(  # self.backbone['out_channels'],
                self.num_ori,
                self.backbone['out_channels'],
                self.backbone['fsize'],
                self.timesteps,
                self.backbone['nl'],)

            self.num_units = self.backbone['out_channels']

        elif self.name.startswith("extrnn"):
            self.rnn = ExtRNNLayer(  # self.backbone['out_channels'],
                self.num_ori,
                self.backbone['out_channels'],
                timesteps=self.timesteps,
                exc_fsize=self.backbone['fsize'],
                init_=self.backbone['init_']
            )
            self.num_units = self.backbone['out_channels']
        
        elif self.name.startswith("convrnn"):
            self.rnn = ConvRNNLayer(  # self.backbone['out_channels'],
                self.num_ori,
                self.backbone['out_channels'],
                fsize=self.backbone['fsize'],
                timesteps=self.timesteps,
                nl=self.backbone['nl'],
            )
            self.num_units = self.backbone['out_channels']
        elif self.name.startswith("cornets"):
            self.rnn = CORblock_S(self.num_ori, 
                                  self.backbone['out_channels'],
                                  fsize=self.backbone['fsize'],
                                  times=self.timesteps,
                                  )
            self.num_units = self.backbone['out_channels']
        else:
            self.rnn = None

        if self.name.startswith('ff'):
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
                                              padding=self.backbone['fsize'] // 2 - 1,
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

        # elif self.name.startswith('dalernn') or self.name == 'hgru' or self.name.startswith('gru') or self.name == 'extrnn':
        elif self.rnn:
            self.backbone = nn.Sequential(
                get_gabor_conv(3, self.num_ori,
                               f_size=11, stride=2),
                nn.BatchNorm2d(self.num_ori),
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
        elif self.name.startswith('resnet_thin_pathfinder'):
            if self.nlayers == 10:
                self.backbone = resnet10_thin_pathfinder(inplanes=self.backbone['out_channels'],
                                                         pretrained=False,
                                                         num_classes=self.num_classes)
            elif self.nlayers == 18:
                self.backbone = resnet18_thin_pathfinder(inplanes=self.backbone['out_channels'],
                                                         pretrained=False,
                                                         num_classes=self.num_classes)
            else:
                raise NotImplementedError("Only 10 and 18 layers are implemented for %s" % self.name)
            self.num_units = self.backbone.out_dim
        elif self.name.startswith('resnet_thin'):
            self.backbone = models_resnet.__dict__['resnet%s_thin' % self.nlayers](inplanes=self.backbone['out_channels'],
                                                                                   pretrained=False,
                                                                                   num_classes=self.num_classes)
            self.num_units = self.backbone.out_dim
        elif self.name.startswith('resnet'):
            self.backbone = models_resnet.__dict__['resnet%s' % self.nlayers](inplanes=self.backbone['out_channels'],
                                                                              pretrained=False,
                                                                              num_classes=self.num_classes)
            self.num_units = self.backbone.out_dim
        elif self.name == 'r_resnet':
            pass
        elif self.name == 'transformer':
            from vit_pytorch import ViT
            self.backbone = ViT(image_size=160,
                                patch_size=8,
                                num_classes=2,
                                dim=512,
                                depth=6,
                                heads=8,
                                mlp_dim=512,
                                dropout=0.1,
                                emb_dropout=0.1
                                )
        else:
            raise NotImplementedError(self.backbone)
        self.final_conv = nn.Conv2d(self.num_units, 2, 1)
        self.final_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.readout = nn.Linear(2, self.num_classes)

    def forward(self, x):
        if self.name.startswith('resnet'):
            x = self.backbone.get_intermediate_layers(
                x, n=1)  # retrieve post-avg-pool output
        # elif self.name.startswith('dalernn'):
        #     # all_outputs = self.backbone(x)
        #     # x = all_outputs[-1]
        else:
            all_outputs = None
            x = self.backbone(x)
        self.rnn_out = x
        x = self.final_conv(x)
        x = self.final_pool(x)
        x = torch.flatten(x, 1)
        x = self.readout(x)
        if all_outputs:
            return x, all_outputs
        return x

    def get_intermediate_layers(self, x, n):
        """Get the n'th layer"""
        if self.name.startswith('resnet') and self.name.count('pathfinder') == 0:
            x = self.backbone.get_intermediate_layers(
                x, n=1)  # retrieve post-avg-pool output
        else:
            x = self.backbone(x)
        if n == 0:
            return x
        x = self.final_conv(x)
        if n == 1:
            return x
        x = self.final_pool(x)
        if n == 2:
            return x
        x = torch.flatten(x, 1)
        x = self.readout(x)
        if n == 3:
            return x
        return x
