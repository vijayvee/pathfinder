"""E/I lesion of DivNormEI."""
import math

import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from numpy.core.numeric import True_


class DivNorm(nn.Module):
    """
    Implements Schwartz and Simoncelli 2001 style divisive normalization.
    Lesion of E/I connections
    params:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
    Example:
      x = torch.zeros(1, 1, 100, 100)
      net = DivNormExcInh(1, 16, 15)
      out = net(x)
    """

    def __init__(self,
                 in_channels,
                 divnorm_fsize=5,
                 exc_lesion=True,
                 inh_lesion=True,
                 exc_fsize=7,
                 inh_fsize=5,
                 padding_mode='zeros',
                 groups=1,
                 ):
        super(DivNorm, self).__init__()
        self.in_channels = in_channels    
        self.hidden_dim = in_channels

        self.div = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            divnorm_fsize,
            padding=(divnorm_fsize - 1) // 2,
            padding_mode=padding_mode,
            groups=groups,
            bias=False)
        if exc_lesion:
            self.e_e = None
        else:
            self.e_e = nn.Conv2d(
                self.hidden_dim, self.hidden_dim, 
                exc_fsize, bias=True, padding=(exc_fsize - 1) // 2,
                )
        if inh_lesion:
            self.i_e = None
        else:
            self.i_e = nn.Conv2d(
                self.hidden_dim, self.hidden_dim, inh_fsize, 
                padding=(inh_fsize - 1) // 2,
                bias=True)
        self.sigma = nn.Parameter(torch.ones([1, self.hidden_dim, 1, 1]))
        self.output_bn = nn.BatchNorm2d(in_channels)
        self.output_relu = nn.ReLU(inplace=True)
    
    def forward(self, x, residual=True, square_act=True):
        """
        params:
            x: Input activation tensor
        returns:
            output: Output post normalization
        """
        return self.forward_divnorm(x=x, residual=residual, square_act=square_act)
        
    def forward_divnorm(self, x, residual=True, square_act=True):
        """
        params:
          x: Input grayscale image tensor
        Returns:
          output: Output post divisive normalization
        """
        identity = x
        simple_cells = nn.Identity()(x)

        if square_act:
            simple_cells = simple_cells ** 2
            norm = self.div(simple_cells) + self.sigma ** 2 + 1e-5
            if (norm == 0).any():
                import ipdb; ipdb.set_trace()
            simple_cells = simple_cells / norm
        else:
            norm = 1 + F.relu(self.div(simple_cells))
            #norm = F.relu(self.div(simple_cells)) + self.sigma ** 2 + 1e-5
            simple_cells = simple_cells / norm
        output = simple_cells
        if self.i_e is not None:
            inhibition = self.i_e(simple_cells)
            output = output - inhibition
        if self.e_e is not None:                
            # Excitatory lateral connections (Center corresponds to self-excitation)
            excitation = self.e_e(simple_cells)
            output = output + excitation
        output = self.output_bn(output)
        if residual:
            output += identity
        output = self.output_relu(output)
        return output
