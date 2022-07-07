"""Schwartz and Simoncelli 2001 + excitation and inhibition, in pytorch."""
from numpy.core.numeric import True_
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
import numpy as np
import math

# define normalized 2D gaussian
def get_gaussian_filterbank(n_filters, f_sz, device='cuda'):
    """Generate a torch tensor for conv2D weights resembling 2D gaussian"""
    def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

    def normalize(x):
        mid_x, mid_y = x.shape[0] // 2, x.shape[1] // 2
        x = x / x[mid_x][mid_y]
        return x
    filters = []
    x = np.linspace(-1, 1, num=f_sz)
    y = np.linspace(-1, 1, num=f_sz)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    z_narrow = normalize(gaus2d(x, y, sx=.5, sy=.5))
    z_low = normalize(gaus2d(x, y, sx=1., sy=1.))
    z_mid = normalize(gaus2d(x, y, sx=2, sy=2))
    z_wide = normalize(gaus2d(x, y, sx=3, sy=3))
    filters = [z_narrow] * (n_filters // 4) + [z_low] * (n_filters // 4) + [z_mid] * (n_filters // 4) + [z_wide] * (n_filters // 4)
    filters = np.array(filters)
    filters = np.random.permutation(filters)
    filters = filters.reshape((n_filters, 1, f_sz, f_sz))
    filters = torch.Tensor(filters).float().to(device)
    return filters

def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)
    else:
        m.data.fill_(0.1)
        
def orthogonal_weights_init(m):
    """Orthogonal initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        m.weight.data.clamp_(0)
        m.bias.data.fill_(0.)
    else:
        m.data.fill_(0.)

def gaussian_weights_init(m):
    """Initialize weights using 2d Gaussian."""
    if isinstance(m, nn.Conv2d):
        n_filters, _, f_sz, _ = m.weight.shape
        weights = get_gaussian_filterbank(n_filters, f_sz)
        m.weight.data = weights


class DivNormExcInh(nn.Module):
    """
    Implements Schwartz and Simoncelli 2001 style divisive normalization.
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
                 exc_fsize=7,
                 inh_fsize=5,
                 padding_mode='zeros',
                 groups=1,
                 device='cuda',
                 gaussian_init=False,
                 ):
        super(DivNormExcInh, self).__init__()
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
        if gaussian_init:
            gaussian_weights_init(self.div)
        self.e_e = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, 
            exc_fsize, bias=True, padding=(exc_fsize - 1) // 2,
            )
        self.i_e = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, inh_fsize, 
            padding=(inh_fsize - 1) // 2,
            bias=True)
        self.sigma = nn.Parameter(torch.ones([1, self.hidden_dim, 1, 1]))
        self.output_bn = nn.BatchNorm2d(in_channels)
        self.output_relu = nn.ReLU(inplace=True)
        
    def forward(self, x, residual=False, square_act=True, hor_conn=True):
        """
        params:
          x: Input grayscale image tensor
        Returns:
          output: Output post divisive normalization
        """
        identity = x
        # Gabor filter bank]
        if self.in_channels <= 3:
            simple_cells = F.relu(self.gfb(x))
            print("| Using Gabor Filter Bank |")
        else:
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

        if hor_conn:
            inhibition = self.i_e(simple_cells)  # + self.i_ff(x)
            # Excitatory lateral connections (Center corresponds to self-excitation)
            excitation = self.e_e(simple_cells)
            #output = simple_cells + excitation - inhibition
            output = self.output_bn(simple_cells + excitation - inhibition)
        else:
            output = self.output_bn(simple_cells)
        
        if residual:
            output += identity

        output = self.output_relu(output)
        return output