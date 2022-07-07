"""Schwartz and Simoncelli 2001 + excitation and inhibition, in pytorch."""
from numpy.core.numeric import True_
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error


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


class ExcInhDivNorm(nn.Module):
    """
    Implements Schwartz and Simoncelli 2001 style divisive normalization w/ lateral E/I connections.
    params:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
    Example:
      x = torch.zeros(1, 1, 100, 100)
      net = ExcInhDivNorm(1, 16, 15)
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
                 ):
        super(ExcInhDivNorm, self).__init__()
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
        nonnegative_weights_init(self.div)
        
    def forward(self, x, residual=False, square_act=True, hor_conn=True):
        """
        params:
          x: Input grayscale image tensor
        Returns:
          output: Output post divisive normalization
        """
        identity = x
        # Gabor filter bank]
        simple_cells = nn.Identity()(x)
        if hor_conn:
            inhibition = self.i_e(simple_cells)  # + self.i_ff(x)
            # Excitatory lateral connections (Center corresponds to self-excitation)
            excitation = self.e_e(simple_cells)
            simple_cells = simple_cells + excitation - inhibition
        if square_act:
            simple_cells = F.relu(simple_cells)
            norm = self.div(simple_cells) + self.sigma ** 2 + 1e-5
            simple_cells = simple_cells / norm
        else:
            norm = 1 + F.relu(self.div(simple_cells))
            simple_cells = simple_cells / norm
        output = self.output_bn(simple_cells)
        if residual:
            output += identity
        output = self.output_relu(output)
        return output