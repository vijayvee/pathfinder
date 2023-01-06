"""ConvRNN, Nayebi et al 2018."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from torch.nn import init


class ConvRNNcell(nn.Module):
    """
    Implements ConvRNN from Nayebi et al 2018
    params:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
    """

    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 fsize=7,
                 device='cuda',
                 nl='relu',
                 ):
        super(ConvRNNcell, self).__init__()
        self.in_channels = in_channels
        self.fsize = fsize
        if hidden_dim is None:
            self.hidden_dim = in_channels
        else:
            self.hidden_dim = hidden_dim
        if nl == 'relu':
            self.act_fn = torch.relu
        elif nl == 'tanh':
            self.act_fn = torch.tanh
        elif nl == 'clipped_relu':
            self.act_fn = nn.ReLU6()
        else:
            raise ValueError('Invalid non-linearity')

        # recurrent gates computation
        self.w_ch = nn.Conv2d(self.hidden_dim, self.hidden_dim, self.fsize, padding=self.fsize//2)
        self.w_cc = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w_hh = nn.Conv2d(self.hidden_dim, self.hidden_dim, self.fsize, padding=self.fsize//2)
        self.w_hc = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)


    def forward(self, input, hidden):
        h, c = hidden
        # recurrent gates computation
        ch_gate = torch.sigmoid(self.w_ch(c))
        hh_gate = torch.sigmoid(self.w_hh(h))
        cc_gate = torch.sigmoid(self.w_cc(c))
        hc_gate = torch.sigmoid(self.w_hc(h))

        c = self.act_fn((1 - hc_gate) * input + (1 - cc_gate) * c)
        h = self.act_fn((1 - ch_gate) * input + (1 - hh_gate) * h)
        return (h, c)


class ConvRNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 fsize=7,
                 timesteps=4,
                 device='cuda',
                 nl='relu',
                 ):
        super(ConvRNNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.fsize = fsize
        self.timesteps = timesteps
        print("%s steps of recurrence" % self.timesteps)
        self.device = device
        self.rnn_cell = ConvRNNcell(in_channels=self.in_channels,
                                    hidden_dim=self.hidden_dim,
                                    fsize=self.fsize,
                                    device=self.device,
                                    nl=nl,
                                    )


    def forward(self, input):
        n_x, _, w_x, h_x = input.shape
        state = (torch.zeros(n_x, self.hidden_dim, w_x, h_x),
                torch.zeros(n_x, self.hidden_dim, w_x, h_x))
        if torch.cuda.is_available():
            state = (state[0].cuda(), state[1].cuda())
        for _ in range(self.timesteps):
            state = self.rnn_cell(input, state)
        return state[0]
