"""Recurrent EI normalization without internal state."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from torch.nn import init


def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)


class ExtRNNcell(nn.Module):
    """
    Implements recurrent inhibitory excitatory normalization w/ lateral connections
    params:
      input_dim: Number of channels in input
      hidden_dim: Number of hidden channels
      kernel_size: Size of kernel in convolutions
    """

    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 exc_fsize=7,
                 init_=None,
                 ):
        super(ExtRNNcell, self).__init__()
        self.in_channels = in_channels
        if hidden_dim is None:
            self.hidden_dim = in_channels
        else:
            self.hidden_dim = hidden_dim
        # recurrent gates computation
        self.g_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        self.ln_e_x = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
        self.g_exc_e = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.ln_e_e = nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
        self.ln_out_e = nn.GroupNorm(
            num_groups=1, num_channels=self.hidden_dim)

        self.w_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)

        # horizontal connections (e->e, i->e, i->i, e->i)
        self.w_exc_ee = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2)
        # disynaptic inhibition with pairs of E-I cells, E -> exciting surround I -> inhibiting surround E

        if init_ == 'ortho':
            print("initializing RNN weights with Orthogonal Initialization")
            init.orthogonal_(self.g_exc_x.weight)
            init.orthogonal_(self.g_exc_e.weight)

            init.orthogonal_(self.w_exc_x.weight)
            init.orthogonal_(self.w_exc_ee.weight)

            init.constant_(self.g_exc_x.bias, 0.)
            init.constant_(self.g_exc_e.bias, 0.)

            init.constant_(self.w_exc_x.bias, 0.)
            init.constant_(self.w_exc_ee.bias, 0.)

    def forward(self, input, exc):
        g_exc = torch.sigmoid(self.ln_e_x(self.g_exc_x(
            input)) + self.ln_e_e(self.g_exc_e(exc)))

        e_hat_t = torch.relu(self.w_exc_x(input) + self.w_exc_ee(exc))

        exc = torch.relu(self.ln_out_e(g_exc * e_hat_t + (1 - g_exc) * exc))
        return exc


class ExtRNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 exc_fsize=11,
                 timesteps=4,
                 temporal_agg=False,
                 init_=None,
                 ):
        super(ExtRNNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.exc_fsize = exc_fsize
        self.timesteps = timesteps
        self.init_ = init_
        print("%s steps of recurrence" % self.timesteps)
        self.rnn_cell = ExtRNNcell(in_channels=self.in_channels,
                                   hidden_dim=self.hidden_dim,
                                   exc_fsize=self.exc_fsize,
                                   init_=self.init_)
        self.emb_exc = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        self.emb_inh = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        if temporal_agg:
            self.temporal_agg = nn.Parameter(torch.ones([1, self.timesteps]))
        else:
            self.temporal_agg = None

    def forward(self, input):
        outputs_e = []
        n_x, _, w_x, h_x = input.shape
        # state = (self.emb_exc(input), self.emb_inh(input))
        # state = (torch.zeros_like(input), torch.zeros_like(input))
        state = torch.zeros(n_x, self.hidden_dim, w_x, h_x)
        if torch.cuda.is_available():
            state = state.cuda()
        for _ in range(self.timesteps):
            state = self.rnn_cell(input, state)
            outputs_e.append(state)
        if self.temporal_agg is not None:
            t_probs = nn.Softmax(dim=1)(self.temporal_agg)
            outputs_e = torch.stack(outputs_e)
            output = torch.einsum('ij,jklmn -> iklmn', t_probs, outputs_e)
            return output[0]
        return outputs_e[-1]
