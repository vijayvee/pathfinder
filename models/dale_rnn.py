"""Recurrent EI normalization."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from torch.nn import init


def nonnegative_weights_init(m):
    """Non-negative initialization of weights."""
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight)


class DaleRNNcell(nn.Module):
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
                 inh_fsize=5,
                 device='cuda',
                 init_ = None,
                 use_ln=True,
                 use_gates=True,
                 nl='relu',
                 ):
        super(DaleRNNcell, self).__init__()
        del init_  # unused
        self.in_channels = in_channels
        if hidden_dim is None:
            self.hidden_dim = in_channels
        else:
            self.hidden_dim = hidden_dim
        self.use_ln = use_ln
        self.use_gates = use_gates
        if nl == 'relu':
            self.act_fn = torch.relu
        elif nl == 'tanh':
            self.act_fn = torch.tanh
        elif nl == 'clipped_relu':
            self.act_fn = nn.ReLU6()
        else:
            raise ValueError('Invalid non-linearity')

        # recurrent gates computation
        
        self.ln_e_x = self.get_ln()
        self.ln_e_e = self.get_ln()
        self.ln_i_x = self.get_ln()
        self.ln_i_i = self.get_ln()
        self.ln_out_e = self.get_ln()
        self.ln_out_i = self.get_ln()

        if self.use_gates:
            self.g_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
            self.g_exc_e = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
            self.g_inh_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
            self.g_inh_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)

        # feedforward stimulus drive
        self.w_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        self.w_inh_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)

        # horizontal connections (e->e, i->e, i->i, e->i)
        self.w_exc_ei = nn.Conv2d(
            self.hidden_dim * 2, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2)
        # disynaptic inhibition with pairs of E-I cells, E -> exciting surround I -> inhibiting surround E
        self.w_inh_ei = nn.Conv2d(
            self.hidden_dim * 2, self.hidden_dim, inh_fsize, padding=(inh_fsize-1) // 2)

    def get_ln(self):
        if self.use_ln:
            return nn.GroupNorm(num_groups=1, num_channels=self.hidden_dim)
        else:
            return nn.Identity()

    def forward(self, input, hidden):
        exc, inh = hidden
        if self.use_gates:
            g_exc = torch.sigmoid(self.ln_e_x(self.g_exc_x(
                input)) + self.ln_e_e(self.g_exc_e(exc)))
            g_inh = torch.sigmoid(self.ln_i_x(self.g_inh_x(
                input)) + self.ln_i_i(self.g_inh_i(inh)))

        e_hat_t = self.act_fn(
            self.w_exc_x(input) +
            self.w_exc_ei(torch.cat((exc, inh), 1)))

        i_hat_t = self.act_fn(
            self.w_inh_x(input) +
            self.w_inh_ei(torch.cat((exc, inh), 1)))

        if self.use_gates:
            exc = self.act_fn(self.ln_out_e(g_exc * e_hat_t + (1 - g_exc) * exc))
            inh = self.act_fn(self.ln_out_i(g_inh * i_hat_t + (1 - g_inh) * inh))
            return (exc, inh)
        else:
            return e_hat_t, i_hat_t


class DaleRNNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=None,
                 exc_fsize=11,
                 inh_fsize=5,
                 timesteps=4,
                 device='cuda',
                 temporal_agg=False,
                 init_=None,
                 return_all_states=False,
                 use_ln=True,
                 use_gates=True,
                 nl='relu',
                 ):
        super(DaleRNNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.exc_fsize = exc_fsize
        self.inh_fsize = inh_fsize
        self.timesteps = timesteps
        self.init_ = init_
        self.return_all_states = return_all_states
        print("%s steps of recurrence" % self.timesteps)
        self.device = device
        self.rnn_cell = DaleRNNcell(in_channels=self.in_channels,
                                    hidden_dim=self.hidden_dim,
                                    exc_fsize=self.exc_fsize,
                                    inh_fsize=self.inh_fsize,
                                    device=self.device,
                                    init_=self.init_,
                                    use_gates=use_gates,
                                    use_ln=use_ln,
                                    nl=nl,
                                    )
        self.emb_exc = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        self.emb_inh = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        if temporal_agg:
            self.temporal_agg = nn.Parameter(torch.ones([1, self.timesteps]))
        else:
            self.temporal_agg = None

    def forward(self, input):
        outputs_e = []
        outputs_i = []
        n_x, _, w_x, h_x = input.shape
        # state = (self.emb_exc(input), self.emb_inh(input))
        # state = (torch.zeros_like(input), torch.zeros_like(input))
        state = (torch.zeros(n_x, self.hidden_dim, w_x, h_x),
                torch.zeros(n_x, self.hidden_dim, w_x, h_x))
        if torch.cuda.is_available():
            state = (state[0].cuda(), state[1].cuda())
        for _ in range(self.timesteps):
            state = self.rnn_cell(input, state)
            outputs_e += [state[0]]
            outputs_i += [state[1]]
        if self.temporal_agg is not None:
            t_probs = nn.Softmax(dim=1)(self.temporal_agg)
            outputs_e = torch.stack(outputs_e)
            output = torch.einsum('ij,jklmn -> iklmn', t_probs, outputs_e)
            return output[0]
        if self.return_all_states:
            return outputs_e
        return outputs_e[-1]
