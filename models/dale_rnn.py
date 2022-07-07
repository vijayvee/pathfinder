"""Recurrent EI normalization."""
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error


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
                 in_h=None,
                 in_w=None,
                 hidden_dim=None,
                 divnorm_fsize=3,
                 exc_fsize=5,
                 inh_fsize=3,
                 device='cuda',
                 ):
        super(DaleRNNcell, self).__init__()
        self.in_channels = in_channels
        if hidden_dim is None:
          self.hidden_dim = in_channels
        else:
          self.hidden_dim = hidden_dim
        self.div = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            divnorm_fsize,
            padding=(divnorm_fsize - 1) // 2,
            bias=False)
        # recurrent gates computation
        self.g_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        self.ln_e_x = nn.LayerNorm([self.hidden_dim, in_h, in_w])
        self.g_exc_e = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.ln_e_e = nn.LayerNorm([self.hidden_dim, in_h, in_w])
        self.g_inh_x = nn.Conv2d(self.in_channels, self.hidden_dim, 1)
        self.ln_i_x = nn.LayerNorm([self.hidden_dim, in_h, in_w])
        self.g_inh_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.ln_i_i = nn.LayerNorm([self.hidden_dim, in_h, in_w])
        self.ln_out = nn.LayerNorm([self.hidden_dim, in_h, in_w])
        # feedforward stimulus drive
        self.w_exc_x = nn.Conv2d(self.in_channels, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2)
        self.w_inh_x = nn.Conv2d(self.in_channels, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2)

        # horizontal connections (e->e, i->e, i->i, e->i)
        self.w_exc_e = nn.Conv2d(self.hidden_dim, self.hidden_dim, exc_fsize, padding=(exc_fsize-1) // 2)
        # disynaptic inhibition with pairs of E-I cells, E -> exciting surround I -> inhibiting surround E
        self.w_exc_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, 1)
        self.w_inh_i = nn.Conv2d(self.hidden_dim, self.hidden_dim, inh_fsize, padding=(inh_fsize-1) // 2)
        self.w_inh_e = nn.Conv2d(self.hidden_dim, self.hidden_dim, inh_fsize, padding=(inh_fsize-1) // 2)
        nonnegative_weights_init(self.div)
        
    def forward(self, input, hidden):
      # TODO (make symmetric horizontal connections)
      exc, inh = hidden
      g_exc = torch.sigmoid(self.ln_e_x(self.g_exc_x(input)) + self.ln_e_e(self.g_exc_e(exc)))
      g_inh = torch.sigmoid(self.ln_i_x(self.g_inh_x(input)) + self.ln_i_i(self.g_inh_i(inh)))
      e_hat_t = torch.relu(self.w_exc_x(input) + self.w_exc_e(exc) - self.w_exc_i(inh))
      # Add a scalar multiplier to i_hat_t to control sub-contrast regime normalization?
      i_hat_t = torch.relu(self.w_inh_x(input) + self.w_inh_e(exc) - self.w_inh_i(inh))
      exc = F.relu(g_exc * e_hat_t + (1 - g_exc) * exc)
      inh = F.relu(g_inh * i_hat_t + (1 - g_inh) * inh)
      norm = self.div(exc) + 1e-5
      exc = F.relu(self.ln_out(exc / norm))
      return (exc, inh)


class DaleRNNLayer(nn.Module):
  def __init__(self, 
               in_channels,
               in_h, in_w,
               hidden_dim=None,
               divnorm_fsize=3,
               exc_fsize=7,
               inh_fsize=3,
               timesteps=4,
               device='cuda',
               temporal_agg=True,
               ):
    super(DaleRNNLayer, self).__init__()
    self.in_channels = in_channels
    self.in_h = in_h
    self.in_w = in_w
    self.hidden_dim = hidden_dim
    self.divnorm_fsize = divnorm_fsize
    self.exc_fsize = exc_fsize
    self.inh_fsize = inh_fsize
    self.timesteps = timesteps
    self.device = device
    self.rnn_cell = DaleRNNcell(in_channels=self.in_channels,
                            in_h=self.in_h, in_w=self.in_w,
                            hidden_dim=self.hidden_dim,
                            divnorm_fsize=self.divnorm_fsize,
                            exc_fsize=self.exc_fsize,
                            inh_fsize=self.inh_fsize,
                            device=self.device)
    if temporal_agg:
      self.temporal_agg = nn.Conv2d(self.hidden_dim * self.timesteps, self.hidden_dim, 1)
    else:
      self.temporal_agg = None
  
  def forward(self, input):
    outputs_e = []
    outputs_i = []
    state = (torch.zeros_like(input), torch.zeros_like(input))
    for _ in range(self.timesteps):
        state = self.rnn_cell(input, state)
        outputs_e += [state[0]]
        outputs_i += [state[1]]
    if self.temporal_agg is not None:
      outputs_e = torch.cat(outputs_e, dim=1)
      output = self.temporal_agg(outputs_e)
      return output
    return outputs_e[-1]