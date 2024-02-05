"""Recurrent EI normalization."""
import numpy as np
import torch  # pylint: disable=import-error
import torch.nn as nn  # pylint: disable=import-error
import torch.nn.functional as F  # pylint: disable=import-error
from torch.nn import init
import pdb

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
        elif nl == 'clipped_relu_1':
            self.act_fn = nn.Hardtanh(min_val=0., max_val=1.)
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
            return (exc, inh), (g_exc, g_inh)
        else:
            return (e_hat_t, i_hat_t)


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
                 interneuron=True,
                 eps=0.01,
                 ):
        super(DaleRNNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.exc_fsize = exc_fsize
        self.inh_fsize = inh_fsize
        # Below is now the MAX number of timesteps allowed for the network to 'ponder'
        self.timesteps = timesteps
        self.init_ = init_
        self.return_all_states = return_all_states
        print("%s steps of recurrence" % self.timesteps)
        self.device = device
        self.interneuron = interneuron
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
        if self.interneuron:
            print("Using interneuron")
        else:
            self.ei_aggregator = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, 1)
        if temporal_agg:
            self.temporal_agg = nn.Parameter(torch.ones([1, self.timesteps]))
        else:
            self.temporal_agg = None

        # ACT params
        self.eps = eps
        self.total_step_count = 0
        self.halt_conv = nn.Conv2d(self.hidden_dim, 2, 1)
        self.halt_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_halt = nn.Linear(2, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([-5.]))  # Set initial bias to use all iterations at the beginning
        
    def forward(self, input):
        self.total_step_count += 1

        outputs_e = [] # This is states []
        outputs_i = []

        gates_e = []
        gates_i = []
        
        n_x, _, w_x, h_x = input.shape

        halt_probs = torch.zeros((n_x, self.timesteps))
        temp_zeros_tensor = torch.zeros(n_x, dtype=torch.int64)
        temp_ones_tensor = torch.ones(n_x)
        temp_max_timestep_tensor = torch.ones(n_x, dtype=torch.int64) * (self.timesteps-1)

        # state = (self.emb_exc(input), self.emb_inh(input))
        # state = (torch.zeros_like(input), torch.zeros_like(input))
        state = (torch.zeros(n_x, self.hidden_dim, w_x, h_x),
                torch.zeros(n_x, self.hidden_dim, w_x, h_x))

        if torch.cuda.is_available():
            state = (state[0].cuda(), state[1].cuda())
            halt_probs = halt_probs.cuda()
            temp_zeros_tensor = temp_zeros_tensor.cuda()
            temp_ones_tensor = temp_ones_tensor.cuda()
            temp_max_timestep_tensor = temp_max_timestep_tensor.cuda()

        for _ in range(self.timesteps):
            generate_halt_prob = self.total_step_count > 60000
            # This part is the addition
            x = self.halt_conv(state[0])
            x = self.halt_pool(x)
            x = torch.flatten(x, 1)
            if generate_halt_prob:
                halt_prob_for_timestep = F.sigmoid(self.fc_halt(x)).flatten()
            else:
                halt_prob_for_timestep = torch.zeros(state[0].shape[0])
            # End addition
            state, gates = self.rnn_cell(input, state)
            halt_probs[:, _] = halt_prob_for_timestep.flatten()
            # halt_probs[:, _] = 0.0 
            outputs_e += [state[0]]
            outputs_i += [state[1]]
            gates_e += [gates[0]]
            gates_i += [gates[1]]

        accumulated_halt_probs = torch.cumsum(halt_probs, dim=1)
        has_not_exceeded_threshold = accumulated_halt_probs < 1-self.eps
        position_exceeded = torch.minimum(has_not_exceeded_threshold.sum(dim=1), temp_max_timestep_tensor)   # Size n_x
        
        # We compute the residuals for all examples which don't end immediately after 1st step
        residuals = 1.0 - torch.gather(accumulated_halt_probs, 
                                dim=1, 
                                index=torch.maximum(position_exceeded-1, temp_zeros_tensor).unsqueeze(1)).flatten()
        # For the examples in the batch where threshold is exceeded in the very first step, set residual to 1
        residuals[position_exceeded == 0] = 1.0
        ''' The above is from their code, but should be we change this since grads don't backprop
                                           in this case? ''' 

        # We update the last halt probability in each example with the residual
        halt_probs.scatter_(dim=1, index=position_exceeded.unsqueeze(1), src=residuals.unsqueeze(1))
        # We then set all probabilities after the position_exceeded ids to 0 with the following 3 lines of code
        # First, a [[0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0]] One hot vector at position_exceeded
        # Then, make it [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0]] using 1 - cumsum
        # Finally, multiply with halt_probs to set all probabilities after the position_exceeded ids to 0
        probs_mask = torch.zeros(n_x, self.timesteps+1)
        if torch.cuda.is_available():
            probs_mask = probs_mask.cuda()
        probs_mask.scatter_(dim=1, index=(position_exceeded+1).unsqueeze(1), src=temp_ones_tensor.unsqueeze(1))
        probs_mask = (1 - torch.cumsum(probs_mask, dim=1))[:, :-1]
        halt_probs = halt_probs * probs_mask
        ponder_cost = residuals.mean()
        
        # return outputs_e[-1], halt_probs, ponder_cost

        if self.temporal_agg is not None:
            t_probs = nn.Softmax(dim=1)(self.temporal_agg)
            outputs_e = torch.stack(outputs_e)
            output = torch.einsum('ij,jklmn -> iklmn', t_probs, outputs_e)
            return output[0]

        if self.return_all_states:
            # return (outputs_e, outputs_i), halt_probs, ponder_cost
            return (outputs_e, outputs_i), (gates_e, gates_i), halt_probs, ponder_cost

        # if self.interneuron:
        #     return outputs_e[-1]
        # else:
        #     return self.ei_aggregator(torch.cat((outputs_e[-1], 
        #                                          outputs_i[-1]), 
        #                                          1))
