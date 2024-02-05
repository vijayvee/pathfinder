# Implements a Convolutional GRU Cell
# Code almost copied from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import pdb

class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, act_fn):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        # self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out1 = nn.Conv2d(input_size, hidden_size, kernel_size, padding=padding)
        self.out2 = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding)
        self.act_fn = act_fn

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out1.weight)
        init.orthogonal(self.out2.weight)
        # init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        # init.constant(self.out_gate.bias, 0.)
        init.constant(self.out1.bias, 0.)
        init.constant(self.out2.bias, 0.)

        # This is technically layernorm but we're using groupNorm for easy implementation
        self.ln1 = nn.GroupNorm(hidden_size, hidden_size)
        self.ln2 = nn.GroupNorm(hidden_size, hidden_size)
        self.ln3 = nn.GroupNorm(hidden_size, hidden_size)
        self.ln4 = nn.GroupNorm(hidden_size, hidden_size)



    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        
        # update = F.sigmoid(self.update_gate(stacked_inputs))
        # reset = F.sigmoid(self.reset_gate(stacked_inputs))
        # # out_inputs = self.act_fn(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        # out_inputs = self.act_fn(self.out1(input_) + self.out2(prev_state * reset))
        # new_state = prev_state * (1 - update) + out_inputs * update

        # With layer norm
        update = F.sigmoid(self.ln1(self.update_gate(stacked_inputs)))
        reset = F.sigmoid(self.ln2(self.reset_gate(stacked_inputs)))
        out_inputs = self.act_fn(self.ln3(self.out1(input_)) + \
                                  self.ln4(self.out2(prev_state * reset)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_size, 
                    kernel_size, timesteps,
                    nl='tanh', eps=0.01):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.timesteps = timesteps
        if nl == 'tanh':
            self.act_fn = nn.Tanh()
        elif nl == 'relu':
            self.act_fn = nn.ReLU()
        else:
            raise ValueError('Unknown non-linearity')
        self.cell = ConvGRUCell(self.input_size, self.hidden_size, self.kernel_size, self.act_fn)

        # ACT params
        self.eps = eps
        self.total_step_count = 0
        self.halt_conv = nn.Conv2d(self.hidden_size, 2, 1)
        self.halt_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_halt = nn.Linear(2, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([-5.]))  # Set initial bias to use all iterations at the beginning

    def forward(self, input):
        '''
        Parameters
        ----------
        input : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        batch_size, _, height, width = input.size()
        hidden = torch.zeros(batch_size, self.hidden_size, height, width).to(input.device)
        
        self.total_step_count += 1        
        n_x, _, _, _ = input.shape
        halt_probs = torch.zeros((n_x, self.timesteps)).to(input.device)
        temp_zeros_tensor = torch.zeros(n_x, dtype=torch.int64).to(input.device)
        temp_ones_tensor = torch.ones(n_x).to(input.device)
        temp_max_timestep_tensor = (torch.ones(n_x, dtype=torch.int64) * (self.timesteps-1)).to(input.device)

        all_hiddens = []
        # print(self.timesteps, "timesteps")
        for _ in range(self.timesteps):
            # This part is the addition
            generate_halt_prob =  self.total_step_count > 60000
            x = self.halt_conv(hidden)
            x = self.halt_pool(x)
            x = torch.flatten(x, 1)
            if generate_halt_prob:
                halt_prob_for_timestep = F.sigmoid(self.fc_halt(x)).flatten()
            else:
                halt_prob_for_timestep = torch.zeros(hidden.shape[0])
            # End addition
            # pass through layer
            hidden = self.cell(input, hidden)
            all_hiddens += [hidden]
            halt_probs[:, _] = halt_prob_for_timestep.flatten()
            # halt_probs[:, _] = 0.0 
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

        # retain tensors in list to allow different hidden sizes
        return all_hiddens, halt_probs, ponder_cost
