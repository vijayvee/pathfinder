""" dt_net_2d.py
    DeepThinking network 2D.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import torch
from torch import nn
import torch.nn.functional as F
import pdb

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class BasicBlock2D(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, 
                 num_blocks, 
                 width, 
                 in_channels=3, 
                 recall=True, 
                 group_norm=False, eps=0.01,
                 **kwargs):
        super().__init__()

        self.recall = recall
        if 'x_to_h' in kwargs.keys():
            self.x_to_h = kwargs['x_to_h']
        else:
            self.x_to_h = False

        if 'split_gate' in kwargs.keys():
            self.split_gate = kwargs['split_gate']
        else:
            self.split_gate = False
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        recur_layers = []
        
        if recall:
            recur_layers.append(conv_recall)

        for curr_num_blocks in num_blocks:
            recur_layers.append(self._make_layer(block, width, curr_num_blocks, stride=1))
        self.block_type = "basicblock"

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        if self.block_type == "basicblock":
            self.recur_block = nn.Sequential(*recur_layers)
        else:
            self.recur_block = recur_layers[0]

        # ACT params
        self.eps = eps
        self.total_step_count = 0
        self.halt_conv = nn.Conv2d(width, 2, 1)
        self.halt_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc_halt = nn.Linear(2, 1)
        self.fc_halt.bias = nn.Parameter(torch.Tensor([-5.]))  # Set initial bias to use all iterations at the beginning

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd,
                          group_norm=self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do=20, interim_thought=None, return_hidden=False, **kwargs):
        if 'force_generate_halt_prob' not in kwargs.keys():
            force_generate_halt_prob = False
        else:
            force_generate_halt_prob = kwargs['force_generate_halt_prob']
        return self.forward_dtnet(x, iters_to_do, interim_thought, return_hidden=return_hidden)
   

    def forward_dtnet(self, x, iters_to_do, interim_thought=None, return_hidden=False, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        self.total_step_count += 1        
        n_x, _, _, _ = x.shape
        halt_probs = torch.zeros((n_x, iters_to_do)).to(x.device)
        temp_zeros_tensor = torch.zeros(n_x, dtype=torch.int64).to(x.device)
        temp_ones_tensor = torch.ones(n_x).to(x.device)
        temp_max_timestep_tensor = (torch.ones(n_x, dtype=torch.int64) * (iters_to_do-1)).to(x.device)

        inter_outputs = []
        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            # This part is the addition
            generate_halt_prob = self.total_step_count > 70000
            x = self.halt_conv(interim_thought)
            x = self.halt_pool(x)
            x = torch.flatten(x, 1)
            if generate_halt_prob:
                halt_prob_for_timestep = F.sigmoid(self.fc_halt(x)).flatten()
            else:
                halt_prob_for_timestep = torch.zeros(interim_thought.shape[0])
            # End addition
            # pdb.set_trace()
            interim_thought = self.recur_block(interim_thought)
            inter_outputs.append(interim_thought)
            halt_probs[:, i] = halt_prob_for_timestep.flatten()
            # halt_probs[:, i] = 0.0

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
        probs_mask = torch.zeros(n_x, iters_to_do+1)
        if torch.cuda.is_available():
            probs_mask = probs_mask.cuda()
        probs_mask.scatter_(dim=1, index=(position_exceeded+1).unsqueeze(1), src=temp_ones_tensor.unsqueeze(1))
        probs_mask = (1 - torch.cumsum(probs_mask, dim=1))[:, :-1]
        halt_probs = halt_probs * probs_mask
        ponder_cost = residuals.mean()

        return inter_outputs, halt_probs, ponder_cost


def dt_net_2d(width, **kwargs):
    return DTNet(BasicBlock2D, [2], width=width, in_channels=kwargs["in_channels"], recall=False)


def dt_net_recall_2d(width, **kwargs):
    return DTNet(BasicBlock2D, [2], width=width, in_channels=kwargs["in_channels"], recall=True, x_to_h=None)


def dt_net_gn_2d(width, **kwargs):
    return DTNet(BasicBlock2D, [2], width=width, in_channels=kwargs["in_channels"], recall=False, group_norm=True)


def dt_net_recall_gn_2d(width, **kwargs):
    return DTNet(BasicBlock2D, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True)