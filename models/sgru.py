import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function

# torch.manual_seed(42)


class dummyhgru(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors

        args = ctx.args
        truncate_iter = args[-1]
        # exp_name = args[-2]
        # i = args[-3]
        # epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(
                last_state,
                state_2nd_last,
                grad_outputs=neumann_v_prev,
                retain_graph=True,
                allow_unused=True,
            )
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)

            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g

            normsv.append(normv.data.item())
            normsg.append(normg.data.item())

        return (None, neumann_g, None, None, None, None)


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        batchnorm=True,
    ):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batchnorm = batchnorm

        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size)
        )
        self.w_gate_exc = nn.Parameter(
            torch.empty(hidden_size, hidden_size, kernel_size, kernel_size)
        )

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList(
            [
                nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True, track_running_stats=False)
                for i in range(4)
            ]
        )

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)

        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)

        for bn in self.bn:
            init.constant_(bn.weight, 0.1)

        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data
        # self.outscale = nn.Parameter(torch.tensor([8.0]))
        # self.outintercpt = nn.Parameter(torch.tensor([-4.0]))
        self.softpl = nn.Softplus()
        # self.softpl.register_backward_hook(
        #     lambda module, grad_i, grad_o: print(len(grad_i))
        # )

    def forward(self, input_, prev_state2, timestep=0):
        activ = F.softplus
        g1_t = torch.sigmoid((self.u1_gate(prev_state2)))
        c1_t = self.bn[1](
            F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding)
        )

        next_state1 = activ(input_ - activ(c1_t * (self.alpha * prev_state2 + self.mu)))

        g2_t = torch.sigmoid((self.u2_gate(next_state1)))
        c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))

        h2_t = activ(
            self.kappa * next_state1 + self.gamma * c2_t + self.w * next_state1 * c2_t
        )
        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        prev_state2 = F.softplus(prev_state2)

        return prev_state2, g2_t


class hConvSGRU(nn.Module):
    def __init__(
        self,
        timesteps=8,
        filt_size=5,
        num_iter=50,
        hidden_dim=25,
        jacobian_penalty=True,
        grad_method="rbp",
    ):
        super(hConvSGRU, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.unit1 = hConvGRUCell(hidden_dim, hidden_dim, filt_size)
        self.bn = nn.BatchNorm2d(hidden_dim, eps=1e-03, track_running_stats=False)

    def forward(
        self, x, iters_to_do=12,
        ):
        testmode = self.training == False
        internal_state = torch.zeros_like(x, requires_grad=False)
        states = []
        if self.grad_method == "rbp":
            with torch.no_grad():
                for i in range(iters_to_do - 1):
                    internal_state, _ = self.unit1(x, internal_state, timestep=i)
                    states.append(internal_state)
            state_2nd_last = internal_state.detach().requires_grad_()
            i += 1
            last_state, _ = self.unit1(x, state_2nd_last, timestep=i)
            internal_state = dummyhgru.apply(
                state_2nd_last, last_state, self.num_iter
            )
            states.append(internal_state)
        elif self.grad_method == "bptt":
            for i in range(iters_to_do):
                internal_state, g2t = self.unit1(x, internal_state, timestep=i)
                if i == iters_to_do - 2:
                    state_2nd_last = internal_state
                elif i == iters_to_do - 1:
                    last_state = internal_state
        # output = self.bn(internal_state)
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.9
        double_neg = False
        if self.training and self.jacobian_penalty:
            norm_1_vect = torch.ones_like(last_state)
            norm_1_vect.requires_grad = False
            jv_prod = torch.autograd.grad(
                last_state,
                state_2nd_last,
                grad_outputs=[norm_1_vect],
                retain_graph=True,
                create_graph=self.jacobian_penalty,
                allow_unused=True,
            )[0]
            jv_penalty = (jv_prod - mu).clamp(0) ** 2
            if double_neg is True:
                neg_norm_1_vect = -1 * norm_1_vect.clone()
                jv_prod = torch.autograd.grad(
                    last_state,
                    state_2nd_last,
                    grad_outputs=[neg_norm_1_vect],
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=True,
                )[0]
                jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                jv_penalty = jv_penalty + jv_penalty2
        return [self.bn(state) for state in states], jv_penalty.mean()