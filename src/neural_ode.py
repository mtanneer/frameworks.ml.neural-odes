"""
NeuralODE module.

Usage:
    func = ODEFunc(dim=2, hidden=64)
    node = NeuralODE(func, method='dopri5')
    z1   = node(z0, t_span)        # z1 shape: (*z0.shape,)
    traj = node(z0, t_eval, return_traj=True)  # shape: (len(t_eval), *z0.shape)
"""

import torch
import torch.nn as nn
from .ode_solver import odeint
from .adjoint import odeint_adjoint


class ODEFunc(nn.Module):
    """Small MLP dynamics: dz/dt = f(z, t)."""

    def __init__(self, dim, hidden=64, time_dependent=True):
        super().__init__()
        self.time_dependent = time_dependent
        in_dim = dim + 1 if time_dependent else dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        # init small weights for stable early training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        self.nfe = 0  # number of function evaluations

    def forward(self, t, z):
        self.nfe += 1
        if self.time_dependent:
            t_vec = t.expand(z.shape[0], 1) if z.dim() > 1 else t.unsqueeze(0)
            inp = torch.cat([z, t_vec], dim=-1)
        else:
            inp = z
        return self.net(inp)

    def reset_nfe(self):
        self.nfe = 0


class NeuralODE(nn.Module):
    """
    Wraps an ODEFunc with a chosen solver.
    Supports adjoint backprop (memory-efficient) or direct autograd.
    """

    def __init__(self, func, method='dopri5', rtol=1e-3, atol=1e-4, adjoint=True):
        super().__init__()
        self.func = func
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.adjoint = adjoint

    def forward(self, z0, t, return_traj=False):
        """
        Args:
            z0:          initial state, shape (batch, dim) or (dim,)
            t:           1-D tensor of time points [t0, ..., t1]
            return_traj: if True return all states, else only final state

        Returns:
            tensor of shape (len(t), *z0.shape) if return_traj else (*z0.shape,)
        """
        self.func.reset_nfe()

        if self.adjoint:
            traj = odeint_adjoint(self.func, z0, t,
                                  method=self.method,
                                  rtol=self.rtol, atol=self.atol)
        else:
            traj = odeint(self.func, z0, t,
                          method=self.method,
                          rtol=self.rtol, atol=self.atol)

        return traj if return_traj else traj[-1]
