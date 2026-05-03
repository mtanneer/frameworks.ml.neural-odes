"""
Adjoint sensitivity method for backpropagating through ODE solvers.

Implements Algorithm 1 from Chen et al. 2018 (Neural ODEs).

Augmented state: s = [z(t), a(t), dL/dθ]
  where a(t) = dL/dz(t)

Backward augmented dynamics (solved t1 → t0):
  dz/dt         =  f(z, t, θ)
  da/dt         = -aᵀ ∂f/∂z
  d(dL/dθ)/dt   = -aᵀ ∂f/∂θ
"""

import torch
import torch.nn as nn
from .ode_solver import odeint


class _AdjointODEFunc(nn.Module):
    """Augmented dynamics for the backward ODE pass."""

    def __init__(self, func, z_shape, theta):
        super().__init__()
        self.func    = func
        self.z_shape = z_shape
        self.theta   = theta
        self.nparams = sum(p.numel() for p in theta)
        self.z_numel = 1
        for s in z_shape:
            self.z_numel *= s

    def forward(self, t, aug_state):
        N = self.z_numel
        z_flat = aug_state[:N]
        a_flat = aug_state[N: 2 * N]

        z = z_flat.reshape(self.z_shape)
        a = a_flat.reshape(self.z_shape)

        with torch.enable_grad():
            z_ = z.detach().requires_grad_(True)
            f  = self.func(t.detach(), z_)

            vjp_z, *vjp_theta = torch.autograd.grad(
                outputs=f,
                inputs=[z_] + self.theta,
                grad_outputs=a,
                allow_unused=True,
                retain_graph=False,
            )

        dev   = f.device
        dz_dt = f.detach().reshape(-1)
        da_dt = -(vjp_z.reshape(-1) if vjp_z is not None else torch.zeros(N, device=dev))

        dtheta_parts = []
        for vt, p in zip(vjp_theta, self.theta):
            if vt is None:
                dtheta_parts.append(torch.zeros(p.numel(), device=dev))
            else:
                dtheta_parts.append(-vt.reshape(-1))
        dtheta_dt = torch.cat(dtheta_parts) if dtheta_parts else torch.zeros(self.nparams, device=dev)

        return torch.cat([dz_dt, da_dt, dtheta_dt])


class _OdeintAdjointFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, z0, t, method, rtol, atol, *params):
        ctx.func   = func
        ctx.method = method
        ctx.rtol   = rtol
        ctx.atol   = atol

        with torch.no_grad():
            traj = odeint(func, z0, t, method=method, rtol=rtol, atol=atol)

        ctx.save_for_backward(t, traj, *params)
        return traj

    @staticmethod
    def backward(ctx, grad_traj):
        func   = ctx.func
        method = ctx.method
        rtol   = ctx.rtol
        atol   = ctx.atol

        t, traj, *params = ctx.saved_tensors
        T      = t.shape[0]
        z_shape = traj.shape[1:]
        N       = traj[0].numel()

        aug_func = _AdjointODEFunc(func, z_shape, list(params))

        # start: adjoint at t[T-1] = ∂L/∂z(tT)
        device   = traj.device
        a        = grad_traj[-1].clone().reshape(z_shape)
        dldtheta = torch.zeros(aug_func.nparams, device=device)

        for i in range(T - 1, 0, -1):
            z_i = traj[i].detach()

            # pack augmented state at t[i]: [z(ti), a(ti), dL/dθ accumulated]
            aug_state = torch.cat([
                z_i.reshape(-1),
                a.reshape(-1),
                dldtheta,
            ])

            # solve augmented ODE backwards: t[i] → t[i-1]
            t_back = torch.stack([t[i], t[i - 1]])
            with torch.no_grad():
                aug_traj = odeint(aug_func, aug_state, t_back,
                                  method=method, rtol=rtol, atol=atol)

            aug_end  = aug_traj[-1]
            a        = aug_end[N: 2 * N].reshape(z_shape)
            dldtheta = aug_end[2 * N:]

            # inject ∂L/∂z(t_{i-1}) from loss at intermediate time point
            a = a + grad_traj[i - 1].reshape(z_shape)

        grad_z0 = a.reshape(traj[0].shape)

        grad_params = []
        idx = 0
        for p in params:
            n = p.numel()
            grad_params.append(dldtheta[idx: idx + n].reshape(p.shape))
            idx += n

        return (None, grad_z0, None, None, None, None, *grad_params)


def odeint_adjoint(func, z0, t, method='rk4', rtol=1e-3, atol=1e-4):
    """
    Drop-in for odeint() with O(1)-memory adjoint backprop.

    Args:
        func:   nn.Module, forward(t, z) -> dz/dt
        z0:     initial state tensor
        t:      1-D time tensor (monotone increasing)
        method: 'euler' | 'rk4' | 'dopri5'

    Returns:
        traj: (len(t), *z0.shape)
    """
    params = list(func.parameters())
    return _OdeintAdjointFunc.apply(func, z0, t, method, rtol, atol, *params)
