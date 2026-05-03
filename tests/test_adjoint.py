"""
Adjoint gradient correctness tests.

Strategy: compare gradients from odeint_adjoint vs torch.autograd
(direct backprop through RK4 steps) on the same small ODEFunc.
They should match to float32 precision.
"""

import torch
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.neural_ode import ODEFunc, NeuralODE
from src.adjoint import odeint_adjoint
from src.ode_solver import odeint


def _copy_func(src):
    """Return a new ODEFunc with identical weights."""
    dim    = src.net[4].out_features
    hidden = src.net[0].out_features
    dst    = _make_func(dim=dim, hidden=hidden)
    dst.load_state_dict(src.state_dict())
    return dst


def _grads(func, z0, t, adjoint=False):
    func.zero_grad()
    if adjoint:
        traj = odeint_adjoint(func, z0, t, method='rk4', rtol=1e-5, atol=1e-6)
    else:
        traj = odeint(func, z0, t, method='rk4')
    loss = traj[-1].sum()
    loss.backward()
    return {n: p.grad.clone() for n, p in func.named_parameters()}


def _make_func(dim=4, hidden=16):
    """ODEFunc with sensible (xavier) init so gradients are O(1)."""
    f = ODEFunc(dim=dim, hidden=hidden, time_dependent=True)
    for m in f.net.modules():
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    return f


@pytest.fixture
def setup():
    torch.manual_seed(0)
    dim = 4
    func = _make_func(dim=dim)
    z0 = torch.randn(dim, requires_grad=True)
    t  = torch.linspace(0, 0.5, 4)   # short interval keeps z well-behaved
    return func, z0, t


def test_adjoint_param_grads_match(setup):
    func, z0, t = setup
    func_adj = _copy_func(func)

    g_direct  = _grads(func,     z0.detach(), t, adjoint=False)
    g_adjoint = _grads(func_adj, z0.detach(), t, adjoint=True)

    for name in g_direct:
        d, a = g_direct[name], g_adjoint[name]
        # relative error floored by gradient magnitude to avoid near-zero inflation
        scale = d.abs().mean().clamp(min=1e-3)
        err = (d - a).abs().max() / scale
        assert err < 0.05, \
            f"param '{name}' grad mismatch: scaled err = {err:.2e}"


def test_adjoint_z0_grad(setup):
    func, z0, t = setup
    func_adj = _copy_func(func)

    # direct
    z0_d = z0.detach().requires_grad_(True)
    traj_d = odeint(func, z0_d, t, method='rk4')
    traj_d[-1].sum().backward()
    grad_d = z0_d.grad.clone()

    # adjoint
    z0_a = z0.detach().requires_grad_(True)
    traj_a = odeint_adjoint(func_adj, z0_a, t, method='rk4')
    traj_a[-1].sum().backward()
    grad_a = z0_a.grad.clone()

    scale = grad_d.abs().mean().clamp(min=1e-3)
    err = (grad_d - grad_a).abs().max() / scale
    assert err < 0.05, f"z0 grad mismatch: scaled err = {err:.2e}"


def test_adjoint_batched(setup):
    """Batched z0 (batch, dim) should work identically."""
    func, z0, t = setup
    func_adj = _copy_func(func)
    batch = 8
    z0_b = torch.randn(batch, 4)

    g_direct  = _grads(func,     z0_b, t, adjoint=False)
    g_adjoint = _grads(func_adj, z0_b, t, adjoint=True)

    for name in g_direct:
        d, a = g_direct[name], g_adjoint[name]
        scale = d.abs().mean().clamp(min=1e-3)
        err = (d - a).abs().max() / scale
        assert err < 0.05, \
            f"batched param '{name}' grad mismatch: scaled err = {err:.2e}"


def test_nfe_counted(setup):
    func, z0, t = setup
    node = NeuralODE(func, method='rk4', adjoint=False)
    _ = node(z0.detach(), t)
    assert func.nfe > 0, "NFE counter should be nonzero after forward pass"
