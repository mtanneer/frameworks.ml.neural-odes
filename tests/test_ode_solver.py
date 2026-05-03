"""
ODE solver tests against problems with known analytical solutions.

1. Exponential decay:  dy/dt = -y,  y(0)=1  =>  y(t) = exp(-t)
2. Harmonic oscillator: d²x/dt² = -x  as 2D system:
       dy/dt = [y1, -y0],  y(0)=[1,0]  =>  y0(t)=cos(t), y1(t)=sin(t)
3. Linear growth:  dy/dt = 2t,  y(0)=0  =>  y(t) = t²
"""

import math
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ode_solver import odeint


# ── Fixtures / helpers ─────────────────────────────────────────────────────────

def rel_err(pred, true):
    return ((pred - true).abs() / (true.abs() + 1e-8)).max().item()


# ── Test 1: exponential decay ──────────────────────────────────────────────────

def exp_decay(t, y):
    return -y

@pytest.mark.parametrize("method,tol", [
    ("euler",  5e-2),
    ("rk4",    1e-6),
    ("dopri5", 1e-4),
])
def test_exp_decay(method, tol):
    t = torch.linspace(0, 2, 101)
    y0 = torch.tensor([1.0])
    ys = odeint(exp_decay, y0, t, method=method)
    exact = torch.exp(-t).unsqueeze(1)
    assert rel_err(ys, exact) < tol, f"{method} exp_decay err={rel_err(ys, exact):.2e} > {tol}"


# ── Test 2: harmonic oscillator ────────────────────────────────────────────────

def harmonic(t, y):
    return torch.stack([y[1], -y[0]])

@pytest.mark.parametrize("method,tol", [
    ("euler",  1e-3),
    ("rk4",    1e-5),
    ("dopri5", 1e-5),
])
def test_harmonic_oscillator(method, tol):
    t = torch.linspace(0, 1.0, 1001)
    y0 = torch.tensor([1.0, 0.0])
    ys = odeint(harmonic, y0, t, method=method)
    exact = torch.stack([torch.cos(t), -torch.sin(t)], dim=1)
    assert rel_err(ys, exact) < tol, f"{method} harmonic err={rel_err(ys, exact):.2e} > {tol}"


# ── Test 3: polynomial (linear growth) ────────────────────────────────────────

def linear_growth(t, y):
    return 2 * t

@pytest.mark.parametrize("method,tol", [
    ("euler",  1e-1),
    ("rk4",    1e-5),
    ("dopri5", 1e-4),
])
def test_polynomial(method, tol):
    # start from t=1 to avoid near-zero exact values inflating relative error
    t = torch.linspace(1, 4, 51)
    y0 = torch.tensor([1.0])   # y(1) = 1^2 = 1
    ys = odeint(linear_growth, y0, t, method=method)
    exact = (t ** 2).unsqueeze(1)
    assert rel_err(ys, exact) < tol, f"{method} poly err={rel_err(ys, exact):.2e} > {tol}"


# ── Test 4: output shape ───────────────────────────────────────────────────────

def test_output_shape():
    t = torch.linspace(0, 1, 10)
    y0 = torch.zeros(4)
    ys = odeint(lambda t, y: -y, y0, t, method='rk4')
    assert ys.shape == (10, 4)


# ── Test 5: invalid method ─────────────────────────────────────────────────────

def test_invalid_method():
    with pytest.raises(ValueError):
        odeint(lambda t, y: -y, torch.tensor([1.0]), torch.linspace(0, 1, 5), method='bogus')
