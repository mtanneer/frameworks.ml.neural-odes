"""
Fixed-step and adaptive ODE solvers.

API mirrors torchdiffeq.odeint:
    odeint(func, y0, t, method='rk4') -> tensor of shape (len(t), *y0.shape)

func signature: func(t, y) -> dy/dt  (same shape as y)
t: 1-D tensor of time points, must be monotonically increasing
"""

import torch


# ── Euler ──────────────────────────────────────────────────────────────────────

def _euler_step(func, t, y, dt):
    return y + dt * func(t, y)


# ── Classic 4th-order Runge-Kutta ──────────────────────────────────────────────

def _rk4_step(func, t, y, dt):
    k1 = func(t,          y)
    k2 = func(t + dt/2,   y + dt/2 * k1)
    k3 = func(t + dt/2,   y + dt/2 * k2)
    k4 = func(t + dt,     y + dt   * k3)
    return y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


# ── Dormand-Prince (dopri5) — adaptive step ────────────────────────────────────
# Butcher tableau coefficients

_DOPRI5_C  = torch.tensor([0., 1/5, 3/10, 4/5, 8/9, 1., 1.])
_DOPRI5_A  = [
    [],
    [1/5],
    [3/40,       9/40],
    [44/45,     -56/15,      32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168,  -355/33,    46732/5247,   49/176,  -5103/18656],
    [35/384,      0.,        500/1113,     125/192, -2187/6784,   11/84],
]
# 5th-order weights
_DOPRI5_B5 = torch.tensor([35/384, 0., 500/1113, 125/192, -2187/6784, 11/84, 0.])
# 4th-order weights (for error estimate)
_DOPRI5_B4 = torch.tensor([5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])


def _dopri5_step(func, t, y, dt, rtol=1e-3, atol=1e-6):
    """Single adaptive dopri5 step. Returns (y_next, dt_used, dt_next)."""
    safety = 0.9
    min_factor = 0.2
    max_factor = 10.0

    while True:
        k = []
        k.append(func(t, y))
        for i in range(1, 7):
            yi = y + dt * sum(_DOPRI5_A[i][j] * k[j] for j in range(i))
            k.append(func(t + _DOPRI5_C[i] * dt, yi))

        y5 = y + dt * sum(_DOPRI5_B5[i] * k[i] for i in range(7))
        y4 = y + dt * sum(_DOPRI5_B4[i] * k[i] for i in range(7))

        err_norm = _error_norm(y, y5, y4, rtol, atol)

        if err_norm <= 1.0:
            factor = min(max_factor, safety * err_norm ** (-0.2))
            return y5, dt, dt * factor

        factor = max(min_factor, safety * err_norm ** (-0.2))
        dt = dt * factor


def _error_norm(y, y5, y4, rtol, atol):
    scale = atol + rtol * torch.maximum(y.abs(), y5.abs())
    return ((y5 - y4) / scale).norm() / y.numel() ** 0.5


# ── Public API ─────────────────────────────────────────────────────────────────

def odeint(func, y0, t, method='rk4', rtol=1e-3, atol=1e-6, **kwargs):
    """
    Integrate ODE from t[0] to t[-1], returning state at each t.

    Args:
        func:   callable(t, y) -> dy/dt
        y0:     initial state tensor
        t:      1-D tensor of evaluation times (monotone increasing)
        method: 'euler' | 'rk4' | 'dopri5'

    Returns:
        ys: tensor of shape (len(t), *y0.shape)
    """
    if method == 'euler':
        return _integrate_fixed(func, y0, t, _euler_step)
    elif method == 'rk4':
        return _integrate_fixed(func, y0, t, _rk4_step)
    elif method == 'dopri5':
        return _integrate_adaptive(func, y0, t, rtol=rtol, atol=atol)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: euler, rk4, dopri5")


def _integrate_fixed(func, y0, t, step_fn):
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        t0, t1 = t[i], t[i + 1]
        dt = t1 - t0
        y = step_fn(func, t0, y, dt)
        ys.append(y)
    return torch.stack(ys)


def _integrate_adaptive(func, y0, t, rtol, atol):
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        t_cur = t[i].item()
        t_end = t[i + 1].item()
        dt = (t_end - t_cur) * 0.1  # initial step guess

        while t_cur < t_end:
            dt = min(dt, t_end - t_cur)
            y_next, dt_used, dt_next = _dopri5_step(func, torch.as_tensor(t_cur, dtype=y.dtype), y, dt, rtol, atol)
            t_cur += dt_used
            y = y_next
            dt = dt_next

        ys.append(y)
    return torch.stack(ys)


# ── Convenience alias ──────────────────────────────────────────────────────────

odeint_rk4 = lambda func, y0, t, **kw: odeint(func, y0, t, method='rk4', **kw)
odeint_euler = lambda func, y0, t, **kw: odeint(func, y0, t, method='euler', **kw)
odeint_dopri5 = lambda func, y0, t, **kw: odeint(func, y0, t, method='dopri5', **kw)
