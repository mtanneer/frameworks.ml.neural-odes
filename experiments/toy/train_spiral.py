"""
Checkpoint 3 — Toy problem: 2D spiral fitting with Neural ODE.

Generates CW/CCW spiral dataset (from paper §4.3), trains Latent ODE
to reconstruct and extrapolate. Saves loss curve + trajectory plot.

Usage:
    python experiments/toy/train_spiral.py
    python experiments/toy/train_spiral.py --niters 2000 --lr 0.01 --adjoint
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ode_solver import odeint
from src.adjoint import odeint_adjoint
from src.neural_ode import ODEFunc


# ── Data ──────────────────────────────────────────────────────────────────────

def generate_spiral(n_points=100, t_end=6.0, clockwise=False, noise=0.05, seed=0):
    """True spiral: dx/dt = A*x  where A encodes rotation+contraction."""
    torch.manual_seed(seed)
    t = torch.linspace(0, t_end, n_points)
    # rotation matrix (CW or CCW)
    sign = -1.0 if clockwise else 1.0
    A = torch.tensor([[-0.1, sign * 2.0],
                      [-sign * 2.0, -0.1]])
    z0 = torch.tensor([0.6, 0.3])

    traj = [z0]
    z = z0.clone()
    dt = t[1] - t[0]
    for _ in range(n_points - 1):
        z = z + dt * (A @ z)
        traj.append(z.clone())
    traj = torch.stack(traj)
    noise_t = noise * torch.randn_like(traj)
    return t, traj + noise_t, traj  # (t, noisy, clean)


# ── Model ─────────────────────────────────────────────────────────────────────

class SpiralODEFunc(nn.Module):
    """ODEFunc for 2D spiral: learns dz/dt = f(z, t)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),   # [z0, z1, t]
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        t_scalar = t.reshape(1) if z.dim() == 1 else t.expand(z.shape[0], 1)
        inp = torch.cat([z, t_scalar], dim=-1)
        return self.net(inp)

    def reset_nfe(self):
        self.nfe = 0


# ── Train ──────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(42)

    # data
    t, noisy, clean = generate_spiral(n_points=args.n_points, noise=args.noise)
    t_cw, noisy_cw, clean_cw = generate_spiral(n_points=args.n_points, clockwise=True,
                                                 noise=args.noise, seed=1)

    func = SpiralODEFunc(hidden=args.hidden)
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.niters // 3, gamma=0.5)

    solve = odeint_adjoint if args.adjoint else odeint
    losses = []

    print(f"Training for {args.niters} iters | lr={args.lr} | adjoint={args.adjoint}")
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        func.reset_nfe()

        # Fit CCW only (one deterministic trajectory per IC)
        pred = solve(func, clean[0], t, method=args.method)
        loss = ((pred - noisy) ** 2).mean()

        loss.backward()
        nn.utils.clip_grad_norm_(func.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if itr % max(1, args.niters // 10) == 0:
            print(f"  iter {itr:4d}/{args.niters}  loss={loss.item():.4f}  nfe={func.nfe}")

    os.makedirs(args.outdir, exist_ok=True)
    _save_results(func, t, noisy, clean, t_cw, noisy_cw, clean_cw, losses, args)
    print(f"Results saved to {args.outdir}")


def _save_results(func, t, noisy, clean, t_cw, noisy_cw, clean_cw, losses, args):
    torch.save({'state_dict': func.state_dict(), 'losses': losses},
               os.path.join(args.outdir, 'spiral_latest.pt'))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # loss curve
    axes[0].semilogy(losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('MSE')
    axes[0].grid(True, alpha=0.3)

    # CCW trajectory
    with torch.no_grad():
        pred = odeint(func, clean[0], t, method=args.method).numpy()
    axes[1].plot(clean[:, 0], clean[:, 1], 'k--', alpha=0.4, label='True')
    axes[1].plot(noisy[:, 0], noisy[:, 1], 'b.', alpha=0.4, label='Observed')
    axes[1].plot(pred[:, 0], pred[:, 1], 'r-', label='Neural ODE')
    axes[1].set_title('CCW Spiral')
    axes[1].legend(fontsize=7)
    axes[1].set_aspect('equal')

    # CW trajectory
    with torch.no_grad():
        pred_cw = odeint(func, clean_cw[0], t_cw, method=args.method).numpy()
    axes[2].plot(clean_cw[:, 0], clean_cw[:, 1], 'k--', alpha=0.4, label='True')
    axes[2].plot(noisy_cw[:, 0], noisy_cw[:, 1], 'b.', alpha=0.4, label='Observed')
    axes[2].plot(pred_cw[:, 0], pred_cw[:, 1], 'r-', label='Neural ODE')
    axes[2].set_title('CW Spiral')
    axes[2].legend(fontsize=7)
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'spiral_fit.png'), dpi=120)
    plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--niters',    type=int,   default=1000)
    p.add_argument('--lr',        type=float, default=0.01)
    p.add_argument('--hidden',    type=int,   default=64)
    p.add_argument('--n-points',  type=int,   default=100)
    p.add_argument('--noise',     type=float, default=0.05)
    p.add_argument('--method',    type=str,   default='rk4')
    p.add_argument('--adjoint',   action='store_true')
    p.add_argument('--outdir',    type=str,   default='results')
    args = p.parse_args()
    train(args)
