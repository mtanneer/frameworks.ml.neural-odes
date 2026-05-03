"""
Latent ODE model for irregular time-series (Chen et al. 2018, §5).

Architecture:
  Encoder : RNN (GRU) reads (x_obs, t_obs) backwards → q(z0 | x)
  Dynamics: Neural ODE integrates z0 forward to query times
  Decoder : MLP maps z(t) → x̂(t)
  Loss    : ELBO = -reconstruction + β * KL(q || p)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ode_solver import odeint
from src.adjoint    import odeint_adjoint


# ── ODE dynamics ──────────────────────────────────────────────────────────────

class LatentODEFunc(nn.Module):
    """dz/dt = f(z, t).  z ∈ R^latent_dim."""

    def __init__(self, latent_dim, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        B = z.shape[0]
        t_vec = t.expand(B, 1)
        return self.net(torch.cat([z, t_vec], dim=-1))

    def reset_nfe(self):
        self.nfe = 0


# ── Encoder: RNN → posterior q(z0) ───────────────────────────────────────────

class RNNEncoder(nn.Module):
    """
    Reads (x, Δt) pairs in REVERSE order and outputs μ, log-σ for z0.

    Input per step: [x_t, Δt]  shape (B, D+1)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.0):
        super().__init__()
        self.gru     = nn.GRU(input_dim + 1, hidden_dim, batch_first=True,
                               dropout=0.0)  # GRU dropout only valid for num_layers>1
        self.drop    = nn.Dropout(dropout)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logv = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_obs, t_obs, obs_mask):
        B, S, D = x_obs.shape

        dt = torch.zeros(B, S, 1, device=x_obs.device)
        dt[:, 1:, 0] = t_obs[:, 1:] - t_obs[:, :-1]

        inp     = torch.cat([x_obs, dt], dim=-1)   # (B, S, D+1)
        inp_rev = inp.flip(1)

        lengths = obs_mask.sum(dim=1).cpu().clamp(min=1)
        packed  = nn.utils.rnn.pack_padded_sequence(
            inp_rev, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)                    # h: (1, B, hidden)
        h = self.drop(h.squeeze(0))                # (B, hidden)

        return self.fc_mu(h), self.fc_logv(h)


# ── Decoder: z(t) → x̂ ────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z):
        return self.net(z)


# ── Full Latent ODE ───────────────────────────────────────────────────────────

class LatentODE(nn.Module):
    """
    VAE with ODE prior for irregular time-series.

    Args:
        input_dim   : observed feature dimension (D)
        latent_dim  : latent state dimension
        hidden_dim  : hidden units for RNN encoder
        ode_hidden  : hidden units for ODE dynamics
        dropout     : dropout rate for encoder + decoder
        adjoint     : use adjoint backprop
        method      : ODE solver method
    """

    def __init__(self, input_dim, latent_dim=16, hidden_dim=64,
                 ode_hidden=64, dropout=0.1,
                 adjoint=True, method='rk4', rtol=1e-3, atol=1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.adjoint    = adjoint
        self.method     = method
        self.rtol       = rtol
        self.atol       = atol

        self.encoder  = RNNEncoder(input_dim, hidden_dim, latent_dim, dropout=dropout)
        self.ode_func = LatentODEFunc(latent_dim, ode_hidden, dropout=dropout)
        self.decoder  = Decoder(latent_dim, input_dim, hidden_dim, dropout=dropout)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def integrate(self, z0, t_query):
        solve = odeint_adjoint if self.adjoint else odeint
        return solve(self.ode_func, z0, t_query,
                     method=self.method, rtol=self.rtol, atol=self.atol)

    def forward(self, x_obs, t_obs, obs_mask, t_query):
        mu, log_var = self.encoder(x_obs, t_obs, obs_mask)
        z0          = self.reparameterize(mu, log_var)

        self.ode_func.reset_nfe()
        z_traj = self.integrate(z0, t_query)       # (L, B, latent_dim)
        z_traj = z_traj.permute(1, 0, 2)           # (B, L, latent_dim)

        x_hat = self.decoder(z_traj)               # (B, L, D)
        return x_hat, mu, log_var

    @property
    def nfe(self):
        return self.ode_func.nfe


# ── ELBO loss ────────────────────────────────────────────────────────────────

def elbo_loss(x_hat, x_target, mu, log_var, mask=None, beta=1.0, free_bits=0.0):
    """
    ELBO = E[log p(x|z)] - β * KL(q(z0)||p(z0))

    free_bits clamps per-dim KL to a minimum to keep latent dims active.
    """
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        recon = ((x_hat - x_target) ** 2 * m).sum() / (m.sum() * x_hat.shape[-1])
    else:
        recon = ((x_hat - x_target) ** 2).mean()

    kl_per_dim = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl = kl_per_dim.sum(dim=-1).mean()

    return recon + beta * kl, recon, kl
