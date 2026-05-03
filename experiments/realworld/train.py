"""
Checkpoint 4 — Latent ODE on ETT irregular time-series.

Architecture: RNN encoder → z0 → Neural ODE → decoder
Loss: ELBO = MSE reconstruction + β·KL divergence
      β is linearly annealed 0→β_max over anneal_epochs to prevent KL collapse.
      free_bits clamps per-dim KL to a minimum to keep latent dims active.

Usage:
    python experiments/realworld/train.py
    python experiments/realworld/train.py --epochs 50 --latent-dim 32
"""

import argparse, os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.realworld.ett_data   import make_loaders
from experiments.realworld.latent_ode import LatentODE, elbo_loss


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── β schedule ────────────────────────────────────────────────────────────────

def beta_schedule(epoch, beta_max, anneal_epochs):
    """Linear warmup from 0 → beta_max over anneal_epochs, then constant."""
    if anneal_epochs <= 0:
        return beta_max
    return min(beta_max, beta_max * epoch / anneal_epochs)


# ── Training / eval loops ─────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, beta, free_bits, train=True):
    model.train(train)
    total_loss = total_recon = total_kl = 0.
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            t_obs, x_obs, obs_mask, t_query, x_target, tgt_mask = [
                b.to(device) for b in batch]

            t_q = t_query[0]   # (L,) same for all items in batch

            x_hat, mu, log_var = model(x_obs, t_obs, obs_mask, t_q)

            loss, recon, kl = elbo_loss(
                x_hat, x_target, mu, log_var,
                mask=tgt_mask, beta=beta, free_bits=free_bits)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss  += loss.item()
            total_recon += recon.item()
            total_kl    += kl.item()
            n_batches   += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_reconstruction(model, loader, device, outdir, epoch):
    model.eval()
    batch = next(iter(loader))
    t_obs, x_obs, obs_mask, t_query, x_target, tgt_mask = [
        b.to(device) for b in batch]
    t_q = t_query[0]

    with torch.no_grad():
        x_hat, _, _ = model(x_obs, t_obs, obs_mask, t_q)

    t_np = t_q.cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for i, ax in enumerate(axes):
        true = x_target[0, :, i].cpu().numpy()
        pred = x_hat[0,   :, i].cpu().numpy()
        mask = tgt_mask[0].cpu().numpy()
        ax.plot(t_np, true, 'k-', alpha=0.4, label='True')
        ax.plot(t_np, pred, 'r-', label='Latent ODE')
        ax.scatter(t_np[mask], true[mask], c='blue', s=10, zorder=5, label='Observed')
        ax.set_title(f'Feature {i} — epoch {epoch}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'ett_recon_ep{epoch:03d}.png'), dpi=100)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    device = get_device()
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(
        data_dir=args.data_dir,
        window=args.window,
        stride=args.stride,
        obs_frac=args.obs_frac,
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = LatentODE(
        input_dim=7,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        ode_hidden=args.ode_hidden,
        dropout=args.dropout,
        adjoint=args.adjoint,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Latent ODE params: {n_params:,}")
    print(f"Config: latent_dim={args.latent_dim}  window={args.window}  stride={args.stride}  "
          f"obs_frac={args.obs_frac}  beta_max={args.beta}  dropout={args.dropout}  "
          f"anneal_epochs={args.anneal_epochs}  free_bits={args.free_bits}  "
          f"method={args.method}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val = float('inf')
    history  = {'train_loss': [], 'val_loss': [],
                 'train_recon': [], 'val_recon': [],
                 'train_kl': [], 'val_kl': [], 'beta': []}

    for epoch in range(1, args.epochs + 1):
        t0   = time.time()
        beta = beta_schedule(epoch, args.beta, args.anneal_epochs)

        tr_loss, tr_recon, tr_kl = run_epoch(
            model, train_loader, optimizer, device, beta, args.free_bits, train=True)
        va_loss, va_recon, va_kl = run_epoch(
            model, val_loader,   optimizer, device, beta, args.free_bits, train=False)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_recon'].append(tr_recon)
        history['val_recon'].append(va_recon)
        history['train_kl'].append(tr_kl)
        history['val_kl'].append(va_kl)
        history['beta'].append(beta)

        print(f"epoch {epoch:3d}/{args.epochs}  β={beta:.3f}  "
              f"train_loss={tr_loss:.4f}  recon={tr_recon:.4f}  kl={tr_kl:.4f}  "
              f"val_loss={va_loss:.4f}  val_recon={va_recon:.4f}  "
              f"nfe={model.nfe}  t={time.time()-t0:.1f}s")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch, 'val_loss': va_loss},
                       os.path.join(args.outdir, 'latent_ode_best.pt'))

        if epoch % max(1, args.epochs // 5) == 0:
            plot_reconstruction(model, val_loader, device, args.outdir, epoch)

    # final test
    te_loss, te_recon, te_kl = run_epoch(
        model, test_loader, optimizer, device, args.beta, args.free_bits, train=False)
    print(f"\nTest  loss={te_loss:.4f}  recon={te_recon:.4f}  kl={te_kl:.4f}")
    print(f"Best val loss: {best_val:.4f}")

    # plots
    epochs_x = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs_x, history['train_loss'], label='train')
    axes[0].plot(epochs_x, history['val_loss'],   label='val')
    axes[0].set_title('ELBO loss')
    axes[0].set_xlabel('epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, history['train_recon'], label='train')
    axes[1].plot(epochs_x, history['val_recon'],   label='val')
    axes[1].set_title('Reconstruction MSE')
    axes[1].set_xlabel('epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    ax2 = axes[2].twinx()
    axes[2].plot(epochs_x, history['train_kl'], 'b-', label='train KL')
    axes[2].plot(epochs_x, history['val_kl'],   'b--', label='val KL')
    ax2.plot(epochs_x, history['beta'], 'r:', label='β', alpha=0.7)
    axes[2].set_title('KL divergence + β schedule')
    axes[2].set_xlabel('epoch')
    axes[2].legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'ett_loss_curve.png'), dpi=120)
    plt.close()
    print(f"Plots saved to {args.outdir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',         type=int,   default=50)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight-decay',   type=float, default=1e-4)
    p.add_argument('--batch-size',     type=int,   default=64)
    p.add_argument('--latent-dim',     type=int,   default=16)
    p.add_argument('--hidden-dim',     type=int,   default=64)
    p.add_argument('--ode-hidden',     type=int,   default=64)
    p.add_argument('--dropout',        type=float, default=0.1)
    p.add_argument('--beta',           type=float, default=0.2,
                   help='max KL weight')
    p.add_argument('--anneal-epochs',  type=int,   default=20,
                   help='ramp β from 0→beta over this many epochs')
    p.add_argument('--free-bits',      type=float, default=0.01,
                   help='min KL per latent dim')
    p.add_argument('--window',         type=int,   default=96)
    p.add_argument('--stride',         type=int,   default=48,
                   help='sliding window stride; <window gives overlapping windows')
    p.add_argument('--obs-frac',       type=float, default=0.5)
    p.add_argument('--method',         type=str,   default='rk4',
                   choices=['euler', 'rk4', 'dopri5'])
    p.add_argument('--adjoint',        action='store_true', default=True)
    p.add_argument('--no-adjoint',     dest='adjoint', action='store_false')
    p.add_argument('--rtol',           type=float, default=1e-3)
    p.add_argument('--atol',           type=float, default=1e-4)
    p.add_argument('--data-dir',       type=str,   default='data')
    p.add_argument('--outdir',         type=str,   default='results')
    p.add_argument('--dataset',        type=str,   default='ett')
    args = p.parse_args()
    train(args)
