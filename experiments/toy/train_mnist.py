"""
Checkpoint 3b — MNIST classification with ODE-Net.

Architecture from paper §3:
  Conv downsampler (28x28 -> 6x6x64) -> ODESolve block -> FC -> softmax

Compares ODE-Net vs ResNet vs RK-Net (direct backprop through RK4).
Memory: ODE-Net O(1), ResNet O(L), RK-Net O(L~).

Usage:
    python experiments/toy/train_mnist.py
    python experiments/toy/train_mnist.py --epochs 10 --model odenet
"""

import argparse, os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.ode_solver import odeint
from src.adjoint import odeint_adjoint


# ── ODE dynamics block ─────────────────────────────────────────────────────────

class ConvODEFunc(nn.Module):
    """Convolutional dynamics: dh/dt = f(h, t). h shape: (B, C, H, W)."""

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels + 1, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        self.nfe = 0

    def forward(self, t, h):
        self.nfe += 1
        B, C, H, W = h.shape
        t_channel = torch.ones(B, 1, H, W, device=h.device) * t
        inp = torch.cat([h, t_channel], dim=1)
        out = torch.relu(self.norm1(self.conv1(inp)))
        out = self.norm2(self.conv2(out))
        return out

    def reset_nfe(self):
        self.nfe = 0


class ODEBlock(nn.Module):
    """Wraps ODEFunc + solver. Supports adjoint or direct backprop."""

    def __init__(self, func, adjoint=True, rtol=1e-3, atol=1e-4, method='dopri5'):
        super().__init__()
        self.func = func
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.t = torch.tensor([0.0, 1.0])

    def forward(self, x):
        t = self.t.to(x.device)
        if self.adjoint:
            out = odeint_adjoint(self.func, x, t,
                                 method=self.method, rtol=self.rtol, atol=self.atol)
        else:
            out = odeint(self.func, x, t,
                         method=self.method, rtol=self.rtol, atol=self.atol)
        return out[-1]

    @property
    def nfe(self):
        return self.func.nfe

    def reset_nfe(self):
        self.func.reset_nfe()


# ── Downsampling / upsampling ──────────────────────────────────────────────────

def conv_downsampler(in_channels=1, out_channels=64):
    """28x28 -> 14x14 -> 7x7 after two stride-2 convs. Paper uses 6x6."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
        nn.GroupNorm(min(32, 32), 32),
        nn.ReLU(),
        nn.Conv2d(32, out_channels, 4, stride=2, padding=1),   # 28->14
        nn.GroupNorm(min(32, out_channels), out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1),  # 14->7
        nn.GroupNorm(min(32, out_channels), out_channels),
        nn.ReLU(),
    )


# ── Full models ────────────────────────────────────────────────────────────────

class ODENet(nn.Module):
    def __init__(self, channels=64, adjoint=True, method='dopri5'):
        super().__init__()
        self.downsampler = conv_downsampler(1, channels)
        self.ode_func    = ConvODEFunc(channels)
        self.ode_block   = ODEBlock(self.ode_func, adjoint=adjoint, method=method)
        self.classifier  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 10),
        )

    def forward(self, x):
        h = self.downsampler(x)
        h = self.ode_block(h)
        return self.classifier(h)

    @property
    def nfe(self):
        return self.ode_func.nfe

    def reset_nfe(self):
        self.ode_func.reset_nfe()


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


class ResNet(nn.Module):
    def __init__(self, channels=64, n_blocks=6):
        super().__init__()
        self.downsampler = conv_downsampler(1, channels)
        self.blocks      = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.classifier  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 10),
        )

    def forward(self, x):
        return self.classifier(self.blocks(self.downsampler(x)))


class RKNet(nn.Module):
    """Same as ODENet but backprop directly through RK4 (no adjoint)."""
    def __init__(self, channels=64, method='rk4'):
        super().__init__()
        self.downsampler = conv_downsampler(1, channels)
        self.ode_func    = ConvODEFunc(channels)
        self.ode_block   = ODEBlock(self.ode_func, adjoint=False, method=method)
        self.classifier  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 10),
        )

    def forward(self, x):
        h = self.downsampler(x)
        h = self.ode_block(h)
        return self.classifier(h)


# ── Training loop ──────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, n = 0., 0, 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if hasattr(model, 'reset_nfe'):
            model.reset_nfe()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, 100. * correct / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n       += x.size(0)
    return 100. * correct / n


def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(args.data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    models = {}
    if 'odenet' in args.models or 'all' in args.models:
        models['ODE-Net'] = ODENet(channels=args.channels, adjoint=True,  method=args.method).to(device)
    if 'rknet' in args.models or 'all' in args.models:
        models['RK-Net']  = RKNet( channels=args.channels, method='rk4').to(device)
    if 'resnet' in args.models or 'all' in args.models:
        models['ResNet']  = ResNet(channels=args.channels, n_blocks=6).to(device)

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name} | params={count_params(model):,}")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
            te_acc = eval_epoch(model, test_loader, device)
            scheduler.step()
            nfe = model.nfe if hasattr(model, 'nfe') else -1
            print(f"  epoch {epoch}/{args.epochs}  "
                  f"loss={tr_loss:.4f}  train={tr_acc:.1f}%  test={te_acc:.1f}%  "
                  f"nfe={nfe}  t={time.time()-t0:.1f}s")

        os.makedirs(args.outdir, exist_ok=True)
        torch.save({'state_dict': model.state_dict(),
                    'test_acc': te_acc,
                    'params': count_params(model)},
                   os.path.join(args.outdir, f'mnist_{name.lower().replace("-","")}.pt'))

    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=5)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--batch-size', type=int,   default=128)
    p.add_argument('--channels',   type=int,   default=64)
    p.add_argument('--method',     type=str,   default='dopri5',
                   choices=['euler', 'rk4', 'dopri5'])
    p.add_argument('--models',     type=str,   nargs='+',
                   default=['all'], choices=['all', 'odenet', 'rknet', 'resnet'])
    p.add_argument('--data-dir',   type=str,   default='data')
    p.add_argument('--outdir',     type=str,   default='results')
    args = p.parse_args()
    train(args)
