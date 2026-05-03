# Neural ODEs — From Scratch

Reimplementation of [Chen et al. 2018](https://arxiv.org/abs/1806.07366) using only PyTorch (autograd only), NumPy, and SciPy. No `torchdiffeq` or other ODE libraries.

---

## Repository Structure

```
neural_odes/
├── src/
│   ├── ode_solver.py      # Euler, RK4, dopri5 solvers
│   ├── neural_ode.py      # NeuralODE module
│   ├── adjoint.py         # Adjoint sensitivity backprop
│   └── models.py          # ODEFunc networks
├── experiments/
│   ├── toy/
│   │   ├── train_spiral.py    # Checkpoint 3a: 2D spiral fitting
│   │   └── train_mnist.py     # Checkpoint 3b: MNIST classification
│   └── realworld/
│       ├── ett_data.py        # ETT dataset loader
│       ├── latent_ode.py      # Latent ODE model
│       └── train.py           # Checkpoint 4: ETT training
├── results/               # Saved checkpoints + plots
├── logs/                  # Training logs (tee'd from make targets)
├── Makefile
└── requirements.txt
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make smoke          # sanity check imports
make toy-spiral     # Checkpoint 3a (~30s)
make toy-mnist      # Checkpoint 3b (~2.5 min on MPS)
make realworld      # Checkpoint 4  (~7.5 min on MPS)
```

Logs written to `logs/spiral.log`, `logs/mnist.log`, `logs/realworld.log`.

---

## Key Math

**Forward pass** — solve ODE initial value problem:

```
z(t₁) = z(t₀) + ∫_{t₀}^{t₁} f(z(t), t, θ) dt
```

**Adjoint sensitivity** — O(1) memory backward pass. Define adjoint `a(t) = dL/dz(t)`. Its dynamics:

```
da/dt         = -a(t)ᵀ ∂f/∂z
d(dL/dθ)/dt   = -a(t)ᵀ ∂f/∂θ
```

Augmented backward ODE state `[z(t), a(t), dL/dθ]` solved in one pass from t₁ → t₀. VJPs computed via `torch.autograd.grad` — no full Jacobian needed.

**Engineering notes:**
- BatchNorm incompatible with adjoint — use GroupNorm
- dopri5 (adaptive step) needed for stiff ODEs
- NFE (number of function evaluations) tracks solver cost per input

---

## Checkpoint 1: Problem Statement

Deep networks like ResNets update hidden state as `h_{t+1} = h_t + f(h_t, θ_t)` — Euler discretization of a continuous ODE. As layers increase and step size shrinks, the limit is `dh(t)/dt = f(h(t), t, θ)`.

**Problems solved:**
1. **Memory:** ResNet backprop stores every layer activation — O(L) for L layers. Adjoint method reduces this to O(1).
2. **Adaptive computation:** ODE solvers auto-adjust evaluation cost per input complexity.
3. **Continuous-time data:** Medical records, sensor streams, financial data arrive at irregular times. Discretizing into fixed bins loses information. Neural ODEs handle arbitrary query times natively.
4. **Invertible generative models:** Continuous normalizing flows replace O(D³) Jacobian determinant with O(D) trace via the instantaneous change-of-variables formula.

---

## Checkpoint 2: Method Description

Replace discrete layer stack with a single ODE IVP: `z(t₁) = z(t₀) + ∫f(z,t,θ)dt`, solved by a black-box ODE solver.

### Comparison

| Property | ResNet | RNN | Neural ODE |
|---|---|---|---|
| Memory (backprop) | O(L) layers | O(T) steps | **O(1)** via adjoint |
| Depth | Fixed | Fixed | Adaptive (solver chooses) |
| Irregular time | No | Hacky | **Native** |
| Invertible | No | No | Yes (reverse ODE) |

### Adjoint Method (Algorithm 1)

Backward dynamics solved from t₁ → t₀:

```
dz/dt          =  f(z, t, θ)       # reconstruct z
da/dt          = -aᵀ ∂f/∂z         # adjoint ODE
d(dL/dθ)/dt    = -aᵀ ∂f/∂θ         # param gradients
```

All three in one ODE call. VJPs `aᵀ ∂f/∂z` and `aᵀ ∂f/∂θ` computed cheaply via autograd — no Jacobian materialised.

---

## Checkpoint 3a: 2D Spiral Fitting

**Setup:** Synthetic spiral from `dz/dt = Az` (rotation matrix A), Gaussian noise σ=0.05. Neural ODE trained to recover dynamics from a single noisy trajectory.

- ODEFunc: 2-layer MLP (64 units, tanh), input `[z, t]`
- Solver: RK4, Adam lr=0.01, 1000 iterations

**Results:**
- Final loss: 0.0021 — converges to noise floor 0.0023
- Model recovers true dynamics; generalises to unseen initial conditions

Plot: `results/spiral_fit.png`

---

## Checkpoint 3b: MNIST Classification

**Architecture:** Conv downsampler (28×28 → 7×7 × 64ch) → ODESolve block → AdaptiveAvgPool → FC(10)

**Results (5 epochs, batch=256, MPS):**

| Model | Test Acc | Params | NFE |
|-------|----------|--------|-----|
| ODE-Net (adjoint) | 99.2% | 174,282 | ~30 |
| RK-Net (direct backprop) | 99.2% | 174,282 | ~30 |
| ResNet (6 blocks) | 99.4% | 543,626 | — |

- ODE-Net matches ResNet with **3.1× fewer parameters**
- Adjoint doesn't hurt convergence vs direct backprop
- GroupNorm required: adjoint reconstructs z backward, batch stats invalid

---

## Checkpoint 4: ETT Irregular Time-Series

**Dataset:** ETT (Electricity Transformer Temperature) — 4 files (ETTh1/2, ETTm1/2), 7 features, ~70,000 total timesteps. Windows of 96 steps, stride 48, 50% of steps randomly masked per window to simulate irregular sampling (2,539 train windows).

**Architecture (Latent ODE):**
- **Encoder:** GRU reads `(x_obs, Δt)` pairs in reverse → posterior `q(z₀ | x)`, outputs μ, log-σ
- **Dynamics:** Neural ODE integrates z₀ forward — `dz/dt = f(z, t, θ)`
- **Decoder:** MLP maps `z(tᵢ) → x̂(tᵢ)` at all query times
- **Loss:** ELBO = MSE reconstruction + β·KL(q(z₀) ∥ p(z₀))
- Params: 24,183 — adjoint backprop through 380 NFE per step

**Training:** β-annealing 0→0.2 over 20 epochs, free bits = 0.01 nats/dim, dropout 0.1, weight decay 1e-4.

**Results (50 epochs, MPS, ~7.5 min):**

| Metric | Value |
|--------|-------|
| Best val ELBO | 0.814 |
| Best val recon MSE | 0.655 |
| Test recon MSE | 0.675 |
| KL (stabilised) | ~0.81 nats |

**Key findings:**
- β-annealing essential: without warmup KL collapses to ~0.004 and model ignores z
- Per-file normalisation required: ETTm (15-min) and ETTh (hourly) have different scales — joint normalisation caused 3× KL spike and train/val divergence
- Adjoint method enables O(1) memory backprop through 380 function evaluations per forward pass
