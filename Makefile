.PHONY: setup install test-solver test-toy test-mnist test-realworld clean all

PYTHON = .venv/bin/python
SRC = src
EXP = experiments
LOGS = logs

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:
	mkdir -p $(SRC) $(EXP)/toy $(EXP)/realworld results

install:
	pip install -r requirements.txt

# ── Unit tests ─────────────────────────────────────────────────────────────────

test-solver:
	$(PYTHON) -m pytest tests/test_ode_solver.py -v

test-adjoint:
	$(PYTHON) -m pytest tests/test_adjoint.py -v

test-all:
	$(PYTHON) -m pytest tests/ -v

# ── Experiments ────────────────────────────────────────────────────────────────

# Checkpoint 3a: 2D spiral — fast sanity check (~30s)
toy-spiral:
	mkdir -p $(LOGS)
	$(PYTHON) $(EXP)/toy/train_spiral.py --niters 1000 --lr 0.01 2>&1 | tee $(LOGS)/spiral.log

# Checkpoint 3b: MNIST ODE-Net
toy-mnist:
	mkdir -p $(LOGS)
	$(PYTHON) $(EXP)/toy/train_mnist.py --epochs 5 --batch-size 256 2>&1 | tee $(LOGS)/mnist.log

# Checkpoint 4: Latent ODE on ETT irregular time-series
realworld:
	mkdir -p $(LOGS)
	$(PYTHON) $(EXP)/realworld/train.py --epochs 50 --window 96 --stride 48 --latent-dim 16 --hidden-dim 64 --ode-hidden 64 --dropout 0.1 --weight-decay 1e-4 --beta 0.2 --anneal-epochs 20 --free-bits 0.01 2>&1 | tee $(LOGS)/realworld.log

# ── Quick smoke tests (no training, just forward pass) ─────────────────────────

smoke:
	$(PYTHON) -c "from src.ode_solver import odeint_rk4; print('solver OK')"
	$(PYTHON) -c "from src.neural_ode import NeuralODE; print('neural_ode OK')"
	$(PYTHON) -c "from src.adjoint import odeint_adjoint; print('adjoint OK')"

# ── Results ────────────────────────────────────────────────────────────────────

plot-spiral:
	$(PYTHON) $(EXP)/toy/plot_spiral.py --checkpoint results/spiral_latest.pt

# ── Housekeeping ───────────────────────────────────────────────────────────────

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf results/*.pt results/*.png

all: install setup smoke toy-spiral
