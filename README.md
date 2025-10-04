# iLQR Solver

JAX-based implementation of iterative Linear Quadratic Regulator (iLQR) with multiple solver variants.

## Installation

### Using the Virtual Environment (Recommended)

The repo includes a `venv` with all dependencies. Activate it:
```bash
source venv/bin/activate
```

The Makefile automatically uses the venv, so you can just run:
```bash
make test
```

### Manual Installation

If setting up a new environment:
```bash
# Create and activate virtual environment (requires Python 3.10+)
python -m venv venv
source venv/bin/activate

# Install package (includes GPU support by default)
pip install -e .
```

**CUDA Compatibility:**
- The default installation includes **CUDA 12** GPU support
- If you have CUDA 11.x, edit `requirements.txt` to use `jax[cuda11_pip]`
- For CPU-only, remove `[cuda12_pip]` from the jax line in `requirements.txt`
- Check your CUDA version: `nvidia-smi`

## Quick Start

```python
from ilqr import iLQR
import jax.numpy as jnp

# Define your dynamics
def dynamics(state, control):
    # Your dynamics here
    return next_state

# Define your cost
def build_cost():
    def stage_cost(x, u):
        # Stage cost
        return cost_value

    def terminal_cost(x_T):
        # Terminal cost
        return cost_value

    def traj_cost(xs, us):
        # Full trajectory cost
        return total_cost

    return {"stage": stage_cost, "terminal": terminal_cost, "traj": traj_cost}

# Setup and solve
cost = build_cost()
dims = {"state": n, "control": m}
ilqr = iLQR(cost, dynamics, horizon, dims)
(states, controls), (success, stats) = ilqr.solve(x0, u_init)
```

## Running Tests

The package includes two test cases that demonstrate the solver:

1. **Unicycle Model** (`test_ilqr_bicycle.py`): Nonlinear dynamics test
2. **Quadratic Problem** (`test_ilqr_unit.py`): Linear-quadratic test case

### Run All Tests
```bash
make test
```

### Run Individual Tests
```bash
make test-bicycle  # Run unicycle test
make test-unit     # Run quadratic test
```

Both tests automatically generate plots in the `figures/` directory.

## Available Solvers

- **LQR**: Standard Linear Quadratic Regulator
- **iLQR**: Iterative LQR for nonlinear systems
- **iLQRAdaptive**: iLQR with variable dynamics parameters
- **iLQRAdaptiveAugmented**: iLQR with constraints via augmented Lagrangian

## Makefile Commands

```bash
make help         # Show available commands
make install      # Install package
make install-dev  # Install with dev dependencies
make test         # Run all tests and generate plots
make test-bicycle # Run bicycle test
make test-unit    # Run unit test
make plots        # Generate all plots
make clean        # Clean build files
```

## Project Structure

```
ilqr/
├── src/ilqr/          # Package source
│   ├── __init__.py
│   └── solvers.py     # All solver implementations
├── tests/             # Test files
│   ├── test_ilqr_bicycle.py
│   └── test_ilqr_unit.py
├── figures/           # Generated plots
├── pyproject.toml     # Package configuration
├── Makefile          # Build automation
└── README.md         # This file
```

## Features

- **JAX-based**: Automatic differentiation and JIT compilation
- **GPU Acceleration**: The venv includes CUDA support for GPU acceleration
- **Multiple Solvers**: LQR, iLQR, and constrained variants
- **Fast Iteration**: Simple Makefile commands for quick testing
- **Visualization**: Automatic plot generation for analysis
- **Modular Design**: Easy to extend and customize

## Dependencies

**Python:** 3.10+

**Core packages** (included `venv` versions):
- JAX 0.6.2 with CUDA 12 support (GPU-accelerated)
- NumPy 2.2.6
- Matplotlib 3.10.6
- SciPy 1.15.3

The `requirements.txt` matches these versions and includes GPU support by default.

**Note:** If JAX installation fails, check your CUDA version with `nvidia-smi` and see the installation notes above.

## Development

For fast iteration during development:

1. Make changes to `src/ilqr/solvers.py`
2. Run `make test` to see results
3. Check `figures/` for updated plots

The package is installed in editable mode, so changes are immediately reflected.
