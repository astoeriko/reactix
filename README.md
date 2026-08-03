![](https://raw.githubusercontent.com/astoeriko/reactix/refs/heads/develop/img/logo.svg)
# Simulating 1-D reactive transport with differentiable models in JAX

[![PyPI](https://img.shields.io/pypi/v/reactix)](https://pypi.org/project/reactix/)
[![Python](https://img.shields.io/pypi/pyversions/reactix)](https://pypi.org/project/reactix/)
[![PyTest](https://github.com/astoeriko/reactix/actions/workflows/pytest.yml/badge.svg)](https://github.com/astoeriko/reactix/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/astoeriko/reactix/branch/develop/graph/badge.svg)](https://codecov.io/gh/astoeriko/reactix)
[![License](https://img.shields.io/github/license/astoeriko/reactix)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://astoeriko.github.io/reactix/)

Reactix is a Python package for simulating reactive transport of chemical species suitable for one-dimensional systems with **advection, dispersion and kinetic reactions**.

Reaction kinetics can be defined flexibly in plain Python. The package combines a finite volume discretization in space with advanced ODE solvers from [diffrax](https://docs.kidger.site/diffrax/) for efficient time integration.

Reactix is built on the JAX ecosystem which provides automatic differentiation, making the models **fully differentiable**.
This enables the use of gradient-based methods for sensitivity analysis and uncertainty quantification. The package integrates seamlessly with [PyMC](https://docs.pymc.io/) for Bayesian modeling and parameter estimation.

## Use cases

Consider using Reactix if you…

- want to easily implement models with custom reaction kinetics,
- can reasonably simplify the system to one spatial dimension (e.g., column experiment, groundwater transport along a flow line),
- want to quantify parameter uncertainty with Bayesian methods,
- like to script your model in Python instead of using graphical interfaces or input files.

Reactix might not be the right choice if…

- your model needs to be 2-D or 3-D,
- the system involves a lot of equilibrium chemical reactions (like in PHREEQC),
- you want to couple groundwater flow and transport simulations.

## Installation

### Create an environment

To avoid possible dependency conflicts, we recommend installing Reactix in a dedicated environment. Reactix requires **Python 3.11 or later**.

You can create an environment:

**with uv:**

```bash
uv venv reactix-env
source reactix-env/bin/activate  # Linux/macOS
reactix-env\Scripts\activate     # Windows
```

**with conda:**

```bash
conda create -n reactix python=3.13
conda activate reactix
```

**or alternatively with venv:**

```bash
python -m venv reactix-env
source reactix-env/bin/activate   # Linux/macOS
reactix-env\Scripts\activate      # Windows
```

Activate the environment before installing the package.

> **Note:** If you don't have `uv` installed, you can get it with:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Dependencies

Reactix depends on [JAX](https://jax.readthedocs.io), [diffrax](https://docs.kidger.site/diffrax/), and [PyMC](https://www.pymc.io). These are installed automatically when installing Reactix via pip, uv, or conda.

> **JAX and GPU support:** GPU support integration is considered for a future release. Currently, Reactix can only be used with CPU-based JAX installations.

### Install Reactix

Below are instructions for installing the package for users and developers.

#### User installation

To install the package you can use conda, pip or uv. The package is available on PyPI and conda-forge. 

- **uv** (from PyPI):

    ```bash
    uv pip install reactix
    ```

- **conda** (from conda-forge):

    ```bash
    conda install -c conda-forge reactix
    ```

- **pip** (from PyPI):

    ```bash
    pip install reactix
    ```

#### Developer installation

If you want to contribute to the development of Reactix, clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/astoeriko/reactix.git
cd reactix
uv pip install -e ".[dev]"
#or if you also want to install the dependencies for building the documentation:
uv pip install -e ".[dev,docs]"
```

Or with pip:

```bash
pip install -e ".[dev]"
```

Make your changes and propose them as described in the [contributing guidelines](./CONTRIBUTING.md).

### Installation verification

<details>
<summary>Click to expand installation verification steps</summary>

After installing, run the following checks to confirm the environment is set up correctly.

**1. Confirm the package is importable:**

```python
import reactix
print(reactix.__version__)
```

**2. Confirm JAX is working:**

```python
import jax
import jax.numpy as jnp

x = jnp.array([1.0, 2.0, 3.0])
print(jax.devices())  # e.g. [CpuDevice(id=0)]
print(x.sum())        # 6.0
```

Note: JAX installs a CPU-only wheel by default; for GPU support follow the JAX installation guide to install the CUDA-enabled wheel that matches your setup.

**3. Run a minimal simulation:**

```python
import jax.numpy as jnp
from reactix.species import declare_species
from reactix.transport import Cells, Advection, Dispersion, System, make_solver

Species = declare_species(["tracer"])

cells = Cells.equally_spaced(length=1.0, n_cells=10)
advection = Advection.build(limiter_type="upwind")
dispersion = Dispersion.build(
    cells=cells,
    dispersivity=jnp.array(0.0),
    pore_diffusion=Species(tracer=jnp.zeros(10)),
)
system = System.build(
    cells=cells,
    advection=advection,
    dispersion=dispersion,
    bcs=[],
    species_is_mobile=Species(tracer=True),
    reactions=[],
    discharge=jnp.array(1.0),
    porosity=jnp.array(0.3),
)

y0 = Species(tracer=jnp.zeros(10).at[0].set(1.0))
solver = make_solver(t_max=1.0, t_points=jnp.linspace(0.0, 1.0, 5))
solution = solver(y0, system)
print(solution.ts)   # expected: [0.0, 0.25, 0.5, 0.75, 1.0]
```

If all three steps complete without errors, Reactix is correctly installed and ready to use.
</details>

## Usage example

This example shows how to set up a simple reactive transport model with a tracer that undergoes first-order decay.

### Step 1: Import Reactix

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from reactix import (
    System, Cells, Advection, Dispersion,
    FixedConcentrationBoundary, declare_species, make_solver
)
```

### Step 2: Declare chemical species

```python
# Define the species in your system
Species = declare_species(["tracer"])

# Specify which species are mobile (can be transported)
species_is_mobile = Species(tracer=True)
```

### Step 3: Set up the domain geometry and transport parameters

```python
# Create a 1-D domain with 100 cells over 10 length units
n_cells = 100
cells = Cells.equally_spaced(length=10.0, n_cells=n_cells)

# Define transport properties
advection = Advection.build(limiter_type="upwind")
dispersion = Dispersion.build(
    cells=cells,
    dispersivity=jnp.array(0.1),  # Longitudinal dispersivity
    pore_diffusion=Species(tracer=jnp.array(1e-9))  # Molecular diffusion
)
```

### Step 4: Set boundary conditions

```python
# Fixed concentration at inlet (left) and outlet (right)
boundary_conditions = [
    FixedConcentrationBoundary(
        boundary="left",
        species_selector=lambda s: s.tracer,
        fixed_concentration=lambda t: jnp.array(1.0)  # Constant injection
    ),
    FixedConcentrationBoundary(
        boundary="right",
        species_selector=lambda s: s.tracer,
        fixed_concentration=lambda t: jnp.array(0.0)  # Clean boundary
    )
]
```

### Step 5: Define reactions

```python
from reactix import KineticReaction, reaction

@reaction
class FirstOrderDecay(KineticReaction):
    decay_coefficient: jax.Array

    def rate(self, time, state, system):
        # Reaction rate proportional to concentration
        return self.decay_coefficient * state.tracer

    def stoichiometry(self, time, state, system):
        # One mole of tracer consumed per reaction
        return {"tracer": -1}

# Create reaction instance
decay_reaction = FirstOrderDecay(decay_coefficient=jnp.array(0.1))
```

### Step 6: Create the transport system

```python
# Define system properties
porosity = jnp.ones(n_cells) * 0.3  # 30% porosity
discharge_rate = lambda t: jnp.array(0.1)  # Constant flow rate

# Build the complete system
system = System.build(
    porosity=porosity,
    discharge=discharge_rate,
    cells=cells,
    advection=advection,
    dispersion=dispersion,
    species_is_mobile=species_is_mobile,
    bcs=boundary_conditions,
    reactions=[decay_reaction],
)
```

### Step 6: Solve the model equations

```python
# Create solver
t_max = 50
t_points = jnp.linspace(0, t_max, num=200)
solver = make_solver(t_points=t_points, t_max=t_max, rtol=1e-6, atol=1e-6)

# Set initial conditions (clean system)
initial_state = Species(tracer=jnp.zeros(n_cells))

# Solve the transport equation
solution = solver(initial_state, system)

# Plot results
plt.figure(figsize=(10, 6))
# Plot every 10th time step
plt.plot(
    cells.centers,
    solution.ys.tracer[::10, :].T,
)
plt.xlabel('Distance')
plt.ylabel('Concentration')
```

### API reference

The full API documentation can be found in the accompanying [documentation](https://astoeriko.github.io/reactix/api/).

### Jupyter notebook examples

In the [`notebooks`](./notebooks) folder there are examples that demonstrate the use of Reactix. The notebooks can also be found on the [documentation site](https://astoeriko.github.io/reactix/examples/).


## Development status

This project is still under development. While most features are implemented by now, the API may still change, and documentation is largely missing at this point. The folder `notebooks` contains Jupyter notebooks that illustrate usage.
If you are interested in trying out the package, feel free to reach out with any questions.

## Contact

For questions about this project, please open an issue or contact Anna Störiko (a.storiko@tudelft.nl).

## Acknowledgement

This project is supported by TU Delft.

## License

This project is under an MIT license. Please see [the license](./LICENSE) for details.

## Waiver

Technische Universiteit Delft hereby disclaims all copyright interest in the program “Reactix” (1-D reactive transport models with JAX) written by the Author(s). 

Stefan Aarninkhof, Dean of faculty of Civil Engineering and Geosciences.

Copyright (c) 2026 Anna Störiko.