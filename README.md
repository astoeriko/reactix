# Reactix – Simulating 1-D reactive transport with differentiable models in JAX

Reactix is a Python package for simulating reactive transport of chemical species suitable for one-dimensional systems with **advection, dispersion and kinetic reactions**.
Reaction kinetics can be defined flexibly in plain Python.
The package combines a finite volume discretization in space with advanced ODE solvers from [diffrax](https://docs.kidger.site/diffrax/) for efficient time integration.

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

To avoid possible dependency conflicts, we recommend installing Reactix in a dedicated environment. Reactix requires **Python 3.11 or later**.

You can create an environment with uv or conda:

With uv:

```bash
uv venv reactix-env
source reactix-env/bin/activate  # Linux/macOS
reactix-env\Scripts\activate     # Windows
```

With conda:

```bash
conda create -n reactix python=3.11
conda activate reactix
```

Or alternatively with venv:

```bash
python -m venv reactix-env
source reactix-env/bin/activate   # Linux/macOS
reactix-env\Scripts\activate      # Windows
```

Activate the environment before installing the package.

> [!NOTE]  
> If you don't have `uv` installed, you can get it with:

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

> **JAX and GPU support:** [TODO: add info on GPU support/development/plans here]

### User installation

To install the package you can use conda, pip or uv. The package is available on PyPI and conda-forge. 

#### **uv** (from PyPI):

```bash
uv pip install reactix
```

#### **conda** (from conda-forge):

```bash
conda install -c conda-forge reactix
```

#### **pip** (from PyPI):

```bash
pip install reactix
```

### Developer installation

Clone the repository and install the package in editable mode with development dependencies:

```bash
git clone https://github.com/astoeriko/reactix.git
cd reactix
uv pip install -e ".[dev]"
```

Or with pip:

```bash
pip install -e ".[dev]"
```

Make your changes and propose them as described in the [contributing guidelines](./CONTRIBUTING.md).

## Usage example

[TODO]: Add a short code snippet and short description here.

[TODO]: Full API documentation is forthcoming.

### Jupyter notebook examples

In the [`notebooks`](./notebooks) folder there are examples that demonstrate the use of Reactix.


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