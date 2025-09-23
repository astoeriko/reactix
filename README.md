# Kinetix – Simulating 1-D reactive transport with differentiable models in JAX

Kinetix is a Python package for simulating reactive transport of chemical species suitable for one-dimensional systems with **advection, dispersion and kinetic reactions**.
Reaction kinetics can be defined flexibly in plain Python.
The package combines a finite volume discretization in space with advanced ODE solvers from diffrax for efficient time integration.

Kinetix is built on the JAX ecosystem which provides automatic differentiation, making the models **fully differentiable**.
This enables the use of gradient-based methods for sensitivity analysis and uncertainty quantification. The package integrates seamlessly with PyMCfor Bayesian modeling and parameter estimation.

## Use cases

Consider using kinetix if you…

- want to easily implement models with custom reaction kinetics,
- can reasonably simplify the system to one spatial dimension (e.g., column experiment, groundwater transport along a flow line),
- want to quantify parameter uncertainty with Bayesian methods,
- like to script your model in Python instead of using graphical interfaces or input files.

Kinetix might not be the right choice if…

- your model needs to be 2-D or 3-D,
- the system involves a lot of equilibrium chemical reactions (like in PHREEQC),
- you want to couple groundwater flow and transport simulations.


## Contact

For questions about this project, please open an issue or contact Anna Störiko (a.storiko@tudelft.nl).

