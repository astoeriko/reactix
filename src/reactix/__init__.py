"""Reactix: a JAX-powered reactive-transport modeling library.

This package exposes the core transport, species, and reaction APIs
for building and solving 1D reactive-transport systems.
"""

from reactix.transport import (
    Cells,
    Advection,
    Dispersion,
    FixedConcentrationBoundary,
)
from reactix.systems import (
    TransportSystem,
    MixedSystem,
    make_solver,
    user_system_parameters,
)
from reactix.species import declare_species
from reactix.reactions import (
    SpatiallyConst,
    SpatiallyVarying,
    KineticReaction,
    reaction,
)

__all__ = [
    "TransportSystem",
    "MixedSystem",
    "Cells",
    "Advection",
    "Dispersion",
    "FixedConcentrationBoundary",
    "make_solver",
    "user_system_parameters",
    "declare_species",
    "SpatiallyConst",
    "SpatiallyVarying",
    "KineticReaction",
    "reaction",
]
