"""Species container utilities for reactive transport models."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace, make_dataclass
from abc import ABC, abstractmethod


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AbstractSpecies(ABC):
    """
    Abstract base class for species concentration containers.

    This class defines the interface for storing and manipulating
    concentrations of chemical species in reactive transport models.
    It provides methods for creating zero-valued instances and
    adding values to specific species.

    Notes
    -----
    Subclasses should be dataclasses with species names as attributes,
    all of type jax.Array.
    """

    @classmethod
    @abstractmethod
    def zeros(cls) -> "AbstractSpecies":
        """
        Create an instance with zero concentrations for all species.

        Returns
        -------
        AbstractSpecies
            Instance with all species set to zero arrays.
        """
        ...

    @classmethod
    @abstractmethod
    def int_zeros(cls) -> "AbstractSpecies":
        """
        Create an instance with zero integers for all species.

        Returns
        -------
        AbstractSpecies
            Instance with all species set to zero integers.
        """
        ...

    def add(self, name, value) -> "AbstractSpecies":
        """
        Add a value to the concentration of a specific species.

        Parameters
        ----------
        name : str
            Name of the species to update.
        value : jax.Array
            Value to add to the species concentration.

        Returns
        -------
        AbstractSpecies
            New instance with updated concentration.
        """
        return replace(self, **{name: value + getattr(self, name)})


def declare_species(
    names: list[str], *, shapes: dict[str, tuple[int, ...]] | None = None
):
    """
    Declare chemical species for reactive transport models.

    Creates a dataclass to hold concentrations of chemical species,
    with methods for creating zero-valued instances. Species can have
    different shapes for spatially varying concentrations.

    Parameters
    ----------
    names : list[str]
        List of species names to declare.
    shapes : dict[str, tuple[int, ...]], optional
        Dictionary mapping species names to their concentration array shapes.
        Species not in this dict default to scalar shape ().

    Returns
    -------
    type
        A dataclass subclass of AbstractSpecies with the specified species
        as attributes, registered for JAX tree operations.

    Notes
    -----
    The returned class includes `zeros()` and `int_zeros()` class methods
    for creating zero-valued instances, and an `add(name, value)` method
    for updating species concentrations.

    For example, the call `declare_species(["tracer"])` is equivalent to:

    .. code-block:: python

        @jax.tree_util.register_dataclass
        @dataclass(frozen=True)
        class Species(AbstractSpecies):
            tracer: jax.Array

            @classmethod
            def zeros(cls) -> "Species":
                return Species(tracer=jnp.zeros(()))

            @classmethod
            def int_zeros(cls):
                return Species(tracer=0)
    """
    if shapes is None:
        shapes = {}

    @classmethod
    def zeros(cls):
        return Species(**{name: jnp.zeros(shapes.get(name, ())) for name in names})

    @classmethod
    def int_zeros(cls):
        return Species(**{name: 0 for name in names})

    Species = make_dataclass(
        "Species",
        [(name, jax.Array) for name in names],
        namespace={"zeros": zeros, "int_zeros": int_zeros},
        bases=(AbstractSpecies,),
        frozen=True,
    )
    return jax.tree_util.register_dataclass(Species)
