from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace, make_dataclass
from abc import ABC, abstractmethod


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AbstractSpecies(ABC):
    @classmethod
    @abstractmethod
    def zeros(cls) -> "AbstractSpecies": ...

    @classmethod
    @abstractmethod
    def int_zeros(cls) -> "AbstractSpecies": ...

    def add(self, name, value) -> "AbstractSpecies":
        return replace(self, **{name: value + getattr(self, name)})


def declare_species(
    names: list[str], *, shapes: dict[str, tuple[int, ...]] | None = None
):
    """
    Declare chemical species used in a chemical/reactive transport model.

    For example, the call `declare_species(["tracer"])` is equivalent to
    the following:

    ```
    @jax.tree_util.register_dataclass
    @dataclass(frozen=True)
    class Species(AbstractSpecies):
        tracer: Array

        @classmethod
        def zeros(cls) -> "Species":
            return Species(tracer=jnp.zeros(()))

        @classmethod
        def int_zeros(cls):
            return Species(tracer=0)
    ```
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
