"""Reaction framework utilities for reactive transport models."""

from __future__ import annotations

import jax
from dataclasses import dataclass, KW_ONLY
from abc import ABC, abstractmethod
from functools import wraps


@dataclass(frozen=True)
class SpatiallyVarying:
    """
    Marker for parameters that vary spatially in the domain.

    This class wraps a parameter value that has spatial variation along
    a specified axis, enabling vectorized computations across the domain.

    Parameters
    ----------
    value : jax.Array
        The parameter values, with shape including the spatial dimension.
    axis : int, optional
        The axis along which the parameter varies spatially. Default is 0.
    """

    value: jax.Array
    _: KW_ONLY
    axis: int = 0


@dataclass(frozen=True)
class SpatiallyConst:
    """
    Marker for parameters that are constant across the domain.

    This class wraps a parameter value that does not vary spatially,
    indicating it should be treated as a scalar or constant array.

    Parameters
    ----------
    value : jax.Array
        The constant parameter value.
    """

    value: jax.Array


def _make_spatial_jaxtree(cls):
    cls = dataclass(frozen=True, kw_only=True)(cls)

    @wraps(cls)
    def make_instance(**kwargs):
        class _WithSpatialAxes(cls):
            @property
            def _spatial_axes(self):
                return spatial_axes_value

        _WithSpatialAxes.__name__ = cls.__name__
        _WithSpatialAxes.__qualname__ = cls.__qualname__
        _WithSpatialAxes.__doc__ = cls.__doc__

        new_cls = _WithSpatialAxes
        new_cls = dataclass(frozen=True, kw_only=True)(new_cls)
        new_cls = jax.tree_util.register_dataclass(new_cls)

        spatial_axes = {}
        clean_kwargs = {}

        for name, val in kwargs.items():
            if isinstance(val, SpatiallyVarying):
                spatial_axes[name] = val.axis
                clean_kwargs[name] = val.value
            elif isinstance(val, SpatiallyConst):
                spatial_axes[name] = None
                clean_kwargs[name] = val.value
            else:
                spatial_axes[name] = None
                clean_kwargs[name] = val

        spatial_axes_value = new_cls(**spatial_axes)
        return new_cls(**clean_kwargs)

    return make_instance


def reaction(cls):
    """
    Decorate a class to create spatially-aware reaction classes.

    This decorator transforms a class into a dataclass that can handle
    spatially varying parameters using SpatiallyVarying and SpatiallyConst
    markers. It enables JAX tree operations and spatial axis tracking.

    Parameters
    ----------
    cls : type
        The class to be decorated, typically a reaction implementation.

    Returns
    -------
    callable
        A factory function that creates instances with spatial awareness.

    Notes
    -----
    The decorated class will automatically handle parameters marked with
    SpatiallyVarying or SpatiallyConst, enabling vectorized computations
    across spatial domains.
    """
    return _make_spatial_jaxtree(cls)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class KineticReaction(ABC):
    """
    Abstract base class for kinetic reactions in reactive transport.

    This class defines the interface for implementing kinetic reactions
    that can be coupled with transport equations. Subclasses must implement
    the rate method and define stoichiometry.

    Attributes
    ----------
    stoichiometry : dict or callable
        Reaction stoichiometry mapping species names to coefficients.
        Can be a dict for constant stoichiometry or a callable for
        time/state-dependent stoichiometry.

    Notes
    -----
    Reactions are evaluated using the _eval_dcdt method, which computes
    the rate of change of species concentrations based on the reaction rate
    and stoichiometry.
    """

    @abstractmethod
    def rate(self, time, state, system) -> jax.Array:
        """
        Compute the reaction rate.

        Parameters
        ----------
        time : jax.Array
            Current time.
        state : AbstractSpecies
            Current species concentrations.
        system : System
            The transport system.

        Returns
        -------
        jax.Array
            The reaction rate.
        """
        ...

    def _eval_dcdt(self, time, state, system):
        """
        Evaluate the rate of change of species concentrations due to this reaction.

        Parameters
        ----------
        time : jax.Array
            Current time.
        state : AbstractSpecies
            Current species concentrations.
        system : System
            The transport system.

        Returns
        -------
        AbstractSpecies
            Rate of change of species concentrations.
        """
        if callable(self.stoichiometry):
            stoichiometry = self.stoichiometry(time, state, system)
        elif isinstance(self.stoichiometry, dict):
            stoichiometry = self.stoichiometry
        else:
            raise ValueError(
                f"Invalid stoichiometry definition for {self}."
                f"Should be dict or method, but is {type(self.stoichiometry)}"
            )
        rate = self.rate(time, state, system)
        dcdt = type(state).zeros()
        for name, coeff in stoichiometry.items():
            dcdt = dcdt.add(name, coeff * rate)
        return dcdt
