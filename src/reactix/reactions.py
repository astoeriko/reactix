from __future__ import annotations

import jax
from dataclasses import dataclass, KW_ONLY
from abc import ABC, abstractmethod
from functools import wraps


@dataclass(frozen=True)
class SpatiallyVarying:
    value: jax.Array
    _: KW_ONLY
    axis: int = 0


@dataclass(frozen=True)
class SpatiallyConst:
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
    return _make_spatial_jaxtree(cls)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class KineticReaction(ABC):
    @abstractmethod
    def rate(self, time, state, stytem) -> jax.Array: ...

    def _eval_dcdt(self, time, state, system):
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
