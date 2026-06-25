import jax.numpy as jnp
import jax
import numpy as np
import pytest

from reactix import (
    MixedSystem,
    make_solver,
    declare_species,
    KineticReaction,
    reaction,
)

Species = declare_species(["tracer"])


def test_mixed_system_construction():
    """Test that MixedSystem can be constructed with required fields."""
    system = MixedSystem(
        discharge=lambda t: jnp.array(1.0),
        inflow_concentration=Species(tracer=jnp.array(1.0)),
        volume=jnp.array(10.0),
    )
    assert system.volume == 10.0
    assert system.parameters is None
    assert system.reactions == []


def test_mixed_rhs_no_reaction():
    """Without reactions, dc/dt should equal Q/V * (c_in - c)."""
    from reactix.systems import mixed_rhs

    Q = jnp.array(2.0)
    V = jnp.array(10.0)
    c_in = jnp.array(5.0)
    c = jnp.array(1.0)

    system = MixedSystem(
        discharge=lambda t: Q,
        inflow_concentration=Species(tracer=c_in),
        volume=V,
    )
    state = Species(tracer=c)
    dcdt = mixed_rhs(jnp.array(0.0), state, system)

    expected = Q / V * (c_in - c)
    np.testing.assert_allclose(dcdt.tracer, expected)

