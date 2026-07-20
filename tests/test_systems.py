import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactix import (
    KineticReaction,
    MixedSystem,
    declare_species,
    make_solver,
    reaction,
)

Species = declare_species(["tracer"])


@reaction
class FirstOrderDecay(KineticReaction):
    """Simple first-order decay reaction used by mixed reactor tests."""

    decay_coefficient: jax.Array

    def rate(self, time, state, system):
        return self.decay_coefficient * state.tracer

    def stoichiometry(self, time, state, system):
        return {"tracer": -1}


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
    dcdt = system._rhs(jnp.array(0.0), state)

    expected = Q / V * (c_in - c)
    np.testing.assert_allclose(dcdt.tracer, expected)


def test_compute_inflow_outflow_with_missing_discharge_is_zero():
    """If discharge is omitted, the inflow/outflow term should vanish."""

    system = MixedSystem(
        discharge=None,
        inflow_concentration=Species(tracer=jnp.array(1.0)),
        volume=jnp.array(10.0),
    )
    state = Species(tracer=jnp.array(0.3))

    inflow_outflow = system.compute_inflow_outflow(jnp.array(0.0), state)

    np.testing.assert_allclose(inflow_outflow.tracer, jnp.zeros_like(state.tracer))


def test_compute_inflow_outflow_with_missing_inflow_concentration_is_zero():
    """If inflow concentration is omitted, the inflow/outflow term should vanish."""

    system = MixedSystem(
        discharge=lambda t: jnp.array(0.5),
        inflow_concentration=None,
        volume=jnp.array(10.0),
    )
    state = Species(tracer=jnp.array(0.3))

    inflow_outflow = system.compute_inflow_outflow(jnp.array(0.0), state)

    np.testing.assert_allclose(inflow_outflow.tracer, jnp.zeros_like(state.tracer))


@pytest.mark.integration
def test_mixed_system_matches_analytical_solution():
    """Verify a simulated mixed reactor matches the closed-form first-order solution."""

    Q = jnp.array(0.5)
    V = jnp.array(10.0)
    k = jnp.array(0.02)
    c_in = jnp.array(1.0)
    c0 = jnp.array(0.0)

    reactions = [FirstOrderDecay(decay_coefficient=k)]
    system = MixedSystem.build(
        reactions=reactions,
        discharge=Q,
        inflow_concentration=Species(tracer=c_in),
        volume=V,
    )

    t_points = jnp.linspace(0.0, 200.0, 300)
    solver = make_solver(t_max=200.0, t_points=t_points, rtol=1e-6, atol=1e-10)

    state0 = Species(tracer=jnp.zeros(()))
    solution = solver(state0, system)

    c_steady = Q / (Q + k * V) * c_in
    c_analytical = c_steady + (c0 - c_steady) * np.exp(-(Q / V + k) * np.asarray(t_points))

    np.testing.assert_allclose(solution.ys.tracer, np.asarray(c_analytical), rtol=5e-4, atol=1e-6)
