import jax.numpy as jnp
import jax
import numpy as np
import pytest
import equinox as eqx
from reactix import reaction, KineticReaction, declare_species, SpatiallyVarying


@reaction
class FirstOrderDecay(KineticReaction):
    decay_coefficient: jax.Array

    def rate(self, time, state, system):
        return self.decay_coefficient * state.A

    def stoichiometry(self, time, state, system):
        return {
            "A": -1,
            "B": 1,
        }


@reaction
class FirstOrderDecayStoichiometryDict(KineticReaction):
    decay_coefficient: jax.Array
    stoichiometry = {
            "A": -1,
            "B": 1,
        }

    def rate(self, time, state, system):
        return self.decay_coefficient * state.A


def test_reaction_stoichiometry():
    """Test that the stoichiometry is correctly applied to the reaction rate."""
    Species = declare_species(["A", "B"])
    state = Species(A=jnp.array(10.0), B=jnp.array(0.0))

    decay = FirstOrderDecay(decay_coefficient=0.1)
    dcdt = decay._eval_dcdt(time=jnp.array(0.0), state=state, system=None)

    # Rate is 0.1 * 10.0 = 1.0
    # Stoichiometry is -1.0 for A, so dA/dt = -1.0 * 1.0 = -1.0
    # Stoichiometry is +1.0 for B, so dB/dt = +1.0 * 1.0 = +1.0
    assert dcdt.A == -1
    assert dcdt.B == 1

def test_stoichiometry_dict():
    """Test that the stoichiometry can be specified as a dictionary."""
    Species = declare_species(["A", "B"])
    state = Species(A=jnp.array(10.0), B=jnp.array(0.0))

    decay = FirstOrderDecayStoichiometryDict(decay_coefficient=0.1)
    dcdt = decay._eval_dcdt(time=jnp.array(0.0), state=state, system=None)

    # Rate is 0.1 * 10.0 = 1.0
    # Stoichiometry is -1.0 for A, so dA/dt = -1.0 * 1.0 = -1.0
    # Stoichiometry is +1.0 for B, so dB/dt = +1.0 * 1.0 = +1.0
    assert dcdt.A == -1
    assert dcdt.B == 1


def test_spatially_varying_rate():
    """Test that a spatially varying decay coefficient is correctly applied to the reaction rate."""
    Species = declare_species(["A", "B"])
    state = Species(A=jnp.array(10.0), B=jnp.array(0.0))

    coefficient = SpatiallyVarying(jnp.array([0.1, 0.2, 0.3]))
    decay = FirstOrderDecay(decay_coefficient=coefficient)
    dcdt = decay._eval_dcdt(time=jnp.array(0.0), state=state, system=None)

    # Rate is [0.1, 0.2, 0.3] * 10.0 = [1.0, 2.0, 3.0]
    assert np.allclose(dcdt.A, jnp.array([-1.0, -2.0, -3.0]))
