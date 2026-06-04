import jax.numpy as jnp
import jax
import numpy as np
import pytest
import equinox as eqx
from reactix import reaction, KineticReaction, declare_species


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


def test_reaction_stoichiometry():
    Species = declare_species(["A", "B"])
    state = Species(A=jnp.array(10.0), B=jnp.array(0.0))

    decay = FirstOrderDecay(decay_coefficient=0.1)
    dcdt = decay._eval_dcdt(time=jnp.array(0.0), state=state, system=None)

    assert dcdt.A == -1
    assert dcdt.B == 1