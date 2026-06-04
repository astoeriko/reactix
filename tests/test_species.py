import pytest
import jax.numpy as jnp
from reactix import declare_species


def test_declare_species_creates_correct_structure():
    """Test that declare_species creates a dataclass with correct attributes."""
    Species = declare_species(['A', 'B'])

    # Test zeros() creates correct shape
    zero_state = Species.zeros()
    assert hasattr(zero_state, 'A')
    assert hasattr(zero_state, 'B')
    assert zero_state.A.shape == ()  # scalar
    assert zero_state.B.shape == ()


def test_species_add_method():
    """Test that the add() method accumulates species concentrations."""
    Species = declare_species(['A', 'B'])
    state = Species(A=jnp.array(5.0), B=jnp.array(2.0))

    # Add to species A
    new_state = state.add('A', jnp.array(3.0))

    assert new_state.A == 8.0  # 5.0 + 3.0
    assert new_state.B == 2.0  # unchanged
    # Original should be unchanged (immutable)
    assert state.A == 5.0


def test_declare_species_with_shapes():
    """Test that declare_species handles custom shapes."""
    Species = declare_species(["tracer"], shapes={"tracer": (5, 3)})

    zero_state = Species.zeros()
    assert zero_state.tracer.shape == (5, 3)
    assert Species.int_zeros().tracer == 0
