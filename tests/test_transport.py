import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from kinetix.transport import (
    Advection,
    Cells,
    Dispersion,
    FixedConcentrationBoundary,
    Species,
    System,
    make_solver,
)


def test_tracer():
    cells = Cells.equally_spaced(10, 200)
    dispersion = Dispersion.build(
        cells=cells,
        dispersivity=jnp.array(0.1),
        pore_diffusion=Species(
            tracer=jnp.array(1e-9 * 3600 * 24),
        ),
    )
    advection = Advection(limiter_type="minmod")
    bcs = [
        FixedConcentrationBoundary(
            boundary="left",
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(10.0),
        ),
    ]
    system = System.build(
        porosity=jnp.array(0.3),
        discharge=lambda t: jnp.array(1 / 365),
        cells=cells,
        advection=advection,
        dispersion=dispersion,
        bcs=bcs,
    )
    t_points = jnp.linspace(0, 1000, 123)
    solver = make_solver(t_max=1000, t_points=t_points, rtol=1e-3, atol=1e-3)
    val0 = jnp.zeros(cells.n_cells)
    state = Species(
        tracer=val0,
    )
    solution = solver(state, system)
    assert solution.ys.tracer.shape == (123, 200)
    np.testing.assert_allclose(solution.ys.tracer[-1, 0], 10, rtol=1e-3)


def test_negative_velocity():
    cells = Cells.equally_spaced(10, 200)
    dispersion = Dispersion.build(
        cells=cells,
        dispersivity=jnp.array(0.1),
        pore_diffusion=Species(
            tracer=jnp.array(1e-9 * 3600 * 24),
        ),
    )
    advection = Advection(limiter_type="minmod")
    bcs = [
        FixedConcentrationBoundary(
            boundary="right",
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(10.0),
        ),
        FixedConcentrationBoundary(
            boundary="left",
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(5.0),
        ),
    ]
    system = System.build(
        porosity=jnp.array(0.3),
        discharge=lambda t: jnp.array(-1 / 365),
        cells=cells,
        advection=advection,
        dispersion=dispersion,
        bcs=bcs,
    )
    t_points = jnp.linspace(0, 1000, 123)
    solver = make_solver(t_max=1000, t_points=t_points, rtol=1e-3, atol=1e-3)
    val0 = jnp.zeros(cells.n_cells)
    state = Species(
        tracer=val0,
    )
    solution = solver(state, system)
    assert solution.ys.tracer.shape == (123, 200)
    np.testing.assert_allclose(solution.ys.tracer[-1, -1], 10, rtol=1e-3)


def test_duplicate_bondaries():
    cells = Cells.equally_spaced(10, 200)
    dispersion = Dispersion.build(
        cells=cells,
        dispersivity=jnp.array(0.1),
        pore_diffusion=Species(
            tracer=jnp.array(1e-9 * 3600 * 24),
        ),
    )
    advection = Advection(limiter_type="minmod")
    bcs = [
        FixedConcentrationBoundary(
            is_active=lambda t, system: t < 1000,
            boundary="left",
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(10.0),
        ),
        FixedConcentrationBoundary(
            is_active=lambda t, system: t >= 500,
            boundary="left",
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(3.0),
        ),
    ]
    system = System.build(
        porosity=jnp.array(0.3),
        discharge=lambda t: jnp.array(1 / 365),
        cells=cells,
        advection=advection,
        dispersion=dispersion,
        bcs=bcs,
    )
    t_points = jnp.linspace(0, 1000, 123)
    solver = make_solver(t_max=1000, t_points=t_points, rtol=1e-3, atol=1e-3)
    val0 = jnp.zeros(cells.n_cells)
    state = Species(
        tracer=val0,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="Duplicate"):
        solver(state, system)
