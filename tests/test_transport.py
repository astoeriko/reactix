import jax.numpy as jnp
import numpy as np

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
    dispersion = Dispersion(
        dispersivity=jnp.array(0.1),
        pore_diffusion=Species(
            tracer=jnp.array(1e-9 * 3600 * 24),
        ),
    )
    advection = Advection(
        limiter_type="minmod"
    )
    bcs = [
        FixedConcentrationBoundary(
            is_active=lambda t, system: jnp.array(True),
            left=True,
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(10.0),
        ),
    ]
    system = System(
        porosity=jnp.array(0.3),
        velocity=lambda t: jnp.array(1 / 365),
        cells=cells,
        advection=advection,
        dispersion=dispersion,
        bcs=bcs
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