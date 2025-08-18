import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx
import xarray as xr
from scipy.special import erfc

from kinetix import (
    Advection,
    Cells,
    Dispersion,
    FixedConcentrationBoundary,
    System,
    make_solver,
    declare_species,
)

TracerSpecies = declare_species(["tracer"])


def test_species_shapes():
    Species = declare_species(["tracer"], shapes={"tracer": (5, 3)})
    x = Species.zeros()
    assert x.tracer.shape == (5, 3)
    assert Species.int_zeros().tracer == 0


def test_tracer():
    cells = Cells.equally_spaced(10, 200)
    dispersion = Dispersion.build(
        cells=cells,
        dispersivity=jnp.array(0.1),
        pore_diffusion=TracerSpecies(
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
    state = TracerSpecies(
        tracer=val0,
    )
    solution = solver(state, system)
    assert solution.ys.tracer.shape == (123, 200)
    np.testing.assert_allclose(solution.ys.tracer[-1, 0], 10, rtol=1e-3)


def ogata_banks_solution(t, x, u, D, c0):
    """Evaluate analytical solution for 1-D advective-dispersive transport

    The solution is for homogeneous coefficients, steady-state conditions
    and a fixed-concentration (Dirichlet) boundary condition at the left domain
    boundary.

    References
    ----------
    Ogata, A., & Banks, R. B. (1961). A solution of the differential equation of
        longitudinal dispersion in porous media (411-A; Geological Survey
        Professional Paper). U.S. Geological Survey. https://doi.org/10.3133/pp411A
    """
    return (
        c0
        / 2
        * (
            erfc((x - u * t) / (2 * np.sqrt(D * t)))
            + np.exp(u * x / D) * erfc((x + u * t) / (2 * np.sqrt(D * t)))
        )
    )


def test_against_analytical_solution():
    """Compare numerical solution with analytical Ogata-Banks solution"""
    cells = Cells.equally_spaced(10, 200)
    dispersivity = 0.5
    pore_diffusion = 1e-9 * 3600 * 24
    dispersion = Dispersion.build(
        cells=cells,
        dispersivity=jnp.array(dispersivity),
        pore_diffusion=TracerSpecies(
            tracer=jnp.array(pore_diffusion),
        ),
    )
    advection = Advection(limiter_type="minmod")
    c0 = 10.0
    bcs = [
        FixedConcentrationBoundary(
            boundary="left",
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(c0),
        ),
        FixedConcentrationBoundary(
            boundary="right",
            species_selector=lambda s: getattr(s, "tracer"),
            fixed_concentration=lambda t: jnp.array(0.0),
        ),
    ]
    porosity = 0.3
    discharge = 1 / 365
    system = System.build(
        porosity=jnp.array(porosity),
        discharge=lambda t: jnp.array(discharge),
        cells=cells,
        advection=advection,
        dispersion=dispersion,
        bcs=bcs,
    )
    t_points = jnp.linspace(0, 500, 123)
    solver = make_solver(t_max=500, t_points=t_points, rtol=1e-3, atol=1e-3)
    val0 = jnp.zeros(cells.n_cells)
    state = TracerSpecies(
        tracer=val0,
    )
    solution = solver(state, system)
    velocity = discharge / porosity
    D = dispersivity * velocity + pore_diffusion
    analytical_solution = ogata_banks_solution(
        t=xr.DataArray(np.array(solution.ts), dims="time"),
        x=xr.DataArray(np.array(cells.centers), dims="x"),
        u=velocity,
        D=D,
        c0=c0,
    )
    # Compare breakthrough curve
    np.testing.assert_allclose(
        actual=solution.ys.tracer[:, 50],
        desired=analytical_solution.isel(x=50),
        rtol=0.02,
        atol=1e-3,
    )
    # Compare concentration profile, cutting the right edge due to boundary effects
    np.testing.assert_allclose(
        actual=solution.ys.tracer[-1, :-25],
        desired=analytical_solution.isel(time=-1, x=slice(None, -25)),
        rtol=0.02,
        atol=1e-3,
    )


def test_negative_velocity():
    cells = Cells.equally_spaced(10, 200)
    dispersion = Dispersion.build(
        cells=cells,
        dispersivity=jnp.array(0.1),
        pore_diffusion=TracerSpecies(
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
    state = TracerSpecies(
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
        pore_diffusion=TracerSpecies(
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
    state = TracerSpecies(
        tracer=val0,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="Duplicate"):
        solver(state, system)


def test_mass_conservation():
    """Test that the total mass in the system does not change with no-flux boundaries

    This also needs to hold in a system with unequl cell sizes.
    """
    areas = jnp.linspace(0, 0.1, 201) + 1
    cells = Cells.equally_spaced(10, 200, interface_area=areas)
    dispersion = Dispersion.build(
        cells=cells,
        dispersivity=jnp.array(0.1),
        pore_diffusion=TracerSpecies(
            tracer=jnp.array(1e-9 * 3600 * 24),
        ),
    )
    advection = Advection(limiter_type="minmod")
    bcs = []
    system = System.build(
        porosity=jnp.array(0.3),
        discharge=lambda t: jnp.array(1 / 100) * jnp.cos(2 * np.pi * 1 / 100 * t),
        cells=cells,
        advection=advection,
        dispersion=dispersion,
        bcs=bcs,
    )
    t_points = jnp.linspace(0, 1000, 123)
    solver = make_solver(t_max=1000, t_points=t_points, rtol=1e-3, atol=1e-3)
    val0 = jnp.zeros(cells.n_cells)
    val0 = val0.at[50:150].set(5)
    state = TracerSpecies(
        tracer=val0,
    )
    solution = solver(state, system)
    y = xr.DataArray(solution.ys.tracer, dims=("time", "x"))
    vol = xr.DataArray(
        np.array(cells.cell_area) * np.array(cells.face_distances), dims="x"
    )
    np.testing.assert_allclose((y * vol).sum("x"), (val0 * vol).sum(), rtol=1e-6)
