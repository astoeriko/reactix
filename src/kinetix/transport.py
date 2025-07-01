from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from dataclasses import dataclass, field, replace
from typing import Callable, Literal


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Species:
    tracer: jax.Array

    @classmethod
    def zeros(cls) -> "Species":
        return Species(tracer=jnp.zeros(()))

    def add(self, name, value) -> "Species":
        return replace(self, **{name: value + getattr(self, name)})


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class System:
    porosity: jax.Array
    velocity: Callable[[jax.Array], jax.Array]
    cells: Cells
    advection: Advection
    dispersion: Dispersion
    reactions: list[BoundaryCondition] = field(
        default_factory=list
    )
    bcs: list[BoundaryCondition] = field(
        default_factory=list
    )  # avoid shared mutable default!

    # TODO get advection and dispersion
    # TODO add reactions
    # TODO rename to Model, but maybe some attrisbutes in System or so?
    # TODO porosity depending in position?

    # retardation_factor: Species


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Cells:
    n_cells: int
    # The x coordinate of the center of each cell. Shape = (n_cells,)
    centers: jax.Array
    # The x coordinate of the point between cells,
    # and the first and last boundary. Shape = (n_cells + 1,)
    nodes: jax.Array

    # TODO add area for each node

    def cell_length(self) -> jax.Array:
        return self.nodes[1:] - self.nodes[:-1]

    @property
    def face_distances(self):
        """
        Returns the distance between cell boundaries (dx for flux divergence).
        Shape = (n_cells,)
        """
        return self.nodes[1:] - self.nodes[:-1]

    @property
    def center_distances(self):
        """
        Returns the distance between cell centers (dx for slope computation).
        Shape = (n_cells - 1,)
        """
        return self.centers[1:] - self.centers[:-1]

    @classmethod
    def equally_spaced(cls, length, n_cells):
        nodes = jnp.linspace(0, length, n_cells + 1)
        return Cells(
            n_cells=n_cells,
            nodes=nodes,
            centers=(nodes[1:] - nodes[:-1]) / 2 + nodes[:-1],
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Advection:
    limiter_type: str = "minmod"  # Options: "minmod", "upwind", "MC"

    def rate(self, time: jax.Array, state: Species, system: System) -> Species:
        def flat_rate(concentration):
            cells = system.cells
            dx_center = cells.center_distances  # (n_cells - 1,)
            dx_face = cells.face_distances  # (n_cells,)

            # Compute slope with chosen limiter
            slope = self.compute_slope(concentration, dx_center)

            # Compute left and right states at interfaces
            QL = concentration - 0.5 * slope * dx_face  # left state for cell i
            QR = concentration + 0.5 * slope * dx_face  # right state for cell i

            # Internal faces (i = 1 to n_cells - 1)
            left_state = QR[:-1]  # right side of cell i-1
            right_state = QL[1:]  # left side of cell i
            velocity = system.velocity(time)
            internal_flux = self.upwind_flux(left_state, right_state, velocity)

            # Handle boundary fluxes using boundary condition object
            left_flux = jnp.array(0.0)
            right_flux = jnp.array(0.0)

            full_flux = jnp.concatenate(
                [left_flux[None], internal_flux, right_flux[None]]
            )

            # Compute flux divergence (flux differences over each cell)
            # TODO use area and porocity
            # TODO better names
            flux_div = (full_flux[:-1] - full_flux[1:]) / dx_face

            return flux_div

        return jax.tree.map(flat_rate, state)

    def compute_slope(self, concentration, dx_center):
        """Compute limited slope with padding at boundaries."""
        delta = concentration[1:] - concentration[:-1]
        raw_slope = delta / dx_center

        if self.limiter_type == "minmod":
            a = raw_slope[:-1]
            b = raw_slope[1:]
            limited = self.minmod(a, b)
            slope = jnp.concatenate([jnp.array([0.0]), limited, jnp.array([0.0])])
            return slope

        elif self.limiter_type == "upwind":
            return jnp.zeros_like(concentration)

        elif self.limiter_type == "MC":
            a = raw_slope[:-1]
            b = raw_slope[1:]
            limited = self.mc_limiter(a, b)
            slope = jnp.concatenate([jnp.array([0.0]), limited, jnp.array([0.0])])
            return slope

        else:
            raise ValueError(f"Unknown limiter type: {self.limiter_type}")

    @staticmethod
    def upwind_flux(left_state, right_state, velocity):
        """Simple upwind flux function."""
        # TODO porocity?
        return jnp.where(velocity >= 0, velocity * left_state, velocity * right_state)

    @staticmethod
    def minmod(a, b):
        """Standard minmod limiter."""
        cond = jnp.sign(a) == jnp.sign(b)
        s = jnp.sign(a)
        return jnp.where(cond, s * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)

    @staticmethod
    def minmod3(a, b, c):
        """Minmod of three values, for MC limiter."""
        cond1 = (jnp.sign(a) == jnp.sign(b)) & (jnp.sign(b) == jnp.sign(c))
        s = jnp.sign(a)
        return jnp.where(
            cond1, s * jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c))), 0.0
        )

    def mc_limiter(self, a, b):
        """Monotonized central (MC) limiter."""
        return self.minmod3(2 * a, 0.5 * (a + b), 2 * b)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Dispersion:
    # Longitudinal dispersivity
    dispersivity: jax.Array
    # pore diffusion coefficient
    pore_diffusion: Species

    def get_coefficient(self, time: jax.Array, system: System):
        velocity = system.velocity(time)
        return jax.tree.map(
            lambda pore_diffusion: jnp.abs(velocity) * self.dispersivity
            + pore_diffusion,
            self.pore_diffusion,
        )

    def rate(self, time: jax.Array, state: Species, system: System) -> Species:
        coeff = self.get_coefficient(time, system)

        def flat_rate(concentration, coeff):
            cells = system.cells
            # TODO add areas and porosity

            diffs = jnp.diff(concentration)
            dc_dx = diffs / (cells.centers[1:] - cells.centers[:-1])
            dx = cells.nodes[1:] - cells.nodes[:-1]
            flux = -dc_dx * coeff

            return (
                jnp.concatenate(
                    [
                        jnp.array(0.0)[None],
                        flux,
                    ]
                )
                - jnp.concatenate(
                    [
                        flux,
                        jnp.array(0.0)[None],
                    ]
                )
            ) / dx

        return jax.tree.map(flat_rate, state, coeff)


@dataclass(frozen=True, kw_only=True)
class BoundaryCondition:
    is_active: Callable[[jax.Array, System], jax.Array] = lambda t, system: jnp.array(True)
    species_selector: Callable
    boundary: Literal["left", "right"]

    def apply(self, t, system, state, rate, apply_count):
        species_rate = self.species_selector(rate)
        species_state = self.species_selector(state)
        species_apply_count = self.species_selector(apply_count)

        location = 0 if self.boundary == "left" else -1
        flux = self.compute_flux(t, system, species_state[location])

        active_val = species_rate.at[location].add(flux)
        inactive_val = species_rate

        active_apply_count = species_apply_count.at[location].add(1)
        inactive_apply_count = species_apply_count

        is_active = self.is_active(t, system)
        new_val = jax.lax.select(is_active, active_val, inactive_val)
        new_apply_count = jax.lax.select(
            is_active, active_apply_count, inactive_apply_count
        )

        new_rate = eqx.tree_at(self.species_selector, rate, new_val)
        new_apply_count = eqx.tree_at(
            self.species_selector, apply_count, new_apply_count
        )

        return new_rate, new_apply_count

    def compute_flux(self, t, system, state):
        raise NotImplementedError()


@dataclass(frozen=True)
class FixedConcentrationBoundary(BoundaryCondition):
    """Boundary condition with fixed concentration on the boundary

    On outflow, the dispersive flux is set to zero, only advection out
    of the domain.
    """

    fixed_concentration: float | Callable[[jax.Array], jax.Array]

    def compute_flux(
        self, time: jax.Array, system: System, boundary_cell_state: jax.Array
    ):
        # TODO use area and porosity
        if isinstance(self.fixed_concentration, float):
            fixed_concentration = self.fixed_concentration
        else:
            fixed_concentration = self.fixed_concentration(time)

        velocity = system.velocity(time)

        if self.boundary == "left":
            c_interface = jax.lax.select(
                velocity >= 0,
                fixed_concentration,
                boundary_cell_state,
            )
            advection_sign = 1
        else:
            c_interface = jax.lax.select(
                velocity <= 0,
                fixed_concentration,
                boundary_cell_state,
            )
            advection_sign = -1

        advection = advection_sign * velocity * c_interface

        # dispersion flow
        diff = boundary_cell_state - fixed_concentration
        location = 0 if self.boundary == "left" else -1
        dx = system.cells.face_distances[location]
        dispersion_coefficient = system.dispersion.get_coefficient(time, system)
        dispersion = -diff * self.species_selector(dispersion_coefficient) / dx * 2

        if self.boundary == "left":
            dispersion = jax.lax.select(
                velocity >= 0, dispersion, jnp.zeros_like(dispersion)
            )
        else:
            dispersion = jax.lax.select(
                velocity <= 0,
                dispersion,
                jnp.zeros_like(dispersion),
            )

        return (advection + dispersion) / dx


def apply_bcs(bcs, t, system, state, rate):
    apply_count = jax.tree.map(jnp.zeros_like, rate)
    for bc in bcs:
        rate, apply_count = bc.apply(t, system, state, rate, apply_count)
    check_bc = lambda rate, apply_count: eqx.error_if(
        rate, (apply_count > 1).any(), "Duplicate boundary conditions."
    )
    rate = jax.tree.map(check_bc, rate, apply_count)
    return rate


def rhs(time, state, system: System):
    vmaped_rates = [
        jax.vmap(reaction.rate, [None, type(state).int_zeros(), None])(time, state, system)
        for reaction in system.reactions
    ]
    rate = jax.tree.map(
        lambda *args: sum(args),
        system.advection.rate(time, state, system),
        system.dispersion.rate(time, state, system),
        *vmaped_rates,
    )
    return apply_bcs(system.bcs, time, system, state, rate)


def make_solver(
    *, t_max, t_points, rtol=1e-8, atol=1e-8, solver=None, t0=0, dt0=None, device=None
):
    if solver is None:
        # solver = diffrax.Dopri5()
        solver = diffrax.Tsit5()
        # root_finder = optimistix.Dogleg(rtol=1e-9, atol=1e-9, norm=optimistix.two_norm)
        # solver = diffrax.Kvaerno3(root_find_max_steps=10, root_finder=root_finder)
        # solver = diffrax.Kvaerno3()

    if device is None:
        device = jax.devices("cpu")[0]

    term = diffrax.ODETerm(rhs)
    stepsize_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
        # dtmax=
        # norm=optimistix.two_norm,
    )
    t_vals = diffrax.SaveAt(ts=t_points)

    @eqx.filter_jit(device=device)
    def solve(y0: Species, args):
        result = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t_max,
            dt0=dt0,
            y0=y0,
            saveat=t_vals,
            args=args,
            stepsize_controller=stepsize_controller,
            max_steps=1024 * 32 * 64,
        )
        return result

    return solve
