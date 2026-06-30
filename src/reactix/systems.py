"""System classes for reactix models (reactive transport and mixed/batch reactors)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from dataclasses import dataclass, field
from typing import Callable, Any

from reactix.species import AbstractSpecies
from reactix.reactions import KineticReaction, _make_spatial_jaxtree
from reactix.transport import (
    Cells,
    Advection,
    Dispersion,
    BoundaryCondition,
    apply_bcs,
)


def user_system_parameters(cls):
    """
    Decorate system parameter classes with spatial awareness.

    This decorator enables classes to handle spatially varying parameters
    by transforming them into dataclasses with JAX tree operations and
    spatial axis tracking.

    Parameters
    ----------
    cls : type
        The class to be decorated.

    Returns
    -------
    callable
        A factory function that creates instances with spatial parameter support.

    Notes
    -----
    Similar to the @reaction decorator, this allows parameters to be marked
    as SpatiallyVarying or SpatiallyConst for vectorized computations.
    """
    return _make_spatial_jaxtree(cls)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class TransportSystem:
    """
    Represents a 1D reactive transport system.

    This class encapsulates all the components needed to model reactive transport
    in a 1D domain, including spatial discretization, advection, dispersion,
    species mobility, reactions, and boundary conditions.

    Parameters
    ----------
    porosity : jax.Array
        Porosity of the medium, either scalar or per cell.
    discharge : Callable[[jax.Array], jax.Array]
        Function that computes discharge given time.
    cells : Cells
        Spatial discretization of the domain.
    advection : Advection
        Advection scheme parameters.
    dispersion : Dispersion
        Dispersion scheme parameters.
    species_is_mobile : AbstractSpecies
        Species mobility information.
    reactions : list[KineticReaction], optional
        List of kinetic reactions. Default is empty list.
    bcs : list[BoundaryCondition], optional
        List of boundary conditions. Default is empty list.
    parameters : Any, optional
        Additional system parameters. Default is None. Must be created with
        @user_system_parameters for spatial awareness and JAX compatibility.
    """

    # Scalar or one value per cell
    porosity: jax.Array
    discharge: Callable[[jax.Array], jax.Array]
    cells: Cells
    advection: Advection
    dispersion: Dispersion
    species_is_mobile: AbstractSpecies
    reactions: list[KineticReaction] = field(default_factory=list)
    bcs: list[BoundaryCondition] = field(
        default_factory=list
    )  # avoid shared mutable default!
    parameters: Any | None = None

    @property
    def _spatial_axes(self):
        """
        Create a tree structure mirroring the system for spatial vectorization.

        This property returns a TransportSystem-like object where each field
        indicates the spatial axis along which that parameter varies. Used
        internally by jax.vmap to vectorize reaction computations across cells.

        For example:
        - Scalar parameters (constant across domain) are set to None
        - Spatially varying parameters are set to the axis index (e.g., 0)

        Returns
        -------
        TransportSystem
            A TransportSystem instance with spatial axis information.
        """
        return TransportSystem(
            cells=self.cells._spatial_axes,
            advection=self.advection._spatial_axes,
            dispersion=self.dispersion._spatial_axes,
            species_is_mobile=None,
            bcs=None,
            reactions=[reaction._spatial_axes for reaction in self.reactions],
            discharge=None,
            porosity=0,
            parameters=self.parameters._spatial_axes if self.parameters else None,
        )

    # TODO rename to Model, but maybe some attrisbutes in System or so?
    # TODO: retardation_factor: Species

    @classmethod
    def build(
        cls,
        *,
        cells: Cells,
        advection: Advection,
        dispersion: Dispersion,
        bcs: list[BoundaryCondition] | None = None,
        species_is_mobile: AbstractSpecies,
        reactions: list[KineticReaction] | None = None,
        discharge: Callable[[jax.Array], jax.Array] | jax.Array,
        porosity: jax.Array,
        parameters: Any | None = None,
    ):
        """
        Build a configured reactive transport system.

        This convenience constructor validates input shapes and
        normalizes scalar inputs before creating a TransportSystem instance.

        Parameters
        ----------
        cells : Cells
            Spatial discretization of the domain.
        advection : Advection
            Advection scheme parameters.
        dispersion : Dispersion
            Dispersion scheme parameters.
        bcs : list[BoundaryCondition] or None, optional
            List of boundary conditions. Default is None.
        species_is_mobile : AbstractSpecies
            Species mobility information.
        reactions : list[KineticReaction] or None, optional
            List of kinetic reactions. Default is None.
        discharge : Callable[[jax.Array], jax.Array] or jax.Array
            Discharge function or scalar discharge value.
        porosity : jax.Array
            Porosity of the medium, either scalar or per cell.
        parameters : Any or None, optional
            Additional system parameters. Default is None. Must be created with
            @user_system_parameters for spatial operations.

        Returns
        -------
        TransportSystem
            Configured reactive transport system.

        Raises
        ------
        ValueError
            If porosity has an invalid shape or a boundary condition is
            incompatible with the selected species.
        """
        if bcs is None:
            bcs = []

        if reactions is None:
            reactions = []

        if callable(discharge):
            discharge_fn = discharge
        else:
            discharge = jnp.asarray(discharge)
            assert discharge.ndim == 0
            def discharge_fn(time):
                return discharge

        if porosity.ndim == 0:
            porosity = jnp.ones(cells.n_cells) * porosity

        if porosity.ndim != 1 or porosity.shape != (cells.n_cells,):
            raise ValueError(
                f"Invalid porosity shape: {porosity.shape}. "
                f"Expected shape {(cells.n_cells,)}"
            )

        for i, bc in enumerate(bcs):
            if not bc.species_selector(species_is_mobile):
                raise ValueError(
                    "Cannot apply boundary condition to immobile species. "
                    f"Please modify or remove boundary condition {i}."
                )

        if parameters is not None and not hasattr(parameters, "_spatial_axes"):
            raise ValueError(
                "System parameters must be created with @user_system_parameters "
                "to support spatial operations. Plain objects are not supported."
            )

        return cls(
            cells=cells,
            advection=advection,
            dispersion=dispersion,
            species_is_mobile=species_is_mobile,
            bcs=bcs,
            reactions=reactions,
            discharge=discharge_fn,
            porosity=porosity,
            parameters=parameters,
        )

    def cell_velocity(self, t):
        """
        Compute the pore-water velocity in each cell.

        Parameters
        ----------
        t : jax.Array
            Current simulation time.

        Returns
        -------
        jax.Array
            Velocity in each cell.
        """
        discharge = self.discharge(t)
        return discharge / self.cells.cell_area / self.porosity

    def _rhs(self, time, state):
        """
        Compute the right-hand side of the reactive transport ODE.

        Evaluates the complete system of equations including advection,
        dispersion, and reactions for all species, then applies boundary
        conditions.

        Parameters
        ----------
        time : jax.Array
            Current simulation time.
        state : AbstractSpecies
            Current species concentrations across the domain.

        Returns
        -------
        AbstractSpecies
            Rate of change of species concentrations (dc/dt).
        """
        compute_spatial_reaction_rates = jax.vmap(
            _sum_reaction_rates_per_species,
            [
                None,
                type(state).int_zeros(),
                self._spatial_axes,
            ],
        )

        reaction_rates = compute_spatial_reaction_rates(time, state, self)

        def sum_rhs_terms(
            species_is_mobile: bool,
            advection: jax.Array,
            dispersion: jax.Array,
            reactions: jax.Array,
        ) -> jax.Array:
            if species_is_mobile:
                return advection + dispersion + reactions
            else:
                return reactions

        rate = jax.tree.map(
            sum_rhs_terms,
            self.species_is_mobile,
            self.advection.rate(time, state, self),
            self.dispersion.rate(time, state, self),
            reaction_rates,
        )
        return apply_bcs(self.bcs, time, self, state, rate)

def _sum_reaction_rates_per_species(time: jax.Array, state: AbstractSpecies, system) -> AbstractSpecies:
    reactions = system.reactions
    if len(reactions) == 0:
        return type(state).zeros()

    rates = [reaction._eval_dcdt(time, state, system) for reaction in reactions]
    return jax.tree.map(lambda *args: sum(args), *rates)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class MixedSystem:
    """
    Represents a mixed (batch) reactor system without spatial transport.

    This class models a well-mixed reactor where species concentrations are
    uniform throughout the volume. It supports kinetic reactions, inflow/outflow,
    and user-defined parameters.

    Parameters
    ----------
    reactions : list[KineticReaction], optional
        List of kinetic reactions. Default is empty list.
    discharge : Callable[[jax.Array], jax.Array] or jax.Array
        Function that computes discharge (volumetric flow rate) given time,
        or a constant discharge value.
    inflow_concentration : AbstractSpecies
        Concentration of species in the inflow.
    volume : jax.Array
        Volume of the mixed reactor.
    parameters : Any, optional
        Additional system parameters. Default is None.
    """

    reactions: list[KineticReaction] = field(default_factory=list)
    discharge: Callable[[jax.Array], jax.Array]
    inflow_concentration: AbstractSpecies
    volume: jax.Array
    parameters: Any | None = None

    def compute_inflow_outflow(self, time, state):
        """
        Compute the inflow/outflow contribution to the rate of change of concentrations.

        Parameters
        ----------
        time : jax.Array
            Current simulation time.
        state : AbstractSpecies
            Current species concentrations.

        Returns
        -------
        AbstractSpecies
            Rate of change due to inflow/outflow (dc/dt).
        """
        discharge = self.discharge(time)
        return jax.tree.map(
            lambda c_in, c: discharge / self.volume * (c_in - c),
            self.inflow_concentration,
            state,
        )


    def _rhs(self, time: jax.Array, state: AbstractSpecies) -> AbstractSpecies:
        """
        Compute the right-hand side of the mixed-system ODE.

        Evaluates the complete system of equations including inflow/outflow and reactions.

        Parameters
        ----------
        time : jax.Array
            Current simulation time.
        state : AbstractSpecies
            Current species concentrations across the domain.

        Returns
        -------
        AbstractSpecies
            Rate of change of species concentrations (dc/dt).
        """
        inflow_outflow = self.compute_inflow_outflow(time, state)
        return jax.tree.map(
            lambda io, r: io + r,
            inflow_outflow,
            _sum_reaction_rates_per_species(time, state, self),
        )


def _rhs(time: jax.Array, state: AbstractSpecies, system: TransportSystem | MixedSystem) -> AbstractSpecies:
    """
    Compute the right-hand side of the ODE for either a TransportSystem or MixedSystem.

    Parameters
    ----------
    time : jax.Array
        Current simulation time.
    state : AbstractSpecies
        Current species concentrations across the domain.
    system : TransportSystem or MixedSystem
        The system object (either reactive transport or mixed reactor).
    """
    return system._rhs(time, state)


def make_solver(
    *, t_max, t_points, rtol=1e-8, atol=1e-8, solver=None, t0=0, dt0=None, device=None
):
    """
    Create a JIT-compiled solver function for reactix simulations.

    This function sets up a differential equation solver using diffrax to solve
    either a TransportSystem or a MixedSystem over time.

    Parameters
    ----------
    t_max : float
        Maximum simulation time.
    t_points : jax.Array
        Time points at which to save the solution.
    rtol : float, optional
        Relative tolerance for the solver. Default is 1e-8.
    atol : float, optional
        Absolute tolerance for the solver. Default is 1e-8.
    solver : diffrax.AbstractSolver, optional
        ODE solver to use. Default is Tsit5.
    t0 : float, optional
        Initial time. Default is 0.
    dt0 : float, optional
        Initial time step. Default is None (auto).
    device : jax.Device, optional
        JAX device to run on. Default is CPU.

    Returns
    -------
    solve : callable
        JIT-compiled function that takes initial state and args, returns solution.

    Notes
    -----
    The returned solver function has signature solve(y0, args) where y0 is the
    initial species concentrations and args are additional arguments passed to
    the RHS function.
    """
    if solver is None:
        solver = diffrax.Tsit5()

    if device is None:
        device = jax.devices("cpu")[0]

    term = diffrax.ODETerm(_rhs)
    stepsize_controller = diffrax.PIDController(
        rtol=rtol,
        atol=atol,
    )
    t_vals = diffrax.SaveAt(ts=t_points)

    @eqx.filter_jit(device=device)
    def solve(y0: AbstractSpecies, args):
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
