"""Core reactive transport model classes and utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

from dataclasses import dataclass, field
from typing import Callable, Literal, Any
import operator

from reactix.species import AbstractSpecies
from reactix.reactions import KineticReaction, _make_spatial_jaxtree


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
class System:
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

        This property returns a System-like object where each field indicates
        the spatial axis along which that parameter varies. Used internally by
        jax.vmap to vectorize reaction computations across spatial cells.

        For example:
        - Scalar parameters (constant across domain) are set to None
        - Spatially varying parameters are set to the axis index (e.g., 0 for cell dimension)

        Returns
        -------
        System
            A System instance with spatial axis information instead of actual values.
        """
        return System(
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
        reactions: list[BoundaryCondition] | None = None,
        discharge: Callable[[jax.Array], jax.Array] | jax.Array,
        porosity: jax.Array,
        parameters: Any | None = None,
    ):
        """
        Build a configured reactive transport system.

        This convenience constructor validates input shapes and
        normalizes scalar inputs before creating a System instance.

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
        reactions : list[BoundaryCondition] or None, optional
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
        System
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
            discharge_fn = lambda time: discharge

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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Cells:
    """
    Represents the spatial discretization of a 1D domain into cells.

    This class defines the geometry and discretization of the 1D transport domain,
    including cell centers, nodes, areas, and distances.

    Parameters
    ----------
    n_cells : int
        Number of cells in the domain.
    centers : jax.Array
        The x coordinate of the center of each cell. Shape = (n_cells,).
    nodes : jax.Array
        The x coordinate of the points between cells, including boundaries.
        Shape = (n_cells + 1,).
    interface_area : jax.Array
        Cross-sectional area at cell interfaces. Shape = (n_cells + 1,).
    cell_area : jax.Array
        Cross-sectional area of each cell. Shape = (n_cells,).
    face_distances : jax.Array
        Distance between cell faces. Shape = (n_cells,).
    """

    n_cells: int
    # The x coordinate of the center of each cell. Shape = (n_cells,)
    centers: jax.Array
    # The x coordinate of the point between cells,
    # and the first and last boundary. Shape = (n_cells + 1,)
    nodes: jax.Array
    interface_area: jax.Array
    cell_area: jax.Array
    face_distances: jax.Array

    @property
    def _spatial_axes(self):
        return Cells(
            n_cells=None,
            centers=0,
            nodes=None,
            interface_area=None,
            cell_area=0,
            face_distances=0,
        )

    @classmethod
    def _cell_area(cls, interface_area):
        return (interface_area[:-1] + interface_area[1:]) / 2

    @classmethod
    def _face_distances(cls, nodes):
        """
        Return the distance between cell boundaries (dx for flux divergence).

        Shape = (n_cells,)
        """
        return nodes[1:] - nodes[:-1]

    @property
    def center_distances(self):
        """
        Return the distance between cell centers (dx for slope computation).

        Shape = (n_cells - 1,)
        """
        return self.centers[1:] - self.centers[:-1]

    @classmethod
    def equally_spaced(cls, length, n_cells, *, interface_area=None):
        """
        Create an equally spaced cell grid for the transport domain.

        Parameters
        ----------
        length : float
            Total length of the domain.
        n_cells : int
            Number of cells in the domain.
        interface_area : jax.Array, optional
            Cross-sectional area at cell interfaces. Default is a constant
            array of ones.

        Returns
        -------
        Cells
            Spatial discretization object for the domain.
        """
        if interface_area is None:
            interface_area = jnp.ones(n_cells + 1)
        nodes = jnp.linspace(0, length, n_cells + 1)
        face_distances = cls._face_distances(nodes)
        cell_area = cls._cell_area(interface_area)
        if (face_distances <= 0).any():
            raise ValueError("Cell nodes must be in strictly increasing order.")

        return Cells(
            n_cells=n_cells,
            nodes=nodes,
            centers=(nodes[1:] - nodes[:-1]) / 2 + nodes[:-1],
            interface_area=interface_area,
            cell_area=cell_area,
            face_distances=face_distances,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Advection:
    """
    Handles advection terms in the 1D transport equation using TVD schemes.

    This class implements numerical advection using Total Variation Diminishing (TVD)
    schemes with selectable limiters to prevent oscillations near discontinuities.

    Parameters
    ----------
    limiter_type : str, optional
        Type of slope limiter to use. Options: "minmod", "upwind", "MC".
        Default is "minmod".

    Notes
    -----
    The advection scheme reconstructs interface values using slope limiters
    to maintain monotonicity and stability in the solution.
    """

    limiter_type: str = "minmod"  # Options: "minmod", "upwind", "MC"

    @property
    def _spatial_axes(self):
        return Advection(limiter_type=None)

    @classmethod
    def build(cls, *, limiter_type):
        """
        Build an Advection instance with a specified slope limiter.

        Parameters
        ----------
        limiter_type : str
            Type of slope limiter to use. Options are "minmod", "upwind",
            or "MC".

        Returns
        -------
        Advection
            Configured advection scheme.
        """
        return cls(limiter_type=limiter_type)

    def rate(
        self, time: jax.Array, state: AbstractSpecies, system: System
    ) -> AbstractSpecies:
        """
        Compute the advection rate for mobile species.

        Parameters
        ----------
        time : jax.Array
            Current simulation time.
        state : AbstractSpecies
            Current species concentrations.
        system : System
            The reactive transport system.

        Returns
        -------
        AbstractSpecies
            Advection contribution to the rate of change.
        """
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
            discharge = system.discharge(time)
            upwind_concentration = self.choose_upwind_concentration(
                left_state, right_state, discharge
            )

            # Fluxes over the domain boundary are handled in a BoundaryCondition object,
            # so we set the concentrations to zero here.
            interface_concentrations = jnp.concatenate(
                [jnp.array([0.0]), upwind_concentration, jnp.array([0.0])]
            )

            # Compute flux divergence (flux differences over each cell)
            flux_div = (
                system.cell_velocity(time)
                * (interface_concentrations[:-1] - interface_concentrations[1:])
                / dx_face
            )

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
    def choose_upwind_concentration(left_state, right_state, discharge):
        """Select the upwind concentration based on the direction of flow."""
        return jax.lax.select(discharge >= 0, left_state, right_state)

    @staticmethod
    def minmod(a, b):
        """Return the standard minmod limited slope."""
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
    """
    Handles dispersion and diffusion terms in the transport equation.

    This class computes hydrodynamic dispersion coefficients combining
    mechanical dispersion and pore diffusion effects.

    Parameters
    ----------
    dispersivity : jax.Array
        Longitudinal dispersivity, either scalar or per cell.
    pore_diffusion : AbstractSpecies
        Pore diffusion coefficient for each species.
    """

    # Scalar or one value per cell
    # Longitudinal dispersivity
    dispersivity: jax.Array
    # pore diffusion coefficient
    pore_diffusion: AbstractSpecies

    @property
    def _spatial_axes(self):
        return Dispersion(
            dispersivity=0,
            pore_diffusion=None,
        )

    @classmethod
    def build(cls, *, cells, dispersivity, pore_diffusion):
        """
        Build a Dispersion instance and validate dispersivity shape.

        Parameters
        ----------
        cells : Cells
            Spatial discretization of the domain.
        dispersivity : jax.Array
            Longitudinal dispersivity, either scalar or per cell.
        pore_diffusion : AbstractSpecies
            Pore diffusion coefficient for each species.

        Returns
        -------
        Dispersion
            Configured dispersion scheme.

        Raises
        ------
        ValueError
            If `dispersivity` does not match the number of cells.
        """
        if dispersivity.ndim == 0:
            dispersivity = jnp.ones(cells.n_cells) * dispersivity

        if dispersivity.ndim != 1 or dispersivity.shape != (cells.n_cells,):
            raise ValueError(
                f"Invalid dispersivity shape: {dispersivity.shape}. "
                f"Expected shape {(cells.n_cells,)}"
            )
        return cls(dispersivity=dispersivity, pore_diffusion=pore_diffusion)

    def get_dispersion_coefficient(
        self, time: jax.Array, system: System
    ) -> AbstractSpecies:
        """
        Compute dispersion coefficients for each species.

        Parameters
        ----------
        time : jax.Array
            Current simulation time.
        system : System
            The reactive transport system.

        Returns
        -------
        AbstractSpecies
            Dispersion coefficients for each species.
        """
        velocity = system.cell_velocity(time)
        return jax.tree.map(
            lambda pore_diffusion: jnp.abs(velocity) * self.dispersivity
            + pore_diffusion,
            self.pore_diffusion,
        )

    def _get_interface_coefficient(
        self, time: jax.Array, system: System
    ) -> AbstractSpecies:
        """Compute dispersion coefficient times porosity for each interior interface.

        Averaging is based on continuity considerations, in analogy to Kirchhoff's
        rule of two resistors in a row.
        """
        dx = system.cells.face_distances
        A = system.cells.cell_area
        # Sum of the lengths of cell i an i+1
        total_length = dx[:-1] + dx[1:]
        # Volume of cells i and i+1 combined
        total_volume = dx[:-1] * A[:-1] + dx[1:] * A[1:]
        geometry_factor = total_length**2 / total_volume

        def inner(coeff):
            total_resistance = dx[:-1] / (
                system.porosity[:-1] * coeff[:-1] * A[:-1]
            ) + dx[1:] / (system.porosity[1:] * coeff[1:] * A[1:])
            return geometry_factor / total_resistance

        return jax.tree.map(inner, self.get_dispersion_coefficient(time, system))

    def rate(
        self, time: jax.Array, state: AbstractSpecies, system: System
    ) -> AbstractSpecies:
        """
        Compute the dispersion contribution to the species rate of change.

        Parameters
        ----------
        time : jax.Array
            Current simulation time.
        state : AbstractSpecies
            Current species concentrations.
        system : System
            The reactive transport system.

        Returns
        -------
        AbstractSpecies
            Dispersion contribution to the rate of change.
        """
        coeff = self._get_interface_coefficient(time, system)

        def flat_rate(concentration, coeff):
            cells = system.cells
            diffs = jnp.diff(concentration)
            dc_dx = diffs / cells.center_distances
            flux = -dc_dx * coeff * cells.interface_area[1:-1]
            flux_div = jnp.concatenate([jnp.array([0.0]), flux]) - jnp.concatenate(
                [flux, jnp.array([0.0])]
            )

            return flux_div / cells.face_distances / (cells.cell_area * system.porosity)

        return jax.tree.map(flat_rate, state, coeff)


@dataclass(frozen=True, kw_only=True)
class BoundaryCondition:
    """
    Base class for boundary conditions in reactive transport models.

    Boundary conditions specify fluxes or concentrations at domain boundaries.
    This abstract base class provides the interface for applying boundary
    conditions to the transport equation.

    Parameters
    ----------
    is_active : Callable[[jax.Array, System], jax.Array], optional
        Function determining if the boundary condition is active at given time.
        Default is always True.
    species_selector : Callable
        Function to select which species this boundary condition applies to.
    boundary : Literal["left", "right"]
        Which boundary this condition applies to.

    Notes
    -----
    Subclasses must implement the `compute_flux` method to define the
    specific boundary condition behavior.
    """

    is_active: Callable[[jax.Array, System], jax.Array] = lambda t, system: jnp.array(
        True
    )
    species_selector: Callable
    boundary: Literal["left", "right"]

    def apply(self, t, system, state, rate, apply_count):
        """
        Apply the boundary condition to the rate equation.

        Parameters
        ----------
        t : jax.Array
            Current time.
        system : System
            The transport system.
        state : AbstractSpecies
            Current species concentrations.
        rate : AbstractSpecies
            Current rate of change.
        apply_count : AbstractSpecies
            Counter for boundary condition applications.

        Returns
        -------
        tuple[AbstractSpecies, AbstractSpecies]
            Updated rate and apply_count.
        """
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
        """
        Compute the boundary flux for the boundary condition.

        Parameters
        ----------
        t : jax.Array
            Current time.
        system : System
            The transport system.
        state : jax.Array
            Boundary cell state.

        Returns
        -------
        jax.Array
            The computed flux at the boundary.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class FixedConcentrationBoundary(BoundaryCondition):
    """
    Boundary condition with fixed concentration at the domain boundary.

    This boundary condition maintains a specified concentration at the boundary
    by adjusting the flux. On outflow, dispersive flux is set to zero to avoid
    unphysical behavior.

    Parameters
    ----------
    fixed_concentration : float or Callable[[jax.Array], jax.Array]
        The fixed concentration value, either constant or time-dependent.
    is_active : Callable[[jax.Array, System], jax.Array], optional
        Function determining if the boundary condition is active. Inherited from BoundaryCondition.
    species_selector : Callable
        Function to select which species this applies to. Inherited from BoundaryCondition.
    boundary : Literal["left", "right"]
        Which boundary this applies to. Inherited from BoundaryCondition.

    Notes
    -----
    For inflow boundaries, both advective and dispersive fluxes contribute.
    For outflow boundaries, only advective flux is considered.
    """

    fixed_concentration: float | Callable[[jax.Array], jax.Array]

    def compute_flux(
        self, time: jax.Array, system: System, boundary_cell_state: jax.Array
    ):
        """
        Compute the advective and dispersive flux at the boundary.

        Parameters
        ----------
        time : jax.Array
            Current simulation time.
        system : System
            The reactive transport system.
        boundary_cell_state : jax.Array
            Concentration in the boundary cell.

        Returns
        -------
        jax.Array
            The total flux (advective + dispersive) at the boundary.
        """
        if isinstance(self.fixed_concentration, float):
            fixed_concentration = self.fixed_concentration
        else:
            fixed_concentration = self.fixed_concentration(time)

        discharge = system.discharge(time)
        location = 0 if self.boundary == "left" else -1

        if self.boundary == "left":
            c_interface = jax.lax.select(
                discharge >= 0,
                fixed_concentration,
                boundary_cell_state,
            )
            advection_sign = 1
        else:
            c_interface = jax.lax.select(
                discharge <= 0,
                fixed_concentration,
                boundary_cell_state,
            )
            advection_sign = -1

        advection = advection_sign * system.cell_velocity(time)[location] * c_interface

        # dispersion flow
        diff = boundary_cell_state - fixed_concentration
        dx = system.cells.face_distances[location]
        dispersion_coefficient = system.dispersion.get_dispersion_coefficient(
            time, system
        )
        interface_dispersion_coefficient = (
            self.species_selector(dispersion_coefficient)[location]
            * system.cells.interface_area[location]
            / system.cells.cell_area[location]
        )
        dispersion = -diff * interface_dispersion_coefficient / dx * 2

        if self.boundary == "left":
            dispersion = jax.lax.select(
                discharge >= 0, dispersion, jnp.zeros_like(dispersion)
            )
        else:
            dispersion = jax.lax.select(
                discharge <= 0,
                dispersion,
                jnp.zeros_like(dispersion),
            )

        return (advection + dispersion) / dx


def apply_bcs(bcs, t, system, state, rate):
    """
    Apply boundary conditions to the transport rate equation.

    Iterates through all boundary conditions and applies them to modify
    the rate of change at boundary cells. Ensures no duplicate boundary
    conditions are applied to the same location.

    Parameters
    ----------
    bcs : list[BoundaryCondition]
        List of boundary conditions to apply.
    t : jax.Array
        Current time.
    system : System
        The transport system.
    state : AbstractSpecies
        Current species concentrations.
    rate : AbstractSpecies
        Current rate of change before boundary conditions.

    Returns
    -------
    AbstractSpecies
        Updated rate of change after applying boundary conditions.

    Raises
    ------
    ValueError
        If multiple boundary conditions try to modify the same boundary location.
    """
    apply_count = jax.tree.map(jnp.zeros_like, rate)
    for bc in bcs:
        rate, apply_count = bc.apply(t, system, state, rate, apply_count)
    check_bc = lambda rate, apply_count: eqx.error_if(
        rate, (apply_count > 1).any(), "Duplicate boundary conditions."
    )
    rate = jax.tree.map(check_bc, rate, apply_count)
    return rate


def rhs(time, state, system: System):
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
    system : System
        The reactive transport system configuration.

    Returns
    -------
    AbstractSpecies
        Rate of change of species concentrations (dc/dt).
    """
    def compute_pointwise_reaction_rates(time, state, system):
        reactions = system.reactions
        if len(reactions) == 0:
            return type(state).zeros()

        rates = [reaction._eval_dcdt(time, state, system) for reaction in reactions]
        return jax.tree.map(lambda *args: sum(args), *rates)

    compute_spatial_reaction_rates = jax.vmap(
        compute_pointwise_reaction_rates,
        [
            None,
            type(state).int_zeros(),
            system._spatial_axes,
        ],
    )

    reaction_rates = compute_spatial_reaction_rates(time, state, system)

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
        system.species_is_mobile,
        system.advection.rate(time, state, system),
        system.dispersion.rate(time, state, system),
        reaction_rates,
    )
    return apply_bcs(system.bcs, time, system, state, rate)


def make_solver(
    *, t_max, t_points, rtol=1e-8, atol=1e-8, solver=None, t0=0, dt0=None, device=None
):
    """
    Create a JIT-compiled solver function for reactive transport simulations.

    This function sets up a differential equation solver using diffrax to solve
    the reactive transport system over time.

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
