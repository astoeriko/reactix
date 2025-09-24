import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
import dataclasses
import equinox as eqx


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Species:
    nitrate: jax.Array
    nitrite: jax.Array
    oxygen_liq: jax.Array
    biomass: jax.Array

    @classmethod
    def zeros(cls) -> "Species":
        return Species(
            nitrate=jnp.zeros(()),
            nitrite=jnp.zeros(()),
            oxygen_liq=jnp.zeros(()),
            biomass=jnp.zeros(()),
        )

    def add(self, name, value) -> "Species":
        return dataclasses.replace(self, **{name: value + getattr(self, name)})


"""
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Kinetic:
    def log_velocity(self, log_concentration: Species) -> (int, jnp.Array):
        pass


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MonodKinetic(Kinetic):
    def __init__(self, log_K, log_r_max, species_name):
        pass

    def log_velocity(self, log_concentration: Species) -> (int, jnp.Array):
        log_c = getattr(log_concentration, self.species_name)




@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class KineticReaction:
    stoichiometric_coefficients: Species
    base_rate: Kinetic

    def specific_rate(self, log_state: Species):
        pass


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EquilibriumReaction:
    stoichiometric_coefficients: Species
    base_rate: Kinetic

    def specific_rate(self, log_state: Species):
        pass

kinetic = MonodKinetic(log_K, log_r_max, lambda species: species.nitrate)
stoichiometry = Species.stoichiometry_from_dict(
    {
        "nitrate": -1,
        "nitrite": 1,
    }
)
nitrite_reaction = KineticReaction(
    "nitrite_reduction",
    kinetic=kinetic,
    stoichiometry=stoichiometry,
)


"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NitrateReduction:
    log_nu_max: jax.Array
    log_K: jax.Array
    log_oxygen_inhib: jax.Array

    # Assuming the state is not on log scale...
    def rate(self, state: Species):
        r_max = jnp.exp(self.log_nu_max) * state.biomass
        raw_rate = r_max * state.nitrate / (state.nitrate + jnp.exp(self.log_K))
        o2_inhibition = jnp.exp(self.log_oxygen_inhib) / (
            jnp.exp(self.log_oxygen_inhib) + state.oxygen_liq
        )

        rate = raw_rate * o2_inhibition
        return Species(
            nitrate=-rate,
            nitrite=rate,
            biomass=jnp.zeros(()),
            oxygen_liq=jnp.zeros(()),
        )

    def specific_rate(self, log_concentration: Species):
        log_r_max = self.log_nu_max + log_concentration.biomass

        log_raw_rate = log_r_max - jax.nn.softplus(
            self.log_K - log_concentration.nitrate
        )
        log_o2_inhibition = -jax.nn.softplus(
            log_concentration.oxygen_liq - self.log_oxygen_inhib
        )

        log_rate = log_raw_rate + log_o2_inhibition
        return Species(
            nitrate=-jnp.exp(log_rate - log_concentration.nitrate),
            nitrite=jnp.exp(log_rate - log_concentration.nitrite),
            biomass=jnp.zeros(()),
            oxygen_liq=jnp.zeros(()),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NitriteReduction:
    log_nu_max: jax.Array
    log_K: jax.Array
    log_oxygen_inhib: jax.Array

    # Assuming the state is not on log scale...
    def rate(self, state: Species):
        r_max = jnp.exp(self.log_nu_max) * state.biomass
        raw_rate = r_max * state.nitrate / (state.nitrate + jnp.exp(self.log_K))
        o2_inhibition = jnp.exp(self.log_oxygen_inhib) / (
            jnp.exp(self.log_oxygen_inhib) + state.oxygen_liq
        )

        rate = raw_rate * o2_inhibition
        return Species(
            nitrate=-rate,
            nitrite=rate,
            biomass=jnp.zeros(()),
            oxygen_liq=jnp.zeros(()),
        )

    def specific_rate(self, log_concentration: Species):
        log_r_max = self.log_nu_max + log_concentration.biomass

        log_raw_rate = log_r_max - jax.nn.softplus(
            self.log_K - log_concentration.nitrite
        )
        log_o2_inhibition = -jax.nn.softplus(
            log_concentration.oxygen_liq - self.log_oxygen_inhib
        )

        log_rate = log_raw_rate + log_o2_inhibition

        rate = -jnp.exp(log_rate - log_concentration.nitrite)
        return Species(
            nitrate=jnp.zeros(()),
            nitrite=rate,
            biomass=jnp.zeros(()),
            oxygen_liq=jnp.zeros(()),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AerobicRespiration:
    log_nu_max: jax.Array
    log_K: jax.Array
    log_growth_yield: jax.Array
    n_biomass: int = field(metadata=dict(static=True), default=2)
    n_oxygen: int = field(metadata=dict(static=True), default=7)

    """
    def rate(self, state: LogState):
        r_max = state.biomass * self.nu_max
        rate = r_max * state.oxygen_liq / (state.oxygen_liq + self.K)
        rates = LogState.zeros()
        rates = rates.add("oxygen_liq", -rate)

        coef = self.n_biomass / self.n_oxygen
        rates = rates.add("biomass", coef * rate * self.growth_yield)
        return rates
    """

    def specific_rate(self, log_concentration: Species):
        log_r_max = log_concentration.biomass + self.log_nu_max
        log_rate = log_r_max - jax.nn.softplus(
            self.log_K - log_concentration.oxygen_liq
        )
        specific_rates = Species.zeros()
        specific_rates = specific_rates.add(
            "oxygen_liq", -jnp.exp(log_rate - log_concentration.oxygen_liq)
        )

        coef = self.n_biomass / self.n_oxygen
        rate = jnp.exp(log_rate + self.log_growth_yield - log_concentration.biomass)
        specific_rates = specific_rates.add("biomass", coef * rate)
        return specific_rates


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class NNReaction:
    nn: eqx.Module
    reference: Species

    def specific_rate(self, log_concentration: Species, log_rates: Species):
        ravelled, unravel = jax.flatten_util.ravel_pytree(log_rates)
        ravelled = jnp.asinh(ravelled)
        predictors = jnp.tanh(self.nn(ravelled)) * 0.01 / 3600
        return unravel(predictors)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Reactions:
    reactions: list
    nn_reaction: NNReaction | None = None

    def specific_rate(self, log_concentration: Species):
        rate = dataclasses.asdict(Species.zeros())
        rate_dicts = [
            dataclasses.asdict(reaction.specific_rate(log_concentration))
            for reaction in self.reactions
        ]
        for rate_val in rate_dicts:
            for key, val in rate_val.items():
                rate[key] = rate[key] + val
        log_rates = Species(**rate)
        if self.nn_reaction is not None:
            nn_rates = self.nn_reaction.specific_rate(
                log_concentration,
                jax.tree.map(lambda x: x * 3600, log_rates),
            )
            log_rates = jax.tree.map(lambda x, y: x + y, log_rates, nn_rates)

        return log_rates
