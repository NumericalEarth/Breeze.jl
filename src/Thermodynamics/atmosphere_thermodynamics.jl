using Adapt
using Oceananigans.Utils: prettysummary

"""
    IdealGas{FT}

A struct representing an ideal gas with molar mass and specific heat capacity.

# Fields
- `molar_mass`: Molar mass of the gas in kg/mol
- `heat_capacity`: Specific heat capacity at constant pressure in J/(kg·K)

# Examples
```jldoctest
dry_air = IdealGas(molar_mass=0.02897, heat_capacity=1005)

# output
"IdealGas{Float64}(molar_mass=0.02897, heat_capacity=1005)"
```
"""
struct IdealGas{FT}
    molar_mass :: FT
    heat_capacity :: FT # specific heat capacity at constant pressure
end

function Base.summary(gas::IdealGas{FT}) where FT
    return string("IdealGas{", FT, "}(",
                  "molar_mass=", prettysummary(gas.molar_mass), ", ",
                  "heat_capacity=", prettysummary(gas.heat_capacity), ")")
end

Base.show(io::IO, gas::IdealGas) = print(io, summary(gas))
Base.eltype(::IdealGas{FT}) where FT = FT

function Adapt.adapt_structure(to, gas::IdealGas)
    molar_mass = adapt(to, gas.molar_mass)
    heat_capacity = adapt(to, gas.heat_capacity)
    FT = typeof(molar_mass)
    return IdealGas{FT}(molar_mass, heat_capacity)
end

function IdealGas(FT = Oceananigans.defaults.FloatType;
                  molar_mass = 0.02897,
                  heat_capacity = 1005)

    return IdealGas{FT}(convert(FT, molar_mass),
                        convert(FT, heat_capacity))
end

struct CondensedPhase{FT}
    latent_heat :: FT
    heat_capacity :: FT
end

function Base.summary(ph::CondensedPhase{FT}) where FT
    return string("CondensedPhase{", FT, "}(",
                  "latent_heat=", prettysummary(ph.latent_heat), ", ",
                  "heat_capacity=", prettysummary(ph.heat_capacity), ")")
end

Base.show(io::IO, ph::CondensedPhase) = print(io, summary(ph))

Adapt.adapt_structure(to, pt::CondensedPhase) =
    CondensedPhase(adapt(to, pt.latent_heat),
                   adapt(to, pt.heat_capacity))

"""
    CondensedPhase(FT = Oceananigans.defaults.FloatType; latent_heat, heat_capacity)

Returns `CondensedPhase` with specified parameters converted to `FT`.

Two examples of `CondensedPhase` are liquid and solid.
When matter is converted from vapor to liquid, water molecules in the
gas phase cluster together and slow down to form liquid with `heat_capacity`,
The lost of molecular kinetic energy is called the `latent_heat`.

Likewise, during deposition, water molecules in the gas phase cluster into ice crystals.

Arguments
=========
- `FT`: Float type to use (defaults to Oceananigans.defaults.FloatType)
- `latent_heat`: Difference between the internal energy of the gaseous phase at the energy_reference_temperature.
- `heat_capacity`: Heat capacity of the phase of matter.
"""
function CondensedPhase(FT = Oceananigans.defaults.FloatType; latent_heat, heat_capacity)
    return CondensedPhase{FT}(convert(FT, latent_heat),
                              convert(FT, heat_capacity))
end

liquid_water(FT) = CondensedPhase(FT; latent_heat=2500800, heat_capacity=4181)
water_ice(FT)    = CondensedPhase(FT; latent_heat=2834000, heat_capacity=2108)

struct AtmosphereThermodynamics{FT, C, S}
    molar_gas_constant :: FT
    gravitational_acceleration :: FT
    energy_reference_temperature :: FT
    triple_point_temperature :: FT
    triple_point_pressure :: FT
    dry_air :: IdealGas{FT}
    vapor :: IdealGas{FT}
    liquid :: C
    solid :: S
end

Base.summary(at::AtmosphereThermodynamics{FT}) where FT = "AtmosphereThermodynamics{$FT}"

function Base.show(io::IO, at::AtmosphereThermodynamics)
    print(io, summary(at), ":", '\n',
        "├── molar_gas_constant: ", at.molar_gas_constant, "\n",
        "├── gravitational_acceleration: ", at.gravitational_acceleration, "\n",
        "├── energy_reference_temperature: ", at.energy_reference_temperature, "\n",
        "├── triple_point_temperature: ", at.triple_point_temperature, "\n",
        "├── triple_point_pressure: ", at.triple_point_pressure, "\n",
        "├── dry_air: ", at.dry_air, "\n",
        "├── vapor: ", at.vapor, "\n",
        "├── liquid: ", at.liquid, "\n",
        "└── solid: ", at.solid)
end

Base.eltype(::AtmosphereThermodynamics{FT}) where FT = FT

function Adapt.adapt_structure(to, thermo::AtmosphereThermodynamics)
    molar_gas_constant = adapt(to, thermo.molar_gas_constant)
    gravitational_acceleration = adapt(to, thermo.gravitational_acceleration)
    dry_air = adapt(to, thermo.dry_air)
    vapor = adapt(to, thermo.vapor)
    energy_reference_temperature = adapt(to, thermo.energy_reference_temperature)
    triple_point_temperature = adapt(to, thermo.triple_point_temperature)
    triple_point_pressure = adapt(to, thermo.triple_point_pressure)
    liquid = adapt(to, thermo.liquid)
    solid = adapt(to, thermo.solid)
    FT = typeof(molar_gas_constant)
    C = typeof(liquid)
    S = typeof(solid)
    return AtmosphereThermodynamics{FT, C, S}(molar_gas_constant,
                                              gravitational_acceleration,
                                              energy_reference_temperature,
                                              triple_point_temperature,
                                              triple_point_pressure,
                                              dry_air,
                                              vapor,
                                              liquid,
                                              solid)
end

"""
    AtmosphereThermodynamics(FT = Oceananigans.defaults.FloatType;
                             gravitational_acceleration = 9.81,
                             molar_gas_constant = 8.314462618,
                             energy_reference_temperature = 273.16,
                             triple_point_temperature = 273.16,
                             triple_point_pressure = 611.657,
                             dry_air_molar_mass = 0.02897,
                             dry_air_heat_capacity = 1005,
                             vapor_molar_mass = 0.018015,
                             vapor_heat_capacity = 1850,
                             liquid = liquid_water(FT),
                             solid = water_ice(FT),
                             condensed_phases = nothing)

Create `AtmosphereThermodynamics` with parameters that represent gaseous mixture of dry "air"
and vapor, as well as condensed liquid and solid phases.
The `triple_point_temperature` and `triple_point_pressure` may be combined with 
internal energy parameters for condensed phases to compute the vapor pressure
at the boundary between vapor and a homogeneous sample of the condensed phase.
The `gravitational_acceleration` parameter is included to compute reference_state
quantities associated with hydrostatic balance.

The Clausius-Clapeyron relation describes the pressure-temperature relationship during phase transitions:

    d/dT log(pⁱ⁺) = ℒⁱ / (Rⁱ * T²)

where:

- `pⁱ⁺` is the saturation vapor pressure for a transition between vapor and the `ⁱ`th phase
- `T` is temperature
- `ℒⁱ` is the latent heat of vaporization
- `Rⁱ` is the specific gas constant for the `ⁱ`th phase

For water vapor, this integrates to:

```math
    pⁱ⁺ = pᵗʳ * exp( ℒⁱ (1/Tᵗʳ - 1/T) / Rⁱ )
```

where

- ``pᵗʳ`` is the triple point pressure
- ``Tᵗʳ`` is the triple point temperature

Note: any reference values for pressure and temperature can be used in principle.
The advantage of using reference values at the triple point is that the same values
can then be used for both condensation (vapor → liquid) and deposition (vapor → ice).

"""
function AtmosphereThermodynamics(FT = Oceananigans.defaults.FloatType;
                                  molar_gas_constant = 8.314462618,
                                  gravitational_acceleration = 9.81,
                                  energy_reference_temperature = 273.16,
                                  triple_point_temperature = 273.16,
                                  triple_point_pressure = 611.657,
                                  dry_air_molar_mass = 0.02897,
                                  dry_air_heat_capacity = 1005,
                                  vapor_molar_mass = 0.018015,
                                  vapor_heat_capacity = 1850,
                                  liquid = liquid_water(FT),
                                  solid = water_ice(FT))

    dry_air = IdealGas(FT; molar_mass = dry_air_molar_mass,
                           heat_capacity = dry_air_heat_capacity)

    vapor = IdealGas(FT; molar_mass = vapor_molar_mass,
                         heat_capacity = vapor_heat_capacity)

    return AtmosphereThermodynamics(convert(FT, molar_gas_constant),
                                    convert(FT, gravitational_acceleration),
                                    convert(FT, energy_reference_temperature),
                                    convert(FT, triple_point_temperature),
                                    convert(FT, triple_point_pressure),
                                    dry_air,
                                    vapor,
                                    liquid,
                                    solid)
end

const AT = AtmosphereThermodynamics
const IG = IdealGas

@inline vapor_gas_constant(thermo::AT)   = thermo.molar_gas_constant / thermo.vapor.molar_mass
@inline dry_air_gas_constant(thermo::AT) = thermo.molar_gas_constant / thermo.dry_air.molar_mass

"""
    mixture_gas_constant(qᵈ, qᵛ, thermo)

Compute the gas constant of moist air given the specific humidity `q` and 
thermodynamic parameters `thermo`.

The mixture gas constant is calculated as a weighted average of the dry air
and water vapor gas constants:

```math
Rᵐ = qᵈ * Rᵈ + qᵛ * Rᵛ
```

where:
- `Rᵈ` is the dry air gas constant
- `Rᵛ` is the water vapor gas constant  
- `qᵈ` is the mass fraction of dry air
- `qᵛ` is the mass fraction of water vapor

# Arguments
- `qᵈ`: Mass fraction of dry air (dimensionless)
- `qᵛ`: Mass fraction of water vapor (dimensionless)
- `thermo`: `AtmosphereThermodynamics` instance containing gas constants

# Returns
- Gas constant of the moist air mixture in J/(kg·K)
"""
@inline function mixture_gas_constant(qᵈ, qᵛ, thermo::AT)
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    return qᵈ * Rᵈ + qᵛ * Rᵛ
end

"""
    mixture_heat_capacity(qᵈ, qᵛ, thermo)

Compute the heat capacity of state air given the total specific humidity q
and assuming that condensate mass ratio qᶜ ≪ q, where qℓ is the mass ratio of
liquid condensate.
"""
@inline function mixture_heat_capacity(qᵈ, qᵛ, thermo::AT)
    cᵖᵈ = thermo.dry_air.heat_capacity
    cᵖᵛ = thermo.vapor.heat_capacity
    return qᵈ * cᵖᵈ + qᵛ * cᵖᵛ
end

#####
##### state thermodynamics for a Boussinesq model
#####

# Organizing information about the state is a WIP
struct ThermodynamicState{FT}
    θ :: FT
    q :: FT
    z :: FT
end

struct ReferenceState{FT}
    p₀ :: FT # base pressure: reference pressure at z=0
    θ :: FT  # constant reference potential temperature
end

Adapt.adapt_structure(to, ref::ReferenceState) =
    ReferenceState(adapt(to, ref.p₀),
                   adapt(to, ref.θ))

function ReferenceState(FT = Oceananigans.defaults.FloatType;
                        base_pressure = 101325,
                        potential_temperature = 288)

    return ReferenceState{FT}(convert(FT, base_pressure),
                              convert(FT, potential_temperature))
end

"""
    reference_density(z, ref, thermo)

Compute the reference density associated with the reference pressure and potential temperature.
The reference density is defined as the density of dry air at the reference pressure and temperature.
"""
@inline function reference_density(z, ref, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p = reference_pressure(z, ref, thermo)
    return p / (Rᵈ * ref.θ)
end

@inline function base_density(ref, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    return ref.p₀ / (Rᵈ * ref.θ)
end

@inline function reference_specific_volume(z, ref, thermo)
    Rᵈ = dry_air_gas_constant(thermo)
    p = reference_pressure(z, ref, thermo)
    return Rᵈ * ref.θ / p
end

@inline function reference_pressure(z, ref, thermo)
    cᵖᵈ = thermo.dry_air.heat_capacity
    Rᵈ = dry_air_gas_constant(thermo)
    inv_ϰᵈ = Rᵈ / cᵖᵈ
    g = thermo.gravitational_acceleration
    return ref.p₀ * (1 - g * z / (cᵖᵈ * ref.θ))^inv_ϰᵈ
end

@inline function saturation_specific_humidity(T, z, ref::ReferenceState, thermo, condensed_phase)
    ρ = reference_density(z, ref, thermo)
    return saturation_specific_humidity(T, ρ, thermo, condensed_phase)
end

@inline function exner_function(state, ref, thermo)
    qᵛ = state.q
    qᵈ = 1 - qᵛ
    Rᵐ = mixture_gas_constant(qᵈ, qᵛ, thermo)
    cᵖᵐ = mixture_heat_capacity(qᵈ, qᵛ, thermo)
    inv_ϰᵐ = Rᵐ / cᵖᵐ
    pᵣ = reference_pressure(state.z, ref, thermo)
    p₀ = ref.base_pressure
    return (pᵣ / p₀)^inv_ϰᵐ
end

condensate_specific_humidity(T, state, ref, thermo) =
    condensate_specific_humidity(T, state.q, state.z, ref, thermo)

function condensate_specific_humidity(T, q, z, ref, thermo)
    qᵛ★ = saturation_specific_humidity(T, z, ref, thermo, thermo.liquid)
    return max(0, q - qᵛ★)
end
