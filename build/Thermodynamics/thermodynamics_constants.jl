using Adapt: Adapt, adapt
using Oceananigans.Grids: prettysummary

"""
$(TYPEDEF)

A struct representing an ideal gas with molar mass and specific heat capacity.

# Fields
- `molar_mass`: Molar mass of the gas in kg/mol
- `heat_capacity`: Specific heat capacity at constant pressure in J/(kg·K)

# Examples
```jldoctest
using Breeze
dry_air = IdealGas(molar_mass=0.02897, heat_capacity=1005)

# output
IdealGas{Float64}(molar_mass=0.02897, heat_capacity=1005.0)
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
    reference_latent_heat :: FT
    heat_capacity :: FT
end

function Base.summary(ph::CondensedPhase{FT}) where FT
    return string("CondensedPhase{", FT, "}(",
                  "reference_latent_heat=", prettysummary(ph.reference_latent_heat), ", ",
                  "heat_capacity=", prettysummary(ph.heat_capacity), ")")
end

Base.show(io::IO, ph::CondensedPhase) = print(io, summary(ph))

Adapt.adapt_structure(to, pt::CondensedPhase) =
    CondensedPhase(adapt(to, pt.reference_latent_heat),
                   adapt(to, pt.heat_capacity))

"""
$(TYPEDSIGNATURES)

Return `CondensedPhase` with specified parameters converted to `FT`.

Two examples of `CondensedPhase` are liquid and ice.
When matter is converted from vapor to liquid, water molecules in the
gas phase cluster together and slow down to form liquid with `heat_capacity`,
The lost of molecular kinetic energy is called the `reference_latent_heat`.

Likewise, during deposition, water molecules in the gas phase cluster into ice crystals.

Arguments
=========
- `FT`: Float type to use (defaults to `Oceananigans.defaults.FloatType`)
- `reference_latent_heat`: Difference between the internal energy of the gaseous phase at
  the `energy_reference_temperature`.
- `heat_capacity`: Heat capacity of the phase of matter.
"""
function CondensedPhase(FT = Oceananigans.defaults.FloatType; reference_latent_heat, heat_capacity)
    return CondensedPhase{FT}(convert(FT, reference_latent_heat),
                              convert(FT, heat_capacity))
end

liquid_water(FT) = CondensedPhase(FT; reference_latent_heat=2500800, heat_capacity=4181)
water_ice(FT)    = CondensedPhase(FT; reference_latent_heat=2834000, heat_capacity=2108)

struct ThermodynamicConstants{FT, C, I}
    molar_gas_constant :: FT
    gravitational_acceleration :: FT
    energy_reference_temperature :: FT
    triple_point_temperature :: FT
    triple_point_pressure :: FT
    dry_air :: IdealGas{FT}
    vapor :: IdealGas{FT}
    liquid :: C
    ice :: I
end

Base.summary(at::ThermodynamicConstants{FT}) where FT = "ThermodynamicConstants{$FT}"

function Base.show(io::IO, at::ThermodynamicConstants)
    print(io, summary(at), ":", '\n',
        "├── molar_gas_constant: ", at.molar_gas_constant, "\n",
        "├── gravitational_acceleration: ", at.gravitational_acceleration, "\n",
        "├── energy_reference_temperature: ", at.energy_reference_temperature, "\n",
        "├── triple_point_temperature: ", at.triple_point_temperature, "\n",
        "├── triple_point_pressure: ", at.triple_point_pressure, "\n",
        "├── dry_air: ", at.dry_air, "\n",
        "├── vapor: ", at.vapor, "\n",
        "├── liquid: ", at.liquid, "\n",
        "└── ice: ", at.ice)
end

Base.eltype(::ThermodynamicConstants{FT}) where FT = FT

function Adapt.adapt_structure(to, thermo::ThermodynamicConstants)
    molar_gas_constant = adapt(to, thermo.molar_gas_constant)
    gravitational_acceleration = adapt(to, thermo.gravitational_acceleration)
    dry_air = adapt(to, thermo.dry_air)
    vapor = adapt(to, thermo.vapor)
    energy_reference_temperature = adapt(to, thermo.energy_reference_temperature)
    triple_point_temperature = adapt(to, thermo.triple_point_temperature)
    triple_point_pressure = adapt(to, thermo.triple_point_pressure)
    liquid = adapt(to, thermo.liquid)
    ice = adapt(to, thermo.ice)
    FT = typeof(molar_gas_constant)
    C = typeof(liquid)
    I = typeof(ice)
    return ThermodynamicConstants{FT, C, I}(molar_gas_constant,
                                            gravitational_acceleration,
                                            energy_reference_temperature,
                                            triple_point_temperature,
                                            triple_point_pressure,
                                            dry_air,
                                            vapor,
                                            liquid,
                                            ice)
end

"""
$(TYPEDSIGNATURES)

Return `ThermodynamicConstants` with parameters that represent gaseous mixture of dry "air"
and vapor, as well as condensed liquid and ice phases.
The `triple_point_temperature` and `triple_point_pressure` may be combined with
internal energy parameters for condensed phases to compute the vapor pressure
at the boundary between vapor and a homogeneous sample of the condensed phase.
The `gravitational_acceleration` parameter is included to compute [`ReferenceState`](@ref)
quantities associated with hydrostatic balance.
"""
function ThermodynamicConstants(FT = Oceananigans.defaults.FloatType;
                                molar_gas_constant = 8.314462618,
                                gravitational_acceleration = 9.81,
                                energy_reference_temperature = 273.15,
                                triple_point_temperature = 273.16,
                                triple_point_pressure = 611.657,
                                dry_air_molar_mass = 0.02897,
                                dry_air_heat_capacity = 1005,
                                vapor_molar_mass = 0.018015,
                                vapor_heat_capacity = 1850,
                                liquid = liquid_water(FT),
                                ice = water_ice(FT))

    dry_air = IdealGas(FT; molar_mass = dry_air_molar_mass,
                           heat_capacity = dry_air_heat_capacity)

    vapor = IdealGas(FT; molar_mass = vapor_molar_mass,
                         heat_capacity = vapor_heat_capacity)

    return ThermodynamicConstants(convert(FT, molar_gas_constant),
                                  convert(FT, gravitational_acceleration),
                                  convert(FT, energy_reference_temperature),
                                  convert(FT, triple_point_temperature),
                                  convert(FT, triple_point_pressure),
                                  dry_air,
                                  vapor,
                                  liquid,
                                  ice)
end

const TC = ThermodynamicConstants

@inline vapor_gas_constant(thermo::TC)   = thermo.molar_gas_constant / thermo.vapor.molar_mass
@inline dry_air_gas_constant(thermo::TC) = thermo.molar_gas_constant / thermo.dry_air.molar_mass

#####
##### Mixtures of dry air with vapor, liquid, and ice
#####

"""
$(TYPEDEF)

A struct representing the moisture mass fractions of a moist air parcel.

# Fields
- `vapor`: the mass fraction of vapor
- `liquid`: the mass fraction of liquid
- `ice`: the mass fraction of ice
"""
struct MoistureMassFractions{FT}
    vapor :: FT
    liquid :: FT
    ice :: FT
end

@inline MoistureMassFractions(vapor::FT) where FT = MoistureMassFractions(vapor, zero(vapor), zero(vapor))
@inline MoistureMassFractions(vapor::FT, liquid::FT) where FT = MoistureMassFractions(vapor, liquid, zero(vapor))

const MMF = MoistureMassFractions
Base.zero(::Type{MMF{FT}}) where FT = MoistureMassFractions(zero(FT), zero(FT), zero(FT))

function Base.summary(q::MoistureMassFractions{FT}) where FT
    return string("MoistureMassFractions{$FT}(vapor=", prettysummary(q.vapor),
                  ", liquid=", prettysummary(q.liquid), ", ice=", prettysummary(q.ice), ")")
end

function Base.show(io::IO, q::MoistureMassFractions{FT}) where FT
    println(io, "MoistureMassFractions{$FT}: \n",
                "├── vapor:  ", prettysummary(q.vapor), "\n",
                "├── liquid: ", prettysummary(q.liquid), "\n",
                "└── ice:    ", prettysummary(q.ice))
end

@inline total_specific_moisture(q::MMF) = q.vapor + q.liquid + q.ice
@inline dry_air_mass_fraction(q::MMF) = 1 - total_specific_moisture(q)

"""
$(TYPEDSIGNATURES)

Return the gas constant of moist air mixture [in J/(kg K)] given the specific humidity
`q` and thermodynamic parameters `thermo`.

The mixture gas constant is calculated as a weighted average of the dry air
and water vapor gas thermo:

```math
Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ ,
```

where:
- `Rᵈ` is the dry air gas constant,
- `Rᵛ` is the water vapor gas constant,
- `qᵈ` is the mass fraction of dry air, and
- `qᵛ` is the mass fraction of water vapor.

# Arguments
- `q`: the moisture mass fractions (vapor, liquid, and ice)
- `thermo`: `ThermodynamicConstants` instance containing gas thermo
"""
@inline function mixture_gas_constant(q::MMF, thermo::TC)
    qᵈ = dry_air_mass_fraction(q)
    qᵛ = q.vapor
    Rᵈ = dry_air_gas_constant(thermo)
    Rᵛ = vapor_gas_constant(thermo)
    return qᵈ * Rᵈ + qᵛ * Rᵛ
end

"""
$(TYPEDSIGNATURES)

Compute the heat capacity of a mixture of dry air, vapor, liquid, and ice, where
the mass fractions of vapor, liquid, and ice are given by `q`.
The heat capacity of moist air is the weighted sum of its constituents:

```math
cᵖᵐ = qᵈ cᵖᵈ + qᵛ cᵖᵛ + qˡ cˡ + qⁱ cⁱ ,
```

where `qᵛ = q.vapor`, `qˡ = q.liquid`, `qⁱ = q.ice` are
the mass fractions of vapor, liquid, and ice constituents, respectively,
and `qᵈ = 1 - qᵛ - qˡ - qⁱ` is the mass fraction of dry air.
The heat capacities `cᵖᵈ`, `cᵖᵛ`, `cˡ`, `cⁱ` are the heat capacities
of dry air, vapor, liquid, and ice at constant pressure, respectively.
The liquid and ice phases are assumed to be incompressible.
"""
@inline function mixture_heat_capacity(q::MMF, thermo::TC)
    qᵈ = dry_air_mass_fraction(q)
    qᵛ = q.vapor
    qˡ = q.liquid
    qⁱ = q.ice
    cᵖᵈ = thermo.dry_air.heat_capacity
    cᵖᵛ = thermo.vapor.heat_capacity
    cˡ = thermo.liquid.heat_capacity
    cⁱ = thermo.ice.heat_capacity
    return qᵈ * cᵖᵈ + qᵛ * cᵖᵛ + qˡ * cˡ + qⁱ * cⁱ
end

#####
##### Equation of state
#####

@inline function density(p, T, q::MMF, thermo::TC)
    Rᵐ = mixture_gas_constant(q, thermo)
    return p / (Rᵐ * T)
end
