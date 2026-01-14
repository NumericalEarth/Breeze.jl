using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, Clock
using Oceananigans.TimeSteppers: TimeSteppers

using Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions,
    LiquidIcePotentialTemperatureState, StaticEnergyState, ThermodynamicConstants,
    temperature, with_moisture, mixture_heat_capacity

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel

#####
##### ParcelState: state of a rising parcel
#####

"""
$(TYPEDEF)

The complete state of a Lagrangian air parcel.

The parcel state contains all variables needed to evolve the parcel through
an atmospheric profile. Position `(x, y, z)` tracks the parcel location,
while thermodynamic variables describe the parcel's internal state.

# Fields
$(TYPEDFIELDS)
"""
mutable struct ParcelState{FT, TH, MI}
    x :: FT
    y :: FT
    z :: FT
    ρ :: FT
    qᵗ :: FT
    𝒰 :: TH
    ℳ :: MI
end

# Accessors
@inline position(state::ParcelState) = (state.x, state.y, state.z)
@inline height(state::ParcelState) = state.z
@inline density(state::ParcelState) = state.ρ
@inline total_moisture(state::ParcelState) = state.qᵗ

Base.eltype(::ParcelState{FT}) where FT = FT

# Property accessors for readable names
Base.getproperty(state::ParcelState, name::Symbol) =
    name === :thermodynamic_state ? getfield(state, :𝒰) :
    name === :microphysical_state ? getfield(state, :ℳ) :
    getfield(state, name)

function Base.show(io::IO, state::ParcelState{FT}) where FT
    print(io, "ParcelState{$FT}(")
    print(io, "z=", state.z, ", ")
    print(io, "ρ=", round(state.ρ, digits=4), ", ")
    print(io, "qᵗ=", round(state.qᵗ * 1000, digits=2), " g/kg)")
end

#####
##### ParcelDynamics: dynamics type for AtmosphereModel
#####

"""
$(TYPEDEF)

Lagrangian parcel dynamics for use with [`AtmosphereModel`](@ref).

ParcelDynamics stores the environmental profile and the current parcel state.
When used with AtmosphereModel, the model evolves a single rising air parcel
through the environmental profile.

# Fields
$(TYPEDFIELDS)

# Example

```julia
using Breeze
using Breeze.ParcelDynamics

profile = EnvironmentalProfile(
    temperature = z -> 288.0 - 0.0065z,
    pressure = z -> 101325.0 * exp(-z/8500),
    density = z -> 1.2 * exp(-z/8500),
    specific_humidity = z -> 0.015 * exp(-z/2500),
    w = z -> 1.0
)

dynamics = ParcelDynamics(profile)
```
"""
mutable struct ParcelDynamics{P, S}
    "Environmental profile providing temperature, pressure, density, and velocity"
    profile :: P

    "Current parcel state"
    state :: S
end

ParcelDynamics(profile) = ParcelDynamics(profile, nothing)

Adapt.adapt_structure(to, dynamics::ParcelDynamics) =
    ParcelDynamics(dynamics.profile, adapt(to, dynamics.state))

Base.summary(::ParcelDynamics) = "ParcelDynamics"

function Base.show(io::IO, dynamics::ParcelDynamics)
    print(io, "ParcelDynamics with ")
    if dynamics.state === nothing
        print(io, "uninitialized state")
    else
        print(io, dynamics.state)
    end
end

# Type alias for AtmosphereModel with ParcelDynamics
const ParcelModel = AtmosphereModel{<:ParcelDynamics}

#####
##### AtmosphereModel constructor for ParcelDynamics
#####

"""
$(TYPEDSIGNATURES)

Construct an [`AtmosphereModel`](@ref) for Lagrangian parcel dynamics.

This constructor creates a minimal `AtmosphereModel` suitable for parcel simulations.
Grid-based fields are set to `nothing` since parcel models don't use spatial grids.

# Arguments
- `dynamics::ParcelDynamics`: The parcel dynamics containing the environmental profile and state

# Keyword Arguments
- `microphysics`: Microphysics scheme (default: `nothing`)
- `thermodynamic_constants`: Thermodynamic constants (default: `ThermodynamicConstants()`)

# Example

```julia
using Breeze
using Breeze.ParcelDynamics

profile = EnvironmentalProfile(...)
state = ParcelState(...)
dynamics = ParcelDynamics(profile, state)

model = AtmosphereModel(dynamics)
time_step!(model, 1.0)
```
"""
function AtmosphereModels.AtmosphereModel(dynamics::ParcelDynamics;
                                          microphysics = nothing,
                                          thermodynamic_constants = ThermodynamicConstants())
    
    clock = Clock(time=0.0)
    
    # ParcelModel doesn't use grid-based fields
    return AtmosphereModel(
        nothing,  # architecture
        nothing,  # grid
        clock,
        dynamics,
        nothing,  # formulation
        thermodynamic_constants,
        nothing,  # momentum
        nothing,  # moisture_density
        nothing,  # specific_moisture
        nothing,  # temperature
        nothing,  # pressure_solver
        nothing,  # velocities
        nothing,  # tracers
        nothing,  # buoyancy
        nothing,  # advection
        nothing,  # coriolis
        nothing,  # forcing
        microphysics,
        nothing,  # microphysical_fields
        nothing,  # timestepper
        nothing,  # closure
        nothing,  # closure_fields
        nothing   # radiation
    )
end

#####
##### set! for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Set the parcel state for a [`ParcelModel`](@ref).

# Arguments
- `model::ParcelModel`: The parcel model
- `state::ParcelState`: The new parcel state
"""
function Oceananigans.set!(model::ParcelModel, state::ParcelState)
    model.dynamics.state = state
    return nothing
end

#####
##### Dynamics interface implementation
#####

AtmosphereModels.dynamics_density(dynamics::ParcelDynamics) = dynamics.state.ρ
AtmosphereModels.dynamics_pressure(dynamics::ParcelDynamics) = environmental_pressure(dynamics.profile, dynamics.state.z)

# ParcelDynamics has no momentum fields
AtmosphereModels.prognostic_momentum_field_names(::ParcelDynamics) = ()

# ParcelDynamics has no grid-based dynamics fields
AtmosphereModels.prognostic_dynamics_field_names(::ParcelDynamics) = ()

#####
##### Time stepping for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Advance the parcel model by one time step `Δt`.

The parcel is advected by the environmental velocity field, and the
thermodynamic state evolves adiabatically. Microphysics tendencies are
computed using the standard microphysics interface.
"""
function TimeSteppers.time_step!(model::ParcelModel, Δt; callbacks=nothing)
    dynamics = model.dynamics
    profile = dynamics.profile
    state = dynamics.state
    microphysics = model.microphysics
    constants = model.thermodynamic_constants

    # Current position and state
    x, y, z = position(state)
    ρ = density(state)
    qᵗ = total_moisture(state)
    𝒰 = state.𝒰
    ℳ = state.ℳ

    # 1. Get environmental velocity at current position
    u, v, w = environmental_velocity(profile, z)

    # 2. Update position (Forward Euler)
    x_new = x + u * Δt
    y_new = y + v * Δt
    z_new = z + w * Δt

    # 3. Get environmental conditions at new height
    p_new = environmental_pressure(profile, z_new)
    ρ_new = environmental_density(profile, z_new)

    # 4. Adiabatic adjustment of thermodynamic state
    𝒰_new = adiabatic_adjustment(𝒰, z_new, p_new, constants)

    # 5. Compute microphysics tendencies and update state
    # Uses the standard microphysical_tendency interface
    ℳ_new = step_microphysics_state(microphysics, ℳ, ρ_new, 𝒰_new, constants, Δt)

    # 6. Update moisture fractions in thermodynamic state
    q_new = compute_parcel_moisture_fractions(ℳ_new, qᵗ)
    𝒰_new = with_moisture(𝒰_new, q_new)

    # Update state in place
    dynamics.state = ParcelState(x_new, y_new, z_new, ρ_new, qᵗ, 𝒰_new, ℳ_new)

    # Advance clock
    model.clock.time += Δt
    model.clock.iteration += 1

    return nothing
end

#####
##### Internal microphysics stepping using the standard interface
#####

# For Nothing microphysics, just return the state unchanged
step_microphysics_state(::Nothing, ℳ, ρ, 𝒰, constants, Δt) = ℳ
step_microphysics_state(::Nothing, ::Nothing, ρ, 𝒰, constants, Δt) = nothing

# For NothingMicrophysicalState, return unchanged
step_microphysics_state(::Nothing, ℳ::NothingMicrophysicalState, ρ, 𝒰, constants, Δt) = ℳ

#####
##### Compute moisture fractions from microphysical state
#####

# For Nothing microphysics, all moisture is vapor
compute_parcel_moisture_fractions(::Nothing, qᵗ) = MoistureMassFractions(qᵗ)
compute_parcel_moisture_fractions(::NothingMicrophysicalState, qᵗ) = MoistureMassFractions(qᵗ)

#####
##### Adiabatic adjustment for different thermodynamic formulations
#####

"""
$(TYPEDSIGNATURES)

Adjust the thermodynamic state for adiabatic ascent/descent to a new height.

For `StaticEnergyState`: The moist static energy is conserved, so we update
the height and reference pressure while keeping `e` constant.

For `LiquidIcePotentialTemperatureState`: The liquid-ice potential temperature
is conserved, so we update the reference pressure while keeping `θˡⁱ` constant.
"""
function adiabatic_adjustment end

# StaticEnergyState: conserve static energy, update height and pressure
@inline function adiabatic_adjustment(𝒰::StaticEnergyState{FT}, z_new, p_new, constants) where FT
    return StaticEnergyState{FT}(𝒰.static_energy, 𝒰.moisture_mass_fractions, z_new, p_new)
end

# LiquidIcePotentialTemperatureState: conserve θˡⁱ, update pressure
@inline function adiabatic_adjustment(𝒰::LiquidIcePotentialTemperatureState{FT}, z_new, p_new, constants) where FT
    return LiquidIcePotentialTemperatureState{FT}(
        𝒰.potential_temperature,
        𝒰.moisture_mass_fractions,
        𝒰.standard_pressure,
        p_new
    )
end
