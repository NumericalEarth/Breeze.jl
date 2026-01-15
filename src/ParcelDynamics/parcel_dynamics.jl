using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, Clock, CenterField
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, set!, interpolate
using Oceananigans.Grids: Center
using Oceananigans.TimeSteppers: TimeSteppers, tick!

using Breeze.Thermodynamics: AbstractThermodynamicState, MoistureMassFractions,
    LiquidIcePotentialTemperatureState, StaticEnergyState, ThermodynamicConstants,
    temperature, with_moisture, mixture_heat_capacity

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel

#####
##### ParcelState: state of a rising parcel
#####

"""
$(TYPEDEF)

State of a Lagrangian air parcel.

# Prognostic variables
- Position: `x`, `y`, `z` [m]
- Total moisture: `qᵗ` [kg/kg]
- Thermodynamic state: `𝒰` (contains static energy or potential temperature)
- Microphysics prognostic variables: `μ` (scheme-dependent, e.g., cloud liquid, rain)

# Diagnostic variables
- Density: `ρ` [kg/m³] (from environmental profile)
"""
mutable struct ParcelState{FT, TH, MP}
    x :: FT
    y :: FT
    z :: FT
    ρ :: FT
    qᵗ :: FT
    𝒰 :: TH
    μ :: MP
end

# Accessors
@inline position(state::ParcelState) = (state.x, state.y, state.z)
@inline height(state::ParcelState) = state.z
@inline parcel_density(state::ParcelState) = state.ρ
@inline total_moisture(state::ParcelState) = state.qᵗ

Base.eltype(::ParcelState{FT}) where FT = FT

# Property accessors for readable names
Base.getproperty(state::ParcelState, name::Symbol) =
    name === :thermodynamic_state ? getfield(state, :𝒰) :
    name === :microphysics_prognostics ? getfield(state, :μ) :
    getfield(state, name)

function Base.show(io::IO, state::ParcelState{FT}) where FT
    print(io, "ParcelState{$FT}(z=", state.z, ", ρ=", round(state.ρ, digits=4),
          ", qᵗ=", round(state.qᵗ * 1000, digits=2), " g/kg)")
end

#####
##### ParcelTendencies: time derivatives of parcel state
#####

"""
$(TYPEDEF)

Tendencies (time derivatives) for parcel prognostic variables:
- Position: `Gx`, `Gy`, `Gz` [m/s]
- Static energy: `Ge` [J/kg/s] (from microphysics, zero for adiabatic)
- Total moisture: `Gqᵗ` [kg/kg/s] (from microphysics, typically zero)
- Microphysics prognostics: `Gμ` (same structure as `μ`, storing tendencies)
"""
mutable struct ParcelTendencies{FT, GM}
    Gx :: FT
    Gy :: FT
    Gz :: FT
    Ge :: FT
    Gqᵗ :: FT
    Gμ :: GM
end

ParcelTendencies(FT::DataType, Gμ::GM) where GM =
    ParcelTendencies{FT, GM}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), Gμ)

#####
##### ParcelDynamics: Lagrangian parcel dynamics for AtmosphereModel
#####

"""
$(TYPEDEF)

Lagrangian parcel dynamics for [`AtmosphereModel`](@ref).

Stores parcel `state`, `tendencies`, environmental `density` and `pressure` fields,
and reference pressures (`surface_pressure`, `standard_pressure`).

# Example

```julia
grid = RectilinearGrid(size=100, z=(0, 10000), topology=(Flat, Flat, Bounded))
model = AtmosphereModel(grid; dynamics=ParcelDynamics())
set!(model, T=z->288-0.0065z, ρ=z->1.2*exp(-z/8500), w=1.0, z=0.0)
```
"""
struct ParcelDynamics{S, G, D, P, FT}
    state :: S
    tendencies :: G
    density :: D
    pressure :: P
    surface_pressure :: FT
    standard_pressure :: FT
end

"""
$(TYPEDSIGNATURES)

Construct `ParcelDynamics` with default (uninitialized) state.

The environmental profiles and parcel state are set using `set!` after
constructing the `AtmosphereModel`.
"""
function ParcelDynamics(FT::DataType=Oceananigans.defaults.FloatType;
                        surface_pressure = 101325,
                        standard_pressure = 1e5)
    return ParcelDynamics{Nothing, Nothing, Nothing, Nothing, FT}(
        nothing,  # state (placeholder, materialized to ParcelState)
        nothing,  # tendencies (placeholder, materialized to ParcelTendencies)
        nothing,  # density
        nothing,  # pressure
        convert(FT, surface_pressure),
        convert(FT, standard_pressure)
    )
end

Base.summary(::ParcelDynamics) = "ParcelDynamics"

function Base.show(io::IO, d::ParcelDynamics)
    print(io, "ParcelDynamics\n")
    state_str = d.state isa ParcelState ? d.state : "uninitialized"
    print(io, "├── state: ", state_str, '\n')
    print(io, "├── tendencies: ", isnothing(d.tendencies) ? "uninitialized" : "ParcelTendencies", '\n')
    print(io, "├── density: ", isnothing(d.density) ? "unset" : summary(d.density), '\n')
    print(io, "├── pressure: ", isnothing(d.pressure) ? "unset" : summary(d.pressure), '\n')
    print(io, "├── surface_pressure: ", d.surface_pressure, '\n')
    print(io, "└── standard_pressure: ", d.standard_pressure)
end

# Type alias for AtmosphereModel with ParcelDynamics
const ParcelModel = AtmosphereModel{<:ParcelDynamics}

#####
##### Dynamics interface implementation
#####

AtmosphereModels.dynamics_density(d::ParcelDynamics) = d.density
AtmosphereModels.dynamics_pressure(d::ParcelDynamics) = d.pressure

AtmosphereModels.prognostic_momentum_field_names(::ParcelDynamics) = ()
AtmosphereModels.prognostic_dynamics_field_names(::ParcelDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::ParcelDynamics) = ()
AtmosphereModels.validate_velocity_boundary_conditions(::ParcelDynamics, bcs) = nothing
AtmosphereModels.velocity_boundary_condition_names(::ParcelDynamics) = (:u, :v, :w)

AtmosphereModels.dynamics_pressure_solver(::ParcelDynamics, grid) = nothing
AtmosphereModels.mean_pressure(d::ParcelDynamics) = d.pressure
AtmosphereModels.pressure_anomaly(::ParcelDynamics) = ZeroField()
AtmosphereModels.total_pressure(d::ParcelDynamics) = d.pressure
AtmosphereModels.surface_pressure(d::ParcelDynamics) = d.surface_pressure
AtmosphereModels.standard_pressure(d::ParcelDynamics) = d.standard_pressure

#####
##### Materialization
#####

function AtmosphereModels.materialize_dynamics(d::ParcelDynamics, grid, bcs, constants)
    FT = eltype(grid)
    p₀ = convert(FT, d.surface_pressure)
    pˢᵗ = convert(FT, d.standard_pressure)
    g = constants.gravitational_acceleration

    # Create density and pressure fields
    ρ = CenterField(grid)
    p = CenterField(grid)

    # Create default parcel state (will be overwritten by set!)
    # Use StaticEnergyState as the default thermodynamic formulation
    q = MoistureMassFractions(zero(FT))
    cᵖᵐ = mixture_heat_capacity(q, constants)
    T_default = FT(288.15)
    z_default = zero(FT)
    e_default = cᵖᵐ * T_default + g * z_default
    𝒰 = StaticEnergyState(e_default, q, z_default, p₀)

    # Microphysics prognostic variables (nothing for no microphysics)
    μ = nothing

    state = ParcelState(zero(FT), zero(FT), z_default, FT(1.2), zero(FT), 𝒰, μ)

    # Microphysics prognostic tendencies (same structure as μ)
    Gμ = zero_microphysics_prognostic_tendencies(μ)
    tendencies = ParcelTendencies(FT, Gμ)

    return ParcelDynamics(state, tendencies, ρ, p, p₀, pˢᵗ)
end

# Create zero-valued microphysics prognostic tendencies
zero_microphysics_prognostic_tendencies(::Nothing) = nothing

function AtmosphereModels.materialize_momentum_and_velocities(::ParcelDynamics, grid, bcs)
    # Parcel models use velocity fields for the environmental wind
    u = CenterField(grid)  # Use CenterField for simplicity in 1D interpolation
    v = CenterField(grid)
    w = CenterField(grid)
    return NamedTuple(), (; u, v, w)
end

#####
##### Adapt and architecture transfer
#####

Adapt.adapt_structure(to, d::ParcelDynamics) =
    ParcelDynamics(adapt(to, d.state), adapt(to, d.tendencies),
                   adapt(to, d.density), adapt(to, d.pressure),
                   d.surface_pressure, d.standard_pressure)

Oceananigans.Architectures.on_architecture(to, d::ParcelDynamics) =
    ParcelDynamics(on_architecture(to, d.state), on_architecture(to, d.tendencies),
                   on_architecture(to, d.density), on_architecture(to, d.pressure),
                   d.surface_pressure, d.standard_pressure)

#####
##### set! for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Set the environmental profiles and initial parcel state for a [`ParcelModel`](@ref).

Environmental profiles are set on the model's fields (temperature, density, pressure,
velocities). The parcel is initialized at the specified position with environmental
conditions interpolated at that height.

# Keyword Arguments
- `T`: Temperature profile T(z) [K] - function, array, or constant
- `ρ`: Density profile ρ(z) [kg/m³] - function, array, or constant
- `p`: Pressure profile p(z) [Pa] - function, array, or constant
- `qᵗ`: Specific humidity profile qᵗ(z) [kg/kg] - function, array, or constant (default: 0)
- `u`: Zonal velocity u(z) [m/s] - function, array, or constant (default: 0)
- `v`: Meridional velocity v(z) [m/s] - function, array, or constant (default: 0)
- `w`: Vertical velocity w(z) [m/s] - function, array, or constant (default: 0)
- `x`: Initial parcel x-position [m] (default: 0)
- `y`: Initial parcel y-position [m] (default: 0)
- `z`: Initial parcel height [m] (required to initialize parcel state)

# Example

```julia
set!(model, T=z->288-0.0065z, ρ=z->1.2*exp(-z/8500), z=0.0, w=1.0)
```
"""
function Oceananigans.set!(model::ParcelModel;
                           T = nothing,
                           ρ = nothing,
                           p = nothing,
                           qᵗ = 0,
                           u = 0,
                           v = 0,
                           w = 0,
                           x = 0,
                           y = 0,
                           z = nothing)

    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    g = constants.gravitational_acceleration

    # Set environmental fields on the model
    !isnothing(T) && set!(model.temperature, T)
    !isnothing(ρ) && set!(dynamics.density, ρ)
    !isnothing(p) && set!(dynamics.pressure, p)

    # Set velocities
    set!(model.velocities.u, u)
    set!(model.velocities.v, v)
    set!(model.velocities.w, w)

    # Set moisture
    set!(model.specific_moisture, qᵗ)

    # Fill halo regions
    fill_halo_regions!(model.temperature)
    fill_halo_regions!(dynamics.density)
    fill_halo_regions!(dynamics.pressure)
    fill_halo_regions!(model.velocities.u)
    fill_halo_regions!(model.velocities.v)
    fill_halo_regions!(model.velocities.w)
    fill_halo_regions!(model.specific_moisture)

    # Initialize parcel state if z is provided
    if !isnothing(z)
        FT = eltype(grid)
        x₀ = convert(FT, x)
        y₀ = convert(FT, y)
        z₀ = convert(FT, z)

        # Interpolate environmental conditions at parcel height
        T₀ = interpolate((z₀,), model.temperature)
        ρ₀ = interpolate((z₀,), dynamics.density)
        p₀ = interpolate((z₀,), dynamics.pressure)
        qᵗ₀ = interpolate((z₀,), model.specific_moisture)

        # Mutate the existing ParcelState fields directly
        state = dynamics.state
        state.x = x₀
        state.y = y₀
        state.z = z₀
        state.ρ = ρ₀
        state.qᵗ = qᵗ₀

        # Update thermodynamic state
        q = MoistureMassFractions(qᵗ₀)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        e = cᵖᵐ * T₀ + g * z₀
        state.𝒰 = StaticEnergyState(e, q, z₀, p₀)
    end

    return nothing
end

#####
##### Update state
#####

"""
$(TYPEDSIGNATURES)

Update the parcel model state, computing tendencies and auxiliary variables.

This function is called at the beginning of each time step and after each
substep in multi-stage time steppers. It mirrors the role of `update_state!`
for [`AtmosphereModel`](@ref) and consolidates all state-dependent computations:

1. Compute position tendencies (Gx, Gy, Gz) from environmental velocity profiles
2. Any other auxiliary state computations (currently none)

# Keyword Arguments
- `compute_tendencies`: If `true` (default), compute tendencies for prognostic variables.
"""
function TimeSteppers.update_state!(model::ParcelModel, callbacks=[]; compute_tendencies=true)
    compute_tendencies && compute_parcel_tendencies!(model)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute tendencies for the parcel prognostic variables.

Position tendencies are interpolated from environmental velocity fields.
Thermodynamic, moisture, and microphysical tendencies come from the microphysics scheme.
"""
function compute_parcel_tendencies!(model::ParcelModel)
    dynamics = model.dynamics
    state = dynamics.state
    tendencies = dynamics.tendencies
    microphysics = model.microphysics
    constants = model.thermodynamic_constants

    z = state.z
    ρ = state.ρ
    qᵗ = state.qᵗ
    𝒰 = state.𝒰
    μ = state.μ

    # Build diagnostic microphysical state from prognostic variables
    ℳ = parcel_microphysical_state(microphysics, ρ, qᵗ, μ, 𝒰, constants)

    # Position tendencies = environmental velocity at current height
    tendencies.Gx = interpolate((z,), model.velocities.u)
    tendencies.Gy = interpolate((z,), model.velocities.v)
    tendencies.Gz = interpolate((z,), model.velocities.w)

    # Thermodynamic and moisture tendencies from microphysics
    tendencies.Ge = microphysical_tendency(microphysics, Val(:ρe), ρ, ℳ, 𝒰, constants)
    tendencies.Gqᵗ = microphysical_tendency(microphysics, Val(:ρqᵗ), ρ, ℳ, 𝒰, constants)

    # Microphysics prognostic tendencies (scheme-dependent)
    tendencies.Gμ = compute_microphysics_prognostic_tendencies(microphysics, ρ, μ, ℳ, 𝒰, constants)

    return nothing
end

# Build diagnostic microphysical state from prognostic variables
parcel_microphysical_state(::Nothing, ρ, qᵗ, μ, 𝒰, constants) = μ
parcel_microphysical_state(::Nothing, ρ, qᵗ, μ::Nothing, 𝒰, constants) = NothingMicrophysicalState(typeof(ρ))

# Compute tendencies for microphysics prognostic variables
compute_microphysics_prognostic_tendencies(::Nothing, ρ, μ, ℳ, 𝒰, constants) = μ
compute_microphysics_prognostic_tendencies(::Nothing, ρ, μ::Nothing, ℳ, 𝒰, constants) = nothing

#####
##### State stepping
#####

"""
$(TYPEDSIGNATURES)

Step the parcel state forward using the computed tendencies.

This applies Forward Euler: `x^(n+1) = x^n + Δt * G^n`

After updating position, the thermodynamic state is adjusted for the
new height (adiabatic adjustment) and environmental conditions are
updated from the profiles.
"""
function step_parcel_state!(model::ParcelModel, Δt)
    dynamics = model.dynamics
    state = dynamics.state
    tendencies = dynamics.tendencies
    constants = model.thermodynamic_constants
    ρ = state.ρ

    # Step position forward (Forward Euler)
    state.x += Δt * tendencies.Gx
    state.y += Δt * tendencies.Gy
    state.z += Δt * tendencies.Gz

    # Step moisture forward (tendency is for ρqᵗ, convert to specific)
    state.qᵗ += Δt * tendencies.Gqᵗ / ρ

    # Get environmental conditions at new height
    z_new = state.z
    p_new = interpolate((z_new,), dynamics.pressure)
    ρ_new = interpolate((z_new,), dynamics.density)

    # Update density from environmental profile
    state.ρ = ρ_new

    # Adiabatic adjustment of thermodynamic state (updates z and p)
    # Then apply energy tendency from microphysics (tendency is for ρe, convert to specific)
    𝒰_adjusted = adiabatic_adjustment(state.𝒰, z_new, p_new, constants)
    𝒰_with_tendency = apply_energy_tendency(𝒰_adjusted, tendencies.Ge, ρ, Δt)
    state.𝒰 = 𝒰_with_tendency

    # Step microphysics prognostics forward using tendencies
    state.μ = apply_microphysical_tendencies(state.μ, tendencies.Gμ, ρ, Δt)

    # Update moisture fractions in thermodynamic state
    q_new = compute_parcel_moisture_fractions(state.μ, state.qᵗ)
    state.𝒰 = with_moisture(state.𝒰, q_new)

    return nothing
end

# Apply tendencies to update microphysics prognostic variables
# Tendencies are for ρ-weighted fields, so we divide by ρ to get specific tendencies
apply_microphysical_tendencies(μ::Nothing, Gμ, ρ, Δt) = nothing

"""
$(TYPEDSIGNATURES)

Apply energy tendency to thermodynamic state.
The tendency `Ge` is for ρe (density-weighted), so we convert to specific: de/dt = Ge/ρ.
"""
function apply_energy_tendency end

@inline function apply_energy_tendency(𝒰::StaticEnergyState{FT}, Ge, ρ, Δt) where FT
    e_new = 𝒰.static_energy + Δt * Ge / ρ
    return StaticEnergyState{FT}(e_new, 𝒰.moisture_mass_fractions, 𝒰.height, 𝒰.reference_pressure)
end

@inline function apply_energy_tendency(𝒰::LiquidIcePotentialTemperatureState{FT}, Ge, ρ, Δt) where FT
    # For potential temperature formulation, Ge would be tendency for ρθ
    # θ_new = θ + Δt * Gθ / ρ
    θ_new = 𝒰.potential_temperature + Δt * Ge / ρ
    return LiquidIcePotentialTemperatureState{FT}(
        θ_new,
        𝒰.moisture_mass_fractions,
        𝒰.standard_pressure,
        𝒰.pressure
    )
end

#####
##### Time stepping for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Advance the parcel model by one time step `Δt` using Forward Euler.

The algorithm is:
1. Update state (compute tendencies): `G = L(u^n)`
2. Step forward: `u^(n+1) = u^n + Δt * G`
3. Update state for new position
4. Advance clock

This follows the standard pattern used by all dynamics types:
1. `update_state!` to compute tendencies
2. Step forward prognostic variables
3. `update_state!` to recompute auxiliary variables
"""
function TimeSteppers.time_step!(model::ParcelModel, Δt; callbacks=nothing)
    # Compute tendencies at current state
    TimeSteppers.update_state!(model, callbacks; compute_tendencies=true)

    # Step forward prognostic variables
    step_parcel_state!(model, Δt)

    # Advance clock
    tick!(model.clock, Δt)

    # Update state for new position (no tendencies needed at end of step)
    TimeSteppers.update_state!(model, callbacks; compute_tendencies=false)

    return nothing
end

#####
##### Compute moisture fractions from microphysical state
#####

compute_parcel_moisture_fractions(::Nothing, qᵗ) = MoistureMassFractions(qᵗ)
compute_parcel_moisture_fractions(::NothingMicrophysicalState, qᵗ) = MoistureMassFractions(qᵗ)

#####
##### Adiabatic adjustment
#####

"""
$(TYPEDSIGNATURES)

Adjust the thermodynamic state for adiabatic ascent/descent to a new height.
"""
function adiabatic_adjustment end

@inline function adiabatic_adjustment(𝒰::StaticEnergyState{FT}, z_new, p_new, constants) where FT
    return StaticEnergyState{FT}(𝒰.static_energy, 𝒰.moisture_mass_fractions, z_new, p_new)
end

@inline function adiabatic_adjustment(𝒰::LiquidIcePotentialTemperatureState{FT}, z_new, p_new, constants) where FT
    return LiquidIcePotentialTemperatureState{FT}(
        𝒰.potential_temperature,
        𝒰.moisture_mass_fractions,
        𝒰.standard_pressure,
        p_new
    )
end
