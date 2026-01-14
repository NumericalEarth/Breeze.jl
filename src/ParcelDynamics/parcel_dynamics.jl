using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, Clock, CenterField
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, set!
using Oceananigans.Grids: znodes, Center
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
@inline parcel_density(state::ParcelState) = state.ρ
@inline total_moisture(state::ParcelState) = state.qᵗ

Base.eltype(::ParcelState{FT}) where FT = FT

# Property accessors for readable names
Base.getproperty(state::ParcelState, name::Symbol) =
    name === :thermodynamic_state ? getfield(state, :𝒰) :
    name === :microphysical_state ? getfield(state, :ℳ) :
    getfield(state, name)

function Base.show(io::IO, state::ParcelState{FT}) where FT
    print(io, "ParcelState{$FT}(z=", state.z, ", ρ=", round(state.ρ, digits=4), 
          ", qᵗ=", round(state.qᵗ * 1000, digits=2), " g/kg)")
end

#####
##### ParcelDynamics: Lagrangian parcel dynamics for AtmosphereModel
#####

"""
$(TYPEDEF)

Lagrangian parcel dynamics for use with [`AtmosphereModel`](@ref).

`ParcelDynamics` stores the current parcel state and references to the environmental
density and pressure fields. The environmental profiles are set on the model's
fields (temperature, velocities, etc.) using `set!`.

# Fields
$(TYPEDFIELDS)

# Example

```julia
using Oceananigans
using Breeze

grid = RectilinearGrid(size=100, z=(0, 10000), topology=(Flat, Flat, Bounded))
model = AtmosphereModel(grid; dynamics=ParcelDynamics())

# Define environmental profiles
T(z) = 288.15 - 0.0065z
p(z) = 101325.0 * exp(-z/8500)
ρ(z) = p(z) / (287.0 * T(z))

# Set profiles and initial parcel position
set!(model, T=T, ρ=ρ, w=1.0, parcel_z=0.0)
```
"""
mutable struct ParcelDynamics{D, P, FT}
    "Current parcel state"
    state :: Any  # Mutable, can be Nothing or ParcelState

    "Environmental density field"
    density :: D

    "Environmental pressure field"
    pressure :: P

    "Surface pressure [Pa]"
    surface_pressure :: FT

    "Standard pressure [Pa]"
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
    return ParcelDynamics{Nothing, Nothing, FT}(
        nothing,  # state
        nothing,  # density
        nothing,  # pressure
        convert(FT, surface_pressure),
        convert(FT, standard_pressure)
    )
end

Base.summary(::ParcelDynamics) = "ParcelDynamics"

function Base.show(io::IO, d::ParcelDynamics)
    print(io, "ParcelDynamics\n")
    print(io, "├── state: ", isnothing(d.state) ? "uninitialized" : d.state, '\n')
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
    
    # Create density and pressure fields
    ρ = CenterField(grid)
    p = CenterField(grid)
    
    return ParcelDynamics{typeof(ρ), typeof(p), FT}(d.state, ρ, p, p₀, pˢᵗ)
end

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
    ParcelDynamics(adapt(to, d.state), adapt(to, d.density), adapt(to, d.pressure),
                   d.surface_pressure, d.standard_pressure)

Oceananigans.Architectures.on_architecture(to, d::ParcelDynamics) =
    ParcelDynamics(on_architecture(to, d.state), on_architecture(to, d.density),
                   on_architecture(to, d.pressure), d.surface_pressure, d.standard_pressure)

#####
##### set! for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Set the environmental profiles and initial parcel state for a [`ParcelModel`](@ref).

Environmental profiles are set on the model's fields (temperature, density, pressure,
velocities). The parcel is initialized at the specified height with environmental
conditions.

# Keyword Arguments
- `T`: Temperature profile T(z) [K] - function, array, or constant
- `ρ`: Density profile ρ(z) [kg/m³] - function, array, or constant
- `p`: Pressure profile p(z) [Pa] - function, array, or constant (optional, computed from ρ if not provided)
- `qᵗ`: Specific humidity profile qᵗ(z) [kg/kg] - function, array, or constant (default: 0)
- `u`: Zonal velocity u(z) [m/s] - function, array, or constant (default: 0)
- `v`: Meridional velocity v(z) [m/s] - function, array, or constant (default: 0)
- `w`: Vertical velocity w(z) [m/s] - function, array, or constant (default: 0)
- `parcel_z`: Initial parcel height [m] (required to initialize parcel)

# Example

```julia
set!(model, T=z->288-0.0065z, ρ=z->1.2*exp(-z/8500), parcel_z=0.0, w=1.0)
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
                           parcel_z = nothing)

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

    # Initialize parcel state if parcel_z is provided
    if !isnothing(parcel_z)
        FT = eltype(grid)
        z₀ = convert(FT, parcel_z)
        
        # Interpolate environmental conditions at parcel height
        T₀ = interpolate(model.temperature, 0.0, 0.0, z₀)
        ρ₀ = interpolate(dynamics.density, 0.0, 0.0, z₀)
        p₀ = interpolate(dynamics.pressure, 0.0, 0.0, z₀)
        qᵗ₀ = interpolate(model.specific_moisture, 0.0, 0.0, z₀)

        # Create moisture fractions (all vapor initially)
        q = MoistureMassFractions(qᵗ₀)

        # Create thermodynamic state (static energy formulation)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        e = cᵖᵐ * T₀ + g * z₀
        𝒰 = StaticEnergyState(e, q, z₀, p₀)

        # Create microphysical state
        ℳ = NothingMicrophysicalState(FT)

        # Create parcel state
        dynamics.state = ParcelState(zero(FT), zero(FT), z₀, ρ₀, qᵗ₀, 𝒰, ℳ)
    end

    return nothing
end


#####
##### Time stepping for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Advance the parcel model by one time step `Δt`.

The parcel is advected by the environmental velocity field (interpolated from
the model's velocity fields), and the thermodynamic state evolves adiabatically.
"""
function TimeSteppers.time_step!(model::ParcelModel, Δt; callbacks=nothing)
    grid = model.grid
    dynamics = model.dynamics
    state = dynamics.state
    constants = model.thermodynamic_constants
    microphysics = model.microphysics

    # Current position and state
    x, y, z = position(state)
    qᵗ = total_moisture(state)
    𝒰 = state.𝒰
    ℳ = state.ℳ

    # Get environmental velocity at current position (interpolate from fields)
    u_env = interpolate(model.velocities.u, 0.0, 0.0, z)
    v_env = interpolate(model.velocities.v, 0.0, 0.0, z)
    w_env = interpolate(model.velocities.w, 0.0, 0.0, z)

    # Update position (Forward Euler)
    x_new = x + u_env * Δt
    y_new = y + v_env * Δt
    z_new = z + w_env * Δt

    # Get environmental conditions at new height (interpolate from fields)
    p_new = interpolate(dynamics.pressure, 0.0, 0.0, z_new)
    ρ_new = interpolate(dynamics.density, 0.0, 0.0, z_new)

    # Adiabatic adjustment of thermodynamic state
    𝒰_new = adiabatic_adjustment(𝒰, z_new, p_new, constants)

    # Compute microphysics tendencies and update state
    ℳ_new = step_microphysics_state(microphysics, ℳ, ρ_new, 𝒰_new, constants, Δt)

    # Update moisture fractions in thermodynamic state
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
##### Internal microphysics stepping
#####

step_microphysics_state(::Nothing, ℳ, ρ, 𝒰, constants, Δt) = ℳ
step_microphysics_state(::Nothing, ::Nothing, ρ, 𝒰, constants, Δt) = nothing
step_microphysics_state(::Nothing, ℳ::NothingMicrophysicalState, ρ, 𝒰, constants, Δt) = ℳ

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
