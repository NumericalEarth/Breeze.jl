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
@inline density(state::ParcelState) = state.ρ
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

`ParcelDynamics` stores environmental profile functions and the current parcel state.
The environmental profiles (temperature, pressure, density, humidity, velocity) are
functions of height that define the atmospheric sounding through which the parcel moves.

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
set!(model, T=T, p=p, ρ=ρ, z=0.0, w=1.0)
```
"""
mutable struct ParcelDynamics{FT}
    "Temperature profile: T(z) [K]"
    temperature::Any

    "Pressure profile: p(z) [Pa]"
    pressure::Any

    "Density profile: ρ(z) [kg/m³]"
    density::Any

    "Specific humidity profile: qᵗ(z) [kg/kg]"
    specific_humidity::Any

    "Zonal velocity profile: u(z) [m/s]"
    u::Any

    "Meridional velocity profile: v(z) [m/s]"
    v::Any

    "Vertical velocity profile: w(z) [m/s]"
    w::Any

    "Current parcel state"
    state::Any

    "Surface pressure [Pa]"
    surface_pressure::FT

    "Standard pressure [Pa]"
    standard_pressure::FT
end

"""
$(TYPEDSIGNATURES)

Construct `ParcelDynamics` with default (uninitialized) profiles.

The environmental profiles and parcel state are set using `set!` after
constructing the `AtmosphereModel`.
"""
function ParcelDynamics(FT::DataType=Float64;
                        surface_pressure = 101325,
                        standard_pressure = 1e5)
    return ParcelDynamics{FT}(
        nothing,  # temperature
        nothing,  # pressure  
        nothing,  # density
        nothing,  # specific_humidity
        nothing,  # u
        nothing,  # v
        nothing,  # w
        nothing,  # state
        convert(FT, surface_pressure),
        convert(FT, standard_pressure)
    )
end

Base.summary(::ParcelDynamics) = "ParcelDynamics"

function Base.show(io::IO, d::ParcelDynamics)
    print(io, "ParcelDynamics\n")
    print(io, "├── temperature: ", isnothing(d.temperature) ? "unset" : "set", '\n')
    print(io, "├── pressure: ", isnothing(d.pressure) ? "unset" : "set", '\n')
    print(io, "├── density: ", isnothing(d.density) ? "unset" : "set", '\n')
    print(io, "├── w: ", isnothing(d.w) ? "unset" : "set", '\n')
    print(io, "├── state: ", isnothing(d.state) ? "uninitialized" : d.state, '\n')
    print(io, "├── surface_pressure: ", d.surface_pressure, '\n')
    print(io, "└── standard_pressure: ", d.standard_pressure)
end

# Type alias for AtmosphereModel with ParcelDynamics
const ParcelModel = AtmosphereModel{<:ParcelDynamics}

#####
##### Dynamics interface implementation
#####

AtmosphereModels.dynamics_density(d::ParcelDynamics) = 
    isnothing(d.state) ? nothing : d.state.ρ

AtmosphereModels.dynamics_pressure(d::ParcelDynamics) = 
    isnothing(d.state) || isnothing(d.pressure) ? nothing : d.pressure(d.state.z)

AtmosphereModels.prognostic_momentum_field_names(::ParcelDynamics) = ()
AtmosphereModels.prognostic_dynamics_field_names(::ParcelDynamics) = ()
AtmosphereModels.additional_dynamics_field_names(::ParcelDynamics) = ()
AtmosphereModels.validate_velocity_boundary_conditions(::ParcelDynamics, bcs) = nothing
AtmosphereModels.velocity_boundary_condition_names(::ParcelDynamics) = ()

AtmosphereModels.dynamics_pressure_solver(::ParcelDynamics, grid) = nothing
AtmosphereModels.mean_pressure(d::ParcelDynamics) = ZeroField()
AtmosphereModels.pressure_anomaly(::ParcelDynamics) = ZeroField()
AtmosphereModels.total_pressure(d::ParcelDynamics) = ZeroField()
AtmosphereModels.surface_pressure(d::ParcelDynamics) = d.surface_pressure
AtmosphereModels.standard_pressure(d::ParcelDynamics) = d.standard_pressure

#####
##### Materialization
#####

function AtmosphereModels.materialize_dynamics(d::ParcelDynamics, grid, bcs, constants)
    FT = eltype(grid)
    p₀ = convert(FT, d.surface_pressure)
    pˢᵗ = convert(FT, d.standard_pressure)
    return ParcelDynamics{FT}(
        d.temperature,
        d.pressure,
        d.density,
        d.specific_humidity,
        d.u, d.v, d.w,
        d.state,
        p₀, pˢᵗ
    )
end

function AtmosphereModels.materialize_momentum_and_velocities(::ParcelDynamics, grid, bcs)
    # Parcel models don't have grid-based momentum/velocity fields
    return NamedTuple(), NamedTuple()
end

#####
##### Adapt and architecture transfer
#####

Adapt.adapt_structure(to, d::ParcelDynamics{FT}) where FT =
    ParcelDynamics{FT}(d.temperature, d.pressure, d.density, d.specific_humidity,
                       d.u, d.v, d.w, adapt(to, d.state),
                       d.surface_pressure, d.standard_pressure)

Oceananigans.Architectures.on_architecture(to, d::ParcelDynamics{FT}) where FT =
    ParcelDynamics{FT}(d.temperature, d.pressure, d.density, d.specific_humidity,
                       d.u, d.v, d.w, on_architecture(to, d.state),
                       d.surface_pressure, d.standard_pressure)

#####
##### set! for ParcelModel
#####

# Convert scalar to constant function
as_function(f::Function) = f
as_function(x::Number) = z -> x
as_function(::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Set the environmental profiles and initial parcel state for a [`ParcelModel`](@ref).

# Keyword Arguments
- `T`: Temperature profile T(z) [K] - function or constant
- `p`: Pressure profile p(z) [Pa] - function or constant
- `ρ`: Density profile ρ(z) [kg/m³] - function or constant
- `qᵗ`: Specific humidity profile qᵗ(z) [kg/kg] - function or constant (default: 0)
- `u`: Zonal velocity u(z) [m/s] - function or constant (default: 0)
- `v`: Meridional velocity v(z) [m/s] - function or constant (default: 0)
- `w`: Vertical velocity w(z) [m/s] - function or constant (default: 0)
- `z`: Initial parcel height [m] (required)

# Example

```julia
set!(model, T=z->288-0.0065z, p=z->101325*exp(-z/8500), ρ=z->1.2*exp(-z/8500), z=0.0, w=1.0)
```
"""
function Oceananigans.set!(model::ParcelModel;
                           T = nothing,
                           p = nothing,
                           ρ = nothing,
                           qᵗ = z -> 0.0,
                           u = z -> 0.0,
                           v = z -> 0.0,
                           w = z -> 0.0,
                           z = nothing)

    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    g = constants.gravitational_acceleration

    # Set environmental profiles
    dynamics.temperature = as_function(T)
    dynamics.pressure = as_function(p)
    dynamics.density = as_function(ρ)
    dynamics.specific_humidity = as_function(qᵗ)
    dynamics.u = as_function(u)
    dynamics.v = as_function(v)
    dynamics.w = as_function(w)

    # Initialize parcel state if z is provided
    if !isnothing(z) && !isnothing(T) && !isnothing(p) && !isnothing(ρ)
        z₀ = z
        T₀ = dynamics.temperature(z₀)
        p₀ = dynamics.pressure(z₀)
        ρ₀ = dynamics.density(z₀)
        qᵗ₀ = dynamics.specific_humidity(z₀)

        # Create moisture fractions (all vapor initially)
        q = MoistureMassFractions(qᵗ₀)

        # Create thermodynamic state (static energy formulation)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        e = cᵖᵐ * T₀ + g * z₀
        𝒰 = StaticEnergyState(e, q, z₀, p₀)

        # Create microphysical state (nothing for now)
        ℳ = NothingMicrophysicalState(typeof(z₀))

        # Create parcel state
        dynamics.state = ParcelState(zero(z₀), zero(z₀), z₀, ρ₀, qᵗ₀, 𝒰, ℳ)
    end

    return nothing
end

#####
##### Time stepping for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Advance the parcel model by one time step `Δt`.

The parcel is advected by the environmental velocity field, and the
thermodynamic state evolves adiabatically.
"""
function TimeSteppers.time_step!(model::ParcelModel, Δt; callbacks=nothing)
    dynamics = model.dynamics
    state = dynamics.state
    constants = model.thermodynamic_constants
    microphysics = model.microphysics

    # Current position and state
    x, y, z = position(state)
    qᵗ = total_moisture(state)
    𝒰 = state.𝒰
    ℳ = state.ℳ

    # Get environmental velocity at current position
    u_env = isnothing(dynamics.u) ? 0.0 : dynamics.u(z)
    v_env = isnothing(dynamics.v) ? 0.0 : dynamics.v(z)
    w_env = isnothing(dynamics.w) ? 0.0 : dynamics.w(z)

    # Update position (Forward Euler)
    x_new = x + u_env * Δt
    y_new = y + v_env * Δt
    z_new = z + w_env * Δt

    # Get environmental conditions at new height
    p_new = dynamics.pressure(z_new)
    ρ_new = dynamics.density(z_new)

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
