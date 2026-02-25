using Adapt: Adapt, adapt

using Oceananigans: Oceananigans, CenterField
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ZeroField, set!, interpolate
using Oceananigans.TimeSteppers: TimeSteppers, tick!
using Oceananigans.Utils: launch!

using KernelAbstractions: @kernel, @index

using Breeze.Thermodynamics: MoistureMassFractions,
    LiquidIcePotentialTemperatureState, StaticEnergyState,
    PlanarLiquidSurface,
    with_moisture, mixture_heat_capacity, density, temperature,
    temperature_from_potential_temperature, saturation_specific_humidity

using Breeze.AtmosphereModels: AtmosphereModels, AtmosphereModel

#####
##### Vertical velocity formulations
#####

"""
    PrescribedVerticalVelocity

Singleton type for prescribed vertical velocity dynamics. The parcel moves
following the prescribed environmental vertical velocity field `w(z)`.

This is the default vertical velocity formulation: `dz/dt = w_env(z)`.
"""
struct PrescribedVerticalVelocity end

"""
    PrognosticVerticalVelocity

Singleton type for prognostic vertical velocity dynamics. The parcel has a
prognostic vertical velocity driven by buoyancy:

```math
dw/dt = B
dz/dt = w
```

where `B = -g (ρ_parcel - ρ_env) / ρ_env` is the net buoyancy from the density
difference, including both the virtual temperature effect and condensate loading.
"""
struct PrognosticVerticalVelocity end

Base.summary(::PrescribedVerticalVelocity) = "PrescribedVerticalVelocity"
Base.summary(::PrognosticVerticalVelocity) = "PrognosticVerticalVelocity"

#####
##### ParcelState: state of a rising parcel
#####

"""
$(TYPEDEF)
$(TYPEDFIELDS)

State of a Lagrangian air parcel with position, thermodynamic state, and microphysics.

The parcel model evolves **specific quantities** (qᵗ, ℰ) directly for exact conservation.
Density-weighted forms (ρqᵗ, ρℰ) are also stored for consistency with the microphysics interface.
"""
mutable struct ParcelState{FT, TH, MP}
    x :: FT
    y :: FT
    z :: FT
    w :: FT
    ρ :: FT
    qᵗ :: FT
    ρqᵗ :: FT
    ℰ :: FT
    ρℰ :: FT
    𝒰 :: TH
    μ :: MP
end

# Accessors
@inline position(state::ParcelState) = (state.x, state.y, state.z)
@inline height(state::ParcelState) = state.z
@inline parcel_density(state::ParcelState) = state.ρ
@inline total_moisture(state::ParcelState) = state.qᵗ

Base.eltype(::ParcelState{FT}) where FT = FT

function Base.show(io::IO, state::ParcelState{FT}) where FT
    print(io, "ParcelState{$FT}(z=", state.z, ", w=", round(state.w, digits=4),
          ", ρ=", round(state.ρ, digits=4),
          ", qᵗ=", round(state.qᵗ * 1000, digits=2), " g/kg)")
end

#####
##### ParcelTendencies: time derivatives of parcel state
#####

"""
$(TYPEDEF)

Tendencies (time derivatives) for parcel prognostic variables.

# Fields
- `Gx`, `Gy`, `Gz`: position tendencies [m/s]
- `Ge`: specific energy tendency [J/kg/s]
- `Gqᵗ`: specific moisture tendency [kg/kg/s]
- `Gμ`: microphysics prognostic tendencies (density-weighted)
"""
mutable struct ParcelTendencies{FT, GM}
    Gx :: FT
    Gy :: FT
    Gz :: FT
    Gw :: FT
    Ge :: FT
    Gqᵗ :: FT
    Gμ :: GM
end

ParcelTendencies(FT::DataType, Gμ::GM) where GM =
    ParcelTendencies{FT, GM}(zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), zero(FT), Gμ)

#####
##### ParcelDynamics: Lagrangian parcel dynamics for AtmosphereModel
#####

"""
$(TYPEDEF)

Lagrangian parcel dynamics for [`AtmosphereModel`](@ref).

# Fields
- `state`: parcel state (position, thermodynamics, microphysics)
- `timestepper`: SSP RK3 timestepper with tendencies
- `density`: environmental density field [kg/m³]
- `pressure`: environmental pressure field [Pa]
- `surface_pressure`: surface pressure [Pa]
- `standard_pressure`: standard pressure for potential temperature [Pa]
"""
struct ParcelDynamics{S, TS, D, P, U, FT}
    state :: S
    timestepper :: TS
    density :: D
    pressure :: P
    vertical_velocity_formulation :: U
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
                        vertical_velocity_formulation = PrescribedVerticalVelocity(),
                        surface_pressure = 101325,
                        standard_pressure = 1e5)
    U = typeof(vertical_velocity_formulation)
    return ParcelDynamics{Nothing, Nothing, Nothing, Nothing, U, FT}(
        nothing,
        nothing,
        nothing,
        nothing,
        vertical_velocity_formulation,
        convert(FT, surface_pressure),
        convert(FT, standard_pressure)
    )
end

Base.summary(::ParcelDynamics) = "ParcelDynamics"

function Base.show(io::IO, d::ParcelDynamics)
    println(io, "ParcelDynamics")
    state_str = d.state isa ParcelState ? d.state : "uninitialized"
    println(io, "├── state: ", state_str)
    println(io, "├── timestepper: ", isnothing(d.timestepper) ? "uninitialized" : "ParcelTimestepper (SSP RK3)")
    println(io, "├── vertical_velocity_formulation: ", summary(d.vertical_velocity_formulation))
    println(io, "├── density: ", isnothing(d.density) ? "unset" : summary(d.density))
    println(io, "├── pressure: ", isnothing(d.pressure) ? "unset" : summary(d.pressure))
    println(io, "├── surface_pressure: ", d.surface_pressure)
    print(io, "└── standard_pressure: ", d.standard_pressure)
end

"""
    ParcelModel

Type alias for `AtmosphereModel{<:ParcelDynamics}`.

A `ParcelModel` represents a Lagrangian adiabatic parcel that rises through a
prescribed environmental atmosphere. The parcel is characterized by its position
`(x, y, z)`, thermodynamic state, and moisture content. The environmental profiles
(temperature, pressure, density, velocities) are defined on a 1D vertical grid.

The parcel's motion is determined by interpolating environmental velocities to the
parcel position, and its thermodynamic evolution follows adiabatic processes with
optional microphysical interactions.

See also [`ParcelDynamics`](@ref), [`AtmosphereModel`](@ref).
"""
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

function AtmosphereModels.materialize_dynamics(d::ParcelDynamics, grid, bcs, constants, microphysics)
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

    # Microphysics prognostic variables based on the microphysics scheme
    μ = materialize_parcel_microphysics_prognostics(FT, microphysics)

    # Initialize state with default values
    ρ_default = FT(1.2)
    w_default = zero(FT)
    qᵗ_default = zero(FT)
    ρqᵗ_default = ρ_default * qᵗ_default
    ℰ_default = e_default  # static energy for default formulation
    ρℰ_default = ρ_default * ℰ_default
    state = ParcelState(zero(FT), zero(FT), z_default, w_default, ρ_default,
                        qᵗ_default, ρqᵗ_default, ℰ_default, ρℰ_default, 𝒰, μ)

    # SSP RK3 timestepper with tendencies
    Gμ = zero_microphysics_prognostic_tendencies(μ)
    timestepper = ParcelTimestepper(state, Gμ)

    return ParcelDynamics(state, timestepper, ρ, p, d.vertical_velocity_formulation, p₀, pˢᵗ)
end

"""
$(TYPEDSIGNATURES)

Create the parcel microphysics prognostic variables for the given microphysics scheme.

Returns `nothing` for microphysics schemes without explicit prognostic variables
(e.g., `Nothing`, `SaturationAdjustment`), or a `NamedTuple` containing the prognostic
density-weighted scalars for schemes with prognostic microphysics.

The prognostic variables use the same ρ-weighted names as the grid-based model
(e.g., `:ρqᶜˡ`, `:ρqʳ`) from `prognostic_field_names(microphysics)`.
"""
function materialize_parcel_microphysics_prognostics(FT, microphysics)
    names = AtmosphereModels.prognostic_field_names(microphysics)
    length(names) == 0 && return nothing
    Nᵃ₀ = FT(AtmosphereModels.initial_aerosol_number(microphysics))
    return NamedTuple{names}(ntuple(i -> names[i] == :ρnᵃ ? Nᵃ₀ : zero(FT), length(names)))
end

function AtmosphereModels.materialize_momentum_and_velocities(::ParcelDynamics, grid, bcs)
    # Parcel models use CenterFields for environmental velocity profiles.
    # This avoids boundary issues when interpolating at arbitrary parcel positions,
    # since cell centers are always in the domain interior.
    u = CenterField(grid)
    v = CenterField(grid)
    w = CenterField(grid)
    return NamedTuple(), (; u, v, w)
end

#####
##### Adapt and architecture transfer
#####

Adapt.adapt_structure(to, d::ParcelDynamics) =
    ParcelDynamics(adapt(to, d.state),
                   adapt(to, d.timestepper),
                   adapt(to, d.density),
                   adapt(to, d.pressure),
                   d.vertical_velocity_formulation,
                   d.surface_pressure,
                   d.standard_pressure)

Oceananigans.Architectures.on_architecture(to, d::ParcelDynamics) =
    ParcelDynamics(on_architecture(to, d.state),
                   on_architecture(to, d.timestepper),
                   on_architecture(to, d.density),
                   on_architecture(to, d.pressure),
                   d.vertical_velocity_formulation,
                   d.surface_pressure,
                   d.standard_pressure)

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

**Thermodynamic profiles** (provide one of `T` or `θ`):
- `T`: Temperature profile T(z) [K] - function, array, Field, or constant
- `θ`: Potential temperature profile θ(z) [K] - function, array, or constant.
       If provided, `T` is computed from `θ` and `p` using thermodynamic relations.
- `ρ`: Density profile ρ(z) [kg/m³] - function, array, Field, or constant
- `p`: Pressure profile p(z) [Pa] - function, array, Field, or constant

**Moisture** (provide one of `qᵗ` or `ℋ`):
- `qᵗ`: Specific humidity profile qᵗ(z) [kg/kg] - function, array, or constant (default: 0)
- `ℋ`: Relative humidity profile ℋ(z) [0-1] - function, array, or constant.
       If provided, `qᵗ` is computed as `qᵗ = ℋ * qᵛ⁺(T, ρ)`.

**Velocities**:
- `u`: Zonal velocity u(z) [m/s] - function, array, or constant (default: 0)
- `v`: Meridional velocity v(z) [m/s] - function, array, or constant (default: 0)
- `w`: Vertical velocity w(z) [m/s] - function, array, or constant (default: 0)

**Parcel state**:
- `x`: Initial parcel x-position [m] (default: 0)
- `y`: Initial parcel y-position [m] (default: 0)
- `z`: Initial parcel height [m] (required to initialize parcel state)
- `w_parcel`: Initial parcel vertical velocity [m/s] (for `PrognosticVerticalVelocity`)
"""
function Oceananigans.set!(model::ParcelModel; T = nothing, θ = nothing,
                           ρ = nothing, p = nothing,
                           qᵗ = nothing, ℋ = nothing,
                           u = 0, v = 0, w = 0,
                           w_parcel = nothing,
                           x = 0, y = 0, z = nothing)

    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    pˢᵗ = dynamics.standard_pressure

    # Set pressure and density first (needed for T from θ and qᵗ from ℋ)
    !isnothing(ρ) && set!(dynamics.density, ρ)
    !isnothing(p) && set!(dynamics.pressure, p)
    fill_halo_regions!(dynamics.density)
    fill_halo_regions!(dynamics.pressure)

    # Compute temperature from potential temperature using thermodynamic functions
    if !isnothing(θ) && isnothing(T)
        isnothing(p) && error("Pressure `p` must be provided when setting potential temperature `θ`")
        set_temperature_from_potential_temperature!(model.temperature, θ, dynamics.pressure, pˢᵗ, constants)
    elseif !isnothing(T)
        set!(model.temperature, T)
    end
    fill_halo_regions!(model.temperature)

    # Set velocities
    set!(model.velocities.u, u)
    set!(model.velocities.v, v)
    set!(model.velocities.w, w)
    fill_halo_regions!(model.velocities.u)
    fill_halo_regions!(model.velocities.v)
    fill_halo_regions!(model.velocities.w)

    # Compute specific humidity from relative humidity if ℋ is provided
    if !isnothing(ℋ) && isnothing(qᵗ)
        set_moisture_from_relative_humidity!(model.specific_moisture, ℋ,
                                              model.temperature, dynamics.density, constants)
    elseif !isnothing(qᵗ)
        set!(model.specific_moisture, qᵗ)
    else
        # Default to zero moisture
        set!(model.specific_moisture, 0)
    end
    fill_halo_regions!(model.specific_moisture)

    # Initialize parcel state if z is provided
    if !isnothing(z)
        initialize_parcel_state!(dynamics.state, z, x, y, model)
    end

    # Set parcel vertical velocity (for PrognosticVerticalVelocity)
    if !isnothing(w_parcel)
        dynamics.state.w = convert(eltype(model.grid), w_parcel)
    end

    return nothing
end

#####
##### Helper functions for set!
#####

@kernel function _set_temperature_from_potential_temperature!(T_field, θ_field, p_field, pˢᵗ, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        θₖ = θ_field[i, j, k]
        pₖ = p_field[i, j, k]
    end
    @inbounds T_field[i, j, k] = @inline temperature_from_potential_temperature(θₖ, pₖ, constants; pˢᵗ)
end

"""
$(TYPEDSIGNATURES)

Set temperature field from potential temperature, using proper thermodynamic relations.
"""
function set_temperature_from_potential_temperature!(T_field, θ, p_field, pˢᵗ, constants)
    grid = T_field.grid
    arch = grid.architecture
    θ_field = CenterField(grid)
    set!(θ_field, θ)
    launch!(arch, grid, :xyz, _set_temperature_from_potential_temperature!,
            T_field, θ_field, p_field, pˢᵗ, constants)
    return nothing
end

@kernel function _set_moisture_from_relative_humidity!(qᵗ_field, ℋ_field, T_field, ρ_field, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ℋₖ = ℋ_field[i, j, k]
        Tₖ = T_field[i, j, k]
        ρₖ = ρ_field[i, j, k]
    end
    qᵛ⁺ = @inline saturation_specific_humidity(Tₖ, ρₖ, constants, PlanarLiquidSurface())
    @inbounds qᵗ_field[i, j, k] = ℋₖ * qᵛ⁺
end

"""
$(TYPEDSIGNATURES)

Set specific humidity field from relative humidity, computing qᵗ = ℋ * qᵛ⁺(T, ρ).
"""
function set_moisture_from_relative_humidity!(qᵗ_field, ℋ, T_field, ρ_field, constants)
    grid = qᵗ_field.grid
    arch = grid.architecture
    ℋ_field = CenterField(grid)
    set!(ℋ_field, ℋ)
    launch!(arch, grid, :xyz, _set_moisture_from_relative_humidity!,
            qᵗ_field, ℋ_field, T_field, ρ_field, constants)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Initialize the parcel state by interpolating environmental conditions at the given position.
"""
function initialize_parcel_state!(state, z₀, x₀, y₀, model)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    g = constants.gravitational_acceleration
    FT = eltype(grid)

    x₀ = convert(FT, x₀)
    y₀ = convert(FT, y₀)
    z₀ = convert(FT, z₀)

    # Interpolate environmental conditions at parcel height
    T₀ = interpolate(z₀, model.temperature)
    ρ₀ = interpolate(z₀, dynamics.density)
    p₀ = interpolate(z₀, dynamics.pressure)
    qᵗ₀ = interpolate(z₀, model.specific_moisture)

    # Set position and zero vertical velocity (can be overridden by set! w_parcel keyword)
    state.x = x₀
    state.y = y₀
    state.z = z₀
    state.w = zero(FT)

    # Set density and moisture
    state.ρ = ρ₀
    state.qᵗ = qᵗ₀
    state.ρqᵗ = ρ₀ * qᵗ₀

    # Compute static energy and thermodynamic state
    q = MoistureMassFractions(qᵗ₀)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    e = cᵖᵐ * T₀ + g * z₀
    state.ℰ = e
    state.ρℰ = ρ₀ * e
    state.𝒰 = StaticEnergyState(e, q, z₀, p₀)

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
Thermodynamic and moisture tendencies come from microphysical sources/sinks.

The parcel model evolves **specific quantities** (e, qᵗ) directly, not
density-weighted quantities. For adiabatic ascent with no microphysics,
specific static energy and moisture are exactly conserved (de/dt = dqᵗ/dt = 0).
This is simpler and more accurate than stepping density-weighted quantities.
"""
function compute_parcel_tendencies!(model::ParcelModel)
    dynamics = model.dynamics
    state = dynamics.state
    tendencies = dynamics.timestepper.G
    microphysics = model.microphysics
    constants = model.thermodynamic_constants

    z = state.z
    ρ = state.ρ
    𝒰 = state.𝒰
    μ = state.μ

    # Horizontal position tendencies = environmental velocity at current height
    tendencies.Gx = interpolate(z, model.velocities.u)
    tendencies.Gy = interpolate(z, model.velocities.v)

    # Vertical position and velocity tendencies dispatched on vertical velocity formulation
    compute_vertical_velocity_tendencies!(tendencies, state, dynamics, model, dynamics.vertical_velocity_formulation)

    # Build diagnostic microphysical state from prognostic variables
    # Pass velocities for microphysics (e.g., aerosol activation uses vertical velocity)
    velocities = (; u = tendencies.Gx, v = tendencies.Gy, w = tendencies.Gz)

    # Temperature guard: skip microphysics at extreme cold (T < 150 K) where
    # saturation vapor pressure underflow produces NaN in condensation/evaporation.
    # This occurs when a buoyancy-driven parcel reaches the domain boundary in an
    # isentropic atmosphere with T → 0 near the top.
    T = temperature(𝒰, constants)

    if !isnothing(microphysics) && T > 150
        ℳ = microphysical_state(microphysics, ρ, μ, 𝒰, velocities)

        # Thermodynamic and moisture tendencies from microphysics (specific, not density-weighted)
        tendencies.Ge = microphysical_tendency(microphysics, Val(:e), ρ, ℳ, 𝒰, constants)
        tendencies.Gqᵗ = microphysical_tendency(microphysics, Val(:qᵗ), ρ, ℳ, 𝒰, constants)

        # Microphysics prognostic tendencies (scheme-dependent)
        tendencies.Gμ = compute_microphysics_prognostic_tendencies(microphysics, ρ, μ, ℳ, 𝒰, constants)
    else
        # Adiabatic: zero tendencies give exact conservation of ℰ and qᵗ
        tendencies.Ge = zero(ρ)
        tendencies.Gqᵗ = zero(ρ)
        tendencies.Gμ = zero_microphysics_prognostic_tendencies(μ)
    end

    return nothing
end

#####
##### Buoyancy computation and vertical velocity tendency dispatch
#####

"""
$(TYPEDSIGNATURES)

Compute the net buoyancy acceleration for a parcel.

The buoyancy is computed from the density difference between the parcel and
environment: `B = -g (ρ_parcel - ρ_env) / ρ_env`.

Here `ρ_parcel = p / (Rᵐ T)` is the total parcel density from the ideal gas law,
where `Rᵐ = qᵈ Rᵈ + qᵛ Rᵛ` with `qᵈ = 1 - qᵛ - qˡ - qⁱ`. This formulation
captures both the virtual temperature effect (from vapor content) and the
condensate loading effect (condensate reduces `qᵈ`, reducing `Rᵐ`, increasing
`ρ_parcel`) in a single term without double-counting.

`ρ_env` is the environmental density interpolated at the parcel height.
"""
@inline function parcel_buoyancy(state, dynamics, constants)
    g = constants.gravitational_acceleration
    ρ_env = state.ρ
    ρ_parcel = density(state.𝒰, constants)

    # Full buoyancy from density difference.
    # density(𝒰, constants) computes p / (R_m T) where R_m = q_d R_d + q_v R_v
    # and q_d = 1 - q_v - q_l - q_i. The condensate loading effect is already
    # captured through the reduced R_m (condensate reduces q_d, reducing R_m,
    # increasing ρ_parcel). No separate water loading term is needed.
    return -g * (ρ_parcel - ρ_env) / ρ_env
end

"""
$(TYPEDSIGNATURES)

Compute vertical velocity tendencies for [`PrescribedVerticalVelocity`](@ref).

The parcel follows the environmental vertical velocity: `dz/dt = w_env(z)`.
The parcel velocity tendency `Gw` is zero (unused prognostic).
"""
@inline function compute_vertical_velocity_tendencies!(tendencies, state, dynamics, model, ::PrescribedVerticalVelocity)
    tendencies.Gz = interpolate(state.z, model.velocities.w)
    tendencies.Gw = zero(state.z)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute vertical velocity tendencies for [`PrognosticVerticalVelocity`](@ref).

The parcel has a prognostic vertical velocity driven by buoyancy:
`dw/dt = B`, `dz/dt = w`.
"""
@inline function compute_vertical_velocity_tendencies!(tendencies, state, dynamics, model, ::PrognosticVerticalVelocity)
    z = state.z
    z_max = model.grid.Lz

    # Absorbing boundary: at the domain top, zero velocity and buoyancy
    # tendencies. This prevents the parcel from accelerating into extreme
    # atmospheric conditions (very cold, low pressure) at the boundary
    # where microphysics computations may produce NaN.
    if z >= z_max
        tendencies.Gz = zero(z)
        tendencies.Gw = zero(z)
    else
        B = parcel_buoyancy(state, dynamics, model.thermodynamic_constants)
        tendencies.Gz = state.w
        tendencies.Gw = B
    end

    return nothing
end

#####
##### Parcel microphysics interface
#####
# These functions implement the parcel-specific microphysics interface.
# The default fallbacks work for schemes with no explicit prognostic microphysics.

# Compute tendencies for microphysics prognostic variables
# Fallback: return nothing for schemes without prognostic microphysics
compute_microphysics_prognostic_tendencies(microphysics, ρ, μ::Nothing, ℳ, 𝒰, constants) = nothing
compute_microphysics_prognostic_tendencies(::Nothing, ρ, μ, ℳ, 𝒰, constants) = μ
compute_microphysics_prognostic_tendencies(::Nothing, ρ, μ::Nothing, ℳ, 𝒰, constants) = nothing
# Disambiguation for Nothing microphysics + NamedTuple
compute_microphysics_prognostic_tendencies(::Nothing, ρ, μ::NamedTuple, ℳ, 𝒰, constants) = μ

# For NamedTuple prognostics, compute tendencies for each field via microphysical_tendency
function compute_microphysics_prognostic_tendencies(microphysics, ρ, μ::NamedTuple, ℳ, 𝒰, constants)
    prog_names = AtmosphereModels.prognostic_field_names(microphysics)
    tendencies = map(prog_names) do name
        microphysical_tendency(microphysics, Val(name), ρ, ℳ, 𝒰, constants)
    end
    return NamedTuple{keys(μ)}(tendencies)
end

# Zero tendencies for microphysics prognostics
zero_microphysics_prognostic_tendencies(::Nothing) = nothing
zero_microphysics_prognostic_tendencies(μ::NamedTuple{names, T}) where {names, T} =
    NamedTuple{names}(ntuple(_ -> zero(eltype(T)), length(names)))

# Apply tendencies to update microphysics prognostic variables
apply_microphysical_tendencies(μ::Nothing, Gμ, Δt) = nothing
function apply_microphysical_tendencies(μ::NamedTuple, Gμ::NamedTuple, Δt)
    # Both μ and Gμ are ρ-weighted, step directly
    new_values = map(keys(μ)) do name
        μ[name] + Δt * Gμ[name]
    end
    return NamedTuple{keys(μ)}(new_values)
end

#####
##### ParcelTimestepper: SSP RK3 time-stepping for parcel models
#####

"""
$(TYPEDEF)

SSP RK3 time-stepper for [`ParcelModel`](@ref).

Stores tendencies, the initial state at the beginning of a time step,
and the SSP RK3 stage coefficients.

# Fields
- `G`: tendencies for prognostic variables
- `U⁰`: initial state storage (position, moisture, thermodynamics, microphysics)
- `α¹`, `α²`, `α³`: SSP RK3 stage coefficients (1, 1/4, 2/3)
"""
struct ParcelTimestepper{GT, U0, FT}
    G :: GT
    U⁰ :: U0
    α¹ :: FT
    α² :: FT
    α³ :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a `ParcelTimestepper` for SSP RK3 time-stepping.
"""
function ParcelTimestepper(state::ParcelState{FT}, Gμ) where FT
    α¹ = FT(1)
    α² = FT(1//4)
    α³ = FT(2//3)
    G = ParcelTendencies(FT, Gμ)
    U⁰ = ParcelInitialState(state)
    return ParcelTimestepper(G, U⁰, α¹, α², α³)
end

"""
$(TYPEDEF)

Storage for the initial parcel prognostic state at the beginning of a time step.
Used by SSP RK3 to combine the initial state with intermediate states.

# Fields
- `x`, `y`, `z`: initial position [m]
- `qᵗ`: initial specific total moisture [kg/kg]
- `ℰ`: initial specific static energy [J/kg] or potential temperature [K]
- `μ`: initial microphysics prognostics (density-weighted)
"""
mutable struct ParcelInitialState{FT, MP}
    x :: FT
    y :: FT
    z :: FT
    w :: FT
    qᵗ :: FT
    ℰ :: FT
    μ :: MP
end

function ParcelInitialState(state::ParcelState{FT, TH, MP}) where {FT, TH, MP}
    return ParcelInitialState{FT, MP}(
        state.x, state.y, state.z, state.w, state.qᵗ, state.ℰ, state.μ
    )
end

"""
$(TYPEDSIGNATURES)

Copy current prognostic state values to the initial state storage.
"""
function store_initial_parcel_state!(U⁰::ParcelInitialState, state::ParcelState)
    U⁰.x = state.x
    U⁰.y = state.y
    U⁰.z = state.z
    U⁰.w = state.w
    U⁰.qᵗ = state.qᵗ
    U⁰.ℰ = state.ℰ
    U⁰.μ = copy_microphysics_prognostics(state.μ)
    return nothing
end

copy_microphysics_prognostics(::Nothing) = nothing
copy_microphysics_prognostics(μ::NamedTuple) = deepcopy(μ)

#####
##### Domain boundary clamping
#####

"""
$(TYPEDSIGNATURES)

Clamp the parcel height to the vertical grid domain `[0, Lz]`.

When the parcel reaches a boundary its vertical velocity is zeroed to
prevent escape from the domain, which would produce unphysical
environmental profiles (pressure, density) from extrapolation.
"""
@inline function clamp_to_domain!(state, grid)
    z_max = grid.Lz
    if state.z >= z_max
        state.z = z_max
        state.w = zero(state.w)
    elseif state.z < 0
        state.z = zero(state.z)
        state.w = zero(state.w)
    end
    return nothing
end

#####
##### SSP RK3 substep
#####

"""
$(TYPEDSIGNATURES)

Apply an SSP RK3 substep with coefficient `α`:

```math
u^{(m)} = (1 - α) u^{(0)} + α (u^{(m-1)} + Δt G^{(m-1)})
```

where `u^{(0)}` is the initial state, `u^{(m-1)}` is the current state,
and `G^{(m-1)}` is the tendency at the current state.

The parcel model steps specific quantities (e, qᵗ) directly for exact conservation.
For adiabatic ascent with no microphysics sources, de/dt = dqᵗ/dt = 0, so these
quantities remain exactly constant throughout the simulation.
"""
function ssp_rk3_parcel_substep!(model::ParcelModel, U⁰::ParcelInitialState, Δt, α)
    # Compute tendencies at current state
    compute_parcel_tendencies!(model)

    dynamics = model.dynamics
    state = dynamics.state
    tendencies = dynamics.timestepper.G

    # Step position and vertical velocity
    state.x = (1 - α) * U⁰.x + α * (state.x + Δt * tendencies.Gx)
    state.y = (1 - α) * U⁰.y + α * (state.y + Δt * tendencies.Gy)
    state.z = (1 - α) * U⁰.z + α * (state.z + Δt * tendencies.Gz)
    state.w = (1 - α) * U⁰.w + α * (state.w + Δt * tendencies.Gw)

    # Clamp parcel to the vertical domain; zero w when hitting a boundary
    clamp_to_domain!(state, model.grid)

    # Step specific quantities directly (exact conservation for adiabatic)
    state.qᵗ = (1 - α) * U⁰.qᵗ + α * (state.qᵗ + Δt * tendencies.Gqᵗ)
    state.ℰ = (1 - α) * U⁰.ℰ + α * (state.ℰ + Δt * tendencies.Ge)

    # Get environmental conditions at new height
    z⁺ = state.z
    p⁺ = interpolate(z⁺, dynamics.pressure)
    ρ⁺ = interpolate(z⁺, dynamics.density)

    # Update density from environmental profile
    state.ρ = ρ⁺

    # Update density-weighted quantities for consistency
    state.ρqᵗ = ρ⁺ * state.qᵗ
    state.ρℰ = ρ⁺ * state.ℰ

    # Reconstruct thermodynamic state with conserved specific energy and updated p, z
    state.𝒰 = reconstruct_thermodynamic_state(state.𝒰, state.ℰ, z⁺, p⁺)

    # Step microphysics prognostics with SSP RK3 formula (density-weighted)
    state.μ = ssp_rk3_microphysics_substep(U⁰.μ, state.μ, tendencies.Gμ, Δt, α)

    # Update moisture fractions in thermodynamic state
    microphysics = model.microphysics
    zero_velocities = (; u = zero(state.ρ), v = zero(state.ρ), w = zero(state.ρ))
    ℳ = microphysical_state(microphysics, state.ρ, state.μ, state.𝒰, zero_velocities)
    q⁺ = moisture_fractions(microphysics, ℳ, state.qᵗ)
    state.𝒰 = with_moisture(state.𝒰, q⁺)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Reconstruct a thermodynamic state with a new conserved variable value and updated z, p.
"""
function reconstruct_thermodynamic_state end

@inline function reconstruct_thermodynamic_state(𝒰::StaticEnergyState{FT}, e⁺, z⁺, p⁺) where FT
    return StaticEnergyState{FT}(e⁺, 𝒰.moisture_mass_fractions, z⁺, p⁺)
end

@inline function reconstruct_thermodynamic_state(𝒰::LiquidIcePotentialTemperatureState{FT}, θ⁺, z⁺, p⁺) where FT
    return LiquidIcePotentialTemperatureState{FT}(θ⁺, 𝒰.moisture_mass_fractions, 𝒰.standard_pressure, p⁺)
end

"""
$(TYPEDSIGNATURES)

Apply SSP RK3 substep formula to microphysics prognostic variables.
"""
ssp_rk3_microphysics_substep(::Nothing, ::Nothing, ::Nothing, Δt, α) = nothing

function ssp_rk3_microphysics_substep(μ⁰::NamedTuple, μᵐ::NamedTuple, Gμ::NamedTuple, Δt, α)
    names = keys(μᵐ)
    μ⁺_values = map(names) do name
        (1 - α) * μ⁰[name] + α * (μᵐ[name] + Δt * Gμ[name])
    end
    return NamedTuple{names}(μ⁺_values)
end

#####
##### State stepping (Forward Euler - used as fallback)
#####

"""
$(TYPEDSIGNATURES)

Step the parcel state forward using Forward Euler: `x^(n+1) = x^n + Δt * G^n`.

Computes tendencies at the current state, then advances all prognostic variables.
After updating position, the thermodynamic state is adjusted for the
new height (adiabatic adjustment) and environmental conditions are
updated from the profiles.
"""
function step_parcel_state!(model::ParcelModel, Δt)
    # Compute tendencies at current state
    compute_parcel_tendencies!(model)

    dynamics = model.dynamics
    state = dynamics.state
    tendencies = dynamics.timestepper.G

    # Step position and vertical velocity forward (Forward Euler)
    state.x += Δt * tendencies.Gx
    state.y += Δt * tendencies.Gy
    state.z += Δt * tendencies.Gz
    state.w += Δt * tendencies.Gw

    # Clamp parcel to the vertical domain; zero w when hitting a boundary
    clamp_to_domain!(state, model.grid)

    # Step specific quantities forward (exact conservation for adiabatic)
    state.qᵗ += Δt * tendencies.Gqᵗ
    state.ℰ += Δt * tendencies.Ge

    # Get environmental conditions at new height
    z⁺ = state.z
    p⁺ = interpolate(z⁺, dynamics.pressure)
    ρ⁺ = interpolate(z⁺, dynamics.density)

    # Update density from environmental profile
    state.ρ = ρ⁺

    # Update density-weighted quantities for consistency
    state.ρqᵗ = ρ⁺ * state.qᵗ
    state.ρℰ = ρ⁺ * state.ℰ

    # Reconstruct thermodynamic state with conserved specific energy and updated p, z
    state.𝒰 = reconstruct_thermodynamic_state(state.𝒰, state.ℰ, z⁺, p⁺)

    # Step microphysics prognostics forward using tendencies (density-weighted)
    state.μ = apply_microphysical_tendencies(state.μ, tendencies.Gμ, Δt)

    # Update moisture fractions in thermodynamic state
    microphysics = model.microphysics
    zero_velocities = (; u = zero(state.ρ), v = zero(state.ρ), w = zero(state.ρ))
    ℳ = microphysical_state(microphysics, state.ρ, state.μ, state.𝒰, zero_velocities)
    q⁺ = moisture_fractions(microphysics, ℳ, state.qᵗ)
    state.𝒰 = with_moisture(state.𝒰, q⁺)

    return nothing
end


#####
##### Time stepping for ParcelModel
#####

"""
$(TYPEDSIGNATURES)

Advance the parcel model by one time step `Δt` using SSP RK3.

The SSP RK3 scheme [Shu and Osher (1988)](@cite Shu1988Efficient) is:
```math
u^{(1)} = u^{(0)} + Δt L(u^{(0)})
u^{(2)} = \\frac{3}{4} u^{(0)} + \\frac{1}{4} u^{(1)} + \\frac{1}{4} Δt L(u^{(1)})
u^{(3)} = \\frac{1}{3} u^{(0)} + \\frac{2}{3} u^{(2)} + \\frac{2}{3} Δt L(u^{(2)})
```

This scheme has CFL coefficient = 1 and is TVD (total variation diminishing).
"""
function TimeSteppers.time_step!(model::ParcelModel, Δt; callbacks=nothing)
    dynamics = model.dynamics
    ts = dynamics.timestepper
    state = dynamics.state
    U⁰ = ts.U⁰

    # Store initial state for SSP RK3 stages
    store_initial_parcel_state!(U⁰, state)

    # Stage 1: u^(1) = u^(0) + Δt * L(u^(0))
    ssp_rk3_parcel_substep!(model, U⁰, Δt, ts.α¹)
    tick!(model.clock, Δt; stage=true)

    # Stage 2: u^(2) = 3/4 u^(0) + 1/4 (u^(1) + Δt * L(u^(1)))
    ssp_rk3_parcel_substep!(model, U⁰, Δt, ts.α²)
    # Don't tick - still at t + Δt for time-dependent forcing

    # Stage 3: u^(3) = 1/3 u^(0) + 2/3 (u^(2) + Δt * L(u^(2)))
    ssp_rk3_parcel_substep!(model, U⁰, Δt, ts.α³)

    # Final clock update (adjust for floating point error)
    tⁿ⁺¹ = model.clock.time + Δt * (1 - ts.α¹)  # Already advanced by α¹*Δt in stage 1
    corrected_Δt = tⁿ⁺¹ - model.clock.time
    tick!(model.clock, corrected_Δt)

    # Set last_Δt
    model.clock.last_Δt = Δt

    # Apply microphysics model update AFTER all RK3 stages and clock update
    # (for schemes like DCMIP2016Kessler that operate via direct state modification)
    microphysics_model_update!(model.microphysics, model)

    return nothing
end



#####
##### Adiabatic adjustment
#####

"""
$(TYPEDSIGNATURES)

Adjust the thermodynamic state for adiabatic ascent/descent to a new height.
Conserves the thermodynamic variable (static energy or potential temperature).
"""
function adjust_adiabatically end

@inline adjust_adiabatically(𝒰::StaticEnergyState, z⁺, p⁺, constants) =
    reconstruct_thermodynamic_state(𝒰, 𝒰.static_energy, z⁺, p⁺)

@inline adjust_adiabatically(𝒰::LiquidIcePotentialTemperatureState, z⁺, p⁺, constants) =
    reconstruct_thermodynamic_state(𝒰, 𝒰.potential_temperature, z⁺, p⁺)
