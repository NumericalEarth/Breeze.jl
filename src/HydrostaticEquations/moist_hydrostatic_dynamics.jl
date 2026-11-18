#####
##### Moist hydrostatic dynamics: a dry-mass coordinate under the pressure of everything
#####
#
# With water, the coordinate, continuity and every prognostic weight remain those of dry air, and
# the pressure is the weight of the dry air and of all the water above:
#
#   Δπ = ΔA + ΔB πˢ               dry pressure thickness of a layer; πˢ is the prognostic surface pressure
#   W  = g Δz Σ ρqˣ               weight of the water of the layer per unit area (Δz the grid's spacing)
#   Δp = Δπ + W                   total pressure thickness
#   pᶠ = A + B πˢ + Σ W above     total pressure of a z-face
#
# The prognostic surface pressure, which keeps the name `pˢ` among the prognostic fields, is
# therefore the dry-mass surface pressure πˢ of a moist model, and the total surface pressure
# pˢ = πˢ + Σ W is diagnosed from it.
#
# The potential temperature of the model is the liquid-ice potential temperature θ with the dry
# Exner exponent, θ = H / (ĉᵖ Π) with H = cᵖᵐ T - ℒˡ₀ qˡ the enthalpy, so that the Exner function
# depends on the pressure alone. The pressure-gradient force and the hydrostatic integral are
# those of dry air with θ replaced by the density potential temperature θᵨ = ϑ θ,
#
#   ϑ = (Rᵐ / Rᵈ) [ĉᵖ / cᵖᵐ + ℒˡ₀ qˡ / (cᵖᵐ Π θ)] ,
#
# which holds the buoyancy of vapor and the loading of condensate, and every factor of which is
# exactly one without water. θ is not conserved in adiabatic motion with this exponent: the
# first law adds the expansion work 𝒲 D(ln p)/Dt to its tendency.
#
# A model is moist when it is built with a microphysics scheme, and only then: its
# `HydrostaticDynamics` carries a `HydrostaticMoisture`. Every method of this file dispatches on
# that, so a model without microphysics runs the dry code, unchanged.

# Who applies the column physics of a moist model. The time steppers apply it once at the end of
# every step, over the step; a driver that applies it itself (`WarmRainColumnPhysics`, with its
# own number of applications per step) first takes it from them, so that it is applied once.
mutable struct ColumnPhysicsDispatch
    in_time_step :: Bool
end

struct HydrostaticMoisture{T, I, S, D, L, C}
    density_potential_temperature :: T  # θᵨ = ϑ θ at cell centers
    interface_pressure :: I             # total pressure pᶠ at z-faces
    total_surface_pressure :: S         # pˢ = πˢ + Σ W (two-dimensional)
    total_density :: D                  # ρ (1 + rᵗ): mass of air and water per unit area and unit of z
    ledger :: L                         # accumulated fluxes through the ground (two-dimensional)
    column_physics :: C                 # ColumnPhysicsDispatch, or nothing inside kernels
end

# Kernels read the fields only
Adapt.adapt_structure(to, moisture::HydrostaticMoisture) =
    HydrostaticMoisture(adapt(to, moisture.density_potential_temperature),
                        adapt(to, moisture.interface_pressure),
                        adapt(to, moisture.total_surface_pressure),
                        adapt(to, moisture.total_density),
                        adapt(to, moisture.ledger),
                        nothing)

@regional HydrostaticMoisture

const MoistHydrostaticDynamics = HydrostaticDynamics{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                     <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:HydrostaticMoisture}

const MoistHydrostaticModel = AtmosphereModel{<:MoistHydrostaticDynamics}
const MoistHydrostaticPotentialTemperatureModel =
    AtmosphereModel{<:MoistHydrostaticDynamics, <:LiquidIcePotentialTemperatureFormulation}

# Fluxes through the ground accumulated over the integration, per unit area. The frictional
# heating is not a flux through a boundary: it is the potential energy that falling rain leaves
# in the air it falls through, reported beside the fluxes.
function precipitation_ledger(grid)
    return (surface_rain = Field{Center, Center, Nothing}(grid),                  # kg m⁻²
            precipitation_energy = Field{Center, Center, Nothing}(grid),          # J m⁻²: P (cˡ T - ℒˡ₀ + Φˢ)
            rain_frictional_heating = Field{Center, Center, Nothing}(grid),       # J m⁻²
            maximum_sedimentation_courant = Field{Center, Center, Nothing}(grid)) # V Δt / (physical depth)
end

function HydrostaticMoisture(grid)
    return HydrostaticMoisture(CenterField(grid),
                               ZFaceField(grid),
                               Field{Center, Center, Nothing}(grid),
                               CenterField(grid),
                               precipitation_ledger(grid),
                               ColumnPhysicsDispatch(true))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Choose whether the time stepper of `model`, an `AtmosphereModel` with moist `HydrostaticDynamics`,
applies the column physics at the end of every time step (`true`, the default of a new model),
or leaves it to the driver (`false`), as [`WarmRainColumnPhysics`](@ref) requires.
"""
function column_physics_in_time_step!(model::MoistHydrostaticModel, in_time_step::Bool)
    model.dynamics.moisture.column_physics.in_time_step = in_time_step
    return nothing
end

column_physics_in_time_step(model::MoistHydrostaticModel) = model.dynamics.moisture.column_physics.in_time_step

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Apply the column physics of `microphysics` over the time step that `model` has just taken, unless
a driver has taken it over with [`column_physics_in_time_step!`](@ref). Called by the time steppers.
"""
time_step_column_physics!(microphysics, model::MoistHydrostaticModel) =
    column_physics_in_time_step(model) ? warm_rain_column_physics!(microphysics, model) : nothing

with_moisture(dynamics, ::Nothing, grid) = dynamics

with_moisture(dynamics, microphysics, grid) =
    HydrostaticDynamics(dynamics.pressure_coefficient,
                        dynamics.surface_pressure_coefficient,
                        dynamics.surface_pressure,
                        dynamics.surface_geopotential,
                        dynamics.density,
                        dynamics.pressure,
                        dynamics.exner_function,
                        dynamics.interface_exner_function,
                        dynamics.geopotential,
                        dynamics.vertical_mass_flux,
                        dynamics.hyperdiffusion,
                        dynamics.vertical_diffusion,
                        dynamics.top_pressure,
                        dynamics.standard_pressure,
                        HydrostaticMoisture(grid))

# A model with a microphysics scheme carries water, and its dynamics the diagnostics of the
# moist column
function AtmosphereModels.materialize_dynamics(dynamics::HydrostaticDynamics, grid, boundary_conditions, constants,
                                               microphysics, formulation)
    dry_dynamics = AtmosphereModels.materialize_dynamics(dynamics, grid, boundary_conditions, constants)
    return with_moisture(dry_dynamics, microphysics, grid)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the total density of moist `HydrostaticDynamics`: the mass of dry air and of all water
of a layer per unit area and unit of the grid's vertical coordinate, ``ρ (1 + rᵗ)``. It recovers
the mass fractions from the prognostic densities, and is not the density of the air in kg m⁻³,
which is ``p / (Rᵐ T)``.
"""
AtmosphereModels.total_density(dynamics::MoistHydrostaticDynamics) = dynamics.moisture.total_density

# The total surface pressure, diagnosed from the dry-mass surface pressure and the water of the column
AtmosphereModels.surface_pressure(dynamics::MoistHydrostaticDynamics) = dynamics.moisture.total_surface_pressure

#####
##### Microphysics on the hydrostatic core
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return whether `microphysics` has been adapted to `HydrostaticDynamics`, whose vertical
coordinate is a label rather than a height and whose densities are masses per unit of that
label. A scheme that has not is rejected when the model is built: its rates and fall speeds
would be computed with the wrong density and the wrong layer depth, without an error.
"""
supports_hydrostatic_dynamics(microphysics) = false
supports_hydrostatic_dynamics(::Nothing) = true

# The interface of a supported scheme, extended where the scheme is defined:
#
#   hydrostatic_water_fields(microphysics, microphysical_fields)
#       the mass fractions (qᵛ, qˡ, qᶜˡ) of vapor, of all liquid and of cloud liquid, the density
#       `ρqʳ` of rain and its sedimentation velocity `wʳ`, as fields
#   rain_source_rates(microphysics, ρ, qᶜˡ, qʳ, q, T, constants)
#       the rates of change of the rain mass fraction by conversion of cloud liquid (≥ 0) and by
#       evaporation (≤ 0), in s⁻¹, with ρ the density of the air in kg m⁻³
#   rain_fall_speed(microphysics, ρ, qʳ)
#       the mass-weighted fall speed of rain in m s⁻¹, positive downward, with the same ρ
function hydrostatic_water_fields end
function rain_source_rates end
function rain_fall_speed end

# The prognostic densities of water: the moisture density and the prognostic densities of the
# microphysics scheme, all of which are water in the supported schemes
water_densities(model) = (model.moisture_density,
                          map(name -> model.microphysical_fields[name],
                              AtmosphereModels.prognostic_field_names(model.microphysics))...)

water_tendencies(model) = map(name -> model.timestepper.Gⁿ[name],
                              (AtmosphereModels.moisture_prognostic_name(model.microphysics),
                               AtmosphereModels.prognostic_field_names(model.microphysics)...))

@inline sum_at(i, j, k, ::Tuple{}) = 0
@inline sum_at(i, j, k, fields::Tuple) = @inbounds first(fields)[i, j, k] + sum_at(i, j, k, Base.tail(fields))

#####
##### Layer mass, total pressure and Exner functions
#####

# The moist counterpart of `_compute_layer_mass_and_exner_functions!`. The weight of the water
# is accumulated from the model top, above which there is none, to give the total pressure of
# the z-faces; the layer Exner function is then the mean of (p / pˢᵗ)^κ over the total mass of
# the layer. Without water every line returns what the dry kernel does.
@kernel function _compute_moist_layer_mass_and_exner_functions!(ρ, ρᵗ, p, Π, Πᶠ, pᶠ, pˢ, grid, A, B, πˢ,
                                                                ρqˣ, pˢᵗ, κ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ΣW = zero(grid)
        pᶠ[i, j, Nz+1] = interface_pressure(i, j, Nz+1, A, B, πˢ)

        for k in Nz:-1:1
            ΣW += g * Δzᶜᶜᶜ(i, j, k, grid) * sum_at(i, j, k, ρqˣ)
            pᶠ[i, j, k] = interface_pressure(i, j, k, A, B, πˢ) + ΣW
        end

        pˢ[i, j, 1] = pᶠ[i, j, 1]

        p⁻ = pᶠ[i, j, 1]
        Πᶠ⁻ = (p⁻ / pˢᵗ)^κ
        Πᶠ[i, j, 1] = Πᶠ⁻

        for k in 1:Nz
            p⁺ = pᶠ[i, j, k+1]
            Πᶠ⁺ = (p⁺ / pˢᵗ)^κ
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            ρqᵗ = sum_at(i, j, k, ρqˣ)
            Δπ = layer_pressure_thickness(i, j, k, A, B, πˢ)
            Δp = Δπ + g * Δz * ρqᵗ
            Πᵏ = (p⁻ * Πᶠ⁻ - p⁺ * Πᶠ⁺) / ((1 + κ) * Δp)
            ρᵏ = Δπ / (g * Δz)

            ρ[i, j, k] = ρᵏ
            ρᵗ[i, j, k] = ρᵏ + ρqᵗ
            Π[i, j, k] = Πᵏ
            p[i, j, k] = pˢᵗ * Πᵏ^(1 / κ)
            Πᶠ[i, j, k+1] = Πᶠ⁺

            p⁻ = p⁺
            Πᶠ⁻ = Πᶠ⁺
        end
    end
end

function AtmosphereModels.compute_total_density!(model::MoistHydrostaticModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    moisture = dynamics.moisture
    constants = model.thermodynamic_constants
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    @apply_regionally launch!(arch, grid, :xy, _compute_moist_layer_mass_and_exner_functions!,
                              dynamics.density,
                              moisture.total_density,
                              dynamics.pressure,
                              dynamics.exner_function,
                              dynamics.interface_exner_function,
                              moisture.interface_pressure,
                              moisture.total_surface_pressure,
                              grid,
                              dynamics.pressure_coefficient,
                              dynamics.surface_pressure_coefficient,
                              dynamics.surface_pressure,
                              water_densities(model),
                              dynamics.standard_pressure, κ, g)

    fill_halo_regions!(dynamics.density)
    fill_halo_regions!(moisture.total_density)
    fill_halo_regions!(dynamics.pressure)
    fill_halo_regions!(dynamics.exner_function)
    fill_halo_regions!(moisture.total_surface_pressure)

    validate_layer_mass(dynamics)

    return nothing
end

#####
##### Density potential temperature, geopotential and mass flux
#####

# ϑ = θᵨ / θ. In dry air the mixture constants are those of dry air and the condensate holds no
# latent energy, so every factor is exactly one.
@inline function density_potential_temperature_factor(θ, Π, q, constants)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
    ℒq = condensate_latent_energy(q, constants)
    Πθ = Π * θ
    safe_Πθ = ifelse(Πθ == 0, one(Πθ), Πθ)
    return Rᵐ / Rᵈ * (ĉᵖ / cᵖᵐ + ℒq / (cᵖᵐ * safe_Πθ))
end

@kernel function _compute_density_potential_temperature!(θᵨ, grid, θ, Π, qᵛ, qˡ, constants)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
        θᵨ[i, j, k] = density_potential_temperature_factor(θ[i, j, k], Π[i, j, k], q, constants) * θ[i, j, k]
    end
end

# Physical depth of layer `k`, in m, from the hydrostatic integral: ΔΦ / g across the layer
@inline physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g) =
    @inbounds cᵖᵈ * θᵨ[i, j, k] * (Πᶠ[i, j, k] - Πᶠ[i, j, k+1]) / g

# Density of the air of layer `k` in kg m⁻³: the equation of state at the layer pressure
@inline function physical_density(i, j, k, p, T, qᵛ, qˡ, constants)
    q = @inbounds MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
    return @inbounds Thermodynamics.density(T[i, j, k], p[i, j, k], q, constants)
end

# The sedimentation velocity of rain through the z-face `k`, the lower face of its donor layer
# `k`: the grid's vertical coordinate is a label, so a fall speed V in m s⁻¹ crosses the layer at
# - V Δz / (physical depth), in units of that coordinate per second.
@kernel function _compute_rain_sedimentation_velocity!(wʳ, grid, microphysics, ρᵗ, ρqʳ, θᵨ, Πᶠ, p, T, qᵛ, qˡ,
                                                       constants, cᵖᵈ, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵖʰ = physical_density(i, j, k, p, T, qᵛ, qˡ, constants)
        V = positive_rain_fall_speed(microphysics, ρᵖʰ, ρqʳ[i, j, k] / ρᵗ[i, j, k])
        depth = physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)
        safe_depth = ifelse(depth > 0, depth, one(depth))
        wʳ[i, j, k] = ifelse(depth > 0, - V * Δzᶜᶜᶜ(i, j, k, grid) / safe_depth, zero(depth))
    end
end

# Rain that is not there does not fall. The fall speed is evaluated for every cell, so it is
# given a positive mass fraction and a finite density, and discarded otherwise.
@inline function positive_rain_fall_speed(microphysics, ρ, qʳ)
    raining = (qʳ > 0) & isfinite(ρ) & (ρ > 0)
    safe_ρ = ifelse(raining, ρ, one(ρ))
    safe_qʳ = ifelse(raining, qʳ, one(qʳ))
    return ifelse(raining, rain_fall_speed(microphysics, safe_ρ, safe_qʳ), zero(ρ))
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Throw an `ArgumentError` unless every prognostic density of water of `model` is a number.
"""
function validate_moist_state(model)
    for ρq in water_densities(model)
        (isfinite(minimum(ρq)) && isfinite(maximum(ρq))) ||
            throw(ArgumentError("A density of water of HydrostaticDynamics is not a number: the integration has failed."))
    end
    return nothing
end

function AtmosphereModels.compute_auxiliary_dynamics_variables!(model::MoistHydrostaticModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    moisture = dynamics.moisture
    constants = model.thermodynamic_constants
    u, v, w = model.velocities
    θ = model.formulation.potential_temperature
    θᵨ = moisture.density_potential_temperature
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration
    water = hydrostatic_water_fields(model.microphysics, model.microphysical_fields)

    validate_moist_state(model)

    @apply_regionally launch!(arch, grid, :xyz, _compute_density_potential_temperature!,
                              θᵨ, grid, θ, dynamics.exner_function, water.qᵛ, water.qˡ, constants)

    @apply_regionally launch!(arch, grid, :xy, _compute_geopotential!,
                              dynamics.geopotential, grid, dynamics.surface_geopotential,
                              θᵨ, dynamics.exner_function, dynamics.interface_exner_function, cᵖᵈ)

    @apply_regionally launch!(arch, grid, :xy, _compute_vertical_mass_flux!,
                              dynamics.vertical_mass_flux, w, grid,
                              dynamics.surface_pressure_coefficient, dynamics.density, u, v)

    @apply_regionally launch!(arch, grid, :xyz, _compute_rain_sedimentation_velocity!,
                              water.wʳ, grid, model.microphysics, moisture.total_density, water.ρqʳ,
                              θᵨ, dynamics.interface_exner_function, dynamics.pressure, model.temperature,
                              water.qᵛ, water.qˡ, constants, cᵖᵈ, g)

    fill_halo_regions!(θᵨ)
    fill_halo_regions!(dynamics.geopotential)
    fill_halo_regions!(dynamics.vertical_mass_flux)
    fill_halo_regions!(w)

    return nothing
end

#####
##### Tendencies
#####

# The pressure-gradient force of moist air is that of dry air with the density potential
# temperature: the kernels are those of the dry model, given θᵨ.
function AtmosphereModels.compute_momentum_tendencies!(model::MoistHydrostaticModel, model_fields)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics

    velocity_args = (model.advection.momentum,
                     model.coriolis,
                     model.velocities,
                     dynamics.density,
                     dynamics.vertical_mass_flux,
                     dynamics.moisture.density_potential_temperature,
                     dynamics.exner_function,
                     dynamics.geopotential,
                     model.thermodynamic_constants.dry_air.heat_capacity,
                     dynamics.hyperdiffusion)

    u_args = tuple(velocity_args..., model.forcing.u, model.clock, model_fields)
    v_args = tuple(velocity_args..., model.forcing.v, model.clock, model_fields)

    launch!(arch, grid, :xyz, _compute_x_velocity_tendency!, model.timestepper.Gⁿ.u, grid, u_args)
    launch!(arch, grid, :xyz, _compute_y_velocity_tendency!, model.timestepper.Gⁿ.v, grid, v_args)

    return nothing
end

# An energy source Q changes ρθ at Q / [(cᵖᵈ + rᵗ cᵖᵛ) Π]: the enthalpy of a layer is
# Δπ (cᵖᵈ + rᵗ cᵖᵛ) Π θ, with rᵗ the mixing ratio of all water. The expansion work is added by
# `compute_dynamics_tendency!`, once the tendencies of the surface pressure and of the water
# that it needs are known.
@inline function moist_hydrostatic_potential_temperature_tendency(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ,
                                                                  cᵖᵈ, cᵖᵛ, ρqˣ, hyperdiffusion, ρθ_forcing,
                                                                  ρE_forcing, clock, model_fields)

    Q = ρE_forcing(i, j, k, grid, clock, model_fields)
    rᵗ = @inbounds sum_at(i, j, k, ρqˣ) / ρ[i, j, k]

    return ( potential_temperature_flux_convergence(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ)
             + potential_temperature_hyperdiffusion(i, j, k, grid, hyperdiffusion, ρ, θ)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + @inbounds Q / ((cᵖᵈ + rᵗ * cᵖᵛ) * Π[i, j, k]))
end

@kernel function _compute_moist_hydrostatic_potential_temperature_tendency!(Gρθ, grid, args)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρθ[i, j, k] = moist_hydrostatic_potential_temperature_tendency(i, j, k, grid, args...)
end

function AtmosphereModels.compute_thermodynamic_tendency!(model::MoistHydrostaticPotentialTemperatureModel,
                                                          common_args, tracer_transport_velocity)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants

    ρθ_args = (model.advection.ρθ,
               dynamics.density,
               model.velocities,
               dynamics.vertical_mass_flux,
               model.formulation.potential_temperature,
               dynamics.exner_function,
               dynamics.interface_exner_function,
               constants.dry_air.heat_capacity,
               constants.vapor.heat_capacity,
               water_densities(model),
               dynamics.hyperdiffusion,
               model.forcing.ρθ,
               model.forcing.ρE,
               model.clock,
               Oceananigans.fields(model))

    Gρθ = model.timestepper.Gⁿ.ρθ
    launch!(arch, grid, :xyz, _compute_moist_hydrostatic_potential_temperature_tendency!, Gρθ, grid, ρθ_args)

    return nothing
end

# The scalars of a moist model are specific to the total mass, c = ρc / [ρ (1 + rᵗ)]. Their
# transport is that of the mixing ratio per unit dry mass, r = ρc / ρ, by the dry mass fluxes
# of continuity: the fluxes that carry θ. A uniform mixing ratio then stays uniform, whatever
# the water does, because water is not in the continuity equation.
struct MixingRatio{FT, C, T, D} <: AbstractArray{FT, 3}
    specific :: C       # c, per unit total mass
    total_density :: T  # ρ (1 + rᵗ)
    density :: D        # ρ, of dry air

    MixingRatio(c::C, ρᵗ::T, ρ::D) where {C, T, D} = new{eltype(ρ), C, T, D}(c, ρᵗ, ρ)
end

Base.size(r::MixingRatio) = size(r.density)

Base.@propagate_inbounds Base.getindex(r::MixingRatio, i, j, k) =
    r.specific[i, j, k] * (r.total_density[i, j, k] / r.density[i, j, k])

# The condensate of the supported schemes does not fall in the stages: rain sedimentation is
# part of the column physics, once per time step, so no sedimentation velocity is added here.
@inline function AtmosphereModels.scalar_tendency(i, j, k, grid,
                                                  c,
                                                  id,
                                                  name,
                                                  c_forcing,
                                                  advection,
                                                  dynamics::MoistHydrostaticDynamics,
                                                  formulation,
                                                  constants,
                                                  specific_prognostic_moisture,
                                                  velocities,
                                                  microphysics,
                                                  microphysical_fields,
                                                  closure,
                                                  closure_fields,
                                                  clock,
                                                  model_fields)

    ρ = dynamics.density
    r = MixingRatio(c, dynamics.moisture.total_density, ρ)

    return ( - div_ρUc(i, j, k, grid, advection, ρ, velocities, r)
             + c_forcing(i, j, k, grid, clock, model_fields))
end

# The microphysics of a hydrostatic model acts once per time step, after the dynamics, and adds
# nothing to the tendencies of the stages
AtmosphereModels.compute_microphysical_tendencies!(::MoistHydrostaticModel) = nothing

#####
##### Expansion work
#####

# Mass flux through an x- or y-face times the difference of the Exner function across it
@inline mass_flux_δxΠ(i, j, k, grid, ρ, u, Π) = mass_flux_x(i, j, k, grid, ρ, u) * δxᶠᵃᵃ(i, j, k, grid, Π)
@inline mass_flux_δyΠ(i, j, k, grid, ρ, v, Π) = mass_flux_y(i, j, k, grid, ρ, v) * δyᵃᶠᵃ(i, j, k, grid, Π)

# 𝒲 = (θ / H) (Rᵐ T - κ H) with H = cᵖᵐ T - ℒˡ₀ qˡ, written with Rᵈ = κ cᵖᵈ taken out so that
# it vanishes exactly without water: Rᵐ T - κ H = T [qᵛ (Rᵛ - κ cᵖᵛ) - κ qˡ cˡ] + κ ℒˡ₀ qˡ.
@inline function expansion_work_coefficient(θ, T, q, κ, constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒq = condensate_latent_energy(q, constants)
    H = mixture_heat_capacity(q, constants) * T - ℒq
    safe_H = ifelse(H == 0, one(H), H)
    return θ / safe_H * (T * (q.vapor * (Rᵛ - κ * cᵖᵛ) - κ * q.liquid * cˡ) + κ * ℒq)
end

# The first law, DH/Dt = α Dp/Dt, gives Dθ/Dt = 𝒲 D(ln p)/Dt for the potential temperature with
# the dry Exner exponent. The rate is that of the layer Exner function following the dry air,
#
#   ρ Δz DΠ/Dt = ρ Δz ∂ₜΠ + [mass fluxes × δΠ, averaged to the cell] / Az
#                + ρw⁺ (Πᶠ⁺ - Π) + ρw⁻ (Π - Πᶠ⁻) ,        D(ln p)/Dt = (DΠ/Dt) / (κ Π) ,
#
# with ∂ₜΠ = [(Πᶠ⁻ - Π) ∂ₜp⁻ + (Π - Πᶠ⁺) ∂ₜp⁺] / Δp from the tendencies of the total pressures
# of the two faces, ∂ₜpᶠ = B ∂ₜπˢ + (tendency of the weight of the water above). The water
# tendencies are those of the transport: the pressure change by falling rain is accounted for
# by the column physics, in the enthalpy it leaves in each layer.
@kernel function _add_expansion_work!(Gρθ, grid, ρ, u, v, ρw, Π, Πᶠ, pᶠ, B, Gπˢ, Gρqˣ, θ, T, qᵛ, qˡ, constants, κ, g)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ∂ₜp⁺ = zero(grid)
        ∂ₜΣW = zero(grid)

        for k in Nz:-1:1
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            ∂ₜΣW += g * Δz * sum_at(i, j, k, Gρqˣ)
            ∂ₜp⁻ = B[i, j, k] * Gπˢ[i, j, 1] + ∂ₜΣW

            Πᵏ = Π[i, j, k]
            Δp = pᶠ[i, j, k] - pᶠ[i, j, k+1]
            ∂ₜΠ = ((Πᶠ[i, j, k] - Πᵏ) * ∂ₜp⁻ + (Πᵏ - Πᶠ[i, j, k+1]) * ∂ₜp⁺) / Δp

            horizontal = Az⁻¹ᶜᶜᶜ(i, j, k, grid) * (ℑxᶜᵃᵃ(i, j, k, grid, mass_flux_δxΠ, ρ, u, Π) +
                                                    ℑyᵃᶜᵃ(i, j, k, grid, mass_flux_δyΠ, ρ, v, Π))
            vertical = ρw[i, j, k+1] * (Πᶠ[i, j, k+1] - Πᵏ) + ρw[i, j, k] * (Πᵏ - Πᶠ[i, j, k])
            ρ_DΠDt = ρ[i, j, k] * ∂ₜΠ + (horizontal + vertical) / Δz

            q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
            𝒲 = expansion_work_coefficient(θ[i, j, k], T[i, j, k], q, κ, constants)
            Gρθ[i, j, k] += 𝒲 * ρ_DΠDt / (κ * Πᵏ)

            ∂ₜp⁺ = ∂ₜp⁻
        end
    end
end

function AtmosphereModels.compute_dynamics_tendency!(model::MoistHydrostaticModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    Gπˢ = model.timestepper.Gⁿ.pˢ
    ρ = dynamics.density
    u, v, _ = model.velocities
    g = constants.gravitational_acceleration
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    water = hydrostatic_water_fields(model.microphysics, model.microphysical_fields)

    launch!(arch, grid, :xy, _compute_surface_pressure_tendency!, Gπˢ, grid, ρ, u, v, g)

    launch!(arch, grid, :xy, _add_expansion_work!,
            model.timestepper.Gⁿ.ρθ, grid, ρ, u, v, dynamics.vertical_mass_flux,
            dynamics.exner_function, dynamics.interface_exner_function, dynamics.moisture.interface_pressure,
            dynamics.surface_pressure_coefficient, Gπˢ, water_tendencies(model),
            model.formulation.potential_temperature, model.temperature, water.qᵛ, water.qˡ, constants, κ, g)

    return nothing
end

#####
##### The gravity-wave operator of the substeps with water (`gravity_wave_substepping.jl`)
#####
#
# The fast system keeps the structure of the dry one, with the coefficients of moist air frozen
# at the start of the step. Writing πˢ′ for the perturbation of the dry-mass surface pressure:
#
#   - continuity and the flux of θᴸ by the perturbation mass flux are those of dry air, because
#     the coordinate and every prognostic weight are those of dry air;
#   - the mixing ratios are frozen, so the weight of the water follows the dry mass,
#     W′ = rᵗᴸ ΔB πˢ′, and the total pressure of the z-face `k` responds as pᶠ′ = B̃ πˢ′ with
#     B̃ₖ = Bₖ + Σ_{j ≥ k} rᵗᴸⱼ ΔBⱼ. With (1 + κ) Δp Π = p⁻ Πᶠ⁻ - p⁺ Πᶠ⁺ of the total pressures,
#
#       ∂Πᶠ/∂πˢ = κ B̃ Πᶠ / pᶠ ,    ∂Π/∂πˢ = [B̃⁻ Πᶠ⁻ - B̃⁺ Πᶠ⁺ - ΔB (1 + rᵗᴸ) Π] / Δp ;
#
#   - the pressure-gradient force and the hydrostatic integral read the density potential
#     temperature with its factor frozen, θᵨ′ = ϑᴸ θ′:
#
#       ∂τ 𝐮′ = - cᵖᵈ ϑᴸ θ′ ∇Πᴸ - cᵖᵈ θᵨᴸ ∇Π′ - ∇Φ′ ,   Φ′ = Φ[ϑᴸ θ′; Πᴸ] + Φ[θᵨᴸ; ∂Π/∂πˢ] πˢ′ .
#
# What the frozen coefficients leave out (the dependence of ϑ on the water and on the phase
# partition, the latent heating of condensation in the waves, the expansion work) is in the slow
# tendency, which is the full tendency less this operator applied to the perturbation: the split
# scheme integrates the equations of the unsplit one. Without water every coefficient is that of
# the dry operator.

@kernel function _compute_density_factor!(ϑ, θᵨ, θ)
    i, j, k = @index(Global, NTuple)
    @inbounds ϑ[i, j, k] = θᵨ[i, j, k] / θ[i, j, k]
end

function freeze_moist_linearization!(substepper, model::MoistHydrostaticModel)
    grid = model.grid
    linearization = substepper.linearization
    hasproperty(linearization, :θᵨ) ||
        throw(ArgumentError("The gravity-wave substepper of this moist model was built without the moist " *
                            "linearization: build GravityWaveSubstepper with moist = true."))

    copy_parent!(linearization.θᵨ, model.dynamics.moisture.density_potential_temperature)

    @apply_regionally launch!(grid.architecture, grid, :xyz, _compute_density_factor!,
                              linearization.ϑ, linearization.θᵨ, linearization.θ)

    return nothing
end

@kernel function _compute_moist_exner_function_derivatives!(dΠdπˢ, dΠᶠdπˢ, grid, B, pᶠ, ρ, ρqˣ, Π, Πᶠ, κ)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        B̃⁺ = B[i, j, Nz+1]
        constant_pressure = B̃⁺ == 0
        safe_p = ifelse(constant_pressure, one(B̃⁺), pᶠ[i, j, Nz+1])
        dΠᶠdπˢ[i, j, Nz+1] = ifelse(constant_pressure, zero(B̃⁺), κ * B̃⁺ * Πᶠ[i, j, Nz+1] / safe_p)

        for k in Nz:-1:1
            ΔB = B[i, j, k] - B[i, j, k+1]
            rᵗ = sum_at(i, j, k, ρqˣ) / ρ[i, j, k]
            B̃⁻ = B̃⁺ + ΔB * (1 + rᵗ)
            Δp = pᶠ[i, j, k] - pᶠ[i, j, k+1]

            constant_pressure = B̃⁻ == 0
            safe_p = ifelse(constant_pressure, one(Δp), pᶠ[i, j, k])
            dΠᶠdπˢ[i, j, k] = ifelse(constant_pressure, zero(Δp), κ * B̃⁻ * Πᶠ[i, j, k] / safe_p)
            dΠdπˢ[i, j, k] = (B̃⁻ * Πᶠ[i, j, k] - B̃⁺ * Πᶠ[i, j, k+1] - ΔB * (1 + rᵗ) * Π[i, j, k]) / Δp

            B̃⁺ = B̃⁻
        end
    end
end

function launch_linearization_derivatives!(substepper, model, moisture::HydrostaticMoisture)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = dry_air_gas_constant(constants) / cᵖᵈ
    (; ρ, θᵨ, Π, Πᶠ, dΠdpˢ, dΠᶠdpˢ, dΦdpˢ) = substepper.linearization

    launch!(arch, grid, :xy, _compute_moist_exner_function_derivatives!, dΠdpˢ, dΠᶠdpˢ, grid,
            dynamics.surface_pressure_coefficient, moisture.interface_pressure, ρ, water_densities(model), Π, Πᶠ, κ)

    # ∂Φ/∂πˢ at fixed θᵨ: the hydrostatic integral is linear in the Exner functions
    launch!(arch, grid, :xy, _compute_geopotential!, dΦdpˢ, grid, ZeroField(eltype(grid)), θᵨ, dΠdpˢ, dΠᶠdpˢ, cᵖᵈ)

    return nothing
end

# The Lamb speed of moist air, √(γᵐ Rᵐ T), is bounded by that of dry air at the density
# temperature, because γᵐ < γᵈ; and Rᵐ / Rᵈ ≤ 1 + (Rᵛ / Rᵈ - 1) qᵛ.
function maximum_gravity_wave_speed(model::MoistHydrostaticModel)
    constants = model.thermodynamic_constants
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
    water = hydrostatic_water_fields(model.microphysics, model.microphysical_fields)
    T = maximum(model.temperature)
    qᵛ = max(0, maximum(water.qᵛ))
    return sqrt(γ * Rᵈ * T * (1 + (Rᵛ / Rᵈ - 1) * qᵛ))
end

#####
##### Column physics of warm rain, once per time step
#####

# Conversion of cloud liquid to rain and evaporation of rain, over Δt. Each sink is limited to
# what its donor holds, which is a limit on a rate between two reservoirs and not a clipping:
# nothing is made negative, and what leaves one density enters the other. The total water and
# the pressure of the layer do not change, and θ is invariant under a phase change at fixed
# pressure and total water, so the latent heat is in the temperature of the next diagnosis.
@kernel function _apply_rain_sources!(ρqᵉ, ρqʳ, grid, microphysics, ρᵗ, p, T, qᵛ, qˡ, qᶜˡ, constants, Δt)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρᵏ = ρᵗ[i, j, k]
        ρqʳᵏ = ρqʳ[i, j, k]
        qʳ = ρqʳᵏ / ρᵏ
        qᶜˡᵏ = qᶜˡ[i, j, k]
        Tᵏ = T[i, j, k]
        q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
        ρᵖʰ = Thermodynamics.density(Tᵏ, p[i, j, k], q, constants)

        conversion, evaporation = rain_source_rates(microphysics, ρᵖʰ, qᶜˡᵏ, qʳ, q, Tᵏ, constants)
        conversion = ifelse(qᶜˡᵏ > 0, conversion, zero(conversion))
        evaporation = ifelse(qʳ > 0, evaporation, zero(evaporation))

        to_rain = min(ρᵏ * Δt * conversion, ρᵏ * max(0, qᶜˡᵏ))
        to_vapor = max(ρᵏ * Δt * evaporation, - max(0, ρqʳᵏ))
        δρqʳ = to_rain + to_vapor

        ρqʳ[i, j, k] = ρqʳᵏ + δρqʳ
        ρqᵉ[i, j, k] -= δρqʳ
    end
end

# Sedimentation of rain over Δt, implicit in time and upwind in the physical layers, from the
# model top down: with λ = V Δt / (physical depth) the Courant number of a layer at the fall
# speed of the start of the step, the layer keeps (Wʳ + inflow) / (1 + λ) of the weight of its
# rain and passes on λ times that. This is positive for any λ, so there is nothing to subcycle.
#
# The rain that leaves a layer carries the enthalpy of liquid water at the temperature of that
# layer, hˡ = cˡ T - ℒˡ₀, into the layer below or through the ground. The potential temperature
# of every layer the rain crossed is then expressed again from its enthalpy at its new pressure
# and composition: the enthalpy of the layers changes only by what the rain carries, so the
# potential energy that the fall releases stays in the air as heat (the frictional dissipation
# of falling rain), and the column loses hˡ + Φˢ per unit mass of rain reaching the ground.
@kernel function _sediment_rain!(ρθ, ρqʳ, ledger, grid, microphysics, ρqᵉ, ρ, A, B, πˢ, Φˢ, Φ, θ, θᵨ, Π, Πᶠ, p, T,
                                 qᵛ, qˡ, constants, pˢᵗ, κ, g, Δt, inflow_fall_speed)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    @inbounds begin
        ΣW = zero(grid)             # weight of the water above the upper face, after sedimentation
        inflow = zero(grid)         # weight of the rain entering through the upper face
        enthalpy_inflow = zero(grid)
        friction = zero(grid)
        courant = zero(grid)
        p⁺ = interface_pressure(i, j, Nz+1, A, B, πˢ)
        Πᶠ⁺ = (p⁺ / pˢᵗ)^κ

        for k in Nz:-1:1
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            Δπ = layer_pressure_thickness(i, j, k, A, B, πˢ)
            ρᵏ = ρ[i, j, k]
            ρqᵉᵏ = ρqᵉ[i, j, k]
            ρqʳᵏ = ρqʳ[i, j, k]
            Tᵏ = T[i, j, k]

            ρᵖʰ = physical_density(i, j, k, p, T, qᵛ, qˡ, constants)
            # The fall speed of the rain of the layer before this sweep, or with `inflow_fall_speed`
            # of that rain together with what enters it from above in this sweep
            ρqʳᵛ = ifelse(inflow_fall_speed, ρqʳᵏ + inflow / (g * Δz), ρqʳᵏ)
            V = positive_rain_fall_speed(microphysics, ρᵖʰ, ρqʳᵛ / (ρᵏ + ρqᵉᵏ + ρqʳᵛ))
            λ = V * Δt / physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)

            Wʳ = g * Δz * ρqʳᵏ
            Wʳ⁺ = (Wʳ + inflow) / (1 + λ)
            outflow = λ * Wʳ⁺
            hˡ = cˡ * Tᵏ - ℒˡ₀
            crossed = (inflow != 0) | (outflow != 0)

            # Enthalpy of the layer, Δπ (cᵖᵈ + rᵗ cᵖᵛ) Π θ, before and after
            Wᵗ = g * Δz * (ρqᵉᵏ + ρqʳᵏ)
            𝓗 = (Δπ * cᵖᵈ + Wᵗ * cᵖᵛ) * Π[i, j, k] * θ[i, j, k]
            𝓗⁺ = 𝓗 + enthalpy_inflow - outflow * hˡ

            # Pressure and Exner function of the layer under the water that is now above it
            ρqʳ⁺ = Wʳ⁺ / (g * Δz)
            Wᵗ⁺ = g * Δz * (ρqᵉᵏ + ρqʳ⁺)
            ΣW += ifelse(crossed, Wᵗ⁺, Wᵗ)
            p⁻ = interface_pressure(i, j, k, A, B, πˢ) + ΣW
            Πᶠ⁻ = (p⁻ / pˢᵗ)^κ
            Π⁺ = (p⁻ * Πᶠ⁻ - p⁺ * Πᶠ⁺) / ((1 + κ) * (Δπ + Wᵗ⁺))
            θ⁺ = 𝓗⁺ / ((Δπ * cᵖᵈ + Wᵗ⁺ * cᵖᵛ) * Π⁺)

            ρθ[i, j, k] = ifelse(crossed, ρᵏ * θ⁺, ρθ[i, j, k])
            ρqʳ[i, j, k] = ifelse(crossed, ρqʳ⁺, ρqʳᵏ)

            # Geopotential the rain falls through to the layer below, or to the ground
            below = max(k - 1, 1)
            Φ⁻ = ifelse(k == 1, Φˢ[i, j, 1], Φ[i, j, below])
            friction += outflow / g * (Φ[i, j, k] - Φ⁻)
            courant = max(courant, λ)

            inflow = outflow
            enthalpy_inflow = outflow * hˡ
            p⁺ = p⁻
            Πᶠ⁺ = Πᶠ⁻
        end

        ledger.surface_rain[i, j, 1] += inflow / g
        ledger.precipitation_energy[i, j, 1] += (enthalpy_inflow + inflow * Φˢ[i, j, 1]) / g
        ledger.rain_frictional_heating[i, j, 1] += friction
        ledger.maximum_sedimentation_courant[i, j, 1] = max(ledger.maximum_sedimentation_courant[i, j, 1], courant)
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Apply the column physics of a warm-rain `microphysics` scheme to `model`, an `AtmosphereModel`
with moist [`HydrostaticDynamics`](@ref), over the time step that has just been taken: the
conversion of cloud liquid to rain and the evaporation of rain, then the sedimentation of rain,
each on the state the previous one leaves (first-order sequential splitting after the
dynamics). `sources = false` or `sedimentation = false` leaves one of the two out. With
`substeps = n` both are applied `n` times over `Δt / n`, each on the state the previous leaves.

The rates and the fall speed are those of `microphysics`, evaluated with the density of the air
in kg m⁻³ and with the physical depth of the layers, neither of which is a quantity of the grid.
The rain reaching the ground, its enthalpy and geopotential, and the heating of the air by the
fall are accumulated in the ledger of the dynamics, `model.dynamics.moisture.ledger`.
"""
function warm_rain_column_physics!(microphysics, model::MoistHydrostaticModel;
                                   Δt=model.clock.last_Δt, sources=true, sedimentation=true, substeps=1,
                                   inflow_fall_speed=false)
    (isfinite(Δt) && Δt > 0) || return nothing
    substeps ≥ 1 || throw(ArgumentError("substeps must be a positive integer, got $substeps"))

    for n in 1:substeps
        apply_warm_rain_column_physics!(microphysics, model, Δt / substeps, sources, sedimentation, n == substeps,
                                        inflow_fall_speed)
    end

    return nothing
end

# One application of the column physics over Δt; tendencies are computed only after the last one.
function apply_warm_rain_column_physics!(microphysics, model, Δt, sources, sedimentation, last, inflow_fall_speed)

    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    moisture = dynamics.moisture
    constants = model.thermodynamic_constants
    water = hydrostatic_water_fields(microphysics, model.microphysical_fields)
    Δt = convert(eltype(grid), Δt)
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    if sources
        @apply_regionally launch!(arch, grid, :xyz, _apply_rain_sources!,
                                  model.moisture_density, water.ρqʳ, grid, microphysics, moisture.total_density,
                                  dynamics.pressure, model.temperature, water.qᵛ, water.qˡ, water.qᶜˡ, constants, Δt)

        update_state!(model; compute_tendencies = !sedimentation & last)
    end

    sedimentation || return nothing

    @apply_regionally launch!(arch, grid, :xy, _sediment_rain!,
                              model.formulation.potential_temperature_density, water.ρqʳ, moisture.ledger,
                              grid, microphysics, model.moisture_density, dynamics.density,
                              dynamics.pressure_coefficient, dynamics.surface_pressure_coefficient,
                              dynamics.surface_pressure, dynamics.surface_geopotential, dynamics.geopotential,
                              model.formulation.potential_temperature, moisture.density_potential_temperature,
                              dynamics.exner_function, dynamics.interface_exner_function, dynamics.pressure,
                              model.temperature, water.qᵛ, water.qˡ, constants, dynamics.standard_pressure, κ, g, Δt,
                              inflow_fall_speed)

    update_state!(model; compute_tendencies=last)

    return nothing
end

# The precipitation ledger accumulates over the integration, so a restart carries it on
AtmosphereModels.dynamics_restart_state(dynamics::MoistHydrostaticDynamics) =
    Oceananigans.prognostic_state(dynamics.moisture.ledger)

function AtmosphereModels.restore_dynamics_restart_state!(dynamics::MoistHydrostaticDynamics, from::NamedTuple)
    Oceananigans.restore_prognostic_state!(dynamics.moisture.ledger, from)
    return dynamics
end

"""
    WarmRainColumnPhysics(; substeps=1)

A callable for `Callback(WarmRainColumnPhysics(; substeps), IterationInterval(1))` that applies
[`warm_rain_column_physics!`](@ref) after every time step, over the step just taken
(`model.clock.last_Δt`), in `substeps` equal applications. The time stepper must not apply the
column physics as well: call `column_physics_in_time_step!(model, false)` first, without which
the callback throws an `ArgumentError`.
"""
struct WarmRainColumnPhysics
    substeps :: Int
end

WarmRainColumnPhysics(; substeps=1) = WarmRainColumnPhysics(substeps)

# It carries no state: a restart rebuilds it from the script
Oceananigans.prognostic_state(::WarmRainColumnPhysics) = nothing
Oceananigans.restore_prognostic_state!(physics::WarmRainColumnPhysics, ::Nothing) = physics

function (physics::WarmRainColumnPhysics)(simulation)
    model = simulation.model
    column_physics_in_time_step(model) &&
        throw(ArgumentError("WarmRainColumnPhysics would apply the column physics a second time: the time " *
                            "stepper applies it at the end of every step. Call " *
                            "column_physics_in_time_step!(model, false) before the first step."))
    return warm_rain_column_physics!(model.microphysics, model; substeps=physics.substeps)
end

#####
##### What a moist model may be built with
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Check the combinations that moist `HydrostaticDynamics` does not support, each of which is
rejected with an `ArgumentError`. A dry model passes: it carries no water to check.
"""
validate_moist_hydrostatic_model(model) = nothing

function validate_moist_hydrostatic_model(model::MoistHydrostaticModel)
    dynamics = model.dynamics

    model.formulation.exner_exponent isa DryExponent ||
        throw(ArgumentError("Moist HydrostaticDynamics requires the liquid-ice potential temperature with the " *
                            "dry Exner exponent, formulation = LiquidIcePotentialTemperatureFormulation(" *
                            "exner_exponent=DryExponent()), got exner_exponent = $(model.formulation.exner_exponent): " *
                            "with any other exponent the Exner function depends on the water."))

    model.thermodynamic_constants isa ClausiusClapeyronThermodynamicConstants ||
        unsupported_by_hydrostatic_dynamics("moisture with a saturation vapor pressure other than ClausiusClapeyron")

    isnothing(AtmosphereModels.negative_moisture_correction(model.microphysics)) ||
        unsupported_by_hydrostatic_dynamics("moisture with a negative_moisture_correction, which moves water " *
                                            "outside the ledger,")

    isnothing(model.radiation) ||
        unsupported_by_hydrostatic_dynamics("radiation with moisture: the pressures of the coordinate surfaces " *
                                            "given to the radiative transfer are not those of moist air")

    isnothing(dynamics.hyperdiffusion) || unsupported_by_hydrostatic_dynamics("hyperdiffusion with moisture")
    isnothing(dynamics.vertical_diffusion) || unsupported_by_hydrostatic_dynamics("vertical_diffusion with moisture")

    validate_moist_energy_boundary_conditions(model)

    return nothing
end

# A flux of energy through a boundary would enter ρθ with the heat capacity of dry air
function validate_moist_energy_boundary_conditions(model::AtmosphereModel)
    ρθ = model.formulation.potential_temperature_density
    @apply_regionally validate_moist_energy_boundary_conditions(ρθ.boundary_conditions)
    return nothing
end

function validate_moist_energy_boundary_conditions(boundary_conditions)
    for side in (:west, :east, :south, :north, :bottom, :top, :immersed)
        is_dry_boundary_condition(getproperty(boundary_conditions, side)) ||
            unsupported_by_hydrostatic_dynamics("moisture with a source of energy on the $side boundary: the " *
                                                "enthalpy of a surface flux into moist air is not implemented")
    end
    return nothing
end

#####
##### Setting the state of a moist model
#####

@inline scale_at!(i, j, k, ::Tuple{}, factor) = nothing

@inline function scale_at!(i, j, k, fields::Tuple, factor)
    @inbounds first(fields)[i, j, k] *= factor
    return scale_at!(i, j, k, Base.tail(fields), factor)
end

# `set!` weights a mass fraction with the continuity density, which here is that of dry air, so
# water given as a mass fraction q arrives as ρ q. The mass fraction is per unit total mass:
# the total density is ρᵗ = (ρ + water given as a density) / (1 - Σ q), and the densities of the
# water given as mass fractions are scaled by ρᵗ / ρ. A total water includes the condensate, so
# then the water of the microphysics is not counted again.
@kernel function _weight_mass_fractions_by_total_density!(ρqᵉ, ρᵗ, grid, ρ, specific_fields, density_fields,
                                                          specific_moisture_given, total_moisture_given)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        ρᵏ = ρ[i, j, k]
        moisture = ρqᵉ[i, j, k]
        others = ifelse(total_moisture_given, zero(ρᵏ), one(ρᵏ))
        specific = ifelse(specific_moisture_given, moisture, zero(ρᵏ)) + others * sum_at(i, j, k, specific_fields)
        given = ifelse(specific_moisture_given, zero(ρᵏ), moisture) + others * sum_at(i, j, k, density_fields)
        factor = (ρᵏ + given) / (ρᵏ - specific)

        ρqᵉ[i, j, k] = ifelse(specific_moisture_given, moisture * factor, moisture)
        scale_at!(i, j, k, specific_fields, factor)
        ρᵗ[i, j, k] = ρᵏ * factor
    end
end

function AtmosphereModels.establish_densities!(model::MoistHydrostaticModel,
                                               total_density_given,
                                               dry_density_given,
                                               moisture_given = false,
                                               specific_moisture_given = false,
                                               total_moisture_given = false,
                                               specific_microphysical_names = ())
    grid = model.grid
    arch = grid.architecture
    μ = model.microphysical_fields
    names = AtmosphereModels.prognostic_field_names(model.microphysics)
    specific_density_names = map(AtmosphereModels.specific_to_density_weighted, specific_microphysical_names)
    specific_fields = map(name -> μ[name], specific_density_names)
    density_fields = Tuple(μ[name] for name in names if name ∉ specific_density_names)
    ρᵗ = model.dynamics.moisture.total_density

    @apply_regionally launch!(arch, grid, :xyz, _weight_mass_fractions_by_total_density!,
                              model.moisture_density, ρᵗ, grid, model.dynamics.density,
                              specific_fields, density_fields, specific_moisture_given, total_moisture_given)

    fill_halo_regions!(ρᵗ)

    return nothing
end

# The dry-mass surface pressure for which the total surface pressure is pˢ, with the mixing
# ratio rᵗ of the water of each layer: πˢ = (pˢ - Σ ΔA rᵗ) / (1 + Σ ΔB rᵗ)
@kernel function _compute_dry_surface_pressure!(πˢ, grid, pˢ, A, B, ρ, ρqˣ)
    i, j = @index(Global, NTuple)

    @inbounds begin
        ΣΔA_rᵗ = zero(grid)
        ΣΔB_rᵗ = zero(grid)

        for k in 1:size(grid, 3)
            rᵗ = sum_at(i, j, k, ρqˣ) / ρ[i, j, k]
            ΣΔA_rᵗ += (A[i, j, k] - A[i, j, k+1]) * rᵗ
            ΣΔB_rᵗ += (B[i, j, k] - B[i, j, k+1]) * rᵗ
        end

        πˢ[i, j, 1] = (pˢ[i, j, 1] - ΣΔA_rᵗ) / (1 + ΣΔB_rᵗ)
    end
end

largest_difference(a, b) = maximum(abs, a - b)

const maximum_set_iterations = 60

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set variables of an `AtmosphereModel` with moist `HydrostaticDynamics`.

`pˢ` is the total surface pressure, the weight of the dry air and of all the water of the
column, which is what a surface pressure means anywhere else. The prognostic variable is the
dry-mass surface pressure, diagnosed here from `pˢ` and the water; without `pˢ` it is left as
it is. The other keyword arguments are set as for any `AtmosphereModel`.

The pressure that relates a temperature or a relative humidity to the prognostic variables
depends on the water, and the densities that weight the mass fractions on the surface
pressure, so the state is set repeatedly until the total surface pressure no longer changes.
"""
function Fields.set!(model::MoistHydrostaticModel; pˢ=nothing, kw...)
    (haskey(kw, :ρu) || haskey(kw, :ρv)) &&
        throw(ArgumentError("Cannot set the momentum density of HydrostaticDynamics, whose prognostic " *
                            "variables are the velocities u and v."))

    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    πˢ = dynamics.surface_pressure
    diagnosed = dynamics.moisture.total_surface_pressure
    FT = eltype(grid)
    tolerance = 100 * eps(FT)

    target = Field{Center, Center, Nothing}(grid)
    previous = Field{Center, Center, Nothing}(grid)
    isnothing(pˢ) || (set!(target, pˢ); set!(πˢ, target))

    for iteration in 1:maximum_set_iterations
        diagnose_layer_mass_of_new_state!(model)
        copy_parent!(previous, diagnosed)
        invoke(set!, Tuple{AtmosphereModel}, model; kw...)

        scale = maximum(abs, diagnosed)
        change = largest_difference(diagnosed, previous)
        mismatch = isnothing(pˢ) ? zero(change) : largest_difference(diagnosed, target)

        if !isnothing(pˢ)
            @apply_regionally launch!(arch, grid, :xy, _compute_dry_surface_pressure!, πˢ, grid, target,
                                      dynamics.pressure_coefficient, dynamics.surface_pressure_coefficient,
                                      dynamics.density, water_densities(model))
            fill_halo_regions!(πˢ)
        end

        if change ≤ tolerance * scale && mismatch ≤ tolerance * scale
            diagnose_layer_mass_of_new_state!(model)
            update_state!(model; compute_tendencies=false)
            return nothing
        end
    end

    throw(ArgumentError("The state of moist HydrostaticDynamics did not converge in $maximum_set_iterations " *
                        "iterations: the total surface pressure keeps changing as the state is set."))
end

#####
##### Not supported with moisture yet
#####

# The checks of the state take the pressure thickness of the layers from the surface pressure
# alone, and the conservation fixer restores integrals that the sources of water change.
hydrostatic_state_check(::MoistHydrostaticModel; kw...) =
    unsupported_by_hydrostatic_dynamics("hydrostatic_state_check with moisture")

HydrostaticConservationFixer(::MoistHydrostaticModel) =
    unsupported_by_hydrostatic_dynamics("the HydrostaticConservationFixer with moisture")
