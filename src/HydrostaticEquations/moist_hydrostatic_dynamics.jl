#####
##### Moist hydrostatic dynamics: a total-mass coordinate
#####
#
# With water, the coordinate, continuity and every prognostic weight are those of the total mass,
# dry air and all of its water together, and the pressure is the weight of everything above:
#
#   Δp = ΔA + ΔB pˢ               total pressure thickness of a layer; pˢ is the prognostic surface pressure
#   pᶠ = A + B pˢ                 total pressure of a z-face
#   ρ                             the layer mass of the coordinate, Δp / (g Δz), of dry air and water
#   q = ρqˣ / ρ                   the mass fractions of the water, per unit total mass
#
# The prognostic surface pressure is the total surface pressure, which is what a surface pressure
# means anywhere else. Nothing in the equations is per unit dry mass and nothing divides by the
# dry density, so they hold as they are in a layer of water vapor alone (ρᵈ → 0). The transport of
# water by the motion does not change the pressure; water that leaves the column through the
# ground (precipitation) or enters it (evaporation) does, by g times its mass per unit area, and
# the column physics that moves it changes pˢ with it (`change_column_surface_pressure!`).
#
# The potential temperature of the model is the liquid-ice potential temperature θ with the dry
# Exner exponent, θ = H / (ĉᵖ Π) with H = cᵖᵐ T - ℒˡ₀ qˡ the enthalpy and ĉᵖ = qᵈ cᵖᵈ + qᵗ cᵖᵛ, so
# that the Exner function depends on the pressure alone. The pressure-gradient force and the
# hydrostatic integral are those of dry air with θ replaced by the density potential temperature
# θᵨ = ϑ θ,
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

# When the state of a moist model is checked during an integration: every `interval` time steps,
# or never with `interval = 0`. See `validate_moist_hydrostatic_state`.
mutable struct StateValidation
    interval :: Int
end

struct HydrostaticMoisture{T, I, S, D, E, L, C, V}
    density_potential_temperature :: T  # θᵨ = ϑ θ at cell centers
    interface_pressure :: I             # total pressure pᶠ = A + B pˢ at z-faces
    total_surface_pressure :: S         # pˢ, the prognostic surface pressure itself
    total_density :: D                  # ρ, the layer mass of the coordinate itself
    column_enthalpy :: E                # scratch of the column physics: the enthalpy a layer is given
    ledger :: L                         # accumulated fluxes through the ground (two-dimensional)
    column_physics :: C                 # ColumnPhysicsDispatch, or nothing inside kernels
    state_validation :: V               # StateValidation, or nothing inside kernels
end

# Kernels read the fields only
Adapt.adapt_structure(to, moisture::HydrostaticMoisture) =
    HydrostaticMoisture(adapt(to, moisture.density_potential_temperature),
                        adapt(to, moisture.interface_pressure),
                        adapt(to, moisture.total_surface_pressure),
                        adapt(to, moisture.total_density),
                        adapt(to, moisture.column_enthalpy),
                        adapt(to, moisture.ledger),
                        nothing,
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
            precipitation_energy = Field{Center, Center, Nothing}(grid),          # J m⁻²: P (cˡ T - ℒˡ₀ + Φˢ) + kinetic
            precipitation_kinetic_energy = Field{Center, Center, Nothing}(grid),  # J m⁻²: the kinetic part, P Σ ΔB 𝒦
            rain_frictional_heating = Field{Center, Center, Nothing}(grid),       # J m⁻²
            maximum_sedimentation_courant = Field{Center, Center, Nothing}(grid)) # V Δt / (physical depth)
end

# The total surface pressure and the total density are those of the coordinate: the same fields
function HydrostaticMoisture(grid, surface_pressure, density)
    return HydrostaticMoisture(CenterField(grid),
                               ZFaceField(grid),
                               surface_pressure,
                               density,
                               CenterField(grid),
                               precipitation_ledger(grid),
                               ColumnPhysicsDispatch(true),
                               StateValidation(1))
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
a driver has taken it over with [`column_physics_in_time_step!`](@ref), and check the state of
the model (see [`state_validation_interval!`](@ref)). Called by the time steppers.
"""
function time_step_column_physics!(microphysics, model::MoistHydrostaticModel)
    column_physics_in_time_step(model) && apply_column_physics!(microphysics, model)

    # The end of the time step, whoever applies the column physics: the state is checked here, once,
    # and not in each of the `update_state!`s of the step
    validate_time_step_state(model)

    return nothing
end

# The column physics of a scheme: warm rain (sources and sedimentation) by default; a scheme
# without rain between steps extends this with `rain_out_column_physics!`
apply_column_physics!(microphysics, model) = warm_rain_column_physics!(microphysics, model)

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
                        HydrostaticMoisture(grid, dynamics.surface_pressure, dynamics.density),
                        dynamics.height_coordinate)

# A model with a microphysics scheme carries water, and its dynamics the diagnostics of the
# moist column
function AtmosphereModels.materialize_dynamics(dynamics::HydrostaticDynamics, grid, boundary_conditions, constants,
                                               microphysics, formulation)
    dry_dynamics = AtmosphereModels.materialize_dynamics(dynamics, grid, boundary_conditions, constants)
    return with_moisture(dry_dynamics, microphysics, grid)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the total density of moist `HydrostaticDynamics`: the layer mass of its coordinate, of dry
air and all water, per unit area and unit of the grid's vertical coordinate. It recovers the mass
fractions from the prognostic densities. On a label grid it is not the density of the air in
kg m⁻³, which is ``p / (Rᵐ T)``.
"""
AtmosphereModels.total_density(dynamics::MoistHydrostaticDynamics) = dynamics.density

# The prognostic surface pressure, which is the total surface pressure
AtmosphereModels.surface_pressure(dynamics::MoistHydrostaticDynamics) = dynamics.surface_pressure

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

# The densities whose mass per unit area a change of the surface pressure keeps: the water and the tracers
# The horizontal velocities, whose kinetic energy the water carries through the ground
horizontal_velocities(model) = (u = model.velocities.u, v = model.velocities.v)

column_mass_densities(model) = (water_densities(model)..., values(model.tracers)...)

# The prognostic densities of water: the moisture density and the prognostic densities of the
# microphysics scheme, all of which are water in the supported schemes
water_densities(model) = (model.moisture_density,
                          map(name -> model.microphysical_fields[name],
                              AtmosphereModels.prognostic_field_names(model.microphysics))...)

@inline sum_at(i, j, k, ::Tuple{}) = 0
@inline sum_at(i, j, k, fields::Tuple) = @inbounds first(fields)[i, j, k] + sum_at(i, j, k, Base.tail(fields))

#####
##### Layer mass, total pressure and Exner functions
#####

# The layer mass and the Exner functions of moist air are those of dry air at the total surface
# pressure: what the kernels of moist air read in addition is the total pressure of the z-faces.
@kernel function _compute_moist_layer_mass_and_exner_functions!(ρ, p, Π, Πᶠ, pᶠ, grid, A, B, pˢ, pˢᵗ, κ, g)
    i, j = @index(Global, NTuple)
    compute_column_moist_layer_mass_and_exner_functions!(i, j, grid, ρ, p, Π, Πᶠ, pᶠ, A, B, pˢ, pˢᵗ, κ, g)
end

@inline function compute_column_moist_layer_mass_and_exner_functions!(i, j, grid, ρ, p, Π, Πᶠ, pᶠ, A, B, pˢ, pˢᵗ, κ, g)
    Nz = size(grid, 3)

    @inbounds begin
        p⁻ = interface_pressure(i, j, 1, A, B, pˢ)
        Πᶠ⁻ = (p⁻ / pˢᵗ)^κ
        pᶠ[i, j, 1] = p⁻
        Πᶠ[i, j, 1] = Πᶠ⁻

        for k in 1:Nz
            p⁺ = interface_pressure(i, j, k+1, A, B, pˢ)
            Πᶠ⁺ = (p⁺ / pˢᵗ)^κ
            Δp = layer_pressure_thickness(i, j, k, A, B, pˢ)
            Πᵏ = layer_exner_function(p⁻, Πᶠ⁻, p⁺, Πᶠ⁺, Δp, κ)

            ρ[i, j, k] = coordinate_layer_density(i, j, k, grid, ρ, Δp, g)
            Π[i, j, k] = Πᵏ
            p[i, j, k] = pˢᵗ * Πᵏ^(1 / κ)
            pᶠ[i, j, k+1] = p⁺
            Πᶠ[i, j, k+1] = Πᶠ⁺

            p⁻ = p⁺
            Πᶠ⁻ = Πᶠ⁺
        end
    end

    return nothing
end

function AtmosphereModels.compute_total_density!(model::MoistHydrostaticModel)
    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    @apply_regionally launch!(arch, grid, :xy, _compute_moist_layer_mass_and_exner_functions!,
                              dynamics.density,
                              dynamics.pressure,
                              dynamics.exner_function,
                              dynamics.interface_exner_function,
                              dynamics.moisture.interface_pressure,
                              grid,
                              dynamics.pressure_coefficient,
                              dynamics.surface_pressure_coefficient,
                              dynamics.surface_pressure,
                              dynamics.standard_pressure, κ, g)

    # The halos that a kernel reads: those of the density (mass fluxes, mass fractions) and of the
    # Exner function (pressure-gradient force, expansion work), filled together. The layer pressure
    # and the interface pressure are read in their own columns only, and their halos are not filled.
    fill_scalar_halos!(grid, (dynamics.density, dynamics.exner_function))

    return nothing
end

# Fill the halos of several scalar fields. Oceananigans fills those that share a location and
# boundary conditions with one kernel per side. On a multi-region grid, where a pair of fields can
# be taken for the two components of a vector, they are filled one by one.
fill_scalar_halos!(grid, fields, args...) = fill_halo_regions!(fields, args...)

function fill_scalar_halos!(::MultiRegion.MultiRegionGrids, fields, args...)
    foreach(field -> fill_halo_regions!(field, args...), fields)
    return nothing
end

# The moist diagnosis does not check the layer mass itself (see `validate_moist_hydrostatic_state`)
validate_new_layer_mass(model::MoistHydrostaticModel) = validate_layer_mass(model.dynamics)

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
    compute_cell_density_potential_temperature!(i, j, k, θᵨ, θ, Π, qᵛ, qˡ, constants)
end

@inline function compute_cell_density_potential_temperature!(i, j, k, θᵨ, θ, Π, qᵛ, qˡ, constants)
    @inbounds begin
        q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
        θᵨ[i, j, k] = density_potential_temperature_factor(θ[i, j, k], Π[i, j, k], q, constants) * θ[i, j, k]
    end
    return nothing
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
@kernel function _compute_rain_sedimentation_velocity!(wʳ, grid, microphysics, ρ, ρqʳ, θᵨ, Πᶠ, p, T, qᵛ, qˡ,
                                                       constants, cᵖᵈ, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ρᵖʰ = physical_density(i, j, k, p, T, qᵛ, qˡ, constants)
        V = positive_rain_fall_speed(microphysics, ρᵖʰ, ρqʳ[i, j, k] / ρ[i, j, k])
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

# What is wrong with a cell, as bits: 1 for a layer whose mass is not positive (or not a number),
# 2 for a density of water that is not a number
@inline all_finite() = true
@inline all_finite(ρq, others...) = isfinite(ρq) & all_finite(others...)

@inline cell_state_defects(ρ, ρqˣ...) = ifelse(ρ > 0, 0x00, 0x01) | ifelse(all_finite(ρqˣ...), 0x00, 0x02)

# The defects of all the cells of `model`, in one reduction over the dry density and the densities of water
state_defects(model) = state_defects(model.grid, model.dynamics.density, water_densities(model))

state_defects(grid, ρ, ρqˣ) = mapreduce(cell_state_defects, |, interior(ρ), map(interior, ρqˣ)...; init=0x00)

function state_defects(grid::MultiRegion.MultiRegionGrids, ρ, ρqˣ)
    defects = 0x00
    for region in regions(grid)
        regional_ρqˣ = map(ρq -> getregion(ρq, region), ρqˣ)
        defects |= state_defects(getregion(grid, region), getregion(ρ, region), regional_ρqˣ)
    end
    return defects
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Throw unless the state of `model`, an `AtmosphereModel` with moist `HydrostaticDynamics`, is one
that can be integrated: a `DomainError` if a layer has a mass that is not positive (see
[`validate_layer_mass`](@ref)), an `ArgumentError` if a density of water is not a number.

Both are found by one reduction over the grid, which copies one number from the device. That
copy waits for the device, so the state is not checked in `update_state!`, which a time step
calls several times: it is checked when the state is set, and at the end of a time step (see
[`state_validation_interval!`](@ref)). A state that fails inside a time step therefore stops the
integration at the end of that step rather than at the stage that produced it; nothing that is
not a number becomes one again in between.
"""
function validate_moist_hydrostatic_state(model)
    iszero(state_defects(model)) && return nothing

    # Something is wrong: find what, and say so
    validate_layer_mass(model.dynamics)
    validate_moist_state(model)

    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Check the state of `model`, an `AtmosphereModel` with moist `HydrostaticDynamics`, at the end of
every `interval`-th time step (see [`validate_moist_hydrostatic_state`](@ref)), or never during
an integration with `interval = 0`. A new model checks every time step. The state is always
checked when it is set.
"""
function state_validation_interval!(model::MoistHydrostaticModel, interval::Integer)
    interval ≥ 0 || throw(ArgumentError("The state validation interval must not be negative, got $interval"))
    model.dynamics.moisture.state_validation.interval = interval
    return nothing
end

state_validation_interval(model::MoistHydrostaticModel) = model.dynamics.moisture.state_validation.interval

function validate_time_step_state(model)
    interval = state_validation_interval(model)
    (interval > 0 && model.clock.iteration % interval == 0) && validate_moist_hydrostatic_state(model)
    return nothing
end

# The halos of the auxiliary thermodynamic variables that a kernel reads across a column (those of
# the potential temperature and of the mass fractions that are transported) are filled with the
# others that the tendencies read, once everything is diagnosed
# (`compute_auxiliary_dynamics_variables!`): no kernel in between reads across a column.
AtmosphereModels.fill_auxiliary_thermodynamic_halos!(::MoistHydrostaticModel) = nothing

# The halos of the prognostic velocities are filled with those of the other prognostic variables,
# at the start of `update_state!`. On a distributed grid that fill may not be complete yet.
function AtmosphereModels.compute_velocities!(model::MoistHydrostaticModel)
    model.grid.architecture isa Distributed &&
        fill_horizontal_velocity_halos!(model.momentum, boundary_condition_args(model)...)
    return nothing
end

# The bounds-preserving limiter of the water is read by the tendencies alone, so it is computed
# only when they are. The time steps of adaptive implicit advection are refreshed either way.
function AtmosphereModels.update_advection_state!(model::MoistHydrostaticModel, compute_tendencies)
    if compute_tendencies
        @apply_regionally update_advection!(model.advection, model)
    else
        @apply_regionally update_adaptive_time_steps!(model.advection, model)
    end
    return nothing
end

function update_adaptive_time_steps!(advection, model)
    foreach(scheme -> update_adaptive_timestep!(scheme, model), values(advection))
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

    @apply_regionally launch!(arch, grid, :xyz, _compute_density_potential_temperature!,
                              θᵨ, grid, θ, dynamics.exner_function, water.qᵛ, water.qˡ, constants)

    @apply_regionally launch!(arch, grid, :xy, _compute_geopotential!,
                              dynamics.geopotential, grid, dynamics.surface_geopotential,
                              θᵨ, dynamics.exner_function, dynamics.interface_exner_function, cᵖᵈ)

    rediagnose_layer_density!(grid, model, θᵨ)

    @apply_regionally launch!(arch, grid, :xy, _compute_vertical_mass_flux!,
                              dynamics.vertical_mass_flux, w, grid,
                              dynamics.surface_pressure_coefficient, dynamics.density, u, v)

    # A scheme without rain between time steps (instantaneous precipitation) has no fall speed
    isnothing(water.wʳ) ||
        @apply_regionally launch!(arch, grid, :xyz, _compute_rain_sedimentation_velocity!,
                                  water.wʳ, grid, model.microphysics, dynamics.density, water.ρqʳ,
                                  θᵨ, dynamics.interface_exner_function, dynamics.pressure, model.temperature,
                                  water.qᵛ, water.qˡ, constants, cᵖᵈ, g)

    # The halos that the limiter and the tendencies read, filled together where the fields share a
    # location and boundary conditions: the potential temperatures, the geopotential and the mass
    # fractions that are transported, then the mass flux and the velocity through the coordinate
    # surfaces. The temperature, the phase partition and the fall speed of rain are read in their
    # own cells only, and their halos are not filled.
    fill_scalar_halos!(grid, (θ, θᵨ, dynamics.geopotential, water_mass_fractions(model)...))
    fill_scalar_halos!(grid, (dynamics.vertical_mass_flux, w))

    compute_hydrostatic_heights!(dynamics.height_coordinate, grid, dynamics, θᵨ, constants)

    # The water that the tendencies transport: its halos across the seam of a Yin-Yang grid, from a
    # limited interpolation of the mixing ratio (`moist_seam_water_halos.jl`)
    fill_water_seam_halos!(model)

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

# An energy source Q changes ρθ at Q / (ĉᵖ Π): the enthalpy of a layer is ρ ĉᵖ Π θ per unit of the
# coordinate, with ĉᵖ = qᵈ cᵖᵈ + qᵗ cᵖᵛ = cᵖᵈ + qᵗ (cᵖᵛ - cᵖᵈ) and qᵗ the mass fraction of all water.
# The expansion work is added by `compute_dynamics_tendency!`, once the tendency of the surface
# pressure that it needs is known.
@inline function moist_hydrostatic_potential_temperature_tendency(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ,
                                                                  cᵖᵈ, cᵖᵛ, ρqˣ, hyperdiffusion, ρθ_forcing,
                                                                  ρE_forcing, clock, model_fields)

    Q = ρE_forcing(i, j, k, grid, clock, model_fields)
    qᵗ = @inbounds sum_at(i, j, k, ρqˣ) / ρ[i, j, k]
    ĉᵖ = cᵖᵈ + qᵗ * (cᵖᵛ - cᵖᵈ)

    return ( potential_temperature_flux_convergence(i, j, k, grid, advection, ρ, velocities, ρw, θ, Π, Πᶠ)
             + potential_temperature_hyperdiffusion(i, j, k, grid, hyperdiffusion, ρ, θ)
             + ρθ_forcing(i, j, k, grid, clock, model_fields)
             + @inbounds Q / (ĉᵖ * Π[i, j, k]))
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

# The scalars of a moist model are specific to the total mass, c = ρc / ρ, and are transported by
# the total mass fluxes of continuity: the fluxes that carry θ. A uniform specific value then
# stays uniform, whatever the water does. `MixingRatio` is the specific value the transport
# reconstructs; it is named for the per-unit-mass quantity the dynamics carries.
struct MixingRatio{FT, C, D} <: AbstractArray{FT, 3}
    specific :: C  # c, per unit total mass
    density :: D   # ρ, the total layer mass of the coordinate

    MixingRatio(c::C, ρ::D) where {C, D} = new{eltype(ρ), C, D}(c, ρ)
end

Base.size(r::MixingRatio) = size(r.density)

# A cell without mass is a halo cell beyond a rigid wall (only the first halo cell of a no-flux
# boundary is filled). Its specific value only ever multiplies the zero mass flux through the wall,
# so it is zero rather than whatever the unfilled halo holds, which could make that flux not a number.
Base.@propagate_inbounds function Base.getindex(r::MixingRatio, i, j, k)
    ρ = r.density[i, j, k]
    return ifelse(ρ == 0, zero(ρ), r.specific[i, j, k])
end

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
    r = MixingRatio(c, ρ)

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
# the dry Exner exponent. The rate is that of the layer Exner function following the air,
#
#   ρ Δz DΠ/Dt = ρ Δz ∂ₜΠ + [mass fluxes × δΠ, averaged to the cell] / Az
#                + ρw⁺ (Πᶠ⁺ - Π) + ρw⁻ (Π - Πᶠ⁻) ,        D(ln p)/Dt = (DΠ/Dt) / (κ Π) ,
#
# with ∂ₜΠ = [(Πᶠ⁻ - Π) ∂ₜp⁻ + (Π - Πᶠ⁺) ∂ₜp⁺] / Δp from the tendencies of the total pressures
# of the two faces, ∂ₜpᶠ = B ∂ₜpˢ. The pressure change by water through the ground is accounted
# for by the column physics, in the enthalpy it leaves in each layer.
@kernel function _add_expansion_work!(Gρθ, grid, ρ, u, v, ρw, Π, Πᶠ, pᶠ, B, Gpˢ, θ, T, qᵛ, qˡ, constants, κ)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        ∂ₜp⁺ = B[i, j, Nz+1] * Gpˢ[i, j, 1]

        for k in Nz:-1:1
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            ∂ₜp⁻ = B[i, j, k] * Gpˢ[i, j, 1]

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
    Gpˢ = model.timestepper.Gⁿ.pˢ
    ρ = dynamics.density
    u, v, _ = model.velocities
    g = constants.gravitational_acceleration
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    water = hydrostatic_water_fields(model.microphysics, model.microphysical_fields)

    launch!(arch, grid, :xy, _compute_surface_pressure_tendency!, Gpˢ, grid, ρ, u, v, g)

    launch!(arch, grid, :xy, _add_expansion_work!,
            model.timestepper.Gⁿ.ρθ, grid, ρ, u, v, dynamics.vertical_mass_flux,
            dynamics.exner_function, dynamics.interface_exner_function, dynamics.moisture.interface_pressure,
            dynamics.surface_pressure_coefficient, Gpˢ,
            model.formulation.potential_temperature, model.temperature, water.qᵛ, water.qˡ, constants, κ)

    # The expansion work has read the transport of the water. A time stepper that transports the
    # scalars with the mass flux of its stages now takes it out of their tendencies.
    leave_scalar_transport_to_stages!(model.timestepper, model)

    return nothing
end

#####
##### The gravity-wave operator of the substeps with water (`gravity_wave_substepping.jl`)
#####
#
# The fast system keeps the structure of the dry one, with the coefficients of moist air frozen
# at the start of the step. Writing pˢ′ for the perturbation of the total surface pressure:
#
#   - continuity and the flux of θᴸ by the perturbation mass flux are those of dry air, because
#     the coordinate and every prognostic weight are those of the total mass;
#   - the pressures of the z-faces are A + B pˢ, so the Exner functions respond to pˢ′ as those
#     of dry air do: the derivatives ∂Π/∂pˢ and ∂Πᶠ/∂pˢ are those of the dry operator;
#   - the pressure-gradient force and the hydrostatic integral read the density potential
#     temperature with its factor frozen, θᵨ′ = ϑᴸ θ′:
#
#       ∂τ 𝐮′ = - cᵖᵈ ϑᴸ θ′ ∇Πᴸ - cᵖᵈ θᵨᴸ ∇Π′ - ∇Φ′ ,   Φ′ = Φ[ϑᴸ θ′; Πᴸ] + Φ[θᵨᴸ; ∂Π/∂pˢ] pˢ′ .
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
"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

What the column physics does over `Δt` at the cell `(i, j, k)`, with density `ρ` (of the grid),
cloud liquid `qᶜˡ` and rain density `ρqʳ`, given the specific rates `conversion ≥ 0` (cloud liquid
to rain) and `evaporation ≤ 0` (rain to vapor): the density `to_rain` converted from cloud liquid,
and the rain density `rain_kept` that evaporation leaves of `ρqʳ`. The rain afterwards is
`rain_kept + to_rain`, a sum of non-negative numbers wherever `ρqʳ ≥ 0`. The default limits each
sink to what its donor holds; a microphysics scheme with stage-aware sink budgets replaces this
with them.
"""
@inline function rain_source_transfers(microphysics, i, j, k, ρqʳ, ρ, qᶜˡ, conversion, evaporation, Δt)
    ρqʳᵏ = @inbounds ρqʳ[i, j, k]
    to_rain = min(ρ * Δt * conversion, ρ * max(0, qᶜˡ))
    rain_kept = ρqʳᵏ + max(ρ * Δt * evaporation, - max(0, ρqʳᵏ))
    return to_rain, rain_kept
end

@inline function apply_cell_rain_sources!(i, j, k, ρqᵉ, ρqʳ, microphysics, ρ, p, T, qᵛ, qˡ, qᶜˡ, constants, Δt)
    @inbounds begin
        ρᵏ = ρ[i, j, k]
        ρqʳᵏ = ρqʳ[i, j, k]
        qʳ = ρqʳᵏ / ρᵏ
        qᶜˡᵏ = qᶜˡ[i, j, k]
        Tᵏ = T[i, j, k]
        q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
        ρᵖʰ = Thermodynamics.density(Tᵏ, p[i, j, k], q, constants)

        conversion, evaporation = rain_source_rates(microphysics, ρᵖʰ, qᶜˡᵏ, qʳ, q, Tᵏ, constants)
        conversion = ifelse(qᶜˡᵏ > 0, conversion, zero(conversion))
        evaporation = ifelse(qʳ > 0, evaporation, zero(evaporation))

        to_rain, rain_kept = rain_source_transfers(microphysics, i, j, k, ρqʳ, ρᵏ, qᶜˡᵏ, conversion, evaporation, Δt)
        ρqʳ⁺ = rain_kept + to_rain
        δρqʳ = ρqʳ⁺ - ρqʳᵏ

        ρqʳ[i, j, k] = ρqʳ⁺
        ρqᵉ[i, j, k] -= δρqʳ
    end

    return nothing
end

#####
##### A change of the mass of a column
#####

# The factor that keeps the mass per unit area of a layer, ρc Δz, when its pressure thickness
# changes from Δp to Δp⁺ at fixed coordinate density: one on a label grid, whose Δz is fixed and
# whose ρ follows Δp; Δp / Δp⁺ on a pressure-level grid, whose ρ is kept and whose Δz follows Δp.
@inline mass_conserving_factor(grid, Δp, Δp⁺) = one(Δp)
@inline mass_conserving_factor(::PressureLevelGrid, Δp, Δp⁺) = Δp / Δp⁺

# The energy of a horizontal area: a quantity per unit area at x-faces (y-faces) is brought to
# cell centers with the areas of the faces for weights, so that the sum over the cells of Az
# times the result is the sum over the faces of Az times the quantity. `f` is a kernel function.
@inline x_face_energy(i, j, k, grid, f, args...) = Azᶠᶜᶜ(i, j, k, grid) * f(i, j, k, grid, args...)
@inline y_face_energy(i, j, k, grid, f, args...) = Azᶜᶠᶜ(i, j, k, grid) * f(i, j, k, grid, args...)

@inline from_x_faces(i, j, k, grid, f, args...) = ℑxᶜᵃᵃ(i, j, k, grid, x_face_energy, f, args...) / Azᶜᶜᶜ(i, j, k, grid)
@inline from_y_faces(i, j, k, grid, f, args...) = ℑyᵃᶜᵃ(i, j, k, grid, y_face_energy, f, args...) / Azᶜᶜᶜ(i, j, k, grid)

@inline squared(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

# Kinetic energy per unit mass of layer `k`, ½ |u|², from the velocity points with their areas
@inline specific_kinetic_energy(i, j, k, grid, u, v) =
    (from_x_faces(i, j, k, grid, squared, u) + from_y_faces(i, j, k, grid, squared, v)) / 2

# The kinetic energy per unit area that a change δpˢ of the surface pressure gives column (i, j)
# at fixed velocity. Layer `k` gains the mass ΔB δpˢ / g (`change_column_surface_pressure!`), at
# the kinetic energy per unit mass of the layer, so the column gains Σ ΔB 𝒦 δpˢ / g. This is the
# kinetic energy that the water entering (δpˢ > 0) or leaving (δpˢ < 0) through the ground
# carries across it, and it belongs in the ledger of that flux beside its enthalpy and its
# geopotential. Summed with the areas of the cells, it is the change of the kinetic energy of the
# velocity points with the masses ℑ(Δp / g) of their two cells.
@inline function surface_pressure_kinetic_energy(i, j, grid, δpˢ, B, u, v, g)
    Nz = size(grid, 3)
    𝒦 = zero(grid)
    @inbounds for k in 1:Nz
        𝒦 += (B[i, j, k] - B[i, j, k+1]) * specific_kinetic_energy(i, j, k, grid, u, v)
    end
    return 𝒦 * δpˢ / g
end

# The enthalpy of a layer per unit of the coordinate, ĉᵖ Π ρθ with ĉᵖ ρ = cᵖᵈ (ρ - ρqᵗ) + cᵖᵛ ρqᵗ
@inline layer_enthalpy_density(ρ, ρqᵗ, Π, θ, cᵖᵈ, cᵖᵛ) = (cᵖᵈ * (ρ - ρqᵗ) + cᵖᵛ * ρqᵗ) * Π * θ

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Change the surface pressure of column `(i, j)` of moist `HydrostaticDynamics` by `δpˢ`, the weight
per unit area of the water that has left (`δpˢ < 0`) or entered the column through the ground,
and express the state of every layer at its new pressure.

The mass per unit area of every density in `scaled` (the water and the tracers) is kept, so the
column budget of each is that of the water moved; the dry air of the column is kept with them,
since its mass is the column mass less the water. The enthalpy of every layer is kept too, or set
to the nonzero value of `column_enthalpy` that the caller has left there (which is then reset to
zero), and θ is the potential temperature of that enthalpy at the new layer Exner function and
composition. Without a change of the surface pressure, a layer whose enthalpy was not given is
left as it is, bit for bit.
"""
@inline function change_column_surface_pressure!(i, j, grid, δpˢ, ρθ, column_enthalpy, scaled, ρqˣ, dynamics,
                                                 constants, κ, g)
    Nz = size(grid, 3)
    A = dynamics.pressure_coefficient
    B = dynamics.surface_pressure_coefficient
    pˢ = dynamics.surface_pressure
    ρ = dynamics.density
    Π = dynamics.exner_function
    pˢᵗ = dynamics.standard_pressure
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity

    @inbounds begin
        pˢᵏ = pˢ[i, j, 1]
        pˢ⁺ = pˢᵏ + δpˢ
        changed = δpˢ != 0

        p⁻ = A[i, j, 1] + B[i, j, 1] * pˢ⁺
        Πᶠ⁻ = (p⁻ / pˢᵗ)^κ

        for k in 1:Nz
            ΔA = A[i, j, k] - A[i, j, k+1]
            ΔB = B[i, j, k] - B[i, j, k+1]
            Δp = ΔA + ΔB * pˢᵏ
            Δp⁺ = ΔA + ΔB * pˢ⁺
            ρᵏ = ρ[i, j, k]
            ρ⁺ = coordinate_layer_density(i, j, k, grid, ρ, Δp⁺, g)
            f = ifelse(changed, mass_conserving_factor(grid, Δp, Δp⁺), one(Δp))

            ℋᵍ = column_enthalpy[i, j, k]
            given = ℋᵍ != 0
            ℋ = ifelse(given, ℋᵍ, layer_enthalpy_density(ρᵏ, sum_at(i, j, k, ρqˣ), Π[i, j, k], ρθ[i, j, k] / ρᵏ, cᵖᵈ, cᵖᵛ))

            scale_at!(i, j, k, scaled, f)
            ρqᵗ⁺ = sum_at(i, j, k, ρqˣ)

            p⁺ = A[i, j, k+1] + B[i, j, k+1] * pˢ⁺
            Πᶠ⁺ = (p⁺ / pˢᵗ)^κ
            Π⁺ = layer_exner_function(p⁻, Πᶠ⁻, p⁺, Πᶠ⁺, Δp⁺, κ)
            θ⁺ = f * ℋ / ((cᵖᵈ * (ρ⁺ - ρqᵗ⁺) + cᵖᵛ * ρqᵗ⁺) * Π⁺)

            ρθ[i, j, k] = ifelse(changed | given, ρ⁺ * θ⁺, ρθ[i, j, k])
            column_enthalpy[i, j, k] = zero(ℋ)

            p⁻ = p⁺
            Πᶠ⁻ = Πᶠ⁺
        end

        pˢ[i, j, 1] = pˢ⁺
    end

    return nothing
end

# Sedimentation of rain over Δt, implicit in time and upwind in the physical layers, from the
# model top down: with λ = V Δt / (physical depth) the Courant number of a layer at the fall
# speed of the start of the step, the layer keeps (Wʳ + inflow) / (1 + λ) of the weight of its
# rain and passes on λ times that. This is positive for any λ, so there is nothing to subcycle.
#
# The rain that leaves a layer carries the enthalpy of liquid water at the temperature of that
# layer, hˡ = cˡ T - ℒˡ₀, into the layer below or through the ground. The enthalpy of the layers
# changes only by what the rain carries, so the potential energy that the fall releases stays in
# the air as heat (the frictional dissipation of falling rain), and the column loses hˡ + Φˢ per
# unit mass of rain reaching the ground. The rain reaching the ground lowers the surface pressure
# by its weight, and every layer is then expressed again at its new pressure from its enthalpy
# (`change_column_surface_pressure!`). The velocities are held, so the column also loses the
# kinetic energy of the mass that the change takes from its layers, P Σ ΔB 𝒦
# (`surface_pressure_kinetic_energy`): the rain carries it through the ground, and the ledger
# books it with the enthalpy and the geopotential of the rain.
@inline function sediment_column_rain!(i, j, grid, ρθ, ρqʳ, scaled, ρqˣ, ledger, column_enthalpy, microphysics,
                                       dynamics, velocities, θ, θᵨ, p, T, qᵛ, qˡ, constants, κ, g, Δt, inflow_fall_speed)
    Nz = size(grid, 3)

    ρ = dynamics.density
    Π = dynamics.exner_function
    Πᶠ = dynamics.interface_exner_function
    Φˢ = dynamics.surface_geopotential
    Φ = dynamics.geopotential
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    @inbounds begin
        inflow = zero(grid)         # weight of the rain entering through the upper face
        enthalpy_inflow = zero(grid)
        friction = zero(grid)
        courant = zero(grid)

        for k in Nz:-1:1
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            ρᵏ = ρ[i, j, k]
            ρqʳᵏ = ρqʳ[i, j, k]
            Tᵏ = T[i, j, k]

            ρᵖʰ = physical_density(i, j, k, p, T, qᵛ, qˡ, constants)
            # The fall speed of the rain of the layer before this sweep, or with `inflow_fall_speed`
            # of that rain together with what enters it from above in this sweep
            ρqʳᵛ = ifelse(inflow_fall_speed, ρqʳᵏ + inflow / (g * Δz), ρqʳᵏ)
            V = positive_rain_fall_speed(microphysics, ρᵖʰ, ρqʳᵛ / (ρᵏ - ρqʳᵏ + ρqʳᵛ))
            # The hydrostatic integral, which is the depth on the physical-height grid wherever its heights are
            # current; the column is diagnosed again inside this kernel, where the height grid is not
            λ = V * Δt / physical_layer_depth(i, j, k, θᵨ, Πᶠ, cᵖᵈ, g)

            Wʳ = g * Δz * ρqʳᵏ
            Wʳ⁺ = (Wʳ + inflow) / (1 + λ)
            outflow = λ * Wʳ⁺
            hˡ = cˡ * Tᵏ - ℒˡ₀
            crossed = (inflow != 0) | (outflow != 0)

            # Enthalpy of the layer per unit of the coordinate, before and after
            ℋ = layer_enthalpy_density(ρᵏ, sum_at(i, j, k, ρqˣ), Π[i, j, k], θ[i, j, k], cᵖᵈ, cᵖᵛ)
            ℋ⁺ = ℋ + (enthalpy_inflow - outflow * hˡ) / (g * Δz)

            column_enthalpy[i, j, k] = ifelse(crossed, ℋ⁺, zero(ℋ⁺))
            ρqʳ[i, j, k] = ifelse(crossed, Wʳ⁺ / (g * Δz), ρqʳᵏ)

            # Geopotential the rain falls through to the layer below, or to the ground
            below = max(k - 1, 1)
            Φ⁻ = ifelse(k == 1, Φˢ[i, j, 1], Φ[i, j, below])
            friction += outflow / g * (Φ[i, j, k] - Φ⁻)
            courant = max(courant, λ)

            inflow = outflow
            enthalpy_inflow = outflow * hˡ
        end

        kinetic_outflow = - surface_pressure_kinetic_energy(i, j, grid, - inflow, dynamics.surface_pressure_coefficient,
                                                            velocities.u, velocities.v, g)
        ledger.surface_rain[i, j, 1] += inflow / g
        ledger.precipitation_energy[i, j, 1] += (enthalpy_inflow + inflow * Φˢ[i, j, 1]) / g + kinetic_outflow
        ledger.precipitation_kinetic_energy[i, j, 1] += kinetic_outflow
        ledger.rain_frictional_heating[i, j, 1] += friction
        ledger.maximum_sedimentation_courant[i, j, 1] = max(ledger.maximum_sedimentation_courant[i, j, 1], courant)
    end

    change_column_surface_pressure!(i, j, grid, - inflow, ρθ, column_enthalpy, scaled, ρqˣ, dynamics, constants, κ, g)

    return nothing
end

# The state of one column that the column physics reads, diagnosed from its prognostic variables
# as `update_state!` diagnoses it: the layer mass, pressures and Exner functions, the temperature
# and the partition of the water, the density potential temperature and the geopotential, each
# with the function its kernel of `update_state!` calls. The vertical mass flux and the fall
# speed of rain are not diagnosed: the column physics does not read them.
@inline function diagnose_moist_column!(i, j, grid, temperature, qᵛᵉ, formulation, dynamics, constants, microphysics,
                                        microphysical_fields, ρqᵉ, ρqˣ, water, cᵖᵈ, κ, g)
    moisture = dynamics.moisture
    θ = formulation.potential_temperature
    θᵨ = moisture.density_potential_temperature
    Π = dynamics.exner_function
    Πᶠ = dynamics.interface_exner_function

    compute_column_moist_layer_mass_and_exner_functions!(i, j, grid, dynamics.density, dynamics.pressure, Π, Πᶠ,
                                                         moisture.interface_pressure,
                                                         dynamics.pressure_coefficient,
                                                         dynamics.surface_pressure_coefficient,
                                                         dynamics.surface_pressure,
                                                         dynamics.standard_pressure, κ, g)

    for k in 1:size(grid, 3)
        AtmosphereModels.compute_cell_thermodynamic_variables!(i, j, k, grid, temperature, qᵛᵉ, formulation, dynamics,
                                                               constants, microphysics, microphysical_fields, ρqᵉ)

        compute_cell_density_potential_temperature!(i, j, k, θᵨ, θ, Π, water.qᵛ, water.qˡ, constants)
    end

    compute_column_geopotential!(i, j, grid, dynamics.geopotential, dynamics.surface_geopotential, θᵨ, Π, Πᶠ, cᵖᵈ)

    return nothing
end

# The column physics of one column over `substeps` applications of Δt each: the sources of rain,
# the diagnosis of the state they leave, the sedimentation of rain, and the diagnosis of the state
# it leaves for the next application. Each of the two operations reads the state that
# `update_state!` would have diagnosed from the prognostic variables the previous one left, so the
# result is that of `substeps` separate applications with an `update_state!` after each
# operation. The state the last one leaves is diagnosed by the `update_state!` that follows the kernel.
@kernel function _apply_warm_rain_column_physics!(ρθ, ρqᵉ, ρqʳ, ρqˣ, scaled, ledger, grid, microphysics, microphysical_fields,
                                                  formulation, dynamics, velocities, temperature, qᵛᵉ, water, constants,
                                                  cᵖᵈ, κ, g, Δt, substeps, sources, sedimentation, inflow_fall_speed)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    moisture = dynamics.moisture
    ρ = dynamics.density
    p = dynamics.pressure
    θ = formulation.potential_temperature
    θᵨ = moisture.density_potential_temperature

    # The state is that of the last `update_state!` before the first operation, and is diagnosed
    # again before every other one
    diagnosed = true

    for n in 1:substeps, operation in 1:2
        apply_sources = sources & (operation == 1)
        apply_sedimentation = sedimentation & (operation == 2)

        if apply_sources | apply_sedimentation
            if !diagnosed
                diagnose_moist_column!(i, j, grid, temperature, qᵛᵉ, formulation, dynamics, constants, microphysics,
                                       microphysical_fields, ρqᵉ, ρqˣ, water, cᵖᵈ, κ, g)
            end

            if apply_sources
                for k in 1:Nz
                    apply_cell_rain_sources!(i, j, k, ρqᵉ, ρqʳ, microphysics, ρ, p, temperature,
                                             water.qᵛ, water.qˡ, water.qᶜˡ, constants, Δt)
                end
            else
                sediment_column_rain!(i, j, grid, ρθ, ρqʳ, scaled, ρqˣ, ledger, moisture.column_enthalpy, microphysics,
                                      dynamics, velocities, θ, θᵨ, p, temperature, water.qᵛ, water.qˡ, constants, κ, g, Δt,
                                      inflow_fall_speed)
            end

            diagnosed = false
        end
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

All the applications are made by one kernel, column by column, which diagnoses the state of the
column between them as `update_state!` does; the state of the model is updated once, after the
last one. The result is that of `n` separate calls over `Δt / n`.

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
    (sources | sedimentation) || return nothing

    grid = model.grid
    arch = grid.architecture
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    water = hydrostatic_water_fields(microphysics, model.microphysical_fields)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ = dry_air_gas_constant(constants) / cᵖᵈ
    g = constants.gravitational_acceleration
    substep_Δt = convert(eltype(grid), Δt / substeps)

    @apply_regionally launch!(arch, grid, :xy, _apply_warm_rain_column_physics!,
                              model.formulation.potential_temperature_density, model.moisture_density, water.ρqʳ,
                              water_densities(model), column_mass_densities(model), dynamics.moisture.ledger, grid,
                              microphysics, model.microphysical_fields,
                              model.formulation, dynamics, horizontal_velocities(model), model.temperature,
                              AtmosphereModels.specific_prognostic_moisture(model), water, constants,
                              cᵖᵈ, κ, g, substep_Δt, substeps, sources, sedimentation, inflow_fall_speed)

    update_state!(model)

    return nothing
end

#####
##### The mass of the water that a generic microphysics precipitates through the ground
#####

# The water P Δt that has left column (i, j) through the ground lowers its surface pressure by its
# weight. Every layer keeps its enthalpy and the mass per unit area of its water and tracers
# (`change_column_surface_pressure!`); the ledger books the water, and the kinetic energy that the
# held velocities lose with the mass, P Δt Σ ΔB 𝒦, which the water carries out. The enthalpy of
# the water is not booked here: the microphysics that took the water out of the layers did so.
@kernel function _change_mass_by_precipitation!(ρθ, column_enthalpy, scaled, ρqˣ, ledger, grid, dynamics, velocities,
                                                surface_precipitation, constants, κ, g, Δt)
    i, j = @index(Global, NTuple)

    @inbounds begin
        outflow = g * Δt * surface_precipitation[i, j, 1]
        kinetic_outflow = - surface_pressure_kinetic_energy(i, j, grid, - outflow, dynamics.surface_pressure_coefficient,
                                                            velocities.u, velocities.v, g)
        ledger.surface_rain[i, j, 1] += outflow / g
        ledger.precipitation_energy[i, j, 1] += kinetic_outflow
        ledger.precipitation_kinetic_energy[i, j, 1] += kinetic_outflow
    end

    change_column_surface_pressure!(i, j, grid, - outflow, ρθ, column_enthalpy, scaled, ρqˣ, dynamics, constants, κ, g)
end

function AtmosphereModels.precipitation_mass_change!(::MoistHydrostaticDynamics, model, Δt, surface_precipitation)
    (isfinite(Δt) && Δt > 0) || return nothing
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    @apply_regionally launch!(grid.architecture, grid, :xy, _change_mass_by_precipitation!,
                              model.formulation.potential_temperature_density, dynamics.moisture.column_enthalpy,
                              column_mass_densities(model), water_densities(model), dynamics.moisture.ledger, grid,
                              dynamics, horizontal_velocities(model), surface_precipitation, constants, κ, g,
                              convert(eltype(grid), Δt))

    update_state!(model)

    return nothing
end

#####
##### Instantaneous rain-out: condensation and the fall of the condensate within the time step
#####
#
# A scheme without rain between time steps (zero-moment, instantaneous precipitation) condenses the
# supersaturation of every layer at its pressure, with θ fixed (θ is invariant under a phase change
# at fixed pressure and total water, so the latent heat is in the temperature), and the condensate
# falls to the ground within the step: the sweep of `sediment_column_rain!` with the condensate of
# each layer in place of its rain and a fall that keeps nothing, so every layer passes on all it
# holds and all that enters it (no Courant number, no λ = ∞). The enthalpy, pressure and ledger
# bookkeeping is that of the sedimentation of rain: the condensate leaving a layer carries
# hˡ = cˡ T - ℒˡ₀ at the temperature of that layer after condensation, the layers it crossed are
# expressed again at their new pressure, and the column loses hˡ + Φˢ per unit mass reaching the
# ground; the potential energy the fall releases is the frictional heating of the ledger.
#
# The scheme provides `rainout_condensate(microphysics, i, j, k, grid, formulation, dynamics, constants, ρqᵛ)`,
# the density of the condensate of layer k (same measure as ρqᵛ) and the temperature after condensation.
function rainout_condensate end

@kernel function _rain_out_columns!(ρθ, ρqᵛ, precipitation_rate, scaled, ρqˣ, ledger, column_enthalpy, grid, microphysics,
                                    formulation, dynamics, velocities, constants, κ, g, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    ρ = dynamics.density
    Φˢ = dynamics.surface_geopotential
    Φ = dynamics.geopotential
    Π = dynamics.exner_function
    θ = formulation.potential_temperature

    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)

    @inbounds begin
        inflow = zero(grid)
        enthalpy_inflow = zero(grid)
        friction = zero(grid)

        for k in Nz:-1:1
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            ρqᵛᵏ = ρqᵛ[i, j, k]

            ρc, T₁ = rainout_condensate(microphysics, i, j, k, grid, formulation, dynamics, constants, ρqᵛ)

            # Instant fall: the layer keeps nothing and passes on its condensate and all that enters it
            outflow = inflow + g * Δz * ρc
            hˡ = cˡ * T₁ - ℒˡ₀
            crossed = (inflow != 0) | (outflow != 0)

            # Enthalpy of the layer per unit of the coordinate, before (all of its water) and after
            ℋ = layer_enthalpy_density(ρ[i, j, k], sum_at(i, j, k, ρqˣ), Π[i, j, k], θ[i, j, k], cᵖᵈ, cᵖᵛ)
            ℋ⁺ = ℋ + (enthalpy_inflow - outflow * hˡ) / (g * Δz)

            column_enthalpy[i, j, k] = ifelse(crossed, ℋ⁺, zero(ℋ⁺))
            ρqᵛ[i, j, k] = ifelse(ρc > 0, ρqᵛᵏ - ρc, ρqᵛᵏ)
            precipitation_rate[i, j, k] = ρc / Δt

            below = max(k - 1, 1)
            Φ⁻ = ifelse(k == 1, Φˢ[i, j, 1], Φ[i, j, below])
            friction += outflow / g * (Φ[i, j, k] - Φ⁻)

            inflow = outflow
            enthalpy_inflow = outflow * hˡ
        end

        kinetic_outflow = - surface_pressure_kinetic_energy(i, j, grid, - inflow, dynamics.surface_pressure_coefficient,
                                                            velocities.u, velocities.v, g)
        ledger.surface_rain[i, j, 1] += inflow / g
        ledger.precipitation_energy[i, j, 1] += (enthalpy_inflow + inflow * Φˢ[i, j, 1]) / g + kinetic_outflow
        ledger.precipitation_kinetic_energy[i, j, 1] += kinetic_outflow
        ledger.rain_frictional_heating[i, j, 1] += friction
    end

    change_column_surface_pressure!(i, j, grid, - inflow, ρθ, column_enthalpy, scaled, ρqˣ, dynamics, constants, κ, g)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Apply the instantaneous rain-out of `microphysics` (a scheme without rain between time steps) to
`model`, an `AtmosphereModel` with moist [`HydrostaticDynamics`](@ref), over the time step just
taken: every layer condenses its supersaturation at its pressure, and the condensate falls to the
ground within the step, with the enthalpy, pressure and ledger bookkeeping of the sedimentation
of rain (see `_rain_out_columns!`). Once per call; the state is updated afterwards.
"""
function rain_out_column_physics!(microphysics, model::MoistHydrostaticModel; Δt=model.clock.last_Δt)
    (isfinite(Δt) && Δt > 0) || return nothing
    grid = model.grid
    constants = model.thermodynamic_constants
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    @apply_regionally launch!(grid.architecture, grid, :xy, _rain_out_columns!,
                              model.formulation.potential_temperature_density, model.moisture_density,
                              model.microphysical_fields.precipitation_rate, column_mass_densities(model),
                              water_densities(model), model.dynamics.moisture.ledger, model.dynamics.moisture.column_enthalpy,
                              grid, microphysics, model.formulation, model.dynamics, horizontal_velocities(model), constants, κ, g,
                              convert(eltype(grid), Δt))

    update_state!(model)

    return nothing
end

# The precipitation ledger accumulates over the integration, so a restart carries it on
# (on a pressure-level grid, with the density of the grid)
AtmosphereModels.dynamics_restart_state(dynamics::MoistHydrostaticDynamics) =
    moist_restart_state(dynamics.density.grid, dynamics)

moist_restart_state(grid, dynamics) = Oceananigans.prognostic_state(dynamics.moisture.ledger)
moist_restart_state(::PressureLevelGrid, dynamics) = (ledger = Oceananigans.prognostic_state(dynamics.moisture.ledger),
                                                      density = coordinate_restart_state(dynamics.density))

function AtmosphereModels.restore_dynamics_restart_state!(dynamics::MoistHydrostaticDynamics, from::NamedTuple)
    restore_moist_restart_state!(dynamics.density.grid, dynamics, from)
    return dynamics
end

restore_moist_restart_state!(grid, dynamics, from) = Oceananigans.restore_prognostic_state!(dynamics.moisture.ledger, from)

function restore_moist_restart_state!(::PressureLevelGrid, dynamics, from)
    Oceananigans.restore_prognostic_state!(dynamics.moisture.ledger, from.ledger)
    restore_coordinate_state!(dynamics.density, from.density)
    return nothing
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

# `set!` weights a mass fraction with the continuity density, which here is the total density of
# the coordinate, so water given as a mass fraction q arrives as ρ q and is per unit total mass
# as it is: there is nothing to establish.
AtmosphereModels.establish_densities!(::MoistHydrostaticModel, args...) = nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Set variables of an `AtmosphereModel` with moist `HydrostaticDynamics`.

`pˢ` is the total surface pressure, the weight of the dry air and of all the water of the
column, which is the prognostic surface pressure; without it the surface pressure is left as it
is. It is set first, because it determines the pressure and the layer mass that weight the other
prognostic variables, none of which depends on the water. The other keyword arguments are set
as for any `AtmosphereModel`.
"""
function Fields.set!(model::MoistHydrostaticModel; pˢ=nothing, kw...)
    (haskey(kw, :ρu) || haskey(kw, :ρv)) &&
        throw(ArgumentError("Cannot set the momentum density of HydrostaticDynamics, whose prognostic " *
                            "variables are the velocities u and v."))

    isnothing(pˢ) || set!(model.dynamics.surface_pressure, pˢ)
    diagnose_layer_mass_of_new_state!(model)
    invoke(set!, Tuple{AtmosphereModel}, model; kw...)
    validate_moist_hydrostatic_state(model)

    return nothing
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
