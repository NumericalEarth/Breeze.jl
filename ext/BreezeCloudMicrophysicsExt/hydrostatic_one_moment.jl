#####
##### Warm-phase one-moment microphysics with saturation adjustment (WP1M) on HydrostaticDynamics
#####
#
# The vertical coordinate of `HydrostaticDynamics` is a label, and its densities are masses per
# unit area and unit of that label. The rates and the fall speed of the one-moment scheme
# depend on the density of the air in kg m⁻³ (through the size distribution and the fall speed
# of rain) and on the depth of the layers in m, so they are evaluated here with the physical
# density that `Breeze.HydrostaticEquations` passes, and never with `total_density`.
#
# The scheme acts once per time step after the dynamics, in `warm_rain_column_physics!`, with
# its sinks limited to their donors over the time step. The numerical timescale `τⁿᵘᵐ` and the
# relaxation of negative rain of the tendency interface are therefore not used.

HydrostaticEquations.supports_hydrostatic_dynamics(::WP1M) = true

HydrostaticEquations.hydrostatic_water_fields(::WP1M, μ) = (; qᵛ=μ.qᵛ, qˡ=μ.qˡ, qᶜˡ=μ.qᶜˡ, ρqʳ=μ.ρqʳ, wʳ=μ.wʳ)

# The rain autoconversion of the default parameters does not depend on the vertical velocity,
# which here would be the velocity through the coordinate surfaces: it is given none.
@inline function HydrostaticEquations.rain_source_rates(bμp::WP1M, ρ, qᶜˡ, qʳ, q, T, constants)
    parameters = bμp.categories.parameters
    rain = parameters.precip.rain
    rain_velocity = parameters.terminal_velocity.rain

    Sᵃᶜⁿᵛ = liquid_autoconversion(parameters, qᶜˡ, zero(qᶜˡ))

    Sᵃᶜᶜ = cloud_precipitation_accretion(parameters.process_params.cloud_liquid_rain_accretion,
                                         parameters.cloud.liquid, rain, rain_velocity, qᶜˡ, qʳ, ρ)

    Sᵉᵛᵃᵖ = rain_evaporation_rate(parameters.processes.rain_condensation_evaporation,
                                  rain, rain_velocity, parameters.air_properties, q, qʳ, ρ, ρ, T, constants)

    return Sᵃᶜⁿᵛ + Sᵃᶜᶜ, min(Sᵉᵛᵃᵖ, zero(Sᵉᵛᵃᵖ))
end

@inline HydrostaticEquations.rain_fall_speed(bμp::WP1M, ρ, qʳ) =
    terminal_velocity(bμp.categories.parameters.precip.rain, bμp.categories.parameters.terminal_velocity.rain, ρ, qʳ)

AM.microphysics_model_update!(bμp::WP1M, model::MoistHydrostaticModel) = warm_rain_column_physics!(bμp, model)
