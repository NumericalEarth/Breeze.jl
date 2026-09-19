module GABLSDiagnostics

export build_gabls_diagnostics,
       install_gabls_diagnostics!,
       boundary_layer_height,
       low_level_jet

using Breeze
using Oceananigans

using Breeze.AtmosphereModels: AtmosphereModelBuoyancy,
                               buoyancy_forceᶜᶜᶜ,
                               closure_scalar_index,
                               pressure_anomaly,
                               specific_prognostic_moisture,
                               thermodynamic_density,
                               thermodynamic_density_name
using Breeze.TurbulenceClosures: Jᶜz, 𝒯_uz, 𝒯_vz
using Breeze.BoundaryConditions: gabls_stability_parameter,
                                 tangential_speed²,
                                 wall_distance
using Oceananigans.AbstractOperations: Average, Integral, KernelFunctionOperation
using Oceananigans.Fields: Field, Reduction, location
using Oceananigans.Grids: Center, Face, znode
using Oceananigans.Models: BoundaryConditionOperation
using Oceananigans.BoundaryConditions: Bottom
using Oceananigans.Operators: ℑzᵃᵃᶠ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ
using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.TurbulenceClosures.Smagorinskys: ΣᵢⱼΣᵢⱼᶜᶜᶜ
using Oceananigans.Units: minute, minutes

const C = Center
const F = Face

# Point functions used by KernelFunctionOperation must remain allocation-free and GPU-safe.

@inline function buoyancy_per_mass(i, j, k, grid, dynamics, temperature,
                                   prognostic_moisture, microphysics,
                                   microphysical_fields, constants, density)
    ρb = buoyancy_forceᶜᶜᶜ(i, j, k, grid, dynamics, temperature,
                          prognostic_moisture, microphysics,
                          microphysical_fields, constants)
    return @inbounds ρb / density[i, j, k]
end

@inline function nonnegative_variance(i, j, k, grid, first_moment, second_moment)
    m₁ = @inbounds first_moment[i, j, k]
    m₂ = @inbounds second_moment[i, j, k]
    return max(0, m₂ - m₁^2)
end

@inline function instantaneous_skewness(i, j, k, grid, variance, third_moment)
    w² = @inbounds variance[i, j, k]
    w³ = @inbounds third_moment[i, j, k]
    denominator = max(w², eps(w²))^(3 / 2)
    ratio = w³ / denominator
    return ifelse(w² > 0, ratio, zero(grid))
end

@inline function scalar_sgs_flux(i, j, k, grid, density, ::Nothing, args...)
    return zero(grid)
end

@inline function scalar_sgs_flux(i, j, k, grid, density, closure, closure_fields,
                                 id, scalar, clock, model_fields, buoyancy)
    discretization = time_discretization(closure)
    dynamic_flux = Jᶜz(i, j, k, grid, density, discretization, closure,
                       closure_fields, id, scalar, clock, model_fields, buoyancy)
    density_at_face = ℑzᵃᵃᶠ(i, j, k, grid, density)
    return dynamic_flux / density_at_face
end

@inline function u_sgs_flux(i, j, k, grid, density, ::Nothing, args...)
    return zero(grid)
end

@inline function u_sgs_flux(i, j, k, grid, density, closure, closure_fields,
                            clock, model_fields)
    discretization = time_discretization(closure)
    dynamic_flux = 𝒯_uz(i, j, k, grid, density, discretization, closure,
                        closure_fields, clock, model_fields, nothing)
    density_at_face = ℑxzᶠᵃᶠ(i, j, k, grid, density)
    return dynamic_flux / density_at_face
end

@inline function v_sgs_flux(i, j, k, grid, density, ::Nothing, args...)
    return zero(grid)
end

@inline function v_sgs_flux(i, j, k, grid, density, closure, closure_fields,
                            clock, model_fields)
    discretization = time_discretization(closure)
    dynamic_flux = 𝒯_vz(i, j, k, grid, density, discretization, closure,
                        closure_fields, clock, model_fields, nothing)
    density_at_face = ℑyzᵃᶠᶠ(i, j, k, grid, density)
    return dynamic_flux / density_at_face
end

@inline function eddy_viscosity_dissipation(i, j, k, grid, viscosity, u, v, w)
    strain_squared = ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w)
    return @inbounds 2 * viscosity[i, j, k] * strain_squared
end

@inline function surface_corrected_flux(i, j, k, grid, resolved, sgs, surface)
    interior_flux = @inbounds resolved[i, j, k] + sgs[i, j, k]
    surface_flux = @inbounds surface[1, 1, 1]
    return ifelse(k == 1, surface_flux, interior_flux)
end

@inline function stress_magnitude(i, j, k, grid, u_flux, v_flux)
    uw = @inbounds u_flux[i, j, k]
    vw = @inbounds v_flux[i, j, k]
    return sqrt(uw^2 + vw^2)
end

@inline function sgs_buoyancy_flux(i, j, k, grid, theta_flux, theta, gravity)
    wtheta = @inbounds theta_flux[i, j, k]
    theta_face = ℑzᵃᵃᶠ(i, j, k, grid, theta)
    return gravity * wtheta / theta_face
end

@inline function obukhov_length_value(i, j, k, grid, friction_velocity,
                                      heat_flux, reference_temperature,
                                      von_karman_constant, gravity)
    ustar = @inbounds friction_velocity[i, j, k]
    wtheta = @inbounds heat_flux[i, j, k]
    denominator = von_karman_constant * gravity * wtheta
    safe_denominator = ifelse(abs(denominator) > eps(denominator), denominator, one(grid))
    length = -ustar^3 * reference_temperature / safe_denominator
    return ifelse(abs(denominator) > eps(denominator), length, zero(grid))
end

@inline function obukhov_length_valid(i, j, k, grid, heat_flux, gravity,
                                      von_karman_constant)
    wtheta = @inbounds heat_flux[i, j, k]
    denominator = von_karman_constant * gravity * wtheta
    return ifelse(abs(denominator) > eps(denominator), one(grid), zero(grid))
end

@inline function surface_bulk_richardson(i, j, k, grid, model_fields, clock,
                                         surface_temperature_initial,
                                         surface_cooling_rate,
                                         reference_temperature, gravity,
                                         minimum_wind_speed)
    side = Bottom()
    height = wall_distance(i, j, 1, grid, side)
    theta₁ = @inbounds model_fields.θ[i, j, 1]
    theta_surface = surface_temperature_initial - surface_cooling_rate * clock.time
    wind_speed² = max(tangential_speed²(i, j, 1, grid, side, nothing, model_fields),
                      minimum_wind_speed^2)
    return gravity * height * (theta₁ - theta_surface) /
           (reference_temperature * wind_speed²)
end

@inline function surface_stability_parameter(i, j, k, grid, bulk_richardson,
                                              momentum_roughness_length,
                                              heat_roughness_length,
                                              stable_momentum_beta,
                                              stable_heat_beta,
                                              maximum_surface_stability)
    height = wall_distance(i, j, 1, grid, Bottom())
    a = log(height / momentum_roughness_length)
    b = log(height / heat_roughness_length)
    Rb = @inbounds bulk_richardson[i, j, 1]
    return gabls_stability_parameter(Rb, a, b, stable_momentum_beta,
                                     stable_heat_beta, maximum_surface_stability)
end

@inline function surface_cap_mask(i, j, k, grid, stability_parameter,
                                  maximum_surface_stability)
    zeta = @inbounds stability_parameter[i, j, 1]
    tolerance = 8 * eps(maximum_surface_stability)
    return ifelse(zeta >= maximum_surface_stability - tolerance, one(grid), zero(grid))
end

@inline function surface_neutral_fallback_mask(i, j, k, grid, bulk_richardson)
    Rb = @inbounds bulk_richardson[i, j, 1]
    return ifelse(Rb <= 0, one(grid), zero(grid))
end

function plane_mean(field)
    horizontal_dimensions = Tuple(d for d in (1, 2) if location(field)[d] !== Nothing)
    isempty(horizontal_dimensions) && return field
    return Field(Average(field, dims=horizontal_dimensions))
end

function plane_central_moments(field)
    mean_field = plane_mean(field)
    perturbation = field - mean_field
    variance = Field(Average(perturbation^2, dims=(1, 2)))
    third_moment = Field(Average(perturbation^3, dims=(1, 2)))
    return mean_field, perturbation, variance, third_moment
end

function plane_central_flux(vertical_velocity, scalar)
    w_mean = plane_mean(vertical_velocity)
    scalar_mean = plane_mean(scalar)
    w_perturbation = vertical_velocity - w_mean
    scalar_perturbation = scalar - scalar_mean
    return Field(Average(w_perturbation * scalar_perturbation, dims=(1, 2)))
end

function scalar_sgs_flux_field(model, scalar, id)
    density = total_density(model.dynamics)
    model_fields = fields(model)
    buoyancy = AtmosphereModelBuoyancy(model.dynamics, model.formulation,
                                       model.thermodynamic_constants)
    operation = KernelFunctionOperation{C, C, F}(
        scalar_sgs_flux, model.grid, density, model.closure, model.closure_fields,
        id, scalar, model.clock, model_fields, buoyancy)
    return Field(operation)
end

function momentum_sgs_flux_fields(model)
    density = total_density(model.dynamics)
    model_fields = fields(model)
    u_operation = KernelFunctionOperation{F, C, F}(
        u_sgs_flux, model.grid, density, model.closure, model.closure_fields,
        model.clock, model_fields)
    v_operation = KernelFunctionOperation{C, F, F}(
        v_sgs_flux, model.grid, density, model.closure, model.closure_fields,
        model.clock, model_fields)
    return Field(u_operation), Field(v_operation)
end

function surface_flux_diagnostics(model, density)
    dynamic_u_flux = Field(BoundaryConditionOperation(model.momentum.ρu, :bottom, model))
    dynamic_v_flux = Field(BoundaryConditionOperation(model.momentum.ρv, :bottom, model))
    dynamic_theta_flux = Field(BoundaryConditionOperation(
        thermodynamic_density(model.formulation), :bottom, model))

    dynamic_u_flux_mean = plane_mean(dynamic_u_flux)
    dynamic_v_flux_mean = plane_mean(dynamic_v_flux)
    dynamic_theta_flux_mean = plane_mean(dynamic_theta_flux)
    density_at_z_faces = Field(@at (C, C, F) density)
    surface_density_mean = plane_mean(view(density_at_z_faces, :, :, 1))

    kinematic_u_flux = Field(dynamic_u_flux_mean / surface_density_mean)
    kinematic_v_flux = Field(dynamic_v_flux_mean / surface_density_mean)
    kinematic_theta_flux = Field(dynamic_theta_flux_mean / surface_density_mean)
    friction_velocity = Field(sqrt(sqrt(kinematic_u_flux^2 + kinematic_v_flux^2)))

    return (; dynamic_u_flux_mean,
              dynamic_v_flux_mean,
              dynamic_theta_flux_mean,
              kinematic_u_flux,
              kinematic_v_flux,
              kinematic_theta_flux,
              surface_density_mean,
              friction_velocity)
end

function corrected_flux_field(grid, resolved, sgs, surface)
    operation = KernelFunctionOperation{Nothing, Nothing, F}(
        surface_corrected_flux, grid, resolved, sgs, surface)
    return Field(operation)
end

function optional_sgs_dissipation(model)
    if isnothing(model.closure)
        zero_dissipation = plane_mean(0 * Field(@at (C, C, C) model.velocities.w))
        return (; sgs_resolved_tke_dissipation=zero_dissipation,
                  available=false)
    end

    hasproperty(model.closure_fields, :νₑ) || return (; profiles=NamedTuple(), available=false)
    u, v, w = model.velocities
    operation = KernelFunctionOperation{C, C, C}(
        eddy_viscosity_dissipation, model.grid, model.closure_fields.νₑ, u, v, w)
    dissipation = plane_mean(Field(operation))
    return (; sgs_resolved_tke_dissipation=dissipation, available=true)
end

function optional_sgs_tke(model)
    haskey(model.tracers, :e) ||
        return (; profiles=NamedTuple(), series=NamedTuple(), available=false)
    sgs_tke = plane_mean(model.tracers.e)
    sgs_tke_integral = Field(Integral(sgs_tke, dims=3))
    return (; profiles=(; sgs_tke),
              series=(; sgs_tke_vertical_integral=sgs_tke_integral),
              available=true)
end

function profile_values(profile)
    compute!(profile)
    return vec(Array(interior(profile, 1, 1, :)))
end

function boundary_layer_height(stress_profile; fraction=0.05)
    values = profile_values(stress_profile)
    coordinates = collect(znodes(stress_profile))
    surface_stress = first(values)
    target = fraction * surface_stress

    if !(isfinite(surface_stress) && surface_stress > 0)
        return (; h_0_05=zero(eltype(values)), h=zero(eltype(values)), valid=false)
    end

    crossing_index = findfirst(k -> values[k] <= target, 2:length(values))
    if isnothing(crossing_index)
        fallback = last(coordinates)
        return (; h_0_05=fallback, h=fallback / (1 - fraction), valid=false)
    end

    k = crossing_index
    lower_value = values[k-1]
    upper_value = values[k]
    lower_height = coordinates[k-1]
    upper_height = coordinates[k]
    difference = upper_value - lower_value
    interpolation_fraction = iszero(difference) ? one(difference) :
                             (target - lower_value) / difference
    height = lower_height + interpolation_fraction * (upper_height - lower_height)
    return (; h_0_05=height, h=height / (1 - fraction), valid=true)
end

function low_level_jet(u_mean, v_mean)
    u = profile_values(u_mean)
    v = profile_values(v_mean)
    coordinates = collect(znodes(u_mean))
    speed = sqrt.(u.^2 .+ v.^2)
    k = argmax(speed)
    return (; height=coordinates[k], speed=speed[k],
              direction=atan(v[k], u[k]), valid=all(isfinite, speed))
end

function build_gabls_diagnostics(model;
                                 reference_temperature=263.5,
                                 surface_temperature_initial=265,
                                 surface_cooling_rate=0.25 / 3600,
                                 von_karman_constant=0.4,
                                 stable_momentum_beta=4.8,
                                 stable_heat_beta=7.8,
                                 momentum_roughness_length=0.1,
                                 heat_roughness_length=0.1,
                                 minimum_surface_wind_speed=0.01,
                                 maximum_surface_stability=10)
    grid = model.grid
    FT = eltype(grid)
    reference_temperature = FT(reference_temperature)
    surface_temperature_initial = FT(surface_temperature_initial)
    surface_cooling_rate = FT(surface_cooling_rate)
    von_karman_constant = FT(von_karman_constant)
    stable_momentum_beta = FT(stable_momentum_beta)
    stable_heat_beta = FT(stable_heat_beta)
    momentum_roughness_length = FT(momentum_roughness_length)
    heat_roughness_length = FT(heat_roughness_length)
    minimum_surface_wind_speed = FT(minimum_surface_wind_speed)
    maximum_surface_stability = FT(maximum_surface_stability)

    u, v, w = model.velocities
    uᶜ = Field(@at (C, C, C) u)
    vᶜ = Field(@at (C, C, C) v)
    wᶜ = Field(@at (C, C, C) w)
    theta = liquid_ice_potential_temperature(model)
    density = total_density(model.dynamics)

    buoyancy_operation = KernelFunctionOperation{C, C, C}(
        buoyancy_per_mass, grid, model.dynamics, model.temperature,
        specific_prognostic_moisture(model), model.microphysics,
        model.microphysical_fields, model.thermodynamic_constants, density)
    buoyancy = Field(buoyancy_operation)

    u_mean, _, u_variance, _ = plane_central_moments(u)
    v_mean, _, v_variance, _ = plane_central_moments(v)
    w_mean, _, w_variance, w_third_central_moment = plane_central_moments(w)
    _, u_prime, _, _ = plane_central_moments(uᶜ)
    _, v_prime, _, _ = plane_central_moments(vᶜ)
    w_center_mean, w_prime, w_center_variance, w_center_third_central_moment =
        plane_central_moments(wᶜ)
    theta_mean, theta_prime, theta_variance, _ = plane_central_moments(theta)
    density_mean = plane_mean(density)
    buoyancy_mean, _, _, _ = plane_central_moments(buoyancy)

    w_at_faces = w
    u_at_faces = Field(@at (C, C, F) u)
    v_at_faces = Field(@at (C, C, F) v)
    theta_at_faces = Field(@at (C, C, F) theta)
    buoyancy_at_faces = Field(@at (C, C, F) buoyancy)

    resolved_u_w_flux = plane_central_flux(w_at_faces, u_at_faces)
    resolved_v_w_flux = plane_central_flux(w_at_faces, v_at_faces)
    resolved_w_theta_flux = plane_central_flux(w_at_faces, theta_at_faces)
    resolved_buoyancy_flux = plane_central_flux(w_at_faces, buoyancy_at_faces)

    thermodynamic_id = closure_scalar_index(
        model, thermodynamic_density_name(model.formulation))
    sgs_theta_flux_3d = scalar_sgs_flux_field(model, theta, thermodynamic_id)
    sgs_u_w_flux_3d, sgs_v_w_flux_3d = momentum_sgs_flux_fields(model)
    sgs_theta_flux = plane_mean(sgs_theta_flux_3d)
    sgs_u_w_flux = plane_mean(Field(@at (C, C, F) sgs_u_w_flux_3d))
    sgs_v_w_flux = plane_mean(Field(@at (C, C, F) sgs_v_w_flux_3d))

    surface = surface_flux_diagnostics(model, density)
    total_u_w_flux = corrected_flux_field(
        grid, resolved_u_w_flux, sgs_u_w_flux, surface.kinematic_u_flux)
    total_v_w_flux = corrected_flux_field(
        grid, resolved_v_w_flux, sgs_v_w_flux, surface.kinematic_v_flux)
    total_w_theta_flux = corrected_flux_field(
        grid, resolved_w_theta_flux, sgs_theta_flux, surface.kinematic_theta_flux)
    total_stress_magnitude = Field(KernelFunctionOperation{Nothing, Nothing, F}(
        stress_magnitude, grid, total_u_w_flux, total_v_w_flux))

    gravity = FT(model.thermodynamic_constants.gravitational_acceleration)
    sgs_buoyancy_flux_profile = plane_mean(Field(KernelFunctionOperation{C, C, F}(
        sgs_buoyancy_flux, grid, sgs_theta_flux_3d, theta, gravity)))
    total_buoyancy_flux = Field(resolved_buoyancy_flux + sgs_buoyancy_flux_profile)

    resolved_tke_3d = Field((u_prime^2 + v_prime^2 + w_prime^2) / 2)
    resolved_tke = plane_mean(resolved_tke_3d)
    resolved_shear_production_face = -resolved_u_w_flux * ∂z(u_mean) -
                                     resolved_v_w_flux * ∂z(v_mean)
    sgs_shear_production_face = -sgs_u_w_flux * ∂z(u_mean) -
                                sgs_v_w_flux * ∂z(v_mean)
    total_shear_production_face = -total_u_w_flux * ∂z(u_mean) -
                                  total_v_w_flux * ∂z(v_mean)
    resolved_tke_shear_production = Field(@at (C, C, C) resolved_shear_production_face)
    sgs_tke_shear_production = Field(@at (C, C, C) sgs_shear_production_face)
    total_tke_shear_production = Field(@at (C, C, C) total_shear_production_face)
    resolved_tke_buoyancy_production = Field(@at (C, C, C) resolved_buoyancy_flux)
    sgs_tke_buoyancy_production = Field(@at (C, C, C) sgs_buoyancy_flux_profile)
    total_tke_buoyancy_production = Field(@at (C, C, C) total_buoyancy_flux)

    density_at_w_faces = plane_mean(Field(@at (C, C, F) density))
    turbulent_tke_flux = plane_central_flux(
        w_at_faces, Field(@at (C, C, F) resolved_tke_3d))
    kinematic_pressure = Field(pressure_anomaly(model.dynamics) / density)
    pressure_tke_flux = plane_central_flux(
        w_at_faces, Field(@at (C, C, F) kinematic_pressure))
    density_weighted_turbulent_tke_flux = Field(density_at_w_faces * turbulent_tke_flux)
    density_weighted_pressure_tke_flux = Field(density_at_w_faces * pressure_tke_flux)
    resolved_tke_turbulent_transport =
        Field(-∂z(density_weighted_turbulent_tke_flux) / density_mean)
    resolved_tke_pressure_transport =
        Field(-∂z(density_weighted_pressure_tke_flux) / density_mean)
    resolved_tke_transport = Field(resolved_tke_turbulent_transport +
                                   resolved_tke_pressure_transport)

    sgs_dissipation = optional_sgs_dissipation(model)
    sgs_tke = optional_sgs_tke(model)
    w_skewness_instantaneous_ratio = Field(KernelFunctionOperation{Nothing, Nothing, F}(
        instantaneous_skewness, grid, w_variance, w_third_central_moment))

    theta_minimum = Field(Reduction(minimum!, theta; dims=(1, 2, 3)))
    theta_maximum = Field(Reduction(maximum!, theta; dims=(1, 2, 3)))
    u_absolute_maximum = Field(Reduction(maximum!, sqrt(uᶜ^2); dims=(1, 2, 3)))
    v_absolute_maximum = Field(Reduction(maximum!, sqrt(vᶜ^2); dims=(1, 2, 3)))
    w_absolute_maximum = Field(Reduction(maximum!, sqrt(wᶜ^2); dims=(1, 2, 3)))
    speed_maximum = Field(Reduction(
        maximum!, sqrt(uᶜ^2 + vᶜ^2 + wᶜ^2); dims=(1, 2, 3)))
    resolved_tke_vertical_integral = Field(Integral(resolved_tke, dims=3))
    w_variance_maximum = Field(Reduction(maximum!, w_variance; dims=3))

    heat_capacity = FT(model.thermodynamic_constants.dry_air.heat_capacity)
    surface_sensible_heat_flux = Field(heat_capacity * surface.dynamic_theta_flux_mean)
    surface_buoyancy_flux = Field(gravity * surface.kinematic_theta_flux /
                                  reference_temperature)
    obukhov_length = Field(KernelFunctionOperation{Nothing, Nothing, F}(
        obukhov_length_value, grid, surface.friction_velocity,
        surface.kinematic_theta_flux, reference_temperature,
        von_karman_constant, gravity))
    obukhov_length_valid_flag = Field(KernelFunctionOperation{Nothing, Nothing, F}(
        obukhov_length_valid, grid, surface.kinematic_theta_flux, gravity,
        von_karman_constant))

    surface_bulk_richardson_field = Field(KernelFunctionOperation{C, C, Nothing}(
        surface_bulk_richardson, grid, fields(model), model.clock,
        surface_temperature_initial, surface_cooling_rate,
        reference_temperature, gravity, minimum_surface_wind_speed))
    surface_stability_parameter_field = Field(KernelFunctionOperation{C, C, Nothing}(
        surface_stability_parameter, grid, surface_bulk_richardson_field,
        momentum_roughness_length, heat_roughness_length,
        stable_momentum_beta, stable_heat_beta, maximum_surface_stability))
    surface_cap_mask_field = Field(KernelFunctionOperation{C, C, Nothing}(
        surface_cap_mask, grid, surface_stability_parameter_field,
        maximum_surface_stability))
    surface_neutral_fallback_mask_field = Field(KernelFunctionOperation{C, C, Nothing}(
        surface_neutral_fallback_mask, grid, surface_bulk_richardson_field))
    surface_bulk_richardson_mean = plane_mean(surface_bulk_richardson_field)
    surface_bulk_richardson_maximum = Field(Reduction(
        maximum!, surface_bulk_richardson_field; dims=(1, 2)))
    surface_stability_parameter_mean = plane_mean(surface_stability_parameter_field)
    surface_stability_parameter_maximum = Field(Reduction(
        maximum!, surface_stability_parameter_field; dims=(1, 2)))
    surface_stability_cap_fraction = plane_mean(surface_cap_mask_field)
    surface_neutral_fallback_fraction = plane_mean(surface_neutral_fallback_mask_field)

    boundary_height_output = model -> boundary_layer_height(total_stress_magnitude).h
    stress_height_output = model -> boundary_layer_height(total_stress_magnitude).h_0_05
    stress_height_valid_output = model ->
        FT(boundary_layer_height(total_stress_magnitude).valid)
    jet_height_output = model -> low_level_jet(u_mean, v_mean).height
    jet_speed_output = model -> low_level_jet(u_mean, v_mean).speed
    jet_turning_output = model -> low_level_jet(u_mean, v_mean).direction
    jet_valid_output = model -> FT(low_level_jet(u_mean, v_mean).valid)
    surface_temperature_output = model ->
        surface_temperature_initial - surface_cooling_rate * model.clock.time

    profile_outputs = (;
        u_mean,
        v_mean,
        w_mean,
        theta_mean,
        density_mean,
        buoyancy_mean,
        u_variance,
        v_variance,
        w_variance,
        w_third_central_moment,
        w_skewness_instantaneous_ratio,
        w_center_mean,
        w_center_variance,
        w_center_third_central_moment,
        theta_variance,
        resolved_u_w_flux,
        resolved_v_w_flux,
        resolved_w_theta_flux,
        sgs_u_w_flux,
        sgs_v_w_flux,
        sgs_w_theta_flux=sgs_theta_flux,
        total_u_w_flux,
        total_v_w_flux,
        total_w_theta_flux,
        total_stress_magnitude,
        resolved_buoyancy_flux,
        sgs_buoyancy_flux=sgs_buoyancy_flux_profile,
        total_buoyancy_flux,
        resolved_tke,
        resolved_tke_shear_production,
        sgs_tke_shear_production,
        total_tke_shear_production,
        resolved_tke_buoyancy_production,
        sgs_tke_buoyancy_production,
        total_tke_buoyancy_production,
        turbulent_tke_flux,
        pressure_tke_flux,
        density_at_w_faces,
        density_weighted_turbulent_tke_flux,
        density_weighted_pressure_tke_flux,
        resolved_tke_turbulent_transport,
        resolved_tke_pressure_transport,
        resolved_tke_transport,
        sgs_resolved_tke_dissipation=sgs_dissipation.sgs_resolved_tke_dissipation,
        sgs_tke.profiles...)

    series_outputs = (;
        friction_velocity=surface.friction_velocity,
        surface_drag_u_dynamic_flux=surface.dynamic_u_flux_mean,
        surface_drag_v_dynamic_flux=surface.dynamic_v_flux_mean,
        surface_drag_u_kinematic_flux=surface.kinematic_u_flux,
        surface_drag_v_kinematic_flux=surface.kinematic_v_flux,
        surface_theta_dynamic_flux=surface.dynamic_theta_flux_mean,
        surface_theta_kinematic_flux=surface.kinematic_theta_flux,
        surface_sensible_heat_flux,
        surface_buoyancy_flux,
        surface_density=surface.surface_density_mean,
        obukhov_length,
        obukhov_length_valid=obukhov_length_valid_flag,
        surface_bulk_richardson_mean,
        surface_bulk_richardson_maximum,
        surface_stability_parameter_mean,
        surface_stability_parameter_maximum,
        surface_stability_cap_fraction,
        surface_neutral_fallback_fraction,
        surface_temperature=surface_temperature_output,
        stress_height_0_05=stress_height_output,
        boundary_layer_height=boundary_height_output,
        boundary_layer_height_valid=stress_height_valid_output,
        low_level_jet_height=jet_height_output,
        low_level_jet_speed=jet_speed_output,
        low_level_jet_turning_from_geostrophic=jet_turning_output,
        low_level_jet_valid=jet_valid_output,
        theta_minimum,
        theta_maximum,
        u_absolute_maximum,
        v_absolute_maximum,
        w_absolute_maximum,
        speed_maximum,
        resolved_tke_vertical_integral,
        w_variance_maximum,
        sgs_tke.series...)

    diagnostic_fields = (; resolved_tke_3d, buoyancy, total_stress_magnitude,
                            surface_bulk_richardson=surface_bulk_richardson_field,
                            surface_stability_parameter=surface_stability_parameter_field,
                            surface_stability_cap_mask=surface_cap_mask_field,
                            surface_neutral_fallback_mask=surface_neutral_fallback_mask_field)
    metadata = (;
        reference_temperature,
        surface_temperature_initial,
        surface_cooling_rate,
        von_karman_constant,
        stable_momentum_beta,
        stable_heat_beta,
        momentum_roughness_length,
        heat_roughness_length,
        minimum_surface_wind_speed,
        maximum_surface_stability,
        interior_sgs_closure_available=!isnothing(model.closure),
        sgs_resolved_tke_dissipation_available=sgs_dissipation.available,
        sgs_tke_available=sgs_tke.available,
        averaging_definition="time average of instantaneous horizontally reduced diagnostics",
        native_w_moment_definition="horizontal central moments computed directly on native z faces before any vertical interpolation",
        instantaneous_skewness_definition="instantaneous w third central moment divided by instantaneous w variance^(3/2), then time averaged by the profile writer; raw moments are retained for ratio-of-window-averaged-moments",
        resolved_tke_definition="one half of centered u, v, and center-interpolated w perturbation variances",
        total_flux_definition="resolved plus interior SGS flux on interior faces; actual MOST BoundaryConditionOperation flux substituted at the bottom face",
        no_closure_definition="interior SGS flux and dissipation profiles are identically zero with explicit availability metadata; actual wall stress and heat flux remain active and are diagnosed independently",
        boundary_layer_height_definition="h = h_0.05 / 0.95; h_0.05 is the first linearly interpolated height where total momentum-stress magnitude falls to 5 percent of its actual MOST surface magnitude",
        missing_stress_crossing_handling="top-face fallback with boundary_layer_height_valid=0; never interpret fallback as a valid h",
        low_level_jet_definition="lowest global maximum of horizontal-mean wind speed; turning is atan(v,u) relative to the eastward geostrophic-wind direction",
        surface_heat_flux_definition="actual bottom thermodynamic-density BoundaryConditionOperation divided by surface density for K m s^-1; multiplied by dry-air heat capacity for W m^-2",
        obukhov_length_definition="L = -u_star^3 theta_ref / (kappa g <w theta>_surface); zero fallback only when obukhov_length_valid=0",
        surface_bulk_richardson_definition="g z1 [theta(z1)-theta_surface] / [theta_ref max(U(z1)^2, minimum_surface_wind_speed^2)], using the exact native-staggered tangential_speed² interpolation used by the scalar wall coefficient and the prescribed surface potential temperature",
        surface_stability_fraction_definition="horizontal fractions at the bottom matching point for which the exact linear-stable solver returns its maximum-stability cap, or Rb<=0 invokes the neutral fallback",
        tke_turbulent_transport_definition="-1/rho_r d/dz [rho_r <w' e'>], where e is per-mass resolved TKE; raw and density-weighted face fluxes are retained",
        tke_pressure_transport_definition="-1/rho_r d/dz [rho_r <w' (p'/rho_r)'>], matching Breeze anelastic pressure-gradient acceleration; raw and density-weighted face fluxes are retained",
        tke_budget_residual_definition="derive storage from successive profile records and form storage minus available production, transport, and dissipation terms; this closure residual is not numerical dissipation")

    return (; profile_outputs, series_outputs, diagnostic_fields, metadata)
end

function initialize_diagnostics_file!(file, model, metadata, kind, interval, window)
    file["metadata/diagnostic_kind"] = kind
    file["metadata/output_interval_seconds"] = interval
    file["metadata/averaging_window_seconds"] = window
    for (name, value) in pairs(metadata)
        file["metadata/$(name)"] = value
    end
    return nothing
end

function install_gabls_diagnostics!(simulation;
                                    dir=".",
                                    prefix="gabls",
                                    profile_interval=30minutes,
                                    series_interval=1minute,
                                    overwrite_files=true,
                                    kwargs...)
    diagnostics = build_gabls_diagnostics(simulation.model; kwargs...)

    profile_init = (file, model) -> initialize_diagnostics_file!(
        file, model, diagnostics.metadata, "true_time_averaged_profiles",
        profile_interval, profile_interval)
    series_init = (file, model) -> initialize_diagnostics_file!(
        file, model, diagnostics.metadata, "instantaneous_reduced_series",
        series_interval, zero(series_interval))

    simulation.output_writers[:gabls_statistics] = JLD2Writer(
        simulation.model, diagnostics.profile_outputs;
        filename="$(prefix)_statistics.jld2",
        dir,
        schedule=AveragedTimeInterval(profile_interval; window=profile_interval, stride=1),
        with_halos=false,
        overwrite_files,
        init=profile_init)

    simulation.output_writers[:gabls_series] = JLD2Writer(
        simulation.model, diagnostics.series_outputs;
        filename="$(prefix)_series.jld2",
        dir,
        schedule=TimeInterval(series_interval),
        with_halos=false,
        overwrite_files,
        init=series_init)

    return diagnostics
end

end # module GABLSDiagnostics
