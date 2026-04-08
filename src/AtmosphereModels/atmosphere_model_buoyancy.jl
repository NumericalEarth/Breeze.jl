using Oceananigans.TurbulenceClosures: TurbulenceClosures as OceanTurbulenceClosures
using Oceananigans.BuoyancyFormulations: BuoyancyFormulations as OceanBuoyancyFormulations
using Oceananigans.Operators: ∂zᶜᶜᶠ

"""
$(TYPEDEF)

Wrapper struct for computing buoyancy for [`AtmosphereModel`](@ref)
in the context of a turbulence closure. Used to interface with Oceananigans
turbulence closures that require buoyancy gradients.
"""
struct AtmosphereModelBuoyancy{D, F, T}
    dynamics :: D
    formulation :: F
    thermodynamic_constants :: T
end

Adapt.adapt_structure(to, b::AtmosphereModelBuoyancy) =
    AtmosphereModelBuoyancy(adapt(to, b.dynamics), adapt(to, b.formulation), adapt(to, b.thermodynamic_constants))

#####
##### Buoyancy interface for AtmosphereModel
#####

OceanTurbulenceClosures.buoyancy_force(model::AtmosphereModel) =
    AtmosphereModelBuoyancy(model.dynamics, model.formulation, model.thermodynamic_constants)

# buoyancy_tracers returns tracers needed for:
# 1. Buoyancy computation (T, qᵗ) used in ∂z_b and AMD viscosity
# 2. Diffusivity computation for each tracer in closure_fields.κₑ
# The energy_density and moisture_density are first (matching closure_names order),
# followed by user tracers, then diagnostic fields for buoyancy.
# TODO: make this microphysics-aware, and also saturation/condensate-aware
function OceanTurbulenceClosures.buoyancy_tracers(model::AtmosphereModel)
    # Diagnostic fields for buoyancy gradient calculation
    buoyancy_tracers = (; T = model.temperature, qᵛ = specific_humidity(model))
    # Prognostic tracer fields for diffusivity computation
    moist_name = moisture_prognostic_name(model.microphysics)
    prognostic_tracers = merge(prognostic_fields(model.formulation), NamedTuple{(moist_name,)}((model.moisture_density,)))
    # Merge with user tracers
    all_prognostic = merge(prognostic_tracers, model.tracers)
    # Final merge - buoyancy tracers at end for named access in ∂z_b
    return merge(all_prognostic, buoyancy_tracers)
end

@inline function OceanBuoyancyFormulations.∂z_b(i, j, k, grid, b::AtmosphereModelBuoyancy, tracers)
    g = b.thermodynamic_constants.gravitational_acceleration
    ∂z_ϑ = ∂zᶜᶜᶠ(i, j, k, grid, virtual_potential_temperature, b.thermodynamic_constants, b.dynamics, tracers.T, tracers.qᵛ)
    ϑ = virtual_potential_temperature(i, j, k, grid, b.thermodynamic_constants, b.dynamics, tracers.T, tracers.qᵛ)
    return g * ∂z_ϑ / ϑ
end

@inline function virtual_potential_temperature(i, j, k, grid, constants, dynamics, T, qᵛ)
    pᵣ_field = dynamics_pressure(dynamics)
    @inbounds pᵣ = pᵣ_field[i, j, k]
    pˢᵗ = standard_pressure(dynamics)
    q = @inbounds MoistureMassFractions(qᵛ[i, j, k])
    Rᵐ = mixture_gas_constant(q, constants)
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    return @inbounds Rᵐ / Rᵈ * T[i, j, k] * (pˢᵗ / pᵣ)^(Rᵐ / cᵖᵐ)
end
