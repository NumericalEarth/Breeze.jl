using Oceananigans.TurbulenceClosures: TurbulenceClosures as OceanTurbulenceClosures
using Oceananigans.BuoyancyFormulations: BuoyancyFormulations as OceanBuoyancyFormulations
using Oceananigans.Operators: ∂zᶜᶜᶠ

"""
    AtmosphereModelBuoyancy{F, T}

Wrapper struct for computing buoyancy for AtmosphereModel
in the context of a turbulence closure. Used to interface with Oceananigans
turbulence closures that require buoyancy gradients.
"""
struct AtmosphereModelBuoyancy{F, T}
    formulation :: F
    thermodynamics :: T
end

Adapt.adapt_structure(to, b::AtmosphereModelBuoyancy) =
    AtmosphereModelBuoyancy(adapt(to, b.formulation), adapt(to, b.thermodynamics))

#####
##### Buoyancy interface for AtmosphereModel
#####

OceanTurbulenceClosures.buoyancy_force(model::AtmosphereModel) =
    AtmosphereModelBuoyancy(model.formulation, model.thermodynamics)

# buoyancy_tracers returns tracers needed for:
# 1. Buoyancy computation (T, qᵗ) used in ∂z_b and AMD viscosity
# 2. Diffusivity computation for each tracer in closure_fields.κₑ
# The energy_density and moisture_density are first (matching closure_names order),
# followed by user tracers, then diagnostic fields for buoyancy.
function OceanTurbulenceClosures.buoyancy_tracers(model::AtmosphereModel)
    # Diagnostic fields for buoyancy gradient calculation
    buoyancy_tracers = (; T = model.temperature, qᵗ = model.specific_moisture)
    # Prognostic tracer fields for diffusivity computation
    prognostic_tracers = (; ρe = model.energy_density, ρqᵗ = model.moisture_density)
    # Merge with user tracers
    all_prognostic = merge(prognostic_tracers, model.tracers)
    # Final merge - buoyancy tracers at end for named access in ∂z_b
    return merge(all_prognostic, buoyancy_tracers)
end

@inline OceanBuoyancyFormulations.∂z_b(i, j, k, grid, b::AtmosphereModelBuoyancy, tracers) =
    ∂zᶜᶜᶠ(i, j, k, grid, turbulence_closure_buoyancy, b, tracers)

@inline turbulence_closure_buoyancy(i, j, k, grid, b::AtmosphereModelBuoyancy, tracers) =
    buoyancy(i, j, k, grid, b.formulation, tracers.T, tracers.qᵗ, b.thermodynamics)
