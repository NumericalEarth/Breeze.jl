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

OceanTurbulenceClosures.buoyancy_tracers(model::AtmosphereModel) =
    (; T = model.temperature, qᵗ = model.specific_moisture)

@inline OceanBuoyancyFormulations.∂z_b(i, j, k, grid, b::AtmosphereModelBuoyancy, tracers) =
    ∂zᶜᶜᶠ(i, j, k, grid, turbulence_closure_buoyancy, b, tracers)

@inline turbulence_closure_buoyancy(i, j, k, grid, b::AtmosphereModelBuoyancy, tracers) =
    buoyancy(i, j, k, grid, b.formulation, tracers.T, tracers.qᵗ, b.thermodynamics)
