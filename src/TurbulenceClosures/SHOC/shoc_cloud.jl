#####
##### Cloud feedback: SHOC's subgrid cloud as the cloud of the model
#####
##### With a microphysics scheme whose subgrid choice is `ClosureCloud`, SHOC owns the cloud liquid
##### and the cloud fraction of the model:
#####
##### * SHOC's total water is the vapor plus the cloud liquid of the model. After every SHOC step
#####   the cloud liquid of the model is set to the liquid `shoc_ql` that the assumed PDF diagnosed,
#####   and the vapor to the rest, so their sum and the liquid-ice potential temperature are
#####   exactly what SHOC's implicit solve left. Temperature, buoyancy and radiation then see
#####   SHOC's cloud through the cloud liquid of the model, like any other cloud.
##### * the microphysics reads the cloud fraction `shoc_cldfrac` through `subgrid_statisticsᶜᶜᶜ`,
#####   forms and evaporates no cloud liquid of its own, and owns the precipitation.
#####
##### During the Runge–Kutta stages the cloud liquid is a transported prognostic that only the
##### warm-rain sinks deplete; the cloud fraction is the one of the last SHOC step.
#####

"""
$(TYPEDEF)

The vapor and cloud liquid densities of a model whose cloud SHOC owns. Their sum over the density
is SHOC's total water.

$(TYPEDFIELDS)
"""
struct VaporAndCloudLiquid{V, C}
    "Vapor density, kg m⁻³"
    vapor :: V
    "Cloud liquid density, kg m⁻³"
    cloud_liquid :: C
end

Adapt.adapt_structure(to, moisture::VaporAndCloudLiquid) =
    VaporAndCloudLiquid(adapt(to, moisture.vapor), adapt(to, moisture.cloud_liquid))

# The moisture SHOC mixes: the moisture density of the model, joined by its cloud liquid when the
# microphysics leaves the cloud to the closure.
shoc_moisture(model) =
    shoc_moisture(AtmosphereModels.subgrid_correlation(model.microphysics), model)

shoc_moisture(subgrid_correlation, model) = model.moisture_density

shoc_moisture(::ClosureCloud, model) =
    VaporAndCloudLiquid(model.moisture_density, model.microphysical_fields.ρqᶜˡ)

# The moisture density whose bottom boundary condition is SHOC's surface moisture flux: the
# prognostic moisture of the model (the vapor, when SHOC owns the cloud).
@inline surface_moisture_density(ρqᵗ) = ρqᵗ
@inline surface_moisture_density(moisture::VaporAndCloudLiquid) = moisture.vapor

@inline total_water_density(i, j, k, ρqᵗ) = @inbounds ρqᵗ[i, j, k]

@inline total_water_density(i, j, k, moisture::VaporAndCloudLiquid) =
    @inbounds moisture.vapor[i, j, k] + moisture.cloud_liquid[i, j, k]

@inline function store_total_water!(i, j, k, ρqᵗ, ρ, qw, ql)
    @inbounds ρqᵗ[i, j, k] = ρ * qw
    return nothing
end

# SHOC's liquid becomes the cloud liquid of the model, bounded by the water there is.
@inline function store_total_water!(i, j, k, moisture::VaporAndCloudLiquid, ρ, qw, ql)
    cloud_liquid = min(max(0, ql), max(0, qw))
    @inbounds moisture.cloud_liquid[i, j, k] = ρ * cloud_liquid
    @inbounds moisture.vapor[i, j, k] = ρ * (qw - cloud_liquid)
    return nothing
end

# The cloud fraction and the liquid of the last SHOC step
@inline function AtmosphereModels.subgrid_statisticsᶜᶜᶜ(i, j, k, grid, ::ClosureCloud, ::FlavorOfSHOC,
                                                         fields::SHOCFields)
    diagnostics = fields.diagnostics
    return @inbounds ClosureCloudStatistics(diagnostics.shoc_cldfrac[i, j, k], diagnostics.shoc_ql[i, j, k])
end
