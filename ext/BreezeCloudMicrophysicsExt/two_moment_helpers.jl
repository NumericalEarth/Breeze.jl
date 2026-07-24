#####
##### Precipitation rate diagnostic for two-moment microphysics
#####

function AtmosphereModels.precipitation_rate(model, microphysics::TwoMomentCloudMicrophysics, ::Val{:liquid})
    grid = model.grid
    qᶜˡ = model.microphysical_fields.qᶜˡ
    nᶜˡ = model.microphysical_fields.nᶜˡ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρ = total_density(model.dynamics)
    kernel = TwoMomentPrecipitationRateKernel(microphysics.categories, qᶜˡ, nᶜˡ, ρqʳ, ρ)
    op = KernelFunctionOperation{Center, Center, Center}(kernel, grid)
    return Field(op)
end

# Ice precipitation not yet implemented for two-moment scheme
AtmosphereModels.precipitation_rate(model, ::TwoMomentCloudMicrophysics, ::Val{:ice}) = nothing

#####
##### Precipitation rate kernel for 2M scheme
#####

struct TwoMomentPrecipitationRateKernel{C, QL, NL, RR, RS}
    categories :: C
    cloud_liquid :: QL
    cloud_number :: NL
    rain_density :: RR
    density :: RS
end

Adapt.adapt_structure(to, k::TwoMomentPrecipitationRateKernel) =
    TwoMomentPrecipitationRateKernel(adapt(to, k.categories),
                                      adapt(to, k.cloud_liquid),
                                      adapt(to, k.cloud_number),
                                      adapt(to, k.rain_density),
                                      adapt(to, k.density))

@inline function (k::TwoMomentPrecipitationRateKernel)(i, j, k_idx, grid)
    sb = k.categories.warm_processes
    @inbounds qᶜˡ = k.cloud_liquid[i, j, k_idx]
    @inbounds nᶜˡ = k.cloud_number[i, j, k_idx]
    @inbounds ρqʳ = k.rain_density[i, j, k_idx]
    @inbounds ρ = k.density[i, j, k_idx]

    qʳ = ρqʳ / ρ
    Nᶜˡ = ρ * max(0, nᶜˡ)

    # Autoconversion: cloud liquid → rain
    au = CM2.autoconversion(sb.acnv, sb.pdf_c, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    ac = CM2.accretion(sb, max(0, qᶜˡ), max(0, qʳ), ρ, Nᶜˡ)

    # Total precipitation production rate (kg/kg/s)
    return au.dq_rai_dt + ac.dq_rai_dt
end

#####
##### Surface precipitation flux (flux out of bottom boundary)
#####

"""
$(TYPEDSIGNATURES)

Return a 2D `Field` representing the precipitation flux at the bottom boundary.

The surface precipitation flux sums every sedimenting prognostic moisture-mass tracer,
using the same advection scheme that transports each tracer during time stepping and
evaluating its flux at the bottom face (`k = 1`).
For explicit advection this is the same boundary flux used by the tendency operator.
For adaptive implicit advection it is the instantaneous split-operator flux.

Units: kg/m²/s (positive = downward, out of domain)
"""
function AtmosphereModels.surface_precipitation_flux(model, microphysics::TwoMomentCloudMicrophysics)
    return sedimenting_moisture_surface_flux(model, microphysics)
end

#####
##### Number concentration diagnostic (2-moment)
#####
#
# For 2-mom microphysics, the total number concentration is the prognostic
# ρnˣ field; the diagnostic just hands it back so consumers see the same
# interface as the 1-mom case (build a `Field` if they need to compute).

Microphysics.number_concentration(model, ::TwoMomentCloudMicrophysics, ::Val{:rain}) =
    get(model.microphysical_fields, :ρnʳ, nothing)

Microphysics.number_concentration(model, ::TwoMomentCloudMicrophysics, ::Val{:cloud_liquid}) =
    get(model.microphysical_fields, :ρnᶜˡ, nothing)

Microphysics.number_concentration(model, ::TwoMomentCloudMicrophysics, ::Val) = nothing

#####
##### show methods for two-moment microphysics
#####

using Oceananigans.Utils: Utils

function Utils.prettysummary(tc::TwoMomentCategories)
    return "TwoMomentCategories(SB2006)"
end

function Utils.prettysummary(sb::SB2006)
    return "SB2006"
end

function Utils.prettysummary(vel::StokesRegimeVelType)
    return "StokesRegimeVelType"
end

function Utils.prettysummary(vel::SB2006VelType)
    return "SB2006VelType"
end

function Utils.prettysummary(vel::Chen2022VelTypeRain)
    return "Chen2022VelTypeRain"
end

function Base.show(io::IO, bμp::BulkMicrophysics{<:Any, <:CM2MCategories})
    categories = bμp.categories
    print(io, summary(bμp), ":\n",
          "├── cloud_formation: ", prettysummary(bμp.cloud_formation), '\n',
          "├── warm_processes: ", prettysummary(categories.warm_processes), '\n',
          "├── air_properties: ", prettysummary(categories.air_properties), '\n',
          "├── cloud_liquid_fall_velocity: ", prettysummary(categories.cloud_liquid_fall_velocity), '\n',
          "├── rain_fall_velocity: ", prettysummary(categories.rain_fall_velocity), '\n',
          "└── precipitation_boundary_condition: ", bμp.precipitation_boundary_condition)
end

Base.summary(bμp::TwoMomentCloudMicrophysics) = "TwoMomentCloudMicrophysics"
