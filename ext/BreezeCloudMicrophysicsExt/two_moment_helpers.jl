#####
##### Precipitation rate diagnostic for two-moment microphysics
#####

function AtmosphereModels.precipitation_rate(model, microphysics::TwoMomentCloudMicrophysics, ::Val{:liquid})
    grid = model.grid
    qᶜˡ = model.microphysical_fields.qᶜˡ
    nᶜˡ = model.microphysical_fields.nᶜˡ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρ = model.dynamics.reference_state.density
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
    reference_density :: RS
end

Adapt.adapt_structure(to, k::TwoMomentPrecipitationRateKernel) =
    TwoMomentPrecipitationRateKernel(adapt(to, k.categories),
                                      adapt(to, k.cloud_liquid),
                                      adapt(to, k.cloud_number),
                                      adapt(to, k.rain_density),
                                      adapt(to, k.reference_density))

@inline function (k::TwoMomentPrecipitationRateKernel)(i, j, k_idx, grid)
    sb = k.categories.warm_processes
    @inbounds qᶜˡ = k.cloud_liquid[i, j, k_idx]
    @inbounds nᶜˡ = k.cloud_number[i, j, k_idx]
    @inbounds ρqʳ = k.rain_density[i, j, k_idx]
    @inbounds ρ = k.reference_density[i, j, k_idx]

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

The surface precipitation flux is computed using the same advection scheme that
transports rain during time stepping, evaluated at the bottom face (`k = 1`).
This ensures numerical consistency between the diagnosed flux and the actual
mass leaving the domain through the advection operator.

Units: kg/m²/s (positive = downward, out of domain)
"""
function AtmosphereModels.surface_precipitation_flux(model, microphysics::TwoMomentCloudMicrophysics)
    grid = model.grid
    ρ = model.dynamics.reference_state.density
    wᵗ = AtmosphereModels.transport_velocities(model).w
    wʳ = model.microphysical_fields.wʳ
    qʳ = model.microphysical_fields.qʳ
    advection = model.advection.ρqʳ
    kernel = SurfacePrecipitationFluxKernel(advection)
    op = KernelFunctionOperation{Center, Center, Nothing}(kernel, grid, ρ, wᵗ, wʳ, qʳ)
    return Field(op)
end

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
