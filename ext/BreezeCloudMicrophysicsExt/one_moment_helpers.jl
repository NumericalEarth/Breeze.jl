#####
##### Precipitation rate diagnostic
#####

function AtmosphereModels.precipitation_rate(model, microphysics::OneMomentLiquidRain, ::Val{:liquid})
    grid = model.grid
    qᶜˡ = model.microphysical_fields.qᶜˡ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρ = total_density(model.dynamics)
    kernel = OneMomentPrecipitationRateKernel(microphysics.categories, qᶜˡ, ρqʳ, ρ)
    op = KernelFunctionOperation{Center, Center, Center}(kernel, grid)
    return Field(op)
end

# Ice precipitation not yet implemented for one-moment scheme
AtmosphereModels.precipitation_rate(model, ::OneMomentCloudMicrophysics, ::Val{:ice}) = nothing

#####
##### Precipitation rate kernel (shared by all 1M schemes)
#####

struct OneMomentPrecipitationRateKernel{C, QL, RR, RS}
    categories :: C
    cloud_liquid :: QL
    rain_density :: RR
    density :: RS
end

Adapt.adapt_structure(to, k::OneMomentPrecipitationRateKernel) =
    OneMomentPrecipitationRateKernel(adapt(to, k.categories),
                                      adapt(to, k.cloud_liquid),
                                      adapt(to, k.rain_density),
                                      adapt(to, k.density))

@inline function (k::OneMomentPrecipitationRateKernel)(i, j, k_idx, grid)
    categories = k.categories
    @inbounds qᶜˡ = k.cloud_liquid[i, j, k_idx]
    @inbounds ρqʳ = k.rain_density[i, j, k_idx]
    @inbounds ρ = k.density[i, j, k_idx]

    qʳ = ρqʳ / ρ
    parameters = categories.parameters

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = liquid_autoconversion(parameters, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = cloud_precipitation_accretion(
        parameters.options.cloud_liquid_rain_accretion,
        parameters.cloud.liquid,
        parameters.precip.rain,
        parameters.terminal_velocity.rain,
        qᶜˡ,
        qʳ,
        ρ,
    )

    # Total precipitation production rate (kg/kg/s)
    return Sᵃᶜⁿᵛ + Sᵃᶜᶜ
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
function AtmosphereModels.surface_precipitation_flux(model, microphysics::OneMomentCloudMicrophysics)
    return sedimenting_moisture_surface_flux(model, microphysics)
end

sedimenting_moisture_surface_flux_operations(model, microphysics, ::Tuple{}) = ()

function sedimenting_moisture_surface_flux_operations(model, microphysics, names::Tuple{Symbol, Vararg})
    name = first(names)
    rest = sedimenting_moisture_surface_flux_operations(model, microphysics, Base.tail(names))
    is_moisture_mass_tracer(name) || return rest
    isnothing(moisture_phase(microphysics, Val(name))) && return rest

    w = sedimentation_velocity(microphysics, model.microphysical_fields, Val(name))
    isnothing(w) && return rest

    grid = model.grid
    ρ = total_density(model.dynamics)
    wᵗ = transport_velocities(model).w
    q = getproperty(model.microphysical_fields, specific_field_name(name))
    advection = getproperty(model.advection, name)
    kernel = SurfacePrecipitationFluxKernel(advection)
    operation = KernelFunctionOperation{Center, Center, Nothing}(kernel, grid, ρ, wᵗ, w, q)
    return (operation, rest...)
end

function sedimenting_moisture_surface_flux(model, microphysics)
    names = prognostic_field_names(microphysics)
    operations = sedimenting_moisture_surface_flux_operations(model, microphysics, names)
    isempty(operations) && return Field{Center, Center, Nothing}(model.grid)
    return Field(reduce(+, operations))
end

#####
##### Number concentration diagnostic (1-moment)
#####
#
# For 1-mom rain and snow, the total number concentration is reconstructed
# from the prognostic mass density ρqˣ and the scheme's assumed size
# distribution as ρnˣ = n₀ · λ⁻¹. Snow's intercept n₀ depends on (q, ρ)
# per Kaul et al. (2015), so this dispatch goes through the scheme's
# `pdf` and `mass` to stay consistent with the model's actual DSD.

function Microphysics.number_concentration(model, microphysics::OneMomentLiquidRain, ::Val{:rain})
    haskey(model.microphysical_fields, :ρqʳ) || return nothing
    pdf = microphysics.categories.parameters.precip.rain.pdf
    mass = microphysics.categories.parameters.precip.rain.mass
    ρq = model.microphysical_fields.ρqʳ
    return build_number_concentration_op(model, pdf, mass, ρq)
end

function Microphysics.number_concentration(model, microphysics::OneMomentLiquidRain, ::Val{:snow})
    haskey(model.microphysical_fields, :ρqˢ) || return nothing
    pdf = microphysics.categories.parameters.precip.snow.pdf
    mass = microphysics.categories.parameters.precip.snow.mass
    ρq = model.microphysical_fields.ρqˢ
    return build_number_concentration_op(model, pdf, mass, ρq)
end

# Species not carried by the 1-mom scheme (e.g. :hail, :graupel, :cloud_liquid).
Microphysics.number_concentration(model, microphysics::OneMomentLiquidRain, ::Val) = nothing

function build_number_concentration_op(model, pdf, mass, ρq)
    ρ = total_density(model.dynamics)  # number conc N = ρ·n uses total air density
    func = NumberConcentrationKernelFunction(pdf, mass, ρq, ρ)
    return KernelFunctionOperation{Center, Center, Center}(func, model.grid)
end

@inline function (k::NumberConcentrationKernelFunction)(i, j, k_idx, grid)
    @inbounds begin
        ρq = max(k.ρq[i, j, k_idx], zero(eltype(k.ρq)))
        ρ  = k.reference_density[i, j, k_idx]
    end
    q   = ρq / ρ
    n0  = get_n0(k.pdf, q, ρ)
    λ⁻¹ = lambda_inverse(k.pdf, k.mass, q, ρ)
    return n0 * λ⁻¹
end

#####
##### show methods
#####

using Oceananigans.Utils: Utils, prettysummary

function Utils.prettysummary(cl::CloudLiquid)
    return string("CloudLiquid(",
                  "ρw=", prettysummary(cl.ρw), ", ",
                  "r_eff=", prettysummary(cl.r_eff), ", ",
                  "N_0=", prettysummary(cl.N_0), ")")
end

function Utils.prettysummary(ci::CloudIce)
    return string("CloudIce(",
                  "r_eff=", prettysummary(ci.r_eff), ", ",
                  "ρᵢ=", prettysummary(ci.ρᵢ), ", ",
                  "N_0=", prettysummary(ci.N_0), ", ",
                  "mass=", prettysummary(ci.mass), ", ",
                  "pdf=", prettysummary(ci.pdf), ")")
end

function Utils.prettysummary(mass::CloudMicrophysics.Parameters.ParticleMass)
    return string("ParticleMass(",
                  "r0=", prettysummary(mass.r0), ", ",
                  "m0=", prettysummary(mass.m0), ", ",
                  "me=", prettysummary(mass.me), ", ",
                  "Δm=", prettysummary(mass.Δm), ", ",
                  "χm=", prettysummary(mass.χm), ")")
end

function Utils.prettysummary(pdf::CloudMicrophysics.Parameters.ParticlePDFIceRain)
    return string("ParticlePDFIceRain(n0=", prettysummary(pdf.n0), ")")
end

Utils.prettysummary(rain::CloudMicrophysics.Parameters.Rain) = "CloudMicrophysics.Parameters.Rain"
Utils.prettysummary(snow::CloudMicrophysics.Parameters.Snow) = "CloudMicrophysics.Parameters.Snow"

function Utils.prettysummary(acnv::CloudMicrophysics.Parameters.Acnv1M)
    return string("Acnv1M(",
                  "τ=", prettysummary(acnv.τ), ", ",
                  "q_threshold=", prettysummary(acnv.q_threshold), ", ",
                  "k=", prettysummary(acnv.k), ")")
end

function Utils.prettysummary(area::CloudMicrophysics.Parameters.ParticleArea)
    return string("ParticleArea(",
                  "a0=", prettysummary(area.a0), ", ",
                  "ae=", prettysummary(area.ae), ", ",
                  "Δa=", prettysummary(area.Δa), ", ",
                  "χa=", prettysummary(area.χa), ")")
end

function Utils.prettysummary(vent::CloudMicrophysics.Parameters.Ventilation)
    return string("Ventilation(",
                  "a=", prettysummary(vent.a), ", ",
                  "b=", prettysummary(vent.b), ")")
end

function Utils.prettysummary(aspr::CloudMicrophysics.Parameters.SnowAspectRatio)
    return string("SnowAspectRatio(",
                  "ϕ=", prettysummary(aspr.ϕ), ", ",
                  "κ=", prettysummary(aspr.κ), ")")
end

Utils.prettysummary(vel::Blk1MVelType) = "Blk1MVelType(...)"
Utils.prettysummary(vel::TerminalVelocityParams) = "TerminalVelocityParams(...)"
Utils.prettysummary(vel::Blk1MVelTypeRain) = "Blk1MVelTypeRain(...)"
Utils.prettysummary(vel::Blk1MVelTypeSnow) = "Blk1MVelTypeSnow(...)"

function Utils.prettysummary(ne::NonEquilibriumCloudFormation)
    liquid_str = isnothing(ne.liquid) ? "nothing" : "liquid(τ=$(prettysummary(1/ne.liquid.rate)))"
    ice_str = isnothing(ne.ice) ? "nothing" : "ice(τ=$(prettysummary(1/ne.ice.rate)))"
    return "NonEquilibriumCloudFormation($liquid_str, $ice_str)"
end

function Base.show(io::IO, bμp::BulkMicrophysics{<:Any, <:CM1MCategories})
    parameters = bμp.categories.parameters
    options = parameters.options
    rain = parameters.precip.rain
    snow = parameters.precip.snow

    print(io, summary(bμp), ":\n",
          "├── cloud_formation: ", prettysummary(bμp.cloud_formation), '\n',
          "├── options: ", summary(options), '\n',
          "├── cloud_liquid: ", prettysummary(parameters.cloud.liquid), '\n',
          "├── cloud_ice: ", prettysummary(parameters.cloud.ice), '\n',
          "├── rain: ", prettysummary(rain), '\n',
          "│   ├── autoconversion: ", prettysummary(options.rain_autoconversion), '\n',
          "│   ├── area:   ", prettysummary(rain.area), '\n',
          "│   ├── vent:   ", prettysummary(rain.vent), '\n',
          "│   └── pdf:    ", prettysummary(rain.pdf), '\n',
          "├── snow: ", prettysummary(snow), "\n",
          "│   ├── autoconversion: ", prettysummary(options.snow_autoconversion), '\n',
          "│   ├── area:   ", prettysummary(snow.area), '\n',
          "│   ├── mass:   ", prettysummary(snow.mass), '\n',
          "│   ├── ρᵢ:     ", prettysummary(snow.ρᵢ), '\n',
          "│   └── aspr:   ", prettysummary(snow.aspr), '\n',
          "├── freezing_temperature: ", prettysummary(bμp.categories.freezing_temperature), '\n',
          "└── velocities: ", prettysummary(bμp.categories.hydrometeor_velocities))
end
