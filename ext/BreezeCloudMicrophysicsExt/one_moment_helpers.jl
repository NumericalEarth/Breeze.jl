#####
##### Precipitation rate diagnostic
#####

function AtmosphereModels.precipitation_rate(model, microphysics::OneMomentLiquidRain, ::Val{:liquid})
    grid = model.grid
    qᶜˡ = model.microphysical_fields.qᶜˡ
    ρqʳ = model.microphysical_fields.ρqʳ
    ρ = model.dynamics.reference_state.density
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
    reference_density :: RS
end

Adapt.adapt_structure(to, k::OneMomentPrecipitationRateKernel) =
    OneMomentPrecipitationRateKernel(adapt(to, k.categories),
                                      adapt(to, k.cloud_liquid),
                                      adapt(to, k.rain_density),
                                      adapt(to, k.reference_density))

@inline function (k::OneMomentPrecipitationRateKernel)(i, j, k_idx, grid)
    categories = k.categories
    @inbounds qᶜˡ = k.cloud_liquid[i, j, k_idx]
    @inbounds ρqʳ = k.rain_density[i, j, k_idx]
    @inbounds ρ = k.reference_density[i, j, k_idx]

    qʳ = ρqʳ / ρ

    # Autoconversion: cloud liquid → rain
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)

    # Accretion: cloud liquid captured by falling rain
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.blk1m.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Total precipitation production rate (kg/kg/s)
    return Sᵃᶜⁿᵛ + Sᵃᶜᶜ
end

#####
##### Surface precipitation flux (flux out of bottom boundary)
#####

"""
$(TYPEDSIGNATURES)

Return a 2D `Field` representing the precipitation flux at the bottom boundary.

The surface precipitation flux is ``wʳ ρqʳ`` at `k = 1` (bottom face), representing
the rate at which rain mass leaves the domain through the bottom boundary.

Units: kg/m²/s (positive = downward, out of domain)

!!! note "Sign convention"
    The returned value is positive when rain is falling out of the domain
    (the terminal velocity ``wʳ`` is negative, and we flip the sign).
"""
function AtmosphereModels.surface_precipitation_flux(model, microphysics::OneMomentCloudMicrophysics)
    grid = model.grid
    wʳ = model.microphysical_fields.wʳ
    ρqʳ = model.microphysical_fields.ρqʳ
    kernel = SurfacePrecipitationFluxKernel(wʳ, ρqʳ)
    op = KernelFunctionOperation{Center, Center, Nothing}(kernel, grid)
    return Field(op)
end

struct SurfacePrecipitationFluxKernel{W, R}
    terminal_velocity :: W
    rain_density :: R
end

Adapt.adapt_structure(to, k::SurfacePrecipitationFluxKernel) =
    SurfacePrecipitationFluxKernel(adapt(to, k.terminal_velocity),
                                    adapt(to, k.rain_density))

@inline function (kernel::SurfacePrecipitationFluxKernel)(i, j, k_idx, grid)
    # Flux at bottom face (k=1), ignore k_idx since this is a 2D field
    # wʳ < 0 (downward), so -wʳ * ρqʳ > 0 represents flux out of domain
    @inbounds wʳ = kernel.terminal_velocity[i, j, 1]
    @inbounds ρqʳ = kernel.rain_density[i, j, 1]

    # Return positive flux for rain leaving domain (downward)
    return -wʳ * ρqʳ
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

"""
$(TYPEDSIGNATURES)

Build a lazy [`NumberConcentration`](@ref) for rain under
`OneMomentCloudMicrophysics`. The returned `KernelFunctionOperation`
computes `n₀ · λ⁻¹` from the scheme's rain pdf and mass parameters.
"""
function Microphysics.number_concentration(model, microphysics::OneMomentLiquidRain, ::Val{:rain})
    haskey(model.microphysical_fields, :ρqʳ) || return nothing
    pdf = microphysics.categories.rain.pdf
    mass = microphysics.categories.rain.mass
    ρq = model.microphysical_fields.ρqʳ
    return build_number_concentration_op(model, pdf, mass, ρq)
end

"""
$(TYPEDSIGNATURES)

Build a lazy [`NumberConcentration`](@ref) for snow under
`OneMomentCloudMicrophysics`. Snow's intercept `n₀` depends on `(q, ρ)` per
Kaul et al. (2015), so a closed-form rain-style expression cannot substitute.
"""
function Microphysics.number_concentration(model, microphysics::OneMomentLiquidRain, ::Val{:snow})
    haskey(model.microphysical_fields, :ρqˢ) || return nothing
    pdf = microphysics.categories.snow.pdf
    mass = microphysics.categories.snow.mass
    ρq = model.microphysical_fields.ρqˢ
    return build_number_concentration_op(model, pdf, mass, ρq)
end

# Species not carried by the 1-mom scheme (e.g. :hail, :graupel, :cloud_liquid).
Microphysics.number_concentration(model, microphysics::OneMomentLiquidRain, ::Val) = nothing

function build_number_concentration_op(model, pdf, mass, ρq)
    ρ = dynamics_density(model.dynamics)
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
                  "τ_relax=", prettysummary(cl.τ_relax))
end

function Utils.prettysummary(ci::CloudIce)
    return string("CloudIce(",
                  "r0=", prettysummary(ci.r0), ", ",
                  "r_eff=", prettysummary(ci.r_eff), ", ",
                  "ρᵢ=", prettysummary(ci.ρᵢ), ", ",
                  "r_ice_snow=", prettysummary(ci.r_ice_snow), ", ",
                  "τ_relax=", prettysummary(ci.τ_relax), ", ",
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

function Utils.prettysummary(eff::CloudMicrophysics.Parameters.CollisionEff)
    return string("CollisionEff(",
                  "e_lcl_rai=", prettysummary(eff.e_lcl_rai), ", ",
                  "e_lcl_sno=", prettysummary(eff.e_lcl_sno), ", ",
                  "e_icl_rai=", prettysummary(eff.e_icl_rai), ", ",
                  "e_icl_sno=", prettysummary(eff.e_icl_sno), ", ",
                  "e_rai_sno=", prettysummary(eff.e_rai_sno), ")")
end

Utils.prettysummary(rain::CloudMicrophysics.Parameters.Rain) = "CloudMicrophysics.Parameters.Rain"
Utils.prettysummary(snow::CloudMicrophysics.Parameters.Snow) = "CloudMicrophysics.Parameters.Snow"

#=
function Utils.prettysummary(rain::CloudMicrophysics.Parameters.Rain)
    return string("Rain(",
                  "acnv1M=", prettysummary(rain.acnv1M), ", ",
                  "area=", prettysummary(rain.area), ", ",
                  "vent=", prettysummary(rain.vent), ", ",
                  "r0=", prettysummary(rain.r0), ", ",
                  "mass=", prettysummary(rain.mass), ", ",
                  "pdf=", prettysummary(rain.pdf), ")")
end
=#

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
    print(io, summary(bμp), ":\n",
          "├── cloud_formation: ", prettysummary(bμp.cloud_formation), '\n',
          "├── collisions: ", prettysummary(bμp.categories.collisions), '\n',
          "├── cloud_liquid: ", prettysummary(bμp.categories.cloud_liquid), '\n',
          "├── cloud_ice: ", prettysummary(bμp.categories.cloud_ice), '\n',
          "├── rain: ", prettysummary(bμp.categories.rain), '\n',
          "│   ├── acnv1M: ", prettysummary(bμp.categories.rain.acnv1M), '\n',
          "│   ├── area:   ", prettysummary(bμp.categories.rain.area), '\n',
          "│   ├── vent:   ", prettysummary(bμp.categories.rain.vent), '\n',
          "│   └── pdf:    ", prettysummary(bμp.categories.rain.pdf), '\n',
          "├── snow: ", prettysummary(bμp.categories.snow), "\n",
          "│   ├── acnv1M: ", prettysummary(bμp.categories.snow.acnv1M), '\n',
          "│   ├── area:   ", prettysummary(bμp.categories.snow.area), '\n',
          "│   ├── mass:   ", prettysummary(bμp.categories.snow.mass), '\n',
          "│   ├── r0:     ", prettysummary(bμp.categories.snow.r0), '\n',
          "│   ├── ρᵢ:     ", prettysummary(bμp.categories.snow.ρᵢ), '\n',
          "│   └── aspr:   ", prettysummary(bμp.categories.snow.aspr), '\n',
          "└── velocities: ", prettysummary(bμp.categories.hydrometeor_velocities))
end
