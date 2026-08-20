module Advection

export div_ρUc,
       surface_advective_tracer_flux,
       SurfacePrecipitationFluxKernel

using Oceananigans.Advection:
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    AdaptiveImplicitVerticalAdvection,
    _biased_interpolate_zᵃᵃᶠ,
    BoundsPreservingWENO,
    LeftBias,
    RightBias,
    upwind_biased_product,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z,
    explicit_velocity_scaleᶜᶜᶠ,
    vertical_scheme,
    _ω̂₁, _ω̂ₙ, _ε₂

using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Azᶜᶜᶠ
using Oceananigans.Utils: SumOfArrays
using Oceananigans.TimeSteppers: time_discretization
using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS
using Adapt: Adapt, adapt

using ..AtmosphereModels: AtmosphereModels, div_ρUc

const AIVA = AdaptiveImplicitVerticalAdvection

struct CFLScaledVerticalVelocity{A, G, W}
    advection :: A
    grid :: G
    velocity :: W
end

@inline function Base.getindex(w::CFLScaledVerticalVelocity, i, j, k)
    scheme = vertical_scheme(w.advection)
    td = time_discretization(scheme)
    scale = explicit_velocity_scaleᶜᶜᶠ(i, j, k, w.grid, scheme, td, w.velocity)
    @inbounds velocity = w.velocity[i, j, k]
    return scale * velocity
end

@inline explicit_vertical_velocity(advection, grid, w) = w
@inline explicit_vertical_velocity(advection::AIVA, grid, w) =
    CFLScaledVerticalVelocity(advection, grid, w)

# Simple wrappers: interpolate ρ to face, multiply existing flux
@inline tracer_mass_flux_x(i, j, k, grid, ρ, args...) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_x(i, j, k, grid, args...)

@inline tracer_mass_flux_y(i, j, k, grid, ρ, args...) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_y(i, j, k, grid, args...)

@inline tracer_mass_flux_z(i, j, k, grid, ρ, args...) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _advective_tracer_flux_z(i, j, k, grid, args...)

# Main operator
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection, ρ, U, c)
    explicit_w = explicit_vertical_velocity(advection, grid, U.w)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
        δxᶜᵃᵃ(i, j, k, grid, tracer_mass_flux_x, ρ, advection, U.u, c) +
        δyᵃᶜᵃ(i, j, k, grid, tracer_mass_flux_y, ρ, advection, U.v, c) +
        δzᵃᵃᶜ(i, j, k, grid, tracer_mass_flux_z, ρ, advection, explicit_w, c))
end

# Fallback for nothing advection
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::Nothing, ρ, U, c) = zero(grid)
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingWENO, ρ, U, c)
    explicit_w = explicit_vertical_velocity(advection, grid, U.w)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, explicit_w, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Advection-consistent surface flux
#####

"""
$(TYPEDSIGNATURES)

Compute the downward advective mass flux per unit area at the bottom face (`k = 1`)
for tracer `c` advected by vertical velocity `w` through density field `ρ`, using the
given `advection` scheme.

For an explicit scheme this evaluates the same face flux that `div_ρUc` uses at
the bottom boundary. For adaptive implicit vertical advection it returns the
instantaneous split-operator flux at the supplied tracer state; the step-integrated
implicit outflow depends on the post-solve tracer and must be accumulated separately.

For adaptive implicit vertical advection, the result includes both the CFL-scaled
high-order explicit flux and its first-order implicit remainder, evaluated at `c`.

Returns a positive value for downward (out-of-domain) flux.
"""
@inline function surface_advective_tracer_flux(i, j, grid, advection, ρ, w, c)
    flux_Az = _advective_tracer_flux_z(i, j, 1, grid, advection, w, c)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * flux_Az / Azᶜᶜᶠ(i, j, 1, grid)
    return explicit_flux + implicit_surface_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

@inline implicit_surface_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c) = 0

@inline function implicit_surface_advective_tracer_flux(i, j, grid, advection::AIVA, ρ_face, w, c)
    scheme = vertical_scheme(advection)
    td = time_discretization(scheme)
    scale = explicit_velocity_scaleᶜᶜᶠ(i, j, 1, grid, scheme, td, w)
    @inbounds w_total = w[i, j, 1]
    w_implicit = (1 - scale) * w_total
    @inbounds c_above = c[i, j, 1]
    return -ρ_face * min(w_implicit, 0) * c_above
end

# Bounds-preserving WENO: replicate the bottom face flux from
# bounded_tracer_flux_divergence_z, which applies a limiting coefficient θ
# to the inward face reconstructions before computing the upwind flux.
@inline function surface_advective_tracer_flux(i, j, grid, advection::BoundsPreservingWENO, ρ, w, c)
    c₊ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, 2, grid, advection, LeftBias,  c)
    c₋ᴸ = _biased_interpolate_zᵃᵃᶠ(i, j, 1, grid, advection, LeftBias,  c)
    c₋ᴿ = _biased_interpolate_zᵃᵃᶠ(i, j, 1, grid, advection, RightBias, c)

    FT = eltype(c)
    ω̂₁ = convert(FT, _ω̂₁)
    ω̂ₙ = convert(FT, _ω̂ₙ)
    ε₂ = convert(FT, _ε₂)

    c_min = @inbounds advection.bounds[1]
    c_max = @inbounds advection.bounds[2]

    @inbounds cᵢⱼ = c[i, j, 1]
    p̃ = (cᵢⱼ - ω̂₁ * c₋ᴿ - ω̂ₙ * c₊ᴸ) / (1 - 2ω̂₁)
    M = max(p̃, c₊ᴸ, c₋ᴿ)
    m = min(p̃, c₊ᴸ, c₋ᴿ)

    θ_max = abs((c_max - cᵢⱼ) / (M - cᵢⱼ + ε₂))
    θ_min = abs((c_min - cᵢⱼ) / (m - cᵢⱼ + ε₂))
    θ = min(θ_max, θ_min, one(grid))

    c₋ᴿ = θ * (c₋ᴿ - cᵢⱼ) + cᵢⱼ

    explicit_w = explicit_vertical_velocity(advection, grid, w)
    @inbounds w⁻ = explicit_w[i, j, 1]
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    explicit_flux = -ρ_face * upwind_biased_product(w⁻, c₋ᴸ, c₋ᴿ)
    return explicit_flux + implicit_surface_advective_tracer_flux(i, j, grid, advection, ρ_face, w, c)
end

"""
$(TYPEDEF)

Kernel for `KernelFunctionOperation`
that computes the advection-consistent precipitation flux at the bottom boundary.

The kernel is called as `kernel(i, j, k, grid, ρ, wᵗ, wˢᵉᵈ, c)` where `ρ` is the
total density used to carry tracer advection, `wᵗ` is the vertical transport velocity,
`wˢᵉᵈ` is the sedimentation velocity, and `c` is the specific tracer (e.g., `qʳ`). Pass
these four fields as arguments to the `KernelFunctionOperation`:

    kernel = SurfacePrecipitationFluxKernel(model.advection.ρqʳ)
    op = KernelFunctionOperation{Center, Center, Nothing}(kernel, grid, ρ, wᵗ, wʳ, qʳ)

$(TYPEDFIELDS)
"""
struct SurfacePrecipitationFluxKernel{A}
    advection :: A
end

Adapt.adapt_structure(to, k::SurfacePrecipitationFluxKernel) =
    SurfacePrecipitationFluxKernel(adapt(to, k.advection))

@inline function (kernel::SurfacePrecipitationFluxKernel)(i, j, k_idx, grid, ρ, wᵗ, wˢᵉᵈ, c)
    w_total = SumOfArrays{2}(wᵗ, wˢᵉᵈ)
    return surface_advective_tracer_flux(i, j, grid, kernel.advection, ρ, w_total, c)
end

end # module
