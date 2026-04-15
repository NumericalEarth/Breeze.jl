module Advection

export div_ρUc,
       surface_advective_tracer_flux,
       SurfacePrecipitationFluxKernel

using Oceananigans.Advection:
    _advective_tracer_flux_x,
    _advective_tracer_flux_y,
    _advective_tracer_flux_z,
    BoundsPreservingWENO,
    bounded_tracer_flux_divergence_x,
    bounded_tracer_flux_divergence_y,
    bounded_tracer_flux_divergence_z

using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: V⁻¹ᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, Azᶜᶜᶠ
using Oceananigans.Utils: SumOfArrays
using Adapt: Adapt, adapt

using ..AtmosphereModels: AtmosphereModels, div_ρUc

# Simple wrappers: interpolate ρ to face, multiply existing flux
@inline tracer_mass_flux_x(i, j, k, grid, ρ, args...) =
    ℑxᶠᵃᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_x(i, j, k, grid, args...)

@inline tracer_mass_flux_y(i, j, k, grid, ρ, args...) =
    ℑyᵃᶠᵃ(i, j, k, grid, ρ) * _advective_tracer_flux_y(i, j, k, grid, args...)

@inline tracer_mass_flux_z(i, j, k, grid, ρ, args...) =
    ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _advective_tracer_flux_z(i, j, k, grid, args...)

# Main operator
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection, ρ, U, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
        δxᶜᵃᵃ(i, j, k, grid, tracer_mass_flux_x, ρ, advection, U.u, c) +
        δyᵃᶜᵃ(i, j, k, grid, tracer_mass_flux_y, ρ, advection, U.v, c) +
        δzᵃᵃᶜ(i, j, k, grid, tracer_mass_flux_z, ρ, advection, U.w, c))
end

# Fallback for nothing advection
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::Nothing, ρ, U, c) = zero(grid)
@inline AtmosphereModels.div_ρUc(i, j, k, grid, ::BoundsPreservingWENO, ρ, U, ::ZeroField) = zero(grid)

# Is this immersed-boundary safe without having to extend it in ImmersedBoundaries.jl? I think so... (velocity on immmersed boundaries is masked to 0)
@inline function AtmosphereModels.div_ρUc(i, j, k, grid, advection::BoundsPreservingWENO, ρ, U, c)
    div_x = bounded_tracer_flux_divergence_x(i, j, k, grid, advection, ρ, U.u, c)
    div_y = bounded_tracer_flux_divergence_y(i, j, k, grid, advection, ρ, U.v, c)
    div_z = bounded_tracer_flux_divergence_z(i, j, k, grid, advection, ρ, U.w, c)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (div_x + div_y + div_z)
end

#####
##### Advection-consistent surface flux
#####

"""
    surface_advective_tracer_flux(i, j, grid, advection, ρ, w, c)

Compute the downward advective mass flux per unit area at the bottom face (`k = 1`)
for tracer `c` advected by vertical velocity `w` through density field `ρ`, using the
given `advection` scheme.

This evaluates the same face flux that [`div_ρUc`](@ref) uses at the bottom boundary,
ensuring numerical consistency between the diagnosed surface flux and the actual mass
leaving the domain during time stepping.

Returns a positive value for downward (out-of-domain) flux.
"""
@inline function surface_advective_tracer_flux(i, j, grid, advection, ρ, w, c)
    flux_Az = _advective_tracer_flux_z(i, j, 1, grid, advection, w, c)
    ρ_face = ℑzᵃᵃᶠ(i, j, 1, grid, ρ)
    return -ρ_face * flux_Az / Azᶜᶜᶠ(i, j, 1, grid)
end

"""
    SurfacePrecipitationFluxKernel(advection)

Kernel for [`KernelFunctionOperation`](@extref Oceananigans.AbstractOperations.KernelFunctionOperation)
that computes the advection-consistent precipitation flux at the bottom boundary.

The kernel is called as `kernel(i, j, k, grid, ρ, wᵗ, wˢᵉᵈ, c)` where `ρ` is the
reference density, `wᵗ` is the vertical transport velocity used by tracer advection,
`wˢᵉᵈ` is the sedimentation velocity, and `c` is the specific tracer (e.g., `qʳ`). Pass
these four fields as arguments to the `KernelFunctionOperation`:

    kernel = SurfacePrecipitationFluxKernel(model.advection.ρqʳ)
    op = KernelFunctionOperation{Center, Center, Nothing}(kernel, grid, ρ, wᵗ, wʳ, qʳ)
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
