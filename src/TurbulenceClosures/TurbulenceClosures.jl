module TurbulenceClosures

export TKEBasedTurbulenceClosure,
       TurbulenceClosureStatistics,
       AbstractMixingLength,
       LocalMinimumMixingLength,
       IntegralMixingLength,
       GradientLimitedMixingLength,
       ConstantStabilityFunctions,
       RationalStabilityFunction,
       TurbulentRichardsonStabilityFunction,
       DryStaticStability,
       MoistStaticStability,
       SaturationWeightedStaticStability,
       GridMeanShear,
       SimplifiedHigherOrderClosure, SHOC,
       BlendedMixingLength,
       ColumnStabilityDamping,
       DowngradientSecondMoments,
       AlgebraicThirdMoment, TKEBudgetThirdMoment,
       HeatingBudgetFlux, heating_budget_flux_bounds,
       RationalSaturatedStratificationNumber,
       PolynomialSaturatedSkewness,
       DoubleGaussianPDF, EqualWidthDoubleGaussianPDF,
       SmoothClipping,
       closure_parameters, with_parameters,
       per_column_closure, set_per_column_coefficients!, PerColumnCoefficient,
       DoubleGaussian, ScalarVarianceTransport, CondensateFluxFraction, LatentStratification, MeanLiquidStratification,
       StratificationLimitedMomentTimeScale, DissipationTimeScale, ReturnToIsotropyTimeScale,
       VarianceStabilityFunctions

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: BoundaryConditions, FieldBoundaryConditions, fill_halo_regions!, compute_x_bcs!, compute_y_bcs!,
                                       update_boundary_conditions!
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.Fields: Field, CenterField, XFaceField, YFaceField, ZFaceField, ZeroField
using Oceananigans.Grids: AbstractGrid, AbstractUnderlyingGrid, Bounded, Center, Face, Flat, inactive_cell

using Oceananigans.Operators:
    # Face-centered difference operators with area metrics
    δxᶠᵃᵃ, δxᶜᵃᵃ, δyᵃᶜᵃ, δyᵃᶠᵃ, δzᵃᵃᶜ, δzᵃᵃᶠ,
    # Cell volumes (inverse)
    V⁻¹ᶠᶜᶜ, V⁻¹ᶜᶠᶜ, V⁻¹ᶜᶜᶠ, V⁻¹ᶜᶜᶜ,
    # Face areas for q-located fields
    Ax_qᶜᶜᶜ, Ax_qᶠᶠᶜ, Ax_qᶠᶜᶠ,
    Ay_qᶠᶠᶜ, Ay_qᶜᶜᶜ, Ay_qᶜᶠᶠ,
    Az_qᶠᶜᶠ, Az_qᶜᶠᶠ, Az_qᶜᶜᶜ,
    Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ, Az_qᶜᶜᶠ,
    # Interpolator functions used for ρᵣ at faces
    ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶠ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ,
    ℑxyᶠᶠᵃ, ℑxzᶠᵃᶠ, ℑyzᵃᶠᶠ, ℑxzᶠᵃᶠ,
    # Vertical derivative at faces, for the static stability
    ∂zᶜᶜᶠ,
    # Cell spacings
    Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ

using Oceananigans.TurbulenceClosures:
    AbstractTurbulenceClosure,
    AbstractScalarDiffusivity,
    VerticalFormulation,
    VerticallyImplicitTimeDiscretization,
    ExplicitTimeDiscretization,
    getclosure,
    _viscous_flux_ux, _viscous_flux_uy, _viscous_flux_uz,
    _viscous_flux_vx, _viscous_flux_vy, _viscous_flux_vz,
    _viscous_flux_wx, _viscous_flux_wy, _viscous_flux_wz,
    _diffusive_flux_x, _diffusive_flux_y, _diffusive_flux_z

using Oceananigans.TimeSteppers: time_discretization
using Oceananigans.Utils: Utils, KernelParameters, launch!, prettysummary

using ..AtmosphereModels: AtmosphereModels
using ..BoundaryConditions: BoundaryConditions as AtmosphereBoundaryConditions, DefaultGustiness, TKEGustiness

using Breeze.AtmosphereModels: grid_moisture_fractions, dynamics_pressure, standard_pressure, total_density, thermodynamic_state_density,
                              precipitating_mass_fraction, standard_score
using Breeze.AtmosphereModels.Diagnostics: microphysics_phase_equilibrium, condensate_phase_equilibrium
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics, SubgridCondensation
using Breeze.Thermodynamics: dry_air_gas_constant, vapor_gas_constant, saturation_specific_humidity,
                             equilibrated_surface, absolute_zero_latent_heat, specific_heat_difference,
                             vapor_equivalent_heat_capacity, vapor_equivalent_gas_constant, MoistureMassFractions
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS, hypotenuse
using Breeze.Thermodynamics: WarmPhaseEquilibrium, MixedPhaseEquilibrium, saturation_vapor_pressure_log_slope,
                             saturation_vapor_pressure_liquid_fraction_log_slope, PlanarLiquidSurface, PlanarIceSurface,
                             liquid_latent_heat, ice_latent_heat, mixture_heat_capacity, exner_exponent, mixture_gas_constant
using SpecialFunctions: erf

#####
##### Fallbacks for closure = nothing
#####

@inline AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)
@inline AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)
@inline AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)
@inline AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid, ρ, ::Nothing, args...) = zero(grid)

#####
##### Density-weighted fluxes for tuples of closures
#####

@inline sum_closure_fluxes(i, j, k, grid, flux, ρ, ::Tuple{}, ::Tuple{}, args...) = zero(grid)

@inline function sum_closure_fluxes(flux, i, j, k, grid, ρ,
                                    closures::Tuple{C, Vararg}, closure_fields::Tuple{F, Vararg},
                                    args...) where {C, F}
    return flux(i, j, k, grid, ρ, first(closures), first(closure_fields), args...) +
           sum_closure_fluxes(flux, i, j, k, grid, ρ, Base.tail(closures), Base.tail(closure_fields), args...)
end

@inline AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid, ρ, closures::Tuple, closure_fields::Tuple,
                                  id, c, clock, model_fields, buoyancy) =
    sum_closure_fluxes(AtmosphereModels.∇_dot_Jᶜ, i, j, k, grid, ρ, closures, closure_fields,
                       id, c, clock, model_fields, buoyancy)

@inline AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρ, closures::Tuple, closure_fields::Tuple,
                                clock, model_fields, buoyancy) =
    sum_closure_fluxes(AtmosphereModels.∂ⱼ_𝒯₁ⱼ, i, j, k, grid, ρ, closures, closure_fields,
                       clock, model_fields, buoyancy)

@inline AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρ, closures::Tuple, closure_fields::Tuple,
                                clock, model_fields, buoyancy) =
    sum_closure_fluxes(AtmosphereModels.∂ⱼ_𝒯₂ⱼ, i, j, k, grid, ρ, closures, closure_fields,
                       clock, model_fields, buoyancy)

@inline AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρ, closures::Tuple, closure_fields::Tuple,
                                clock, model_fields, buoyancy) =
    sum_closure_fluxes(AtmosphereModels.∂ⱼ_𝒯₃ⱼ, i, j, k, grid, ρ, closures, closure_fields,
                       clock, model_fields, buoyancy)

#####
##### Scalar (tracer) dynamic fluxes: J = ρᵣ τ
#####

# Face flux wrappers that call Oceananigans' kinematic diffusive fluxes and
# multiply by ρᵣ at the appropriate face.
# Note: args must include (disc, closure, closure_fields, id, c, clock, model_fields, buoyancy)
# where id is the tracer index (Val(n)) and c is the tracer field.

@inline Jᶜx(i, j, k, grid, ρ, args...) = ℑxᶠᵃᵃ(i, j, k, grid, ρ) * _diffusive_flux_x(i, j, k, grid, args...)
@inline Jᶜy(i, j, k, grid, ρ, args...) = ℑyᵃᶠᵃ(i, j, k, grid, ρ) * _diffusive_flux_y(i, j, k, grid, args...)
@inline Jᶜz(i, j, k, grid, ρ, args...) = ℑzᵃᵃᶠ(i, j, k, grid, ρ) * _diffusive_flux_z(i, j, k, grid, args...)

@inline function AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, id, c, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶜ, Jᶜx, ρᵣ, disc, closure, closure_fields, id, c, clock, model_fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶜ, Jᶜy, ρᵣ, disc, closure, closure_fields, id, c, clock, model_fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, Jᶜz, ρᵣ, disc, closure, closure_fields, id, c, clock, model_fields, buoyancy))
end

#####
##### Momentum dynamic stresses: 𝒯 = ρᵣ τ
#####

# Face stress wrappers for u-momentum
@inline 𝒯_ux(i, j, k, grid, ρ, args...) = @inbounds ρ[i, j, k]     * _viscous_flux_ux(i, j, k, grid, args...)
@inline 𝒯_uy(i, j, k, grid, ρ, args...) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * _viscous_flux_uy(i, j, k, grid, args...)
@inline 𝒯_uz(i, j, k, grid, ρ, args...) = ℑxzᶠᵃᶠ(i, j, k, grid, ρ) * _viscous_flux_uz(i, j, k, grid, args...)

@inline 𝒯_vx(i, j, k, grid, ρ, args...) = ℑxyᶠᶠᵃ(i, j, k, grid, ρ) * _viscous_flux_vx(i, j, k, grid, args...)
@inline 𝒯_vy(i, j, k, grid, ρ, args...) = @inbounds ρ[i, j, k]     * _viscous_flux_vy(i, j, k, grid, args...)
@inline 𝒯_vz(i, j, k, grid, ρ, args...) = ℑyzᵃᶠᶠ(i, j, k, grid, ρ) * _viscous_flux_vz(i, j, k, grid, args...)

@inline 𝒯_wx(i, j, k, grid, ρ, args...) = ℑxzᶠᵃᶠ(i, j, k, grid, ρ) * _viscous_flux_wx(i, j, k, grid, args...)
@inline 𝒯_wy(i, j, k, grid, ρ, args...) = ℑyzᵃᶠᶠ(i, j, k, grid, ρ) * _viscous_flux_wy(i, j, k, grid, args...)
@inline 𝒯_wz(i, j, k, grid, ρ, args...) = @inbounds ρ[i, j, k]     * _viscous_flux_wz(i, j, k, grid, args...)

@inline function AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶠᶜᶜ(i, j, k, grid) * (
          δxᶠᵃᵃ(i, j, k, grid, Ax_qᶜᶜᶜ, 𝒯_ux, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶠᶠᶜ, 𝒯_uy, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, 𝒯_uz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
end

@inline function AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶠᶜ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶠᶜ, 𝒯_vx, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δyᵃᶠᵃ(i, j, k, grid, Ay_qᶜᶜᶜ, 𝒯_vy, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, 𝒯_vz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
end

# Face stress wrappers for w-momentum

@inline function AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid, ρᵣ, closure::AbstractTurbulenceClosure, closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    return V⁻¹ᶜᶜᶠ(i, j, k, grid) * (
          δxᶜᵃᵃ(i, j, k, grid, Ax_qᶠᶜᶠ, 𝒯_wx, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δyᵃᶜᵃ(i, j, k, grid, Ay_qᶜᶠᶠ, 𝒯_wy, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy)
        + δzᵃᵃᶠ(i, j, k, grid, Az_qᶜᶜᶜ, 𝒯_wz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
end

#####
##### Vertically implicit vertical diffusivities: the flux divergences without their identically-zero terms
#####
##### For a scalar diffusivity of `VerticalFormulation` stepped with a `VerticallyImplicitTimeDiscretization` (e.g.
##### `TKEBasedTurbulenceClosure`), every horizontal viscous and diffusive flux is Oceananigans' literal `zero(grid)`,
##### and on a vertically bounded grid the vertical flux is the literal zero of the implicit part on every interior
##### face; only the bottom and top faces carry an explicit flux. The generic operators above still multiply each
##### zero by an interpolated density and an area, difference and sum them. The methods below form the vertical
##### difference alone and select it on the bottom and top levels with `ifelse` on the level index (the selection
##### is by value, as the kernels require; the horizontal terms are not formed).
#####
##### With densities and areas finite and positive, each dropped term is exactly +0: A·(ρ·(+0)) = +0, δ(+0, +0) = +0.
##### So the generic bracket `(δx + δy) + δz` is `(+0) + δz`, and on interior levels `δz = +0` too, so the operator is
##### V⁻¹ · (+0) = +0. The methods below return exactly that. On the bottom and top levels the literal +0 is kept,
##### because `(+0) + δz` maps a −0 to +0 as the generic sum does. The vertical flux of w is the implicit zero on
##### every face, so the w operator is +0 at every level.
#####

const VerticallyImplicitVerticalDiffusivity = AbstractScalarDiffusivity{<:VerticallyImplicitTimeDiscretization, <:VerticalFormulation}

# Non-immersed grids bounded in z (an immersed grid keeps the generic operators and their conditional fluxes).
const VerticallyBoundedUnderlyingGrid = AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Bounded}

# The levels whose cell has a bottom or top face with an explicit vertical flux.
@inline explicit_vertical_flux_level(k, grid) = (k == 1) | (k == size(grid, 3))

@inline function AtmosphereModels.∂ⱼ_𝒯₁ⱼ(i, j, k, grid::VerticallyBoundedUnderlyingGrid, ρᵣ,
                                          closure::VerticallyImplicitVerticalDiffusivity,
                                          closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    divergence = V⁻¹ᶠᶜᶜ(i, j, k, grid) *
                 (zero(grid) + δzᵃᵃᶜ(i, j, k, grid, Az_qᶠᶜᶠ, 𝒯_uz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
    return ifelse(explicit_vertical_flux_level(k, grid), divergence, zero(grid))
end

@inline function AtmosphereModels.∂ⱼ_𝒯₂ⱼ(i, j, k, grid::VerticallyBoundedUnderlyingGrid, ρᵣ,
                                          closure::VerticallyImplicitVerticalDiffusivity,
                                          closure_fields, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    divergence = V⁻¹ᶜᶠᶜ(i, j, k, grid) *
                 (zero(grid) + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶠᶠ, 𝒯_vz, ρᵣ, disc, closure, closure_fields, clock, model_fields, buoyancy))
    return ifelse(explicit_vertical_flux_level(k, grid), divergence, zero(grid))
end

@inline AtmosphereModels.∂ⱼ_𝒯₃ⱼ(i, j, k, grid::VerticallyBoundedUnderlyingGrid, ρᵣ,
                                 closure::VerticallyImplicitVerticalDiffusivity,
                                 closure_fields, clock, model_fields, buoyancy) = zero(grid)

@inline function AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid::VerticallyBoundedUnderlyingGrid, ρᵣ,
                                           closure::VerticallyImplicitVerticalDiffusivity,
                                           closure_fields, id, c, clock, model_fields, buoyancy)
    disc = time_discretization(closure)
    divergence = V⁻¹ᶜᶜᶜ(i, j, k, grid) *
                 (zero(grid) + δzᵃᵃᶜ(i, j, k, grid, Az_qᶜᶜᶠ, Jᶜz, ρᵣ, disc, closure, closure_fields, id, c, clock, model_fields, buoyancy))
    return ifelse(explicit_vertical_flux_level(k, grid), divergence, zero(grid))
end

# The floating-point type of the grid's numbers inside a kernel. A grid whose eltype is a traced number type (a
# `ReactantState` grid with stretched z, adapted into a kernel) has no `floatmin` or `eps`, but its zero is the plain
# float; for any other grid this is `eltype(grid)`.
@inline grid_float_type(grid) = typeof(zero(eltype(grid)))

include("turbulence_closure_statistics.jl")
include("static_stability.jl")
include("tke_based_turbulence_closure.jl")
include("per_column_coefficients.jl")
include("shear.jl")
include("richardson_number_stability_functions.jl")
include("turbulent_richardson_stability_functions.jl")

include("closure_parameters.jl")
include("clipping.jl")
include("column_utilities.jl")
include("SHOC/mixing_lengths.jl")
include("SHOC/turbulence_time_scales.jl")
include("second_moments.jl")
include("SHOC/third_moments.jl")
include("third_moment_limiters.jl")
include("pdf_saturation.jl")
include("SHOC/assumed_pdfs.jl")

include("subgrid_distribution.jl")
include("scalar_variance_transport.jl")
include("variance_stability_functions.jl")
include("applied_flux_energetics.jl")
include("tke_budget_third_moment.jl")
include("heating_budget_flux.jl")
include("SHOC/simplified_higher_order_closure.jl")

end # module TurbulenceClosures
