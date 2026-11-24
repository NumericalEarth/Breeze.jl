"""
    SimplifiedHigherOrderClosures

The relations of the Simplified Higher-Order Closure (SHOC) of Bogenschutz and Krueger (2013),
re-expressed from E3SM's `shoc.F90` (pinned at E3SM `ba53243f`) as components of
[`TKEBasedTurbulenceClosure`](@ref): the mixing length ([`BlendedMixingLength`](@ref)), the
return-to-isotropy time scale and diffusivities as stability functions ([`ColumnStabilityDamping`](@ref)),
and, for the subgrid distribution [`DoubleGaussian`](@ref), the second moments, the third moment and its
limiters and the assumed double-Gaussian PDF, which condenses with the thermodynamics of the model on the
phase of the condensate of its microphysics. [`SimplifiedHigherOrderClosure`](@ref) is the configuration of
the TKE closure that assembles them with the coefficients of `shoc.F90`.
"""
module SimplifiedHigherOrderClosures

export BlendedMixingLength, ColumnStabilityDamping,
       DowngradientSecondMoments,
       AlgebraicThirdMoment,
       UnboundedStratificationNumber, RationalSaturatedStratificationNumber,
       ResetAboveThresholdSkewness, PolynomialSaturatedSkewness, TanhSaturatedSkewness, ClampedSkewness,
       DoubleGaussianPDF, SmoothDoubleGaussianPDF, EqualWidthDoubleGaussianPDF,
       HardClipping, SmoothClipping,
       closure_parameters, with_parameters

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using KernelAbstractions: @kernel, @index
using SpecialFunctions: erf

using Oceananigans: Oceananigans
using Oceananigans.Fields: Field
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.TurbulenceClosures: getclosure, height_above_bottomᶜᶜᶠ, height_above_bottomᶜᶜᶜ
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: turbulent_velocityᶜᶜᶜ, ℑbzᵃᵃᶜ
using Oceananigans.Utils: launch!, prettysummary

using Breeze.AtmosphereModels: AtmosphereModels
using Breeze.AtmosphereModels.Diagnostics: condensate_phase_equilibrium
using Breeze.TurbulenceClosures: TurbulenceClosures, TKEBasedTurbulenceClosure, AbstractMixingLength,
                                 HardMinimum, HarmonicBlend, TurbulentRichardsonStabilityFunction,
                                 face_valueᶜᶜᶠ, grid_float_type, column_kernel_parameters
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS
using Breeze.Thermodynamics: WarmPhaseEquilibrium, MixedPhaseEquilibrium, equilibrated_surface, saturation_vapor_pressure,
                             PlanarLiquidSurface, PlanarIceSurface, dry_air_gas_constant, vapor_gas_constant,
                             liquid_latent_heat, ice_latent_heat, MoistureMassFractions, vapor_equivalent_heat_capacity,
                             exner_exponent, saturation_specific_humidity, mixture_gas_constant

include("subgrid_distribution_components.jl")
include("clipping.jl")
include("column_utilities.jl")
include("mixing_lengths.jl")
include("turbulence_time_scales.jl")
include("second_moments.jl")
include("third_moments.jl")
include("pdf_saturation.jl")
include("assumed_pdfs.jl")
include("smooth_assumed_pdf.jl")
include("shoc_parameters.jl")

end # module SimplifiedHigherOrderClosures
