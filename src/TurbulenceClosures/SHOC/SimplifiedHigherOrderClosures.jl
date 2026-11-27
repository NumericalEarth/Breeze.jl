module SimplifiedHigherOrderClosures

export BlendedMixingLength, ColumnStabilityDamping,
       DowngradientSecondMoments,
       AlgebraicThirdMoment,
       RationalSaturatedStratificationNumber,
       PolynomialSaturatedSkewness,
       DoubleGaussianPDF, EqualWidthDoubleGaussianPDF,
       SmoothClipping,
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
                                 TurbulentRichardsonStabilityFunction,
                                 face_valueᶜᶜᶠ, grid_float_type, column_kernel_parameters
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS, hypotenuse
using Breeze.Thermodynamics: WarmPhaseEquilibrium, MixedPhaseEquilibrium, equilibrated_surface, saturation_vapor_pressure_log_slope,
                             saturation_vapor_pressure_liquid_fraction_log_slope,
                             PlanarLiquidSurface, PlanarIceSurface, dry_air_gas_constant, vapor_gas_constant,
                             liquid_latent_heat, ice_latent_heat, absolute_zero_latent_heat, MoistureMassFractions,
                             vapor_equivalent_heat_capacity, vapor_equivalent_gas_constant, mixture_heat_capacity,
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
include("shoc_parameters.jl")

end # module SimplifiedHigherOrderClosures
