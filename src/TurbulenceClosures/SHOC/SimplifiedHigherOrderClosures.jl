"""
    SimplifiedHigherOrderClosures

The Simplified Higher-Order Closure (SHOC) of Bogenschutz and Krueger (2013), re-expressed from
E3SM's `shoc.F90` (pinned at E3SM `ba53243f`) as allocation-free column functions that run one
thread per column on CPUs and GPUs, and wired into `AtmosphereModel` as
[`SimplifiedHigherOrderClosure`](@ref). The closure is assembled from swappable components, one
per relation of the scheme, each a keyword argument of the constructor; the defaults are those
of `shoc.F90`.
"""
module SimplifiedHigherOrderClosures

export SimplifiedHigherOrderClosure, SHOC,
       smooth_shoc_components,
       BlendedMixingLength,
       ColumnStabilityDamping, RichardsonStabilityFunctions, StratificationLimitedTimeScale,
       ForwardEulerTKE, ImplicitSinkTKE,
       PDFBuoyancyFlux, DowngradientBuoyancyFlux,
       DowngradientSecondMoments,
       AlgebraicThirdMoment,
       UnboundedStratificationNumber, RationalSaturatedStratificationNumber,
       ResetAboveThresholdSkewness, PolynomialSaturatedSkewness, ClampedSkewness,
       DoubleGaussianPDF, SmoothDoubleGaussianPDF, EqualWidthDoubleGaussianPDF,
       FixedConstantsLiquidSaturation, ModelThermodynamicsSaturation,
       HardClipping, SmoothClipping,
       BulkRichardsonBoundaryLayerDepth,
       SHOCConstants,
       closure_parameters, with_parameters

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using GPUArraysCore: @allowscalar
using KernelAbstractions: KernelAbstractions, @kernel, @index
using SpecialFunctions: erf

using Oceananigans: Oceananigans
using Oceananigans.Architectures: device
using Oceananigans.BoundaryConditions: BoundaryConditions, BoundaryCondition, Flux, Bottom,
                                       fill_halo_regions!, total_boundary_flux
using Oceananigans.Fields: Field, CenterField, ZFaceField
using Oceananigans.Models: boundary_condition_args
using Oceananigans.Grids: AbstractGrid, Center, Face, Flat, znode, peripheral_node
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.TimeSteppers: ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, getclosure
using Oceananigans.Utils: Utils, launch!, prettysummary

using Breeze.AtmosphereModels: AtmosphereModels, ClosureCloud, ClosureCloudStatistics
using Breeze.TurbulenceClosures: TurbulenceClosures, convert_eltype,
                                 GradientLimitedMixingLength, HardMinimum, HarmonicBlend, fill_mixing_length!, mixing_lengthᶜᶜᶜ,
                                 RationalStabilityFunction, richardson_number, momentum_stability_function,
                                 tracer_stability_function, dissipation_stability_function
using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS
using Breeze.Thermodynamics: WarmPhaseEquilibrium, MixedPhaseEquilibrium, equilibrated_surface, saturation_vapor_pressure,
                             PlanarLiquidSurface, PlanarIceSurface, dry_air_gas_constant, vapor_gas_constant,
                             liquid_latent_heat, ice_latent_heat, MoistureMassFractions, vapor_equivalent_heat_capacity,
                             exner_exponent

include("shoc_constants.jl")
include("clipping.jl")
include("column_utilities.jl")
include("boundary_layer_depth.jl")
include("mixing_lengths.jl")
include("turbulence_time_scales.jl")
include("implicit_diffusion.jl")
include("tke_updates.jl")
include("buoyancy_sources.jl")
include("second_moments.jl")
include("third_moments.jl")
include("pdf_saturation.jl")
include("assumed_pdfs.jl")
include("smooth_assumed_pdf.jl")
include("shoc_column.jl")
include("component_sets.jl")
include("simplified_higher_order_closure.jl")
include("shoc_parameters.jl")
include("shoc_cloud.jl")

end # module SimplifiedHigherOrderClosures
