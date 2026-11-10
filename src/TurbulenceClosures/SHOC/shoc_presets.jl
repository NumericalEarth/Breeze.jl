#####
##### Presets: named choices of the components of SHOC
#####

"""
Supertype of the presets of [`SimplifiedHigherOrderClosure`](@ref): a preset names one component
for each slot of the closure, any of which a keyword argument of the constructor may replace.
"""
abstract type AbstractSHOCPreset end

"""
$(TYPEDEF)

SHOC as E3SM `shoc.F90` (pinned at E3SM `ba53243f`) computes it, with the published coefficients:
[`SHOCMixingLength`](@ref), [`ColumnStabilityDamping`](@ref), [`ForwardEulerTKE`](@ref),
[`PDFBuoyancyFlux`](@ref), [`DowngradientSecondMoments`](@ref), [`CanutoThirdMoment`](@ref) with
[`ResetBound`](@ref), [`DoubleGaussianPDF`](@ref) and [`HardClipping`](@ref). With these components
the column physics reproduces `shoc.F90` to round-off.
"""
struct OriginalSHOC <: AbstractSHOCPreset end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the component that `preset` puts in the slot `Val(name)` of
[`SimplifiedHigherOrderClosure`](@ref).
"""
default_component(::OriginalSHOC, ::Val{:mixing_length}) = SHOCMixingLength()
default_component(::OriginalSHOC, ::Val{:turbulence_time_scale}) = ColumnStabilityDamping()
default_component(::OriginalSHOC, ::Val{:tke_update}) = ForwardEulerTKE()
default_component(::OriginalSHOC, ::Val{:buoyancy_source}) = PDFBuoyancyFlux()
default_component(::OriginalSHOC, ::Val{:second_moments}) = DowngradientSecondMoments()
default_component(::OriginalSHOC, ::Val{:third_moment}) = CanutoThirdMoment()
default_component(::OriginalSHOC, ::Val{:assumed_pdf}) = DoubleGaussianPDF()
default_component(::OriginalSHOC, ::Val{:clipping}) = HardClipping()
