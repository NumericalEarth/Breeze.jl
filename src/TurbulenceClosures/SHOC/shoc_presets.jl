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
default_component(::OriginalSHOC, ::Val{:mixing_length}) = SHOCMixingLength(Float64)
default_component(::OriginalSHOC, ::Val{:turbulence_time_scale}) = ColumnStabilityDamping(Float64)
default_component(::OriginalSHOC, ::Val{:tke_update}) = ForwardEulerTKE(Float64)
default_component(::OriginalSHOC, ::Val{:buoyancy_source}) = PDFBuoyancyFlux()
default_component(::OriginalSHOC, ::Val{:second_moments}) = DowngradientSecondMoments(Float64)
default_component(::OriginalSHOC, ::Val{:third_moment}) = CanutoThirdMoment(Float64)
default_component(::OriginalSHOC, ::Val{:assumed_pdf}) = DoubleGaussianPDF(Float64)
default_component(::OriginalSHOC, ::Val{:clipping}) = HardClipping()

# The published coefficients are built in double precision, whatever the default float type, and
# converted once by the constructor of the closure.

"""
$(TYPEDEF)

SHOC with every hard switch and clip of `shoc.F90` smoothed, so that a calibration loss is a
differentiable function of the coefficients:

| slot | component | what it replaces in [`OriginalSHOC`](@ref) |
|:-----|:----------|:-------------------------------------------|
| `clipping` | [`SmoothClipping`](@ref) of width 0.01 | every `max`, `min` and bound the state can reach |
| `assumed_pdf` | [`SmoothDoubleGaussianPDF`](@ref) | the calm and uniform switches, the all-or-nothing saturation, the ramp of the skewness |
| `third_moment` | [`CanutoThirdMoment`](@ref) with [`TanhBound`](@ref) | the reset of the third moment to 0.02 beyond its bound |
| `tke_update` | [`ImplicitSinkTKE`](@ref) | the forward Euler step that the floor of the turbulent kinetic energy catches |
| `mixing_length`, `turbulence_time_scale` | chosen by `core` | see below |

The second moments and the buoyancy source are those of `OriginalSHOC`
([`DowngradientSecondMoments`](@ref), [`PDFBuoyancyFlux`](@ref)).

Smoothing the switches is not enough for a smooth loss. With SHOC's own mixing length and
return-to-isotropy time scale the BOMEX column fluctuates from step to step after a few hours, and
a loss on its state after six hours is as rough in the coefficients with every switch smoothed
as without (neighbouring points of a scan differ by a tenth of the loss or more); with the
mixing length and the stability functions of [`TKEBasedTurbulenceClosure`](@ref) the same loss
is smooth and the skill against large eddy simulations is better at the published coefficients.
The keyword `core` therefore selects the eddy diffusivity core:

- `core = :tke_closure`, the default: [`GradientLimitedMixingLength`](@ref) and
  [`RichardsonStabilityFunctions`](@ref) with the coefficients published with
  `TKEBasedTurbulenceClosure`, and the shear production coefficient `ℂᴾ = 1` they go with. These
  two reused components are continuous but not smooth: the length is a lower envelope (a `min`
  over the column) and the stability functions have a kink at neutral stratification. No
  coefficient of this configuration is dimensional.
- `core = :shoc`: [`SHOCMixingLength`](@ref) and [`ColumnStabilityDamping`](@ref), SHOC's own
  mixing length and time scale with every bound through the smooth clipping policy. Every switch
  and clip is then smooth, and what distinguishes SHOC from an eddy diffusivity closure is
  kept whole.

Left hard in both: the discrete search for the boundary layer top (a diagnostic), the
cold-surface fallback of the diffusivities (below 182 K), and guards against round-off whose
bound the exact value already satisfies.

The published coefficients are a starting point for calibration, not a tuned set.

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure(DifferentiableSHOC())

# output
SimplifiedHigherOrderClosure{Float64}
├── mixing_length: GradientLimitedMixingLength{Float64}
├── turbulence_time_scale: RichardsonStabilityFunctions with RationalStabilityFunction{Float64}
├── tke_update: ImplicitSinkTKE{Float64}
├── buoyancy_source: PDFBuoyancyFlux
├── second_moments: DowngradientSecondMoments{Float64}
├── third_moment: CanutoThirdMoment{Float64} with TanhBound
├── assumed_pdf: SmoothDoubleGaussianPDF{Float64}
├── boundary_layer_depth: HoltslagBovilleDepth{Float64}
├── clipping: SmoothClipping{Float64}
├── surface fluxes: wthl_sfc = 0.0, wqw_sfc = 0.0, uw_sfc = 0.0, vw_sfc = 0.0, friction_velocity = 0.0
└── host_grid_spacing: 100000.0
```
"""
struct DifferentiableSHOC{C} <: AbstractSHOCPreset end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the preset [`DifferentiableSHOC`](@ref) with the eddy diffusivity `core` of the TKE
closure (`:tke_closure`) or SHOC's own mixing length and time scale (`:shoc`).
"""
function DifferentiableSHOC(; core::Symbol = :tke_closure)
    core ∈ (:tke_closure, :shoc) ||
        throw(ArgumentError("The core of DifferentiableSHOC is :tke_closure or :shoc, not :$core."))
    return DifferentiableSHOC{core}()
end

# The smoothed switches, whatever the core
default_component(::DifferentiableSHOC, ::Val{:buoyancy_source}) = PDFBuoyancyFlux()
default_component(::DifferentiableSHOC, ::Val{:second_moments}) = DowngradientSecondMoments(Float64)
default_component(::DifferentiableSHOC, ::Val{:third_moment}) = CanutoThirdMoment(Float64; bound = TanhBound())
default_component(::DifferentiableSHOC, ::Val{:assumed_pdf}) = SmoothDoubleGaussianPDF(Float64)
default_component(::DifferentiableSHOC, ::Val{:clipping}) = SmoothClipping(Float64; width = 1e-2)

# The mixing length and stability functions of the TKE closure, with its calibrated coefficients
# (the pair of the constructor of `TKEBasedTurbulenceClosure`, in double precision whatever the default float type)
default_component(::DifferentiableSHOC{:tke_closure}, ::Val{:mixing_length}) =
    GradientLimitedMixingLength(ℂˢ = 0.7877315198717979, ℂⁱ = 1.777871783491512)
default_component(::DifferentiableSHOC{:tke_closure}, ::Val{:turbulence_time_scale}) = RichardsonStabilityFunctions(Float64)
default_component(::DifferentiableSHOC{:tke_closure}, ::Val{:tke_update}) = ImplicitSinkTKE(Float64; ℂᴾ = 1)

# SHOC's own mixing length and time scale
default_component(::DifferentiableSHOC{:shoc}, ::Val{:mixing_length}) = SHOCMixingLength(Float64)
default_component(::DifferentiableSHOC{:shoc}, ::Val{:turbulence_time_scale}) = ColumnStabilityDamping(Float64)
default_component(::DifferentiableSHOC{:shoc}, ::Val{:tke_update}) = ImplicitSinkTKE(Float64)
