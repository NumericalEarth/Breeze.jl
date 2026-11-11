#####
##### The components each keyword argument of `SimplifiedHigherOrderClosure` takes, and named sets of them
#####

# The components of each slot. The constructor checks against it, so that a component in a slot
# it does not belong to fails there and not inside a kernel.
const SLOT_COMPONENTS = (mixing_length = (BlendedMixingLength, GradientLimitedMixingLength),
                         turbulence_time_scale = (ColumnStabilityDamping, RichardsonStabilityFunctions),
                         tke_update = (ForwardEulerTKE, ImplicitSinkTKE),
                         buoyancy_source = (PDFBuoyancyFlux, DowngradientBuoyancyFlux),
                         second_moments = (DowngradientSecondMoments,),
                         third_moment = (AlgebraicThirdMoment,),
                         stratification_number_bound = (UnboundedStratificationNumber, RationalSaturatedStratificationNumber),
                         skewness = (ResetAboveThresholdSkewness, PolynomialSaturatedSkewness, ClampedSkewness),
                         assumed_pdf = (DoubleGaussianPDF, SmoothDoubleGaussianPDF),
                         boundary_layer_depth = (BulkRichardsonBoundaryLayerDepth,),
                         clipping = (HardClipping, SmoothClipping),
                         constants = (SHOCConstants,))

function check_component_slots(components::NamedTuple)
    for slot in keys(components)
        component = components[slot]
        allowed = SLOT_COMPONENTS[slot]
        any(Component -> component isa Component, allowed) ||
            throw(ArgumentError("The keyword argument `$slot` of SimplifiedHigherOrderClosure takes " *
                                join(allowed, " or ") * ", not $(summary(component))."))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the components of a [`SimplifiedHigherOrderClosure`](@ref) without the hard switches and
clips of E3SM `shoc.F90`, so that a calibration loss varies smoothly with the coefficients. The
result is a `NamedTuple` of keyword arguments of the constructor, with scalars of type `FT`, to
be splatted into it; a keyword argument given after the splat replaces one
component of the set, like any other keyword argument of the constructor.

| keyword | component | what it replaces of `shoc.F90` |
|:--------|:----------|:-------------------------------|
| `clipping` | [`SmoothClipping`](@ref) of width 0.01 | every `max`, `min` and bound the state can reach |
| `assumed_pdf` | [`SmoothDoubleGaussianPDF`](@ref) | the calm and uniform switches, the all-or-nothing saturation, the ramp of the skewness |
| `stratification_number_bound` | [`RationalSaturatedStratificationNumber`](@ref) with `pole_fraction = 1/2` | the pole of the third moment at the stratification number `1 / (a₁ + a₃)` |
| `skewness` | [`PolynomialSaturatedSkewness`](@ref) | the reset of the third moment to 0.02 beyond its bound |
| `tke_update` | [`ImplicitSinkTKE`](@ref) with `ℂᴾ = 1` | the forward Euler step that the floor of the turbulent kinetic energy catches |
| `mixing_length` | [`GradientLimitedMixingLength`](@ref) with `ℂˢ = 0.788`, `ℂⁱ = 1.778` | the mixing length of `shoc.F90` |
| `turbulence_time_scale` | [`RichardsonStabilityFunctions`](@ref) of [`RationalStabilityFunction`](@ref) | the damping by the stability of the whole column |

The second moments, the buoyancy source and the boundary layer depth are not in the set: the
closure keeps their defaults.

The first five rows smooth the switches and the pole of SHOC. The last two replace SHOC's mixing length
and return-to-isotropy time scale by the mixing length and the stability functions that
[`TKEBasedTurbulenceClosure`](@ref) also uses, with the coefficients published with that
closure as a starting point and the shear production coefficient `ℂᴾ = 1` they go with. These
two components are continuous but not smooth (the length is a lower envelope, a `min` over the
column, and the stability functions have a kink at neutral stratification), and with them no
coefficient of the closure is dimensional. The closure holds its own copies of their
coefficients (`ℂˢ`, `ℂⁱ`, the `ℂᵘ`, `ℂᶜ`, `ℂᵉ`, `ℂᴰ` endpoints, `Ri⁻`, `Ri⁺`): all are listed by
[`closure_parameters`](@ref) and are to be calibrated within SHOC, independently of
`TKEBasedTurbulenceClosure`. (SHOC does not use the stability function of the diffusivity of
turbulent kinetic energy, so `ℂᵉ⁻`, `ℂᵉ⁰`, `ℂᵉ⁺` have no effect.)

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure(; smooth_shoc_components()...)

# output
SimplifiedHigherOrderClosure{Float64}
├── mixing_length: GradientLimitedMixingLength{Float64}
├── turbulence_time_scale: RichardsonStabilityFunctions with RationalStabilityFunction{Float64}
├── tke_update: ImplicitSinkTKE{Float64}
├── buoyancy_source: PDFBuoyancyFlux
├── second_moments: DowngradientSecondMoments{Float64}
├── third_moment: AlgebraicThirdMoment{Float64}
├── stratification_number_bound: RationalSaturatedStratificationNumber{Float64}
├── skewness: PolynomialSaturatedSkewness{Float64}
├── assumed_pdf: SmoothDoubleGaussianPDF{Float64}
├── boundary_layer_depth: BulkRichardsonBoundaryLayerDepth{Float64}
├── clipping: SmoothClipping{Float64}
├── surface fluxes: wthl_sfc = 0.0, wqw_sfc = 0.0, uw_sfc = 0.0, vw_sfc = 0.0, friction_velocity = 0.0
└── host_grid_spacing: 100000.0
```

To smooth the switches and keep SHOC's own mixing length and time scale, with the coefficients
of `shoc.F90` (every bound of the two then goes through the smooth clipping, so that every
switch and clip of the closure is smooth), name them after the splat:

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure(; smooth_shoc_components()...,
                                       mixing_length = BlendedMixingLength(),
                                       turbulence_time_scale = ColumnStabilityDamping(),
                                       tke_update = ImplicitSinkTKE())

# output
SimplifiedHigherOrderClosure{Float64}
├── mixing_length: BlendedMixingLength{Float64}
├── turbulence_time_scale: ColumnStabilityDamping{Float64}
├── tke_update: ImplicitSinkTKE{Float64}
├── buoyancy_source: PDFBuoyancyFlux
├── second_moments: DowngradientSecondMoments{Float64}
├── third_moment: AlgebraicThirdMoment{Float64}
├── stratification_number_bound: RationalSaturatedStratificationNumber{Float64}
├── skewness: PolynomialSaturatedSkewness{Float64}
├── assumed_pdf: SmoothDoubleGaussianPDF{Float64}
├── boundary_layer_depth: BulkRichardsonBoundaryLayerDepth{Float64}
├── clipping: SmoothClipping{Float64}
├── surface fluxes: wthl_sfc = 0.0, wqw_sfc = 0.0, uw_sfc = 0.0, vw_sfc = 0.0, friction_velocity = 0.0
└── host_grid_spacing: 100000.0
```

`tke_update = ImplicitSinkTKE()` restores the shear production coefficient `ℂᴾ = 0.1` of
`shoc.F90`, which goes with `ColumnStabilityDamping`.

Smoothing the switches is not by itself enough for a smooth loss over long windows: with SHOC's
own mixing length and time scale the BOMEX column fluctuates from step to step after a few
hours, and a loss on its state after six hours is as rough in the coefficients with every
switch smoothed as without, while with the two components of the set the same loss is smooth.

Left hard in either case: the discrete search for the boundary layer top (a diagnostic), the
cold-surface fallback of the diffusivities of `ColumnStabilityDamping` (below 182 K), and guards
against round-off whose bound the exact value already satisfies.

The coefficients are a starting point for calibration, not a tuned set.
"""
function smooth_shoc_components(FT::DataType = Oceananigans.defaults.FloatType)
    # Built in double precision, whatever the default float type, and converted once
    components = (mixing_length = GradientLimitedMixingLength(ℂˢ = 0.7877315198717979, ℂⁱ = 1.777871783491512),
                  turbulence_time_scale = RichardsonStabilityFunctions(RationalStabilityFunction(), Float64),
                  tke_update = ImplicitSinkTKE(Float64; ℂᴾ = 1),
                  stratification_number_bound = RationalSaturatedStratificationNumber(Float64),
                  skewness = PolynomialSaturatedSkewness(Float64),
                  assumed_pdf = SmoothDoubleGaussianPDF(Float64),
                  clipping = SmoothClipping(Float64; width = 1e-2))

    return map(component -> convert_eltype(FT, component), components)
end
