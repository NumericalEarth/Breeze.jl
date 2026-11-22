#####
##### `SimplifiedHigherOrderClosure`: SHOC as a configuration of `TKEBasedTurbulenceClosure`
#####
##### SHOC's relations are components of the TKE closure: its mixing length (`BlendedMixingLength`), its
##### return-to-isotropy time scale and eddy diffusivities as stability functions (`ColumnStabilityDamping`),
##### and its diagnosed second and third moments and assumed double-Gaussian PDF as the subgrid distribution
##### (`DoubleGaussian` with SHOC's components and the moment time scale of the stability functions). The TKE
##### is the closure's prognostic `ρe`, stepped with the model's time stepper: diffusion and sinks in the
##### vertically implicit solve of each stage, sources in the stage tendency, the surface fluxes from the
##### model's boundary conditions, and the energy conversions of the applied fluxes.
#####

"""
    SimplifiedHigherOrderClosure

The Simplified Higher-Order Closure of [Bogenschutz and Krueger (2013)](@cite Bogenschutz2013), as E3SM
implements it (`shoc.F90`, pinned at E3SM `ba53243f`), as a configuration of [`TKEBasedTurbulenceClosure`](@ref):
the TKE closure with SHOC's mixing length ([`BlendedMixingLength`](@ref)), SHOC's return-to-isotropy time scale
and diffusivities as stability functions ([`ColumnStabilityDamping`](@ref)) and a [`DoubleGaussian`](@ref)
subgrid distribution. A closure is a `SimplifiedHigherOrderClosure` when it has these three component types.
"""
const SimplifiedHigherOrderClosure{TD} = TKEBasedTurbulenceClosure{TD, <:BlendedMixingLength, <:ColumnStabilityDamping,
                                                                   <:Any, <:Any, <:Any, <:DoubleGaussian} where TD

"""
A shorter name for [`SimplifiedHigherOrderClosure`](@ref).
"""
const SHOC = SimplifiedHigherOrderClosure

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return the [`DoubleGaussian`](@ref) subgrid distribution of [`SimplifiedHigherOrderClosure`](@ref) with the
components and coefficients of E3SM `shoc.F90`: the moment time scale of the stability functions
([`ReturnToIsotropyTimeScale`](@ref)), [`DowngradientSecondMoments`](@ref), [`AlgebraicThirdMoment`](@ref) and the
double Gaussian of `shoc.F90` ([`DoubleGaussianPDF`](@ref)) saturating with the model's thermodynamics. Two
limiters replace switches of `shoc.F90` by polynomial and rational forms: the third moment saturates along a
polynomial toward its bound `1.2 √2 (w′²)^{3/2}` ([`PolynomialSaturatedSkewness`](@ref)) where `shoc.F90` resets it
to `0.02` m³ s⁻³, and the stratification number of its diagnosis is bounded below the first pole
([`RationalSaturatedStratificationNumber`](@ref)) where `shoc.F90` leaves it unbounded. Every keyword argument
replaces a component.
"""
shoc_distribution(FT::DataType = Oceananigans.defaults.FloatType; kw...) =
    DoubleGaussian(FT; time_scale = ReturnToIsotropyTimeScale(),
                       second_moments = DowngradientSecondMoments(FT),
                       third_moment = AlgebraicThirdMoment(FT),
                       stratification_number_bound = RationalSaturatedStratificationNumber(FT),
                       skewness = PolynomialSaturatedSkewness(FT),
                       assumed_pdf = DoubleGaussianPDF(FT),
                       saturation = ModelThermodynamicsSaturation(),
                       clipping = HardClipping(),
                       constants = SHOCConstants(FT),
                       kw...)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SimplifiedHigherOrderClosure`](@ref): a [`TKEBasedTurbulenceClosure`](@ref) with the components of
E3SM `shoc.F90` and its published coefficients.

- `mixing_length`: [`BlendedMixingLength`](@ref);
- `stability_functions`: [`ColumnStabilityDamping`](@ref), SHOC's return-to-isotropy time scale damped by the
  stability of the column, which gives `Kᶜ = ℂᶜ τ e`, `Kᵘ = ℂᵘ τ e` and the dissipation `ε = 𝕊ᴰ e^{3/2} / ℓ`;
- `subgrid_distribution`: SHOC's double Gaussian ([`shoc_distribution`](@ref Breeze.TurbulenceClosures.shoc_distribution));
- `static_stability`: `SaturationWeightedStaticStability(CondensateFluxFraction(latent_static_stability = false))`:
  the buoyancy production of the TKE is the conversion of the fluxes the closure applies with the latent part of
  the subgrid condensation that the distribution implies (the counterpart of the buoyancy flux of the assumed PDF
  of `shoc.F90`), while the mixing length and the stability functions read the stratification of a displacement
  without phase change (`DryStaticStability`, the counterpart of the Brunt–Väisälä frequency of `shoc.F90`);

The other keyword arguments are those of [`TKEBasedTurbulenceClosure`](@ref), and every component may be
replaced. Replacing the mixing length, the stability functions or the subgrid distribution by a component of
another type gives a `TKEBasedTurbulenceClosure` that is not a `SimplifiedHigherOrderClosure`.

How this differs from the operator-split `shoc.F90` step:

- the TKE, θˡⁱ, total water and momentum are stepped by the model's time stepper, with the vertical diffusion
  and the sinks of the TKE implicit in each stage;
- the shear and buoyancy production of the TKE are the energy the applied fluxes take from the mean state: the
  TKE receives the fraction `ℂᴾ = Ck_sh = 0.1` of the kinetic energy the momentum fluxes and the surface stress
  take from the mean flow and the rest heats the air (`shoc.F90` produces `Ck_sh Kᵘ S²` and discards the rest),
  and the surface stress acts through its work on the lowest level, where `shoc.F90` adds `u★³`; the dissipated
  energy heats the air;
- the surface fluxes are the model's boundary conditions;
- the bounds of `shoc.F90` on the TKE (`mintke = 4 × 10⁻⁴` m² s⁻², `maxtke`), the mixing length (`minlen`, `maxlen`) and the time
  scale (`maxiso`) are not imposed as clips: the TKE stays positive through its implicit sinks, and the upper
  bounds of the length and the time scale are sums of inverse squares;
- the cold-surface fallback of the diffusivities and the boundary layer depth (a diagnostic of `shoc.F90`
  that only that fallback uses) are not included.

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure()

# output
SHOW_OUTPUT
```
"""
function SimplifiedHigherOrderClosure(time_discretization = VerticallyImplicitTimeDiscretization(),
                                      FT = Oceananigans.defaults.FloatType;
                                      mixing_length = BlendedMixingLength(Float64),
                                      stability_functions = ColumnStabilityDamping(Float64),
                                      static_stability = SaturationWeightedStaticStability(CondensateFluxFraction(latent_static_stability = false)),
                                      subgrid_distribution = shoc_distribution(Float64),
                                      kw...)

    return TKEBasedTurbulenceClosure(time_discretization, FT; mixing_length, stability_functions, static_stability,
                                     subgrid_distribution, kw...)
end

SimplifiedHigherOrderClosure(FT::DataType; kw...) =
    SimplifiedHigherOrderClosure(VerticallyImplicitTimeDiscretization(), FT; kw...)

# The moment time scale of the stability functions needs stability functions that have one
validate_moment_time_scale(::ColumnStabilityDamping, ::ReturnToIsotropyTimeScale) = nothing
validate_moment_time_scale(stability_functions, ::ReturnToIsotropyTimeScale) =
    throw(ArgumentError("ReturnToIsotropyTimeScale takes the time scale of the moments from the stability functions, " *
                        "which $(summary(stability_functions)) do not have; use ColumnStabilityDamping or another time scale."))

function Base.summary(closure::SimplifiedHigherOrderClosure)
    TD = nameof(typeof(time_discretization(closure)))
    return string("SimplifiedHigherOrderClosure{$TD}")
end

function Base.show(io::IO, closure::SimplifiedHigherOrderClosure)
    d = closure.subgrid_distribution
    print(io, summary(closure), '\n',
              "├── mixing_length: ", closure.mixing_length, '\n',
              "├── stability_functions: ", summary(closure.stability_functions), '\n')
    show_stability_function_lines(io, closure.stability_functions, "│   ")
    print(io, '\n',
              "├── static_stability: ", summary(closure.static_stability), '\n',
              "├── subgrid_distribution: DoubleGaussian", '\n',
              "│   ├── time_scale: ", summary(d.time_scale), '\n',
              "│   ├── second_moments: ", summary(d.second_moments), '\n',
              "│   ├── third_moment: ", summary(d.third_moment), '\n',
              "│   ├── stratification_number_bound: ", summary(d.stratification_number_bound), '\n',
              "│   ├── skewness: ", summary(d.skewness), '\n',
              "│   ├── assumed_pdf: ", summary(d.assumed_pdf), '\n',
              "│   ├── saturation: ", summary(d.saturation), '\n',
              "│   └── clipping: ", summary(d.clipping), '\n',
              "├── minimum_tke: ", prettysummary(closure.minimum_tke), '\n',
              "└── solver: ", summary(closure.solver))
end
