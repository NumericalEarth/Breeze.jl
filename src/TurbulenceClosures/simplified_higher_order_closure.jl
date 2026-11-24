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
double Gaussian of `shoc.F90` ([`DoubleGaussianPDF`](@ref), its bounds imposed by `HardClipping` as in `shoc.F90`). Two
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
                       kw...)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`SimplifiedHigherOrderClosure`](@ref): a [`TKEBasedTurbulenceClosure`](@ref) with the components of
E3SM `shoc.F90` and its published coefficients.

- `mixing_length`: [`BlendedMixingLength`](@ref);
- `stability_functions`: [`ColumnStabilityDamping`](@ref), SHOC's return-to-isotropy time scale damped by the
  stability of the column, which gives `Kᶜ = ℂᶜ τ e`, `Kᵘ = ℂᵘ τ e` and the dissipation `ε = 𝕊ᴰ e^{3/2} / ℓ`;
- `subgrid_distribution`: SHOC's double Gaussian ([`shoc_distribution`](@ref Breeze.TurbulenceClosures.shoc_distribution));
- `static_stability`: `SaturationWeightedStaticStability(CondensateFluxFraction(stratification = MeanLiquidStratification()))`:
  the mixing length and the stability functions read the stratification of the grid-mean state with the mean liquid of
  the distribution as its condensate (the Brunt–Väisälä frequency of the virtual potential temperature of `shoc.F90`),
  and the buoyancy production of the TKE carries the latent part of the subgrid condensation that the distribution
  implies (the counterpart of the buoyancy flux of the assumed PDF of `shoc.F90`). The treatment of
  [`TKEBasedTurbulenceClosure`](@ref), `CondensateFluxFraction()` ([`LatentStratification`](@ref)), puts the latent part
  in the static stability as well.

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
- the return-to-isotropy time scale is damped in unstable air too, `τ₀ / (1 + (ℂᶜ² / 4) |N²| τ₀²)`, where `shoc.F90`
  damps it in stable air only, so that the buoyant growth rate of the TKE, `ℂᶜ τ |N²|`, stays below `|N|` for any
  dissipation coefficient (see [`ColumnStabilityDamping`](@ref));
- the cold-surface fallback of the diffusivities and the boundary layer depth (a diagnostic of `shoc.F90`
  that only that fallback uses) are not included;
- the constants are the model's: the assumed PDF condenses with the model's thermodynamic constants (gas
  constants, heat capacities, temperature-dependent latent heats, saturation vapor pressure), its standard
  pressure and the potential temperature of each level in place of `shoc.F90`'s fixed constants, `basepres`
  and `basetemp`, and the third moment takes the model's gravitational acceleration in place of `ggr`;
- the assumed PDF condenses on the phase of the condensate in the water it distributes
  (`condensate_phase_equilibrium`): liquid, as `shoc.F90`, for warm microphysics and for the cloud liquid of P3,
  and liquid turning to ice between the freezing and homogeneous freezing temperatures, with the latent heat of
  the condensate, for mixed-phase saturation adjustment. There is no phase equilibrium of the closure's own.

Five of these differences are deliberate and change the solution measurably; on the same column state (BOMEX and
RICO states handed to the port and to `shoc.F90`'s routines) the first four account for the differences of the port:

- **Saturation.** The assumed PDF condenses at the model's saturation specific humidity
  `qᵛ⁺ = pᵛ⁺ / (ρ Rᵛ T)`, within about 0.1 % of the exact `ε e / (p - (1 - ε) e)`. `shoc.F90` uses
  `0.622 e / (p - e)`, about 1.3 % (≈ 3 × 10⁻⁴ kg kg⁻¹ in the trade-cumulus layer) too large, so its PDF condenses less:
  on one BOMEX state the port's PDF with `shoc.F90`'s saturation reproduces `shoc.F90`'s cloud to round-off, and
  with the model's saturation it has about twice the cloud fraction and liquid water path.
- **The asymptotic length.** `l∞` weights the height by `√e` with `e` floored at `minimum_tke`; `shoc.F90` floors
  `e` at `mintke = 4 × 10⁻⁴` m² s⁻², which gives the free troposphere a weight `√e = 0.02` m s⁻¹. The port's `l∞`
  is 20–60 % smaller (and equal to that of `shoc.F90` with the same floor), its mixing length 10–30 % smaller,
  and the time scale, the diffusivities and the variances (`∝ τ² e`) follow.
- **The bound of the third moment.** It saturates along a polynomial toward `1.2 √2 (w′²)^{3/2}`
  ([`PolynomialSaturatedSkewness`](@ref)); `shoc.F90` resets a third moment beyond its bound to `0.02` m³ s⁻³, which
  at an inversion is a large positive skewness.
- **The surface source of TKE.** The TKE receives `ℂᴾ` times the work of the surface stress on the lowest level,
  `ℂᴾ u★² |u₁|` per unit area, about `0.1 |u₁| / u★` (2–3 in BOMEX and RICO) times the `u★³` that `shoc.F90` adds.
- **The time scale in unstable air.** `shoc.F90` sets `lambda = 0` where `N² ≤ 0`, so the return-to-isotropy time
  scale there is the undamped `τ₀ = 2 ℓ / (𝕊ᴰ √e)`; the port damps it with `ℂᶜ² / 4` ([`ColumnStabilityDamping`](@ref)),
  which keeps the buoyant growth rate `ℂᶜ τ |N²|` of the TKE below `|N|`. With the undamped time scale a small
  dissipation coefficient (`𝕊ᴰ ∝ ℂᴰ₁³ / ℂᴰ₂⁴`, so the ratio `ℂᴰ₂ / ℂᴰ₁` enters to the fourth power) makes the TKE,
  the diffusivities and the variances run away: at `ℂᴰ₂ / ℂᴰ₁ = 2.25` the port without the damping fails on RICO
  within 20 minutes while `shoc.F90` runs, and 30–50 % of calibration ensembles around the default ratio went
  non-finite. At the published coefficients it moves the port's RMS differences from `shoc.F90` by 5–25 %.

```jldoctest
using Breeze

closure = SimplifiedHigherOrderClosure()

# output
SimplifiedHigherOrderClosure{VerticallyImplicitTimeDiscretization}
├── mixing_length: BlendedMixingLength{Float64} (ℂˡ₁ = 0.5, ℂˡ₂ = 2.8284, ℂˡ₃ = 0.01, ℂˡ₄ = 0.1, ℂˡ₅ = 400.0)
├── stability_functions: ColumnStabilityDamping{Float64}
│   ├── ℂᵘ: 0.1, ℂᶜ: 0.1
│   ├── ℂᵝ₁: 0.001, ℂᵝ₂: 0.08, ℂᵝ₃: 2.65, ℂᵝ₄: 0.02
│   ├── 𝕊ᴰ: 1.97531
│   └── ℂᴾ: 0.1
├── static_stability: SaturationWeightedStaticStability(CondensateFluxFraction(DryStaticStability, MeanLiquidStratification()))
├── subgrid_distribution: DoubleGaussian
│   ├── time_scale: ReturnToIsotropyTimeScale()
│   ├── second_moments: DowngradientSecondMoments{Float64}
│   ├── third_moment: AlgebraicThirdMoment{Float64}
│   ├── stratification_number_bound: RationalSaturatedStratificationNumber{Float64}
│   ├── skewness: PolynomialSaturatedSkewness{Float64}
│   └── assumed_pdf: DoubleGaussianPDF{Float64} with HardClipping
├── minimum_tke: 1.0e-6
└── solver: SinglePass()
```
"""
function SimplifiedHigherOrderClosure(time_discretization = VerticallyImplicitTimeDiscretization(),
                                      FT = Oceananigans.defaults.FloatType;
                                      mixing_length = BlendedMixingLength(Float64),
                                      stability_functions = ColumnStabilityDamping(Float64),
                                      static_stability = SaturationWeightedStaticStability(CondensateFluxFraction(stratification = MeanLiquidStratification())),
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

static_stability_summary(static_stability) = summary(static_stability)
static_stability_summary(ss::SaturationWeightedStaticStability) = string(summary(ss), "(", sprint(show, ss.fraction), ")")

function Base.show(io::IO, closure::SimplifiedHigherOrderClosure)
    d = closure.subgrid_distribution
    print(io, summary(closure), '\n',
              "├── mixing_length: ", closure.mixing_length, '\n',
              "├── stability_functions: ", summary(closure.stability_functions), '\n')
    show_stability_function_lines(io, closure.stability_functions, "│   ")
    print(io, '\n',
              "├── static_stability: ", static_stability_summary(closure.static_stability), '\n',
              "├── subgrid_distribution: DoubleGaussian", '\n',
              "│   ├── time_scale: ", summary(d.time_scale), '\n',
              "│   ├── second_moments: ", summary(d.second_moments), '\n',
              "│   ├── third_moment: ", summary(d.third_moment), '\n',
              "│   ├── stratification_number_bound: ", summary(d.stratification_number_bound), '\n',
              "│   ├── skewness: ", summary(d.skewness), '\n',
              "│   ", isnothing(d.liquid_flux) ? "└" : "├", "── assumed_pdf: ", assumed_pdf_summary(d.assumed_pdf), '\n')
    isnothing(d.liquid_flux) || print(io, "│   └── liquid_flux: ", summary(d.liquid_flux), '\n')
    isnothing(closure.heating_budget_flux) || print(io, "├── heating_budget_flux: ", summary(closure.heating_budget_flux), '\n')
    isnothing(closure.total_water_budget_flux) || print(io, "├── total_water_budget_flux: ", summary(closure.total_water_budget_flux), '\n')
    print(io, "├── minimum_tke: ", prettysummary(closure.minimum_tke), '\n',
              "└── solver: ", summary(closure.solver))
end

# An assumed PDF with the policy that imposes its bounds, or one without bounds
assumed_pdf_summary(pdf) = hasproperty(pdf, :clipping) ? string(summary(pdf), " with ", summary(pdf.clipping)) : summary(pdf)
