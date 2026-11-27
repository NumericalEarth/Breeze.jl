#####
##### The third moment of vertical velocity from the vertical integral of the TKE budget
#####
##### A third moment of the `DoubleGaussian` subgrid distribution of `TKEBasedTurbulenceClosure`
##### (`third_moment = TKEBudgetThirdMoment(...)`), next to `AlgebraicThirdMoment`.
#####
##### Large-eddy simulations give w′³ ≈ r w′e′ with r ≈ 1.5, and the
##### flux of TKE is what the TKE budget leaves over: ∂z (ρ F) = ρ (P + B - u), with P + B the shear
##### and buoyancy production the TKE equation receives and u a sink. The model takes the sink as
##### u = c ω with a shape ω that vanishes with e (ω = e, one turnover rate, as `turbulent_shape` forms it; ω = ε, the closure's
##### dissipation) and an amplitude c fixed by the budget, so that F vanishes at the ground and at
##### the top of the column and wherever the turbulence does. No depth of a turbulent layer is
##### diagnosed.
#####
#####   * ℂʷᴸ = Inf: one amplitude for the column, c = Σ ρ Δz (P + B) / Σ ρ Δz ω.
#####   * finite ℂʷᴸ: c varies from level to level and is exchanged between neighbouring levels over
#####     the length Λ = ℂʷᴸ ℓ, so that every turbulent layer balances its own budget and no flux
#####     crosses air without turbulence between two layers:
#####
#####         (ρF)₊ - (ρF)₋ = ρ Δz (P + B - u),    (ρF)₊ = -ρ₊ Λ₊² ω̃₊ (c₊ - c) / Δz₊,
#####
#####     ω̃ the harmonic mean of ω at the two levels. Written for u = c ω this is one tridiagonal
#####     solve whose coefficients are Λ² / Δz times the bounded shares ω / (ω + ω′); its columns sum
#####     to ρ Δz, so Σ ρ Δz u = Σ ρ Δz (P + B) for any Λ, and it is an M-matrix.
#####
##### The damped local third moment is added to the flux members: the two sources of the stationary
##### budget of w′³, ∂t w′³ = 3 w′²b′ - 3 w′² ∂z w′² - w′³ / τ₃ + ..., relaxed over τ₃ = ℓ / √e,
#####
#####     w′³_local = τ₃ (√e (ℂʷᵇ B₊ + ℂʷˡ (1 - aᶜ) Bˡ₊) - ℂʷᵍ w′² ∂z w′²) / D,    D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ),    αᴺ = τ₃² N².
#####
##### The positive parts are smooth: x₊ = (x + √(x² + δ²)) / 2 = δ smooth_positive_part(x / δ), with a scale δ of the
##### dimension of x from the same place. B₊ has the scale of the dissipation ε = 𝕊ᴰ e^{3/2} / ℓ, the other term of the
##### budget that B belongs to; Bˡ₊ has the scale aᶜ ε, the dissipation in the cloudy part of the cell, which carries the
##### flux of liquid, so that the latent source vanishes in clear air (aᶜ = 0) as it does in saturated air. Both vanish
##### where the turbulence does.
#####
##### √e B₊ stands for the buoyant source w′²b′ (the buoyancy flux carried at the turbulent velocity; it
##### is not closed with w′³), and D ≥ 1 for the feedback of the stratification on the third moment:
##### solving the coupled budgets of w′³, w′²θ′ and w′θ′² exactly gives a determinant that is 1 + c αᴺ to
##### first order and vanishes at finite αᴺ (the poles of `AlgebraicThirdMoment`); D keeps the damping
##### of stable air, αᴺ > 0, and neither amplifies nor vanishes in unstable air. δ is the width in αᴺ over
##### which D passes from one to 1 + ℂʷᴺ αᴺ.
#####
##### The latent source, ℂʷˡ √e (1 - aᶜ) Bˡ₊, is the part of w′²b′ that condensation makes. The buoyancy is linear in
##### θˡⁱ and qᵗ but for the liquid, b′ = b_θ θ′ + b_q qᵗ′ + (b_qˡ - b_qᵛ) qˡ′, and the liquid is one-sided,
##### qˡ = max(s, 0) with s the saturation excess. For w and s jointly Gaussian with correlation ρ, spread σₛ and
##### t = s̄ / σₛ, the triple moments of the linear part vanish and
#####
#####     w′²qˡ′ = ρ² w′² σₛ φ(t) = ρ √(w′²) (φ(t) / Φ(t)) w′qˡ′,    aᶜ = Φ(t),    w′qˡ′ = ρ √(w′²) σₛ Φ(t),
#####
##### (φ and Φ the density and the distribution function of the standard normal): cloudy air is the tail of the
##### distribution that rises fastest, and it alone is heated. The ratio φ / Φ is between 1.6 (1 - aᶜ) and 2.5 (1 - aᶜ)
##### for 0.02 ≤ aᶜ ≤ 0.98 (1.6 at aᶜ = 1/2): it vanishes in saturated air, where the heating has no preference for
##### updrafts, and is largest where the cloud is a small fraction. With Bˡ = (b_qˡ - b_qᵛ) w′qˡ′, the part of the
##### buoyancy flux the liquid flux carries, the stationary budget gives w′³ = 3 τ₃ w′²b′ = ℂʷˡ τ₃ √e (1 - aᶜ) Bˡ with
##### ℂʷˡ = 3 √(2/3) ρ (1.6 to 2.5) ≈ 2 to 4 for ρ = 0.5 to 0.7 (the default is 8).
##### aᶜ and Bˡ are those of the distribution without skewness (the first pass), so the source is a function of the
##### prognostic state, and it does not need the net buoyancy flux to be positive: in a layer of cumulus the mean
##### stratification is stable, B < 0 without skewness, and Bˡ > 0.
#####
##### The buoyancy production needs the distribution (through aᴺ, N² and the diffusivities), and the
##### distribution needs w′³. The loop is not carried from one evaluation of the closure fields to the
##### next: within each evaluation the third moment is first diagnosed from the budget of the
##### distribution without skewness (a function of the prognostic state alone), and with `passes > 1`
##### again from the budget of the distribution that has the third moment of the pass before, a
##### fixed-point iteration from that start. The distribution, the stability and the conversions that the
##### stage uses are then computed with that third moment. The third moment is therefore a function of the
##### prognostic state and of what the budget fluxes read; of these only the microphysical tendency of the water of a
##### `TotalWaterBudgetFlux` (𝒬ᵠᵐ) is carried from the stage before, and it is checkpointed with the model.
#####
##### With ℂʷᵀ > 0 the same flux, with the sink shape e, is also a transport of TKE in the TKE
##### equation: ∂t e += ℂʷᵀ (uᵉ - (P + B)). It moves the share ℂʷᵀ of the production from where it is
##### produced to where e is, conserves the column's TKE exactly, and leaves the stage tendency
##### (1 - ℂʷᵀ)(P + B⁺) - ℂʷᵀ B⁻ + ℂʷᵀ uᵉ non-negative for 0 ≤ ℂʷᵀ ≤ 1 where uᵉ ≥ 0.
#####

"""
$(TYPEDEF)

The third moment of vertical velocity at the cell faces (interpolated to the centers with the other moments of the
distribution) from the vertical integral of the
budget of turbulent kinetic energy, with a damped local third moment,

    w′³ = r (ℂʷᵉ Fᵉ + ℂʷᵋ Fᵋ) + τ₃ (√e (ℂʷᵇ B₊ + ℂʷˡ (1 - aᶜ) Bˡ₊) - ℂʷᵍ w′² ∂z w′²) / D,
    Fʸ(z) = (1 / ρ) ∫₀ᶻ ρ (P + B - uʸ) dz′,    uʸ = cʸ ωʸ,    ωᵉ = e,    ωᵋ = ε = 𝕊ᴰ e^{3/2} / ℓ,
    τ₃ = ℓ / √e,    αᴺ = τ₃² N²,    D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ),    r = 1 + ℂʷʳ αᴺ / (1 + |αᴺ|),

where `P + B` is the shear and buoyancy production the TKE equation receives (the conversions of the
fluxes the closure applies), `B₊ = (B + √(B² + ε²)) / 2` its smooth positive part with the scale of the dissipation
`ε` there (a buoyancy sink makes little skewness of its own), `aᶜ` the cloud fraction and
`Bˡ₊ = (Bˡ + √(Bˡ² + (aᶜ ε)²)) / 2` the smooth positive part, with the scale `aᶜ ε`, of the part of the buoyancy flux that
the liquid flux carries, `Bˡ = (b_qˡ - b_qᵛ) w′qˡ′`, both of the distribution without skewness,
`ℓ` the closure's mixing length and `αᴺ` the stratification number at the face (`|ℂʷʳ| < 1`
keeps `r` positive).

The amplitude `cʸ` of each sink is fixed by the budget so that `Fʸ` vanishes at the ground and at
the top of the column: one number for the column with `ℂʷᴸ = Inf`, or a profile exchanged between
neighbouring levels over the length `ℂʷᴸ ℓ` (one tridiagonal solve per sink shape and column). No
depth of a turbulent layer enters. `Fᵋ` is the flux the closure's own dissipation leaves over: where
the TKE equation is steady it is the flux `-Kᵉ ∂z e` the closure applies to `e`. `Fᵉ` is the flux
with one turnover rate for the column. `ℂʷᵀ > 0` adds `ℂʷᵀ Fᵉ` to the flux of TKE in the TKE equation.

The damped local third moment holds the two sources of the stationary budget of `w′³`, the buoyant
one as `√e B₊` and the gradient production `-w′² ∂z w′²`, relaxed over `τ₃`. The latent source
`√e (1 - aᶜ) Bˡ₊` is the part of the buoyant source `w′²b′` that condensation makes in a partly cloudy
layer, where the cloudy air is the fastest-rising tail of the distribution and is the only air
heated; it vanishes in saturated air (`aᶜ = 1`) and in clear air (`Bˡ = 0`), and it is positive in a
layer of cumulus, whose net buoyancy flux without skewness is negative. The damping `D ≥ 1`
stands for the suppression of the third moment in stably stratified air; it has no zero, so the
member has no pole. It tends to one in unstable air and to `1 + ℂʷᴺ αᴺ` in stable air, over the width
`δ = stratification_damping_width` in `αᴺ`.

No third moment is stored between evaluations of the closure fields (the budget it integrates reads the conversion
of a [`TotalWaterBudgetFlux`](@ref), whose microphysical tendency of the water is that of the stage before and is
checkpointed with the model). Within one evaluation the
third moment is diagnosed `passes` times: first from the budget of the distribution without
skewness, then from the budget of the distribution with the third moment of the pass before.
The limiter in the `skewness` slot of [`DoubleGaussian`](@ref) ([`TanhSaturatedSkewness`](@ref), ...)
keeps the result realizable.

The initial values of the coefficients are a joint least-squares fit to coarse-grained large-eddy
simulations of BOMEX, ARM cumulus, ASTEX and DYCOMS.

$(TYPEDFIELDS)
"""
# The amplitude of the sinks that balance the budget: one per column, or one per level exchanged between neighbours
struct ColumnSinkAmplitude end
struct ExchangedSinkAmplitude end

struct TKEBudgetThirdMoment{N, FT, L, A} <: SimplifiedHigherOrderClosures.AbstractSubgridDistributionComponent
    "Coefficient of the budget flux with the sink shape `e`"
    ℂʷᵉ :: FT
    "Coefficient of the budget flux with the sink shape `ε`"
    ℂʷᵋ :: FT
    "Exchange length of the sink amplitude in mixing lengths, `Λ = ℂʷᴸ ℓ`; `Inf` for one amplitude per column"
    ℂʷᴸ :: FT
    "Amplitude of the dependence of the ratio `r` on the stratification number, in (-1, 1)"
    ℂʷʳ :: FT
    "Coefficient of the buoyant source `√e B₊` of the damped local third moment"
    ℂʷᵇ :: FT
    "Coefficient of the latent source `√e (1 - aᶜ) Bˡ₊` of the damped local third moment"
    ℂʷˡ :: FT
    "Coefficient of the gradient production `-w′² ∂z w′²` of the damped local third moment"
    ℂʷᵍ :: FT
    "Coefficient of the stratification number in the damping `D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ)` of the local third moment; not negative"
    ℂʷᴺ :: FT
    "Share of the production that the budget flux with the sink shape `e` transports in the TKE equation, in [0, 1]"
    ℂʷᵀ :: FT
    "Width `δ` in the stratification number over which the damping passes from one to `1 + ℂʷᴺ αᴺ`; positive"
    stratification_damping_width :: FT
    "Share of the TKE of the cell below in the sink shape `e` of a cell, `ωᵉₖ = eₖ + ℂʷᵒ eₖ₋₁`: the overshoot of the budget flux into the cell above a turbulent layer; not negative"
    ℂʷᵒ :: FT
    "Where the third moment is diagnosed: `Face()` (the default) or `Center()`"
    location :: L
    "The amplitude of the sinks: `ExchangedSinkAmplitude()`, or `ColumnSinkAmplitude()` (constructed with `ℂʷᴸ = Inf`)"
    sink_amplitude :: A

    TKEBudgetThirdMoment{N}(ℂʷᵉ::FT, ℂʷᵋ::FT, ℂʷᴸ::FT, ℂʷʳ::FT, ℂʷᵇ::FT, ℂʷˡ::FT, ℂʷᵍ::FT, ℂʷᴺ::FT, ℂʷᵀ::FT,
                            stratification_damping_width::FT, ℂʷᵒ::FT, location::L = Face(),
                            sink_amplitude::A = ExchangedSinkAmplitude()) where {N, FT, L, A} =
        new{N, FT, L, A}(ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷᵇ, ℂʷˡ, ℂʷᵍ, ℂʷᴺ, ℂʷᵀ, stratification_damping_width, ℂʷᵒ, location, sink_amplitude)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`TKEBudgetThirdMoment`](@ref). The defaults exchange the sink amplitude over eight mixing
lengths, have a constant ratio and no transport of TKE, and of the damped local third moment only
the latent source (`ℂʷᵇ = ℂʷᵍ = 0`: the buoyant source `B₊` cannot tell heating from below, carried
by updrafts, from cooling from above, carried by downdrafts, so the budget fluxes carry that part
of the third moment; `ℂʷˡ = 8`: the latent source is the heating of the cloudy updrafts of a partly
cloudy layer, which the budget of the distribution without skewness does not hold). They diagnose
the third moment once per evaluation of the closure fields, from the budget of the distribution
without skewness.

`ℂʷˡ = 8` is twice the upper end, 4, of the estimate `3 √(2/3) ρ (1.6 to 2.5)` for a correlation `ρ = 0.5` to
`0.7` of `w` with the saturation excess, and near a fit of `w′³` in the cloud layer of large-eddy
simulations of cumulus (3.4 with their mixing length). It is
an initial value: the closure's own `ρ` and `w′²` differ from the simulations', and the simulated columns
come closer to them with a larger coefficient: with the hand-tuned stability functions
8 lowers the error of the BOMEX and cfsite14_04 columns against 4 and leaves the others nearly
unchanged.
"""
function TKEBudgetThirdMoment(FT::DataType = Oceananigans.defaults.FloatType;
                              ℂʷᵉ = 0.12, ℂʷᵋ = 0.34, ℂʷᴸ = 8, ℂʷʳ = 0, ℂʷᵇ = 0, ℂʷˡ = 8, ℂʷᵍ = 0, ℂʷᴺ = 0, ℂʷᵀ = 0,
                              stratification_damping_width = 1, passes = 1, ℂʷᵒ = 0, location = Face())
    abs(ℂʷʳ) < 1 || throw(ArgumentError("ℂʷʳ must lie in (-1, 1) for a positive ratio, got $(ℂʷʳ)."))
    0 <= ℂʷᵀ <= 1 || throw(ArgumentError("ℂʷᵀ must lie in [0, 1], got $(ℂʷᵀ)."))
    ℂʷᴺ >= 0 || throw(ArgumentError("ℂʷᴺ must not be negative (the damping must not vanish), got $(ℂʷᴺ)."))
    ℂʷᴸ > 0 || throw(ArgumentError("ℂʷᴸ must be positive (Inf for one sink amplitude per column), got $(ℂʷᴸ)."))
    stratification_damping_width > 0 ||
        throw(ArgumentError("stratification_damping_width must be positive, got $(stratification_damping_width)."))
    ℂʷᵒ >= 0 || throw(ArgumentError("ℂʷᵒ must not be negative, got $(ℂʷᵒ)."))
    passes isa Integer && passes >= 1 || throw(ArgumentError("passes must be a positive integer, got $(passes)."))
    scalars = (ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷᵇ, ℂʷˡ, ℂʷᵍ, ℂʷᴺ, ℂʷᵀ, stratification_damping_width, ℂʷᵒ)
    SimplifiedHigherOrderClosures.validate_third_moment_location(location)
    # The amplitude is chosen here, once: a kernel does not branch on a coefficient
    sink_amplitude = isfinite(ℂʷᴸ) ? ExchangedSinkAmplitude() : ColumnSinkAmplitude()
    return TKEBudgetThirdMoment{Int(passes)}(map(c -> convert(FT, c), scalars)..., location, sink_amplitude)
end

# The number of passes is a type parameter, so that the loop over them is unrolled when traced
third_moment_passes(::TKEBudgetThirdMoment{N}) where N = N

# The amplitude of the sinks is chosen by the constructor from ℂʷᴸ; a rebuild (`with_parameters`) that replaces ℂʷᴸ
# keeps it, so a replacement that would need the other amplitude is an error (an exchange over Λ = Inf ℓ gives NaN,
# and a finite ℂʷᴸ would be ignored by the amplitude of the column). Traced scalars are not checked.
SimplifiedHigherOrderClosures.component_constructor(::Type{<:TKEBudgetThirdMoment{N}}) where N =
    (args...) -> validate_sink_amplitude(TKEBudgetThirdMoment{N}(args...))

function validate_sink_amplitude(tm::TKEBudgetThirdMoment)
    validate_sink_amplitude(tm.sink_amplitude, tm.ℂʷᴸ)
    return tm
end

validate_sink_amplitude(amplitude, ℂʷᴸ) = nothing

validate_sink_amplitude(::ExchangedSinkAmplitude, ℂʷᴸ::AbstractFloat) =
    isfinite(ℂʷᴸ) || throw(ArgumentError("ℂʷᴸ = $(ℂʷᴸ) replaces the finite exchange length of a TKEBudgetThirdMoment " *
                                         "that exchanges its sink amplitude; construct it with ℂʷᴸ = Inf for one " *
                                         "amplitude per column."))

validate_sink_amplitude(::ColumnSinkAmplitude, ℂʷᴸ::AbstractFloat) =
    isinf(ℂʷᴸ) || throw(ArgumentError("ℂʷᴸ = $(ℂʷᴸ) replaces Inf in a TKEBudgetThirdMoment with one sink amplitude " *
                                      "per column; construct it with a finite ℂʷᴸ to exchange the amplitude."))

# The scalars are adapted one by one and the location passes through unchanged, so that the adapted closure infers
# (inside a kernel, `getclosure` adapts the closure of each column)
Adapt.adapt_structure(to, tm::TKEBudgetThirdMoment{N}) where N =
    TKEBudgetThirdMoment{N}(adapt(to, tm.ℂʷᵉ), adapt(to, tm.ℂʷᵋ), adapt(to, tm.ℂʷᴸ), adapt(to, tm.ℂʷʳ), adapt(to, tm.ℂʷᵇ),
                            adapt(to, tm.ℂʷˡ), adapt(to, tm.ℂʷᵍ), adapt(to, tm.ℂʷᴺ), adapt(to, tm.ℂʷᵀ),
                            adapt(to, tm.stratification_damping_width), adapt(to, tm.ℂʷᵒ), tm.location, tm.sink_amplitude)

SimplifiedHigherOrderClosures.type_parameter_summary(tm::TKEBudgetThirdMoment{N, FT}) where {N, FT} = string("{", N, ", ", FT, "}")

# The third moment before its limiter, at its location
third_moment_location_field(grid, ::Face) = ZFaceField(grid)
third_moment_location_field(grid, ::Center) = CenterField(grid)

const TKEBudgetDistribution = DoubleGaussian{<:Any, <:Any, <:TKEBudgetThirdMoment}

#####
##### Fields: the two sinks, the two fluxes, the dissipation, the third moment before its limiter, the transport tendency of
##### TKE and a work array
#####

third_moment_fields(grid, third_moment) = NamedTuple()

third_moment_fields(grid, tm::TKEBudgetThirdMoment) =
    (Fᵉ = CenterField(grid), Fᵋ = CenterField(grid), uᵉ = CenterField(grid), uᵋ = CenterField(grid), ε = CenterField(grid),
     w′³ᵇ = third_moment_location_field(grid, tm.location), Tᵉ = CenterField(grid), budget_work = CenterField(grid),
     NamedTuple(pass_name(name) => ZFaceField(grid) for name in PASS_CLOSURE_FIELD_NAMES)...)

# The closure fields that a pass recomputes (the stability, the mixing lengths and the diffusivities). A pass
# writes them into fields of its own, so that those of the prognostic state are never overwritten and need not be
# kept and put back.
const PASS_CLOSURE_FIELD_NAMES = (:N², :ℓᵐ, :ℓˢ, :Kᵘ, :Kᶜ, :Kᵉ)
pass_name(name) = Symbol(:pass_, name)

# The closure fields of a pass: those of the prognostic state with the stability, the mixing lengths and the
# diffusivities replaced by the pass's own (the per-tracer lookup of the diffusivities included). The column value of
# the stability functions (`column_stability`, `nothing` for pointwise ones) is shared: each pass fills it from its own
# static stability before its diffusivities, as the prognostic state does.
function pass_closure_fields(closure_fields::TKEClosureFields, cl)
    Kᵘ, Kᶜ, Kᵉ = cl.pass_Kᵘ, cl.pass_Kᶜ, cl.pass_Kᵉ
    tracer_diffusivities = map(K -> K === closure_fields.Kᵉ ? Kᵉ : Kᶜ, closure_fields.tupled_tracer_diffusivities)
    return TKEClosureFields(Kᵘ, Kᶜ, Kᵉ, cl.pass_N², cl.pass_ℓᵐ, cl.pass_ℓˢ, closure_fields.Lᵉ, tracer_diffusivities,
                            closure_fields.tupled_implicit_linear_coefficients, closure_fields.subgrid_cloud,
                            closure_fields.column_stability)
end

# The pass starts from the closure fields of the prognostic state, halos included: one kernel for the six copies
@kernel function _copy_parents!(destinations, sources)
    i, j, k = @index(Global, NTuple)
    # `map` over the tuples unrolls the copies
    map((destination, source) -> @inbounds(destination[i, j, k] = source[i, j, k]), destinations, sources)
end

function start_pass_closure_fields!(pass_fields, closure_fields, grid)
    destinations = map(name -> parent(getproperty(pass_fields, name)), PASS_CLOSURE_FIELD_NAMES)
    sources = map(name -> parent(getproperty(closure_fields, name)), PASS_CLOSURE_FIELD_NAMES)
    launch!(grid.architecture, grid, size(first(sources)), _copy_parents!, destinations, sources)
    return nothing
end

#####
##### The sinks that balance the budget, in place: each `u` holds its shape ω on entry and its sink on exit
#####

# 2 ρ Λ² / Δz at the face k, between the centers k - 1 and k
@inline function sink_exchange_coefficientᶜᶜᶠ(i, j, k, grid, ℂᴸ, ρ, ℓᵐ)
    Λ = @inbounds ℂᴸ * ℓᵐ[i, j, k]
    return 2 * ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Λ * Λ / Δzᶜᶜᶠ(i, j, k, grid)
end

# The shape e of a sink or of a redistributed source, from the TKE e of a cell (which transport can leave negative) and
# the TKE eᵐⁱⁿ that the closure gives still air (`minimum_tke`): e₊² / (e₊ + eᵐⁱⁿ), with e₊ the positive part of e. It
# is zero, with zero slope, where e ≤ 0, so that still air has no shape (the levels of still air do not exchange and
# no flux reaches them, whatever eᵐⁱⁿ); it is e²/eᵐⁱⁿ below eᵐⁱⁿ and e - eᵐⁱⁿ far above. The smooth positive part
# (e + √(e² + eᵐⁱⁿ²)) / 2 would not do here: it is eᵐⁱⁿ / 2 in still air, and the redistribution reads the shape, not
# its magnitude, so still air would exchange as uniformly weak turbulence does.
@inline function turbulent_shape(e, eᵐⁱⁿ)
    turbulent = e > 0
    e₊ = ifelse(turbulent, e, one(e))
    return ifelse(turbulent, e₊ * (e₊ / (e₊ + eᵐⁱⁿ)), zero(e))
end

# The shape of an array of the centers and the sum of two arrays, as arrays of the centers
struct TurbulentShape{X, S}
    e :: X
    minimum_tke :: S
end

@inline Base.getindex(shape::TurbulentShape, i, j, k) = @inbounds turbulent_shape(shape.e[i, j, k], shape.minimum_tke)

struct SumOfArrays{A, B}
    a :: A
    b :: B
end

@inline Base.getindex(s::SumOfArrays, i, j, k) = @inbounds s.a[i, j, k] + s.b[i, j, k]

# A second source of `nothing` is none
@inline source_value(source, i, j, k, FT) = @inbounds source[i, j, k]
@inline source_value(::Nothing, i, j, k, FT) = zero(FT)
@inline store_source!(s, i, j, k, value) = @inbounds s[i, j, k] = value
@inline store_source!(::Nothing, i, j, k, value) = nothing
@inline eliminate_source!(s, work, i, j, k) = @inbounds s[i, j, k] -= work[i, j, k] * s[i, j, k+1]
@inline eliminate_source!(::Nothing, work, i, j, k) = nothing

# Two sources of the column, `source₁` and `source₂` (arrays of the centers, per unit mass), as the turbulence
# redistributes them, `s₁` and `s₂`: each c ω with the `shape` ω (an array of the centers) and an amplitude c exchanged
# between neighbouring levels over the length ℂᴸ ℓ,
#
#     [mₖ + a₊θ₊ + a₋θ₋] sₖ - a₊(1 - θ₊) sₖ₊₁ - a₋(1 - θ₋) sₖ₋₁ = mₖ sourceₖ,    m = ρ Δz,    θ± = ωₖ±₁ / (ωₖ + ωₖ±₁),
#
# with the exchange coefficients a± of the faces (`sink_exchange_coefficientᶜᶜᶠ`). The Thomas algorithm eliminates the
# matrix, which is diagonally dominant by columns, once for the two right-hand sides; `work` holds the eliminated upper
# diagonal. `s₁` may be the array of the shape (the shape of a level is read before its `s₁` is stored). The one solve
# of the sinks of `TKEBudgetThirdMoment` and of the budget fluxes of the heating and of the total water.
@inline function exchange_amplitudes!(s₁, s₂, work, i, j, grid, ℂᴸ, shape, source₁, source₂, ρ, ℓᵐ)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    tiny = sqrt(floatmin(FT))
    @inbounds begin
        ω⁻ = zero(FT)
        c′ = zero(FT)
        d′₁ = zero(FT)
        d′₂ = zero(FT)
        a⁻ = zero(FT) # the exchange coefficient at the face below
        ω = shape[i, j, 1]
        for k in 1:Nz
            m = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            diagonal = m
            lower = zero(FT)
            upper = zero(FT)
            ω⁺ = zero(FT)
            if k > 1
                share = ω + ω⁻ + tiny
                diagonal += a⁻ * ω⁻ / share
                lower = -a⁻ * ω / share
            end
            if k < Nz
                a = sink_exchange_coefficientᶜᶜᶠ(i, j, k + 1, grid, ℂᴸ, ρ, ℓᵐ)
                ω⁺ = shape[i, j, k+1]
                share = ω + ω⁺ + tiny
                diagonal += a * ω⁺ / share
                upper = -a * ω / share
                a⁻ = a
            end
            pivot = diagonal - lower * c′
            d′₁ = (m * source₁[i, j, k] - lower * d′₁) / pivot
            d′₂ = (m * source_value(source₂, i, j, k, FT) - lower * d′₂) / pivot
            c′ = upper / pivot
            work[i, j, k] = c′
            s₁[i, j, k] = d′₁
            store_source!(s₂, i, j, k, d′₂)
            ω⁻ = ω
            ω = ω⁺
        end
        for k in Nz-1:-1:1
            s₁[i, j, k] -= work[i, j, k] * s₁[i, j, k+1]
            eliminate_source!(s₂, work, i, j, k)
        end
    end
    return nothing
end

# The sink of the column, `u` holding its shape ω on entry and the sink that balances ρ Δz (P + B) on exit
@inline balance_sink!(u, work, i, j, grid, ::ExchangedSinkAmplitude, ℂʷᴸ, cl, ρ, ℓᵐ) =
    exchange_amplitudes!(u, nothing, work, i, j, grid, ℂʷᴸ, u, SumOfArrays(cl.P, cl.B), nothing, ρ, ℓᵐ)

# One amplitude for the column, c = Σ ρ Δz (P + B) / Σ ρ Δz ω
@inline function balance_sink!(u, work, i, j, grid, ::ColumnSinkAmplitude, ℂʷᴸ, cl, ρ, ℓᵐ)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    tiny = sqrt(floatmin(FT))
    @inbounds begin
        Π = zero(FT)
        Ω = zero(FT)
        for k in 1:Nz
            m = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            Π += m * (cl.P[i, j, k] + cl.B[i, j, k])
            Ω += m * u[i, j, k]
        end
        c = Π / (Ω + tiny)
        for k in 1:Nz
            u[i, j, k] *= c
        end
    end
    return nothing
end

#####
##### The damped local third moment
#####

# smooth_positive_part(x) = (x + √(x² + 1)) / 2 = 1 / (2 (√(x² + 1) - x)): smooth, increasing, positive, x + 1/(4x) for
# large x and 1/(4|x|) for large negative x, with slope in (0, 1); one square root and three divisions, written in
# the form without cancellation on each side of zero; √(x² + 1) is scaled by max(|x|, 1) so that it does not overflow
@inline function smooth_positive_part(x)
    a = max(abs(x), one(x))
    s = a * sqrt((x / a)^2 + (1 / a)^2)
    return ifelse(x >= 0, (x + s) / 2, 1 / (2 * (s - x)))
end

# x₊ = δ smooth_positive_part(x / δ) = (x + √(x² + δ²)) / 2 with the scale δ ≥ 0 of x, written without the division: it
# is max(x, 0) where δ = 0, δ / 2 at x = 0, and x + δ² / (4x) and δ² / (4|x|) far on either side. The negative part
# and the minimum of two numbers follow from it: x₋ = -(-x)₊ and min(a, b) = b - (b - a)₊, which is symmetric,
# (a + b - √((a - b)² + δ²)) / 2.
@inline function smooth_positive_part(x, δ)
    s = sqrt(x * x + δ * δ)
    return ifelse(x >= 0, (x + s) / 2, δ * δ / (2 * (s - x)))
end

@inline smooth_negative_part(x, δ) = -smooth_positive_part(-x, δ)
@inline smooth_minimum(a, b, δ) = b - smooth_positive_part(b - a, δ)

# D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ) ≥ 1: one in unstable air, 1 + ℂʷᴺ αᴺ in stable air, with slope in (0, ℂʷᴺ)
@inline stratification_damping(ℂʷᴺ, δ, αᴺ) = 1 + ℂʷᴺ * δ * smooth_positive_part(αᴺ / δ)

# τ₃ (√e (ℂʷᵇ B₊ + ℂʷˡ L) - ℂʷᵍ w′² ∂z w′²) / D at the `location` k, with the latent source L = (1 - aᶜ) Bˡ₊ given;
# w′² ∂z w′² = ∂z (w′²)² / 2 is differenced across the face, between the centers k - 1 and k (compact, as `shoc.F90`
# differences center quantities for its interface third moment). `ε` is the dissipation there, the scale of the smooth
# positive part B₊ and of the smooth minimum. With a `HeatingBudgetFlux` the buoyant source is signed by origin
# (`cooling = (Gᑫᶜ, ℂʷᑫ)`, `nothing` without one): of B₊, the part Gᶜ = min(Gᑫᶜ, B₊) that the flux driven by cooling
# converts is carried by downdrafts, and the source is ℂʷᵇ (B₊ - Gᶜ) - ℂʷᑫ Gᶜ.
@inline buoyant_source(::Nothing, ℂʷᵇ, B₊, ε) = ℂʷᵇ * B₊

@inline function buoyant_source(cooling::Tuple, ℂʷᵇ, B₊, ε)
    Gᑫᶜ, ℂʷᑫ = cooling
    Gᶜ = smooth_minimum(Gᑫᶜ, B₊, ε)
    return ℂʷᵇ * (B₊ - Gᶜ) - ℂʷᑫ * Gᶜ
end

# The latent source (1 - aᶜ) Bˡ₊ at the center k, with the scale aᶜ ε of the positive part of Bˡ: none in clear air
@inline latent_sourceᶜᶜᶜ(i, j, k, cl) =
    @inbounds (1 - cl.aᶜ[i, j, k]) * smooth_positive_part(cl.Bˡ[i, j, k], cl.aᶜ[i, j, k] * cl.ε[i, j, k])

# At the face k: (w′²)² differenced across the face, between the centers k - 1 and k
@inline function local_variance_gradient(::Face, i, j, k, grid, w′²)
    w′²ᵘ = @inbounds w′²[i, j, k]
    w′²ˡ = @inbounds w′²[i, j, k-1]
    return (w′²ᵘ * w′²ᵘ - w′²ˡ * w′²ˡ) / (2 * Δzᶜᶜᶠ(i, j, k, grid))
end

# At the center k: (w′²)² differenced across the neighbouring centers (one-sided at the two ends of the column)
@inline function local_variance_gradient(::Center, i, j, k, grid, w′²)
    Nz = size(grid, 3)
    kᵘ = min(k + 1, Nz)
    kˡ = max(k - 1, 1)
    Δz = ifelse(k < Nz, Δzᶜᶜᶠ(i, j, k + 1, grid), zero(grid_float_type(grid))) +
         ifelse(k > 1, Δzᶜᶜᶠ(i, j, k, grid), zero(grid_float_type(grid)))
    w′²ᵘ = @inbounds w′²[i, j, kᵘ]
    w′²ˡ = @inbounds w′²[i, j, kˡ]
    return ifelse(kᵘ > kˡ, (w′²ᵘ * w′²ᵘ - w′²ˡ * w′²ˡ) / (2 * Δz), zero(Δz))
end

@inline function damped_local_third_moment(location, i, j, k, grid, tm, w′², τ, u, αᴺ, B, ε, cooling, latent_source)
    w′²∂zw′² = local_variance_gradient(location, i, j, k, grid, w′²)
    B₊ = smooth_positive_part(B, ε)
    source = buoyant_source(cooling, tm.ℂʷᵇ, B₊, ε) + tm.ℂʷˡ * latent_source
    return τ * (u * source - tm.ℂʷᵍ * w′²∂zw′²) / stratification_damping(tm.ℂʷᴺ, tm.stratification_damping_width, αᴺ)
end

#####
##### The budget of one evaluation: sinks, fluxes, the transport of TKE and, with `diagnose = Val(true)`, the third moment
#####

@kernel function _compute_tke_budget_flux!(cl, grid, closure, closure_fields, e, velocities, ρ, diagnose)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    tm = closure_ij.subgrid_distribution.third_moment
    ℓᵐ = closure_fields.ℓᵐ
    eᵐⁱⁿ = closure_ij.minimum_tke
    @inbounds for k in 1:size(grid, 3)
        # The shape e (`turbulent_shape`: none where e ≤ 0). With ℂʷᵒ > 0 the shape reaches one cell above the turbulence
        # (the overshoot); the cell below the first is none
        e⁻ = ifelse(k > 1, turbulent_shape(e[i, j, max(k - 1, 1)], eᵐⁱⁿ), zero(grid_float_type(grid)))
        cl.uᵉ[i, j, k] = turbulent_shape(e[i, j, k], eᵐⁱⁿ) + tm.ℂʷᵒ * e⁻
        cl.ε[i, j, k] = dissipationᶜᶜᶜ(i, j, k, grid, closure_ij, e, velocities, closure_fields)
        cl.uᵋ[i, j, k] = cl.ε[i, j, k]
    end
    balance_sink!(cl.uᵉ, cl.budget_work, i, j, grid, tm.sink_amplitude, tm.ℂʷᴸ, cl, ρ, ℓᵐ)
    balance_sink!(cl.uᵋ, cl.budget_work, i, j, grid, tm.sink_amplitude, tm.ℂʷᴸ, cl, ρ, ℓᵐ)
    N² = closure_fields.N²
    # The fluxes F at the centers: the integral of ρ (P + B - u) from the ground, the cell of the center counted half
    belowᵉ = zero(grid_float_type(grid))
    belowᵋ = zero(grid_float_type(grid))
    @inbounds for k in 1:size(grid, 3)
        ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        rᵉ = ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵉ[i, j, k])
        rᵋ = ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵋ[i, j, k])
        cl.Fᵉ[i, j, k] = (belowᵉ + rᵉ / 2) / ρ[i, j, k]
        cl.Fᵋ[i, j, k] = (belowᵋ + rᵋ / 2) / ρ[i, j, k]
        belowᵉ += rᵉ
        belowᵋ += rᵋ

        cl.Tᵉ[i, j, k] = tm.ℂʷᵀ * (cl.uᵉ[i, j, k] - cl.P[i, j, k] - cl.B[i, j, k])

        diagnose_center_third_moment!(tm.location, diagnose, cl, i, j, k, grid, tm, closure_ij, e, ℓᵐ, N², cl.Fᵉ[i, j, k], cl.Fᵋ[i, j, k])
    end
    diagnose_face_third_moments!(tm.location, diagnose, cl, i, j, grid, tm, closure_ij, e, ℓᵐ, N², ρ)
end

# The third moment at the interior face k, with ℓ, √e and N² of the face; one stratification number αᴺ = τ₃² N² for
# the ratio and for the damping. The fluxes at the face are the integrals ρF of the cells below it over the density
# at the face; the buoyancy production and the latent source are averaged from the two centers. Only in the passes
# of `diagnose_third_moment!` (`Val(true)`). Diagnosed at the faces, as `shoc.F90` diagnoses its third moment, and
# interpolated to the centers with the other moments.
@inline diagnose_face_third_moments!(location, ::Val{false}, args...) = nothing
@inline diagnose_face_third_moments!(::Center, ::Val{true}, args...) = nothing

# The faces of the column in a loop of their own, after the fluxes at the centers: a branch on `k > 1` in the loop
# over the centers lets LLVM peel the first center and leave the rest a loop whose exit flag is carried, whose
# iteration count the reverse pass of Enzyme (under Reactant) cannot recover to cache its values. The integrals
# of the cells below each face are summed again, in the order and with the operations of the loop over the centers.
@inline function diagnose_face_third_moments!(::Face, ::Val{true}, cl, i, j, grid, tm, closure_ij, e, ℓᵐ, N², ρ)
    belowᵉ = zero(grid_float_type(grid))
    belowᵋ = zero(grid_float_type(grid))
    @inbounds for k in 1:size(grid, 3)-1
        ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        belowᵉ += ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵉ[i, j, k])
        belowᵋ += ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵋ[i, j, k])
        diagnose_face_third_moment!(cl, i, j, k + 1, grid, tm, closure_ij, e, ℓᵐ, N², ρ, belowᵉ, belowᵋ)
    end
    return nothing
end

@inline function diagnose_face_third_moment!(cl, i, j, k, grid, tm, closure_ij, e, ℓᵐ, N², ρ, ρFᵉ, ρFᵋ)
    @inbounds begin
        ρᶠ = (ρ[i, j, k] + ρ[i, j, k-1]) / 2
        ℓ = ℓᵐ[i, j, k]
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)
        τ = ℓ / u
        αᴺ = τ * τ * N²[i, j, k]
        r = 1 + tm.ℂʷʳ * αᴺ / (1 + abs(αᴺ))
        B = (cl.B[i, j, k] + cl.B[i, j, k-1]) / 2
        ε = (cl.ε[i, j, k] + cl.ε[i, j, k-1]) / 2
        Gᑫᶜ = (cooling_flux_conversion(cl, i, j, k) + cooling_flux_conversion(cl, i, j, k-1)) / 2
        cooling = cooling_conversion(closure_ij.heating_budget_flux, Gᑫᶜ)
        latent = (latent_sourceᶜᶜᶜ(i, j, k, cl) + latent_sourceᶜᶜᶜ(i, j, k-1, cl)) / 2
        cl.w′³ᵇ[i, j, k] = r * (tm.ℂʷᵉ * ρFᵉ + tm.ℂʷᵋ * ρFᵋ) / ρᶠ +
                           damped_local_third_moment(Face(), i, j, k, grid, tm, cl.w′², τ, u, αᴺ, B, ε, cooling, latent)
    end
    return nothing
end

# The third moment at the center, with ℓ, √e and N² as the distribution forms them there; one stratification number
# αᴺ = τ₃² N² for the ratio and for the damping. Only in the passes of `diagnose_third_moment!` (`Val(true)`).
@inline diagnose_center_third_moment!(location, ::Val{false}, args...) = nothing
@inline diagnose_center_third_moment!(::Face, ::Val{true}, args...) = nothing

@inline function diagnose_center_third_moment!(::Center, ::Val{true}, cl, i, j, k, grid, tm, closure_ij, e, ℓᵐ, N², Fᵉ, Fᵋ)
    @inbounds begin
        ℓ = (ℓᵐ[i, j, k] + ℓᵐ[i, j, k+1]) / 2
        u = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure_ij, e)
        τ = ℓ / u
        αᴺ = τ * τ * (N²[i, j, k] + N²[i, j, k+1]) / 2
        r = 1 + tm.ℂʷʳ * αᴺ / (1 + abs(αᴺ))
        cooling = cooling_conversion(closure_ij.heating_budget_flux, cooling_flux_conversion(cl, i, j, k))
        cl.w′³ᵇ[i, j, k] = r * (tm.ℂʷᵉ * Fᵉ + tm.ℂʷᵋ * Fᵋ) +
                           damped_local_third_moment(Center(), i, j, k, grid, tm, cl.w′², τ, u, αᴺ, cl.B[i, j, k],
                                                     cl.ε[i, j, k], cooling, latent_sourceᶜᶜᶜ(i, j, k, cl))
    end
    return nothing
end

function compute_tke_budget_flux!(fields, closure_fields, closure, model, tracers, diagnose)
    grid = model.grid
    launch!(grid.architecture, grid, :xy, _compute_tke_budget_flux!, fields.cloud, grid, closure, closure_fields,
            tracers[TKE_NAME], model.velocities, total_density(model.dynamics), diagnose)
    return nothing
end

#####
##### The third moment of one evaluation of the closure fields
#####

diagnose_third_moment!(fields, distribution, closure_fields, closure, model, tracers, parameters) = nothing
compute_tke_budget!(fields, distribution, closure_fields, closure, model, tracers) = nothing

# Within `update_state!`, after the first static stability, mixing length and diffusivities (which read only the
# prognostic state) and before the distribution the stage uses. Each pass: the distribution with the third moment of
# the pass before (none in the first), its stability and diffusivities, the conversions of the applied fluxes, and the
# third moment from their budget. The distribution reads the stability, mixing length and diffusivities of the
# prognostic state (whose stability reads the distribution of the previous update); the pass computes its own into
# `pass_closure_fields`, which every pass starts from those of the prognostic state, so the final evaluation starts
# from them too. The buoyancy derivatives of the prognostic state computed here serve the final evaluation as well.
function diagnose_third_moment!(fields::SubgridCloudFields, distribution::TKEBudgetDistribution, closure_fields, closure, model, tracers, parameters)
    cl = fields.cloud
    pass_fields = pass_closure_fields(closure_fields, cl)
    fill!(parent(cl.w′³ᵇ), 0)
    compute_buoyancy_derivatives!(fields, model)
    # The heating budget flux reads the prognostic state and the heating alone, like the buoyancy derivatives
    compute_budget_fluxes!(fields, closure_fields, closure, model, tracers)
    passes = third_moment_passes(distribution.third_moment)
    for pass in 1:passes
        start_pass_closure_fields!(pass_fields, closure_fields, model.grid)
        compute_subgrid_distribution!(fields, closure_fields, closure, model)
        solve_static_stability!(closure.static_stability, fields.buoyancy_flux, fields, pass_fields, closure, model, tracers, parameters)
        compute_applied_flux_conversions!(fields, pass_fields, closure, model, tracers)
        compute_tke_budget_flux!(fields, pass_fields, closure, model, tracers, Val(true))
    end
    return nothing
end

# The buoyancy derivatives depend on the prognostic state alone: those of `diagnose_third_moment!` stand,
# and so do the budget fluxes
compute_subgrid_cloud_buoyancy_derivatives!(fields, ::TKEBudgetDistribution, model) = nothing
compute_subgrid_cloud_budget_fluxes!(fields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers) = nothing

# After the conversions of the applied fluxes the stage uses: their budget, for the transport of TKE and as a diagnostic
compute_tke_budget!(fields::SubgridCloudFields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers) =
    compute_tke_budget_flux!(fields, closure_fields, closure, model, tracers, Val(false))

#####
##### The third moment at its location k: the one diagnosed from the budget
#####

@inline third_moment_diagnosis(tm::TKEBudgetThirdMoment, i, j, k, d, cl, g, algebraic_arguments) =
    @inbounds cl.w′³ᵇ[i, j, k]

#####
##### The transport of TKE by the budget flux, in the stage tendency of ρe
#####

@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, distribution, fields) = zero(grid)
@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, ::TKEBudgetDistribution, fields) = @inbounds fields.cloud.Tᵉ[i, j, k]
