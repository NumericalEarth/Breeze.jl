#####
##### The third moment of vertical velocity from the vertical integral of the TKE budget
#####
##### A third moment of the `DoubleGaussian` subgrid distribution of `TKEBasedTurbulenceClosure`
##### (`third_moment = TKEBudgetThirdMoment(...)`), next to `AlgebraicThirdMoment`.
#####
##### Large-eddy simulations give w′³ ≈ r w′e′ with r ≈ 1.5 (harness notes/w3-model.md), and the
##### flux of TKE is what the TKE budget leaves over: ∂z (ρ F) = ρ (P + B - u), with P + B the shear
##### and buoyancy production the TKE equation receives and u a sink. The model takes the sink as
##### u = c ω with a shape ω that vanishes with e (ω = e, one turnover rate; ω = ε, the closure's
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
#####     w′³_local = τ₃ (√e (ℂʷᵇ B⁺ + ℂʷˡ (1 - aᶜ) Bˡ⁺) - ℂʷᵍ w′² ∂z w′²) / D,    D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ),    αᴺ = τ₃² N².
#####
##### √e B⁺ stands for the buoyant source w′²b′ (the buoyancy flux carried at the turbulent velocity; it
##### is not closed with w′³), and D ≥ 1 for the feedback of the stratification on the third moment:
##### solving the coupled budgets of w′³, w′²θ′ and w′θ′² exactly gives a determinant that is 1 + c αᴺ to
##### first order and vanishes at finite αᴺ (the poles of `AlgebraicThirdMoment`); D keeps the damping
##### of stable air, αᴺ > 0, and neither amplifies nor vanishes in unstable air. δ is the width in αᴺ over
##### which D passes from one to 1 + ℂʷᴺ αᴺ.
#####
##### The latent source, ℂʷˡ √e (1 - aᶜ) Bˡ⁺, is the part of w′²b′ that condensation makes. The buoyancy is linear in
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
##### stage uses are then computed with that third moment. The closure fields are therefore functions
##### of the prognostic state: nothing is stored between evaluations.
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

    w′³ = r (ℂʷᵉ Fᵉ + ℂʷᵋ Fᵋ) + τ₃ (√e (ℂʷᵇ B⁺ + ℂʷˡ (1 - aᶜ) Bˡ⁺) - ℂʷᵍ w′² ∂z w′²) / D,
    Fʸ(z) = (1 / ρ) ∫₀ᶻ ρ (P + B - uʸ) dz′,    uʸ = cʸ ωʸ,    ωᵉ = e,    ωᵋ = ε = 𝕊ᴰ e^{3/2} / ℓ,
    τ₃ = ℓ / √e,    αᴺ = τ₃² N²,    D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ),    r = 1 + ℂʷʳ αᴺ / (1 + |αᴺ|),

where `P + B` is the shear and buoyancy production the TKE equation receives (the conversions of the
fluxes the closure applies), `B⁺ = max(B, 0)` (a buoyancy sink makes no skewness of its own), `aᶜ` the cloud fraction and
`Bˡ⁺ = max(Bˡ, 0)` the part of the buoyancy flux that the liquid flux carries, `Bˡ = (b_qˡ - b_qᵛ) w′qˡ′`,
both of the distribution without skewness,
`ℓ` the closure's mixing length and `αᴺ` the stratification number at the face (`|ℂʷʳ| < 1`
keeps `r` positive).

The amplitude `cʸ` of each sink is fixed by the budget so that `Fʸ` vanishes at the ground and at
the top of the column: one number for the column with `ℂʷᴸ = Inf`, or a profile exchanged between
neighbouring levels over the length `ℂʷᴸ ℓ` (one tridiagonal solve per sink shape and column). No
depth of a turbulent layer enters. `Fᵋ` is the flux the closure's own dissipation leaves over: where
the TKE equation is steady it is the flux `-Kᵉ ∂z e` the closure applies to `e`. `Fᵉ` is the flux
with one turnover rate for the column. `ℂʷᵀ > 0` adds `ℂʷᵀ Fᵉ` to the flux of TKE in the TKE equation.

The damped local third moment holds the two sources of the stationary budget of `w′³`, the buoyant
one as `√e B⁺` and the gradient production `-w′² ∂z w′²`, relaxed over `τ₃`. The latent source
`√e (1 - aᶜ) Bˡ⁺` is the part of the buoyant source `w′²b′` that condensation makes in a partly cloudy
layer, where the cloudy air is the fastest-rising tail of the distribution and is the only air
heated; it vanishes in saturated air (`aᶜ = 1`) and in clear air (`Bˡ = 0`), and it is positive in a
layer of cumulus, whose net buoyancy flux without skewness is negative. The damping `D ≥ 1`
stands for the suppression of the third moment in stably stratified air; it has no zero, so the
member has no pole. It tends to one in unstable air and to `1 + ℂʷᴺ αᴺ` in stable air, over the width
`δ = stratification_damping_width` in `αᴺ`.

No third moment is stored between evaluations of the closure fields. Within one evaluation the
third moment is diagnosed `passes` times: first from the budget of the distribution without
skewness, then from the budget of the distribution with the third moment of the pass before.
The limiter in the `skewness` slot of [`DoubleGaussian`](@ref) ([`TanhSaturatedSkewness`](@ref), ...)
keeps the result realizable.

The initial values of the coefficients are a joint least-squares fit to coarse-grained large-eddy
simulations of BOMEX, ARM cumulus, ASTEX and DYCOMS (harness `notes/w3-proposal.md`).

$(TYPEDFIELDS)
"""
struct TKEBudgetThirdMoment{N, FT} <: SimplifiedHigherOrderClosures.AbstractSHOCComponent
    "Coefficient of the budget flux with the sink shape `e`"
    ℂʷᵉ :: FT
    "Coefficient of the budget flux with the sink shape `ε`"
    ℂʷᵋ :: FT
    "Exchange length of the sink amplitude in mixing lengths, `Λ = ℂʷᴸ ℓ`; `Inf` for one amplitude per column"
    ℂʷᴸ :: FT
    "Amplitude of the dependence of the ratio `r` on the stratification number, in (-1, 1)"
    ℂʷʳ :: FT
    "Coefficient of the buoyant source `√e B⁺` of the damped local third moment"
    ℂʷᵇ :: FT
    "Coefficient of the latent source `√e (1 - aᶜ) Bˡ⁺` of the damped local third moment"
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

    TKEBudgetThirdMoment{N}(ℂʷᵉ::FT, ℂʷᵋ::FT, ℂʷᴸ::FT, ℂʷʳ::FT, ℂʷᵇ::FT, ℂʷˡ::FT, ℂʷᵍ::FT, ℂʷᴺ::FT, ℂʷᵀ::FT,
                            stratification_damping_width::FT, ℂʷᵒ::FT) where {N, FT} =
        new{N, FT}(ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷᵇ, ℂʷˡ, ℂʷᵍ, ℂʷᴺ, ℂʷᵀ, stratification_damping_width, ℂʷᵒ)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`TKEBudgetThirdMoment`](@ref). The defaults exchange the sink amplitude over eight mixing
lengths, have a constant ratio and no transport of TKE, and of the damped local third moment only
the latent source (`ℂʷᵇ = ℂʷᵍ = 0`: the buoyant source `B⁺` cannot tell heating from below, carried
by updrafts, from cooling from above, carried by downdrafts, so the budget fluxes carry that part
of the third moment; `ℂʷˡ = 8`: the latent source is the heating of the cloudy updrafts of a partly
cloudy layer, which the budget of the distribution without skewness does not hold). They diagnose
the third moment once per evaluation of the closure fields, from the budget of the distribution
without skewness.

`ℂʷˡ = 8` is twice the upper end, 4, of the estimate `3 √(2/3) ρ (1.6 to 2.5)` for a correlation `ρ = 0.5` to
`0.7` of `w` with the saturation excess, and near a fit of `w′³` in the cloud layer of large-eddy
simulations of cumulus (3.4 with their mixing length; harness `notes/cumulus-skewness.md`). It is
an initial value: the closure's own `ρ` and `w′²` differ from the simulations', and the columns
of the note come closer to them with a larger coefficient: with the hand-tuned stability functions
8 lowers the error of the BOMEX and cfsite14_04 columns against 4 and leaves the others nearly
unchanged (harness `notes/hand-tuning.md`).
"""
function TKEBudgetThirdMoment(FT::DataType = Oceananigans.defaults.FloatType;
                              ℂʷᵉ = 0.12, ℂʷᵋ = 0.34, ℂʷᴸ = 8, ℂʷʳ = 0, ℂʷᵇ = 0, ℂʷˡ = 8, ℂʷᵍ = 0, ℂʷᴺ = 0, ℂʷᵀ = 0,
                              stratification_damping_width = 1, passes = 1, ℂʷᵒ = 0)
    abs(ℂʷʳ) < 1 || throw(ArgumentError("ℂʷʳ must lie in (-1, 1) for a positive ratio, got $(ℂʷʳ)."))
    0 <= ℂʷᵀ <= 1 || throw(ArgumentError("ℂʷᵀ must lie in [0, 1], got $(ℂʷᵀ)."))
    ℂʷᴺ >= 0 || throw(ArgumentError("ℂʷᴺ must not be negative (the damping must not vanish), got $(ℂʷᴺ)."))
    ℂʷᴸ > 0 || throw(ArgumentError("ℂʷᴸ must be positive (Inf for one sink amplitude per column), got $(ℂʷᴸ)."))
    stratification_damping_width > 0 ||
        throw(ArgumentError("stratification_damping_width must be positive, got $(stratification_damping_width)."))
    ℂʷᵒ >= 0 || throw(ArgumentError("ℂʷᵒ must not be negative, got $(ℂʷᵒ)."))
    passes isa Integer && passes >= 1 || throw(ArgumentError("passes must be a positive integer, got $(passes)."))
    scalars = (ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷᵇ, ℂʷˡ, ℂʷᵍ, ℂʷᴺ, ℂʷᵀ, stratification_damping_width, ℂʷᵒ)
    return TKEBudgetThirdMoment{Int(passes)}(map(c -> convert(FT, c), scalars)...)
end

# The number of passes is a type parameter, so that the loop over them is unrolled when traced
third_moment_passes(::TKEBudgetThirdMoment{N}) where N = N

SimplifiedHigherOrderClosures.component_constructor(::Type{<:TKEBudgetThirdMoment{N}}) where N = TKEBudgetThirdMoment{N}

Adapt.adapt_structure(to, tm::TKEBudgetThirdMoment{N}) where N =
    TKEBudgetThirdMoment{N}(map(name -> adapt(to, getfield(tm, name)), fieldnames(typeof(tm)))...)

SimplifiedHigherOrderClosures.type_parameter_summary(tm::TKEBudgetThirdMoment{N, FT}) where {N, FT} = string("{", N, ", ", FT, "}")

const TKEBudgetDistribution = DoubleGaussian{<:Any, <:Any, <:TKEBudgetThirdMoment}

#####
##### Fields: the two sinks, the two fluxes, the third moment before its limiter, the transport tendency of TKE and a work array
#####

third_moment_fields(grid, third_moment) = NamedTuple()

third_moment_fields(grid, ::TKEBudgetThirdMoment) =
    (Fᵉ = CenterField(grid), Fᵋ = CenterField(grid), uᵉ = CenterField(grid), uᵋ = CenterField(grid),
     w′³ᵇ = ZFaceField(grid), Tᵉ = CenterField(grid), budget_work = CenterField(grid),
     NamedTuple(pass_name(name) => ZFaceField(grid) for name in PASS_CLOSURE_FIELD_NAMES)...)

# The closure fields that a pass recomputes (the stability, the mixing lengths and the diffusivities). A pass
# writes them into fields of its own, so that those of the prognostic state are never overwritten and need not be
# kept and put back.
const PASS_CLOSURE_FIELD_NAMES = (:N², :ℓᵐ, :ℓˢ, :Kᵘ, :Kᶜ, :Kᵉ)
pass_name(name) = Symbol(:pass_, name)

# The closure fields of a pass: those of the prognostic state with the stability, the mixing lengths and the
# diffusivities replaced by the pass's own (the per-tracer lookup of the diffusivities included)
function pass_closure_fields(closure_fields::TKEClosureFields, cl)
    Kᵘ, Kᶜ, Kᵉ = cl.pass_Kᵘ, cl.pass_Kᶜ, cl.pass_Kᵉ
    tracer_diffusivities = map(K -> K === closure_fields.Kᵉ ? Kᵉ : Kᶜ, closure_fields.tupled_tracer_diffusivities)
    return TKEClosureFields(Kᵘ, Kᶜ, Kᵉ, cl.pass_N², cl.pass_ℓᵐ, cl.pass_ℓˢ, closure_fields.Lᵉ, tracer_diffusivities,
                            closure_fields.tupled_implicit_linear_coefficients, closure_fields.subgrid_cloud)
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
@inline function sink_exchange_coefficientᶜᶜᶠ(i, j, k, grid, ℂʷᴸ, ρ, ℓᵐ)
    Λ = @inbounds ℂʷᴸ * ℓᵐ[i, j, k]
    return 2 * ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Λ * Λ / Δzᶜᶜᶠ(i, j, k, grid)
end

# The two sinks of the column at once, `uᵉ` and `uᵋ` holding their shapes on entry; the matrices share the exchange
# coefficients and the right-hand side ρ Δz (P + B). `workᵉ` and `workᵋ` hold the eliminated upper diagonals.
@inline function balance_sinks!(uᵉ, uᵋ, workᵉ, workᵋ, i, j, grid, ℂʷᴸ, cl, ρ, ℓᵐ)
    Nz = size(grid, 3)
    FT = grid_float_type(grid)
    tiny = sqrt(floatmin(FT))
    @inbounds if isfinite(ℂʷᴸ)
        # Thomas algorithm; each matrix is diagonally dominant by columns
        ωᵉ⁻ = zero(FT)
        ωᵋ⁻ = zero(FT)
        c′ᵉ = zero(FT)
        d′ᵉ = zero(FT)
        c′ᵋ = zero(FT)
        d′ᵋ = zero(FT)
        a⁻ = zero(FT) # the exchange coefficient at the face below
        for k in 1:Nz
            ωᵉ = uᵉ[i, j, k]
            ωᵋ = uᵋ[i, j, k]
            m = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            diagonalᵉ = m
            diagonalᵋ = m
            lowerᵉ = zero(FT)
            lowerᵋ = zero(FT)
            upperᵉ = zero(FT)
            upperᵋ = zero(FT)
            if k > 1
                sᵉ = ωᵉ + ωᵉ⁻ + tiny
                sᵋ = ωᵋ + ωᵋ⁻ + tiny
                diagonalᵉ += a⁻ * ωᵉ⁻ / sᵉ
                diagonalᵋ += a⁻ * ωᵋ⁻ / sᵋ
                lowerᵉ = -a⁻ * ωᵉ / sᵉ
                lowerᵋ = -a⁻ * ωᵋ / sᵋ
            end
            if k < Nz
                a = sink_exchange_coefficientᶜᶜᶠ(i, j, k + 1, grid, ℂʷᴸ, ρ, ℓᵐ)
                ωᵉ⁺ = uᵉ[i, j, k+1]
                ωᵋ⁺ = uᵋ[i, j, k+1]
                sᵉ = ωᵉ + ωᵉ⁺ + tiny
                sᵋ = ωᵋ + ωᵋ⁺ + tiny
                diagonalᵉ += a * ωᵉ⁺ / sᵉ
                diagonalᵋ += a * ωᵋ⁺ / sᵋ
                upperᵉ = -a * ωᵉ / sᵉ
                upperᵋ = -a * ωᵋ / sᵋ
                a⁻ = a
            end
            source = m * (cl.P[i, j, k] + cl.B[i, j, k])
            pivotᵉ = diagonalᵉ - lowerᵉ * c′ᵉ
            pivotᵋ = diagonalᵋ - lowerᵋ * c′ᵋ
            d′ᵉ = (source - lowerᵉ * d′ᵉ) / pivotᵉ
            d′ᵋ = (source - lowerᵋ * d′ᵋ) / pivotᵋ
            c′ᵉ = upperᵉ / pivotᵉ
            c′ᵋ = upperᵋ / pivotᵋ
            workᵉ[i, j, k] = c′ᵉ
            workᵋ[i, j, k] = c′ᵋ
            uᵉ[i, j, k] = d′ᵉ
            uᵋ[i, j, k] = d′ᵋ
            ωᵉ⁻ = ωᵉ
            ωᵋ⁻ = ωᵋ
        end
        for k in Nz-1:-1:1
            uᵉ[i, j, k] -= workᵉ[i, j, k] * uᵉ[i, j, k+1]
            uᵋ[i, j, k] -= workᵋ[i, j, k] * uᵋ[i, j, k+1]
        end
    else
        Π = zero(FT)
        Ωᵉ = zero(FT)
        Ωᵋ = zero(FT)
        for k in 1:Nz
            m = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            Π += m * (cl.P[i, j, k] + cl.B[i, j, k])
            Ωᵉ += m * uᵉ[i, j, k]
            Ωᵋ += m * uᵋ[i, j, k]
        end
        cᵉ = Π / (Ωᵉ + tiny)
        cᵋ = Π / (Ωᵋ + tiny)
        for k in 1:Nz
            uᵉ[i, j, k] *= cᵉ
            uᵋ[i, j, k] *= cᵋ
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

# D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ) ≥ 1: one in unstable air, 1 + ℂʷᴺ αᴺ in stable air, with slope in (0, ℂʷᴺ)
@inline stratification_damping(ℂʷᴺ, δ, αᴺ) = 1 + ℂʷᴺ * δ * smooth_positive_part(αᴺ / δ)

# τ₃ (ℂʷᵇ √e B⁺ - ℂʷᵍ w′² ∂z w′²) / D at the interior face k; w′² ∂z w′² = ∂z (w′²)² / 2 is differenced across the
# face, between the centers k - 1 and k (compact, as `shoc.F90` differences center quantities for its interface
# third moment). With a `HeatingBudgetFlux` the buoyant source is signed by origin: of B⁺, the part
# Gᶜ = min(Gᑫᶜ, B⁺) that the flux driven by cooling converts is carried by downdrafts, and the source is
# ℂʷᵇ (B⁺ - Gᶜ) - ℂʷᑫ Gᶜ.
@inline function damped_local_third_moment(i, j, k, grid, tm, w′², τ, u, αᴺ, B, Gᑫᶜ = zero(B), ℂʷᑫ = zero(B), latent_source = zero(B))
    w′²ᵘ = @inbounds w′²[i, j, k]
    w′²ˡ = @inbounds w′²[i, j, k-1]
    w′²∂zw′² = (w′²ᵘ * w′²ᵘ - w′²ˡ * w′²ˡ) / (2 * Δzᶜᶜᶠ(i, j, k, grid))
    B⁺ = max(0, B)
    Gᶜ = min(Gᑫᶜ, B⁺)
    buoyant_source = tm.ℂʷᵇ * (B⁺ - Gᶜ) - ℂʷᑫ * Gᶜ + tm.ℂʷˡ * latent_source
    return τ * (u * buoyant_source - tm.ℂʷᵍ * w′²∂zw′²) / stratification_damping(tm.ℂʷᴺ, tm.stratification_damping_width, αᴺ)
end

#####
##### The budget of one evaluation: sinks, fluxes, the transport of TKE and, with `diagnose = Val(true)`, the third moment
#####

@kernel function _compute_tke_budget_flux!(cl, grid, closure, closure_fields, e, velocities, ρ, diagnose)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    tm = closure_ij.subgrid_distribution.third_moment
    ℓᵐ = closure_fields.ℓᵐ
    @inbounds for k in 1:size(grid, 3)
        # With ℂʷᵒ > 0 the shape reaches one cell above the turbulence (the overshoot); the cell below the first is none
        e⁻ = ifelse(k > 1, max(e[i, j, max(k - 1, 1)], zero(grid_float_type(grid))), zero(grid_float_type(grid)))
        cl.uᵉ[i, j, k] = max(e[i, j, k], zero(grid_float_type(grid))) + tm.ℂʷᵒ * e⁻
        cl.uᵋ[i, j, k] = dissipationᶜᶜᶜ(i, j, k, grid, closure_ij, e, velocities, closure_fields)
    end
    # Fᵋ holds the eliminated upper diagonal of the second solve until the fluxes overwrite it
    balance_sinks!(cl.uᵉ, cl.uᵋ, cl.budget_work, cl.Fᵋ, i, j, grid, tm.ℂʷᴸ, cl, ρ, ℓᵐ)
    N² = closure_fields.N²
    # The fluxes F at the centers: the integral of ρ (P + B - u) from the ground, the cell of the center counted half
    belowᵉ = zero(grid_float_type(grid))
    belowᵋ = zero(grid_float_type(grid))
    @inbounds for k in 1:size(grid, 3)
        # The third moment at the face k, below this center, from the fluxes there (the integral of the cells below)
        k > 1 && diagnose_face_third_moment!(diagnose, cl, i, j, k, grid, tm, closure_ij, e, ℓᵐ, N², ρ, belowᵉ, belowᵋ)

        ρΔz = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
        rᵉ = ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵉ[i, j, k])
        rᵋ = ρΔz * (cl.P[i, j, k] + cl.B[i, j, k] - cl.uᵋ[i, j, k])
        cl.Fᵉ[i, j, k] = (belowᵉ + rᵉ / 2) / ρ[i, j, k]
        cl.Fᵋ[i, j, k] = (belowᵋ + rᵋ / 2) / ρ[i, j, k]
        belowᵉ += rᵉ
        belowᵋ += rᵋ

        cl.Tᵉ[i, j, k] = tm.ℂʷᵀ * (cl.uᵉ[i, j, k] - cl.P[i, j, k] - cl.B[i, j, k])
    end
end

# The third moment at the interior face k, with ℓ, √e and N² of the face; one stratification number αᴺ = τ₃² N² for
# the ratio and for the damping. The fluxes at the face are the integrals ρF of the cells below it over the density
# at the face; the buoyancy production and the latent source are averaged from the two centers. Only in the passes
# of `diagnose_third_moment!` (`Val(true)`). Diagnosed at the faces, as `shoc.F90` diagnoses its third moment, and
# interpolated to the centers with the other moments.
@inline diagnose_face_third_moment!(::Val{false}, args...) = nothing

@inline function diagnose_face_third_moment!(::Val{true}, cl, i, j, k, grid, tm, closure_ij, e, ℓᵐ, N², ρ, ρFᵉ, ρFᵋ)
    @inbounds begin
        ρᶠ = (ρ[i, j, k] + ρ[i, j, k-1]) / 2
        ℓ = ℓᵐ[i, j, k]
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure_ij, e)
        τ = ℓ / u
        αᴺ = τ * τ * N²[i, j, k]
        r = 1 + tm.ℂʷʳ * αᴺ / (1 + abs(αᴺ))
        ℂʷᑫ = cooling_third_moment_coefficient(closure_ij.heating_budget_flux, grid_float_type(grid))
        B = (cl.B[i, j, k] + cl.B[i, j, k-1]) / 2
        Gᑫᶜ = (cooling_flux_conversion(cl, i, j, k) + cooling_flux_conversion(cl, i, j, k-1)) / 2
        latent = ((1 - cl.aᶜ[i, j, k]) * max(cl.Bˡ[i, j, k], 0) + (1 - cl.aᶜ[i, j, k-1]) * max(cl.Bˡ[i, j, k-1], 0)) / 2
        cl.w′³ᵇ[i, j, k] = r * (tm.ℂʷᵉ * ρFᵉ + tm.ℂʷᵋ * ρFᵋ) / ρᶠ +
                           damped_local_third_moment(i, j, k, grid, tm, cl.w′², τ, u, αᴺ, B, Gᑫᶜ, ℂʷᑫ, latent)
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
    compute_heating_budget_flux!(fields, heating_budget_flux(closure), closure_fields, closure, model, tracers)
    passes = third_moment_passes(distribution.third_moment)
    for pass in 1:passes
        start_pass_closure_fields!(pass_fields, closure_fields, model.grid)
        compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
        solve_static_stability!(closure.static_stability, fields.buoyancy_flux, fields, pass_fields, closure, model, tracers, parameters)
        compute_applied_flux_conversions!(fields, pass_fields, closure, model, tracers)
        compute_tke_budget_flux!(fields, pass_fields, closure, model, tracers, Val(true))
    end
    return nothing
end

# The buoyancy derivatives depend on the prognostic state alone: those of `diagnose_third_moment!` stand,
# and so does the heating budget flux
compute_subgrid_cloud_buoyancy_derivatives!(fields, ::TKEBudgetDistribution, model) = nothing
compute_subgrid_cloud_heating_budget_flux!(fields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers) = nothing

# After the conversions of the applied fluxes the stage uses: their budget, for the transport of TKE and as a diagnostic
compute_tke_budget!(fields::SubgridCloudFields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers) =
    compute_tke_budget_flux!(fields, closure_fields, closure, model, tracers, Val(false))

#####
##### The third moment at the face k: the one diagnosed from the budget
#####

@inline face_third_moment_diagnosis(tm::TKEBudgetThirdMoment, i, j, k, d, cl, c³, algebraic_arguments) =
    @inbounds cl.w′³ᵇ[i, j, k]

#####
##### The transport of TKE by the budget flux, in the stage tendency of ρe
#####

@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, distribution, fields) = zero(grid)
@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, ::TKEBudgetDistribution, fields) = @inbounds fields.cloud.Tᵉ[i, j, k]
