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
#####     w′³_local = τ₃ (ℂʷᵇ √e B⁺ - ℂʷᵍ w′² ∂z w′²) / D,    D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ),    αᴺ = τ₃² N².
#####
##### √e B⁺ stands for the buoyant source w′²b′ (the buoyancy flux carried at the turbulent velocity; it
##### is not closed with w′³), and D ≥ 1 for the feedback of the stratification on the third moment:
##### solving the coupled budgets of w′³, w′²θ′ and w′θ′² exactly gives a determinant that is 1 + c αᴺ to
##### first order and vanishes at finite αᴺ (the poles of `AlgebraicThirdMoment`); D keeps the damping
##### of stable air, αᴺ > 0, and neither amplifies nor vanishes in unstable air. δ is the width in αᴺ over
##### which D passes from one to 1 + ℂʷᴺ αᴺ.
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

The third moment of vertical velocity at the cell centers from the vertical integral of the
budget of turbulent kinetic energy, with a damped local third moment,

    w′³ = r (ℂʷᵉ Fᵉ + ℂʷᵋ Fᵋ) + τ₃ (ℂʷᵇ √e B⁺ - ℂʷᵍ w′² ∂z w′²) / D,
    Fʸ(z) = (1 / ρ) ∫₀ᶻ ρ (P + B - uʸ) dz′,    uʸ = cʸ ωʸ,    ωᵉ = e,    ωᵋ = ε = 𝕊ᴰ e^{3/2} / ℓ,
    τ₃ = ℓ / √e,    αᴺ = τ₃² N²,    D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ),    r = 1 + ℂʷʳ αᴺ / (1 + |αᴺ|),

where `P + B` is the shear and buoyancy production the TKE equation receives (the conversions of the
fluxes the closure applies), `B⁺ = max(B, 0)` (a buoyancy sink makes no skewness of its own),
`ℓ` the closure's mixing length and `αᴺ` the stratification number at the cell center (`|ℂʷʳ| < 1`
keeps `r` positive).

The amplitude `cʸ` of each sink is fixed by the budget so that `Fʸ` vanishes at the ground and at
the top of the column: one number for the column with `ℂʷᴸ = Inf`, or a profile exchanged between
neighbouring levels over the length `ℂʷᴸ ℓ` (one tridiagonal solve per sink shape and column). No
depth of a turbulent layer enters. `Fᵋ` is the flux the closure's own dissipation leaves over: where
the TKE equation is steady it is the flux `-Kᵉ ∂z e` the closure applies to `e`. `Fᵉ` is the flux
with one turnover rate for the column. `ℂʷᵀ > 0` adds `ℂʷᵀ Fᵉ` to the flux of TKE in the TKE equation.

The damped local third moment holds the two sources of the stationary budget of `w′³`, the buoyant
one as `√e B⁺` and the gradient production `-w′² ∂z w′²`, relaxed over `τ₃`. The damping `D ≥ 1`
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
    "Coefficient of the gradient production `-w′² ∂z w′²` of the damped local third moment"
    ℂʷᵍ :: FT
    "Coefficient of the stratification number in the damping `D = 1 + ℂʷᴺ δ smooth_positive_part(αᴺ / δ)` of the local third moment; not negative"
    ℂʷᴺ :: FT
    "Share of the production that the budget flux with the sink shape `e` transports in the TKE equation, in [0, 1]"
    ℂʷᵀ :: FT
    "Width `δ` in the stratification number over which the damping passes from one to `1 + ℂʷᴺ αᴺ`; positive"
    stratification_damping_width :: FT

    TKEBudgetThirdMoment{N}(ℂʷᵉ::FT, ℂʷᵋ::FT, ℂʷᴸ::FT, ℂʷʳ::FT, ℂʷᵇ::FT, ℂʷᵍ::FT, ℂʷᴺ::FT, ℂʷᵀ::FT,
                            stratification_damping_width::FT) where {N, FT} =
        new{N, FT}(ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷᵇ, ℂʷᵍ, ℂʷᴺ, ℂʷᵀ, stratification_damping_width)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`TKEBudgetThirdMoment`](@ref). The defaults exchange the sink amplitude over eight mixing
lengths, have a constant ratio, no transport of TKE and no damped local third moment
(`ℂʷᵇ = ℂʷᵍ = 0`: the buoyant source `B⁺` cannot tell heating from below, carried by updrafts, from
cooling from above, carried by downdrafts, so the budget fluxes carry the third moment), and diagnose the third moment once per
evaluation of the closure fields, from the budget of the distribution without skewness.
"""
function TKEBudgetThirdMoment(FT::DataType = Oceananigans.defaults.FloatType;
                              ℂʷᵉ = 0.12, ℂʷᵋ = 0.34, ℂʷᴸ = 8, ℂʷʳ = 0, ℂʷᵇ = 0, ℂʷᵍ = 0, ℂʷᴺ = 0, ℂʷᵀ = 0,
                              stratification_damping_width = 1, passes = 1)
    abs(ℂʷʳ) < 1 || throw(ArgumentError("ℂʷʳ must lie in (-1, 1) for a positive ratio, got $(ℂʷʳ)."))
    0 <= ℂʷᵀ <= 1 || throw(ArgumentError("ℂʷᵀ must lie in [0, 1], got $(ℂʷᵀ)."))
    ℂʷᴺ >= 0 || throw(ArgumentError("ℂʷᴺ must not be negative (the damping must not vanish), got $(ℂʷᴺ)."))
    ℂʷᴸ > 0 || throw(ArgumentError("ℂʷᴸ must be positive (Inf for one sink amplitude per column), got $(ℂʷᴸ)."))
    stratification_damping_width > 0 ||
        throw(ArgumentError("stratification_damping_width must be positive, got $(stratification_damping_width)."))
    passes isa Integer && passes >= 1 || throw(ArgumentError("passes must be a positive integer, got $(passes)."))
    scalars = (ℂʷᵉ, ℂʷᵋ, ℂʷᴸ, ℂʷʳ, ℂʷᵇ, ℂʷᵍ, ℂʷᴺ, ℂʷᵀ, stratification_damping_width)
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
     w′³ᵇ = CenterField(grid), Tᵉ = CenterField(grid), budget_work = CenterField(grid),
     NamedTuple(saved_name(name) => ZFaceField(grid) for name in SAVED_CLOSURE_FIELD_NAMES)...)

# The closure fields of the prognostic state that the passes overwrite: kept before them and put back after
const SAVED_CLOSURE_FIELD_NAMES = (:N², :ℓᵐ, :ℓˢ, :Kᵘ, :Kᶜ, :Kᵉ)
saved_name(name) = Symbol(:saved_, name)

function save_closure_fields!(cloud, closure_fields)
    for name in SAVED_CLOSURE_FIELD_NAMES
        parent(getproperty(cloud, saved_name(name))) .= parent(getproperty(closure_fields, name))
    end
    return nothing
end

function restore_closure_fields!(closure_fields, cloud)
    for name in SAVED_CLOSURE_FIELD_NAMES
        parent(getproperty(closure_fields, name)) .= parent(getproperty(cloud, saved_name(name)))
    end
    return nothing
end

#####
##### The sink that balances the budget, in place: `u` holds the shape ω on entry and the sink on exit
#####

# 2 ρ Λ² / Δz at the face k, between the centers k - 1 and k
@inline function sink_exchange_coefficientᶜᶜᶠ(i, j, k, grid, ℂʷᴸ, ρ, ℓᵐ)
    Λ = @inbounds ℂʷᴸ * ℓᵐ[i, j, k]
    return 2 * ℑzᵃᵃᶠ(i, j, k, grid, ρ) * Λ * Λ / Δzᶜᶜᶠ(i, j, k, grid)
end

@inline function balance_sink!(u, work, i, j, grid, ℂʷᴸ, cl, ρ, ℓᵐ)
    Nz = size(grid, 3)
    FT = eltype(grid)
    tiny = sqrt(floatmin(FT))
    @inbounds if isfinite(ℂʷᴸ)
        # Thomas algorithm; the matrix is diagonally dominant by columns
        ω⁻ = zero(FT)
        c′ = zero(FT)
        d′ = zero(FT)
        for k in 1:Nz
            ω = u[i, j, k]
            m = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid)
            diagonal = m
            lower = zero(FT)
            upper = zero(FT)
            if k > 1
                a = sink_exchange_coefficientᶜᶜᶠ(i, j, k, grid, ℂʷᴸ, ρ, ℓᵐ)
                s = ω + ω⁻ + tiny
                diagonal += a * ω⁻ / s
                lower = -a * ω / s
            end
            if k < Nz
                ω⁺ = u[i, j, k+1]
                a = sink_exchange_coefficientᶜᶜᶠ(i, j, k + 1, grid, ℂʷᴸ, ρ, ℓᵐ)
                s = ω + ω⁺ + tiny
                diagonal += a * ω⁺ / s
                upper = -a * ω / s
            end
            pivot = diagonal - lower * c′
            d′ = (m * (cl.P[i, j, k] + cl.B[i, j, k]) - lower * d′) / pivot
            c′ = upper / pivot
            work[i, j, k] = c′
            u[i, j, k] = d′
            ω⁻ = ω
        end
        for k in Nz-1:-1:1
            u[i, j, k] -= work[i, j, k] * u[i, j, k+1]
        end
    else
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

# F at the centers: the integral of ρ (P + B - u) from the ground, the cell of the center counted half
@inline function budget_flux!(F, u, i, j, grid, cl, ρ)
    below = zero(eltype(grid))
    @inbounds for k in 1:size(grid, 3)
        r = ρ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid) * (cl.P[i, j, k] + cl.B[i, j, k] - u[i, j, k])
        F[i, j, k] = (below + r / 2) / ρ[i, j, k]
        below += r
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

# τ₃ (ℂʷᵇ √e B⁺ - ℂʷᵍ w′² ∂z w′²) / D at the center k; w′² ∂z w′² = ∂z (w′²)² / 2 is differenced across the
# neighbouring centers (one-sided at the two ends of the column)
@inline function damped_local_third_moment(i, j, k, grid, tm, w′², τ, u, αᴺ, B)
    Nz = size(grid, 3)
    kᵘ = min(k + 1, Nz)
    kˡ = max(k - 1, 1)
    ccc = (Center(), Center(), Center())
    Δz = znode(i, j, kᵘ, grid, ccc...) - znode(i, j, kˡ, grid, ccc...)
    w′²ᵘ = @inbounds w′²[i, j, kᵘ]
    w′²ˡ = @inbounds w′²[i, j, kˡ]
    w′²∂zw′² = ifelse(kᵘ > kˡ, (w′²ᵘ * w′²ᵘ - w′²ˡ * w′²ˡ) / (2 * Δz), zero(Δz))
    return τ * (tm.ℂʷᵇ * u * max(0, B) - tm.ℂʷᵍ * w′²∂zw′²) / stratification_damping(tm.ℂʷᴺ, tm.stratification_damping_width, αᴺ)
end

#####
##### The budget of one evaluation: sinks, fluxes, the transport of TKE and, with `diagnose`, the third moment
#####

@kernel function _compute_tke_budget_flux!(cl, grid, closure, closure_fields, e, velocities, ρ, diagnose)
    i, j = @index(Global, NTuple)
    closure_ij = getclosure(i, j, closure)
    tm = closure_ij.subgrid_distribution.third_moment
    ℓᵐ = closure_fields.ℓᵐ
    @inbounds for k in 1:size(grid, 3)
        cl.uᵉ[i, j, k] = max(e[i, j, k], zero(eltype(grid)))
        cl.uᵋ[i, j, k] = dissipationᶜᶜᶜ(i, j, k, grid, closure_ij, e, velocities, closure_fields)
    end
    balance_sink!(cl.uᵉ, cl.budget_work, i, j, grid, tm.ℂʷᴸ, cl, ρ, ℓᵐ)
    balance_sink!(cl.uᵋ, cl.budget_work, i, j, grid, tm.ℂʷᴸ, cl, ρ, ℓᵐ)
    budget_flux!(cl.Fᵉ, cl.uᵉ, i, j, grid, cl, ρ)
    budget_flux!(cl.Fᵋ, cl.uᵋ, i, j, grid, cl, ρ)
    N² = closure_fields.N²
    @inbounds for k in 1:size(grid, 3)
        cl.Tᵉ[i, j, k] = tm.ℂʷᵀ * (cl.uᵉ[i, j, k] - cl.P[i, j, k] - cl.B[i, j, k])

        # The third moment at the center, with ℓ, √e and N² as the distribution forms them there; one
        # stratification number αᴺ = τ₃² N² for the ratio and for the damping
        ℓ = (ℓᵐ[i, j, k] + ℓᵐ[i, j, k+1]) / 2
        u = turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure_ij, e)
        τ = ℓ / u
        αᴺ = τ * τ * (N²[i, j, k] + N²[i, j, k+1]) / 2
        r = 1 + tm.ℂʷʳ * αᴺ / (1 + abs(αᴺ))
        w′³ = r * (tm.ℂʷᵉ * cl.Fᵉ[i, j, k] + tm.ℂʷᵋ * cl.Fᵋ[i, j, k]) +
              damped_local_third_moment(i, j, k, grid, tm, cl.w′², τ, u, αᴺ, cl.B[i, j, k])
        cl.w′³ᵇ[i, j, k] = ifelse(diagnose, w′³, cl.w′³ᵇ[i, j, k])
    end
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
# third moment from their budget. The stability, mixing length and diffusivities of the prognostic state (whose
# stability reads the distribution of the previous update) are saved before the passes, every pass after the first
# starts from them, and they are put back after the last, so the final evaluation starts from them too.
function diagnose_third_moment!(fields::SubgridCloudFields, distribution::TKEBudgetDistribution, closure_fields, closure, model, tracers, parameters)
    save_closure_fields!(fields.cloud, closure_fields)
    fill!(parent(fields.cloud.w′³ᵇ), 0)
    compute_buoyancy_derivatives!(fields, model)
    passes = third_moment_passes(distribution.third_moment)
    for pass in 1:passes
        pass > 1 && restore_closure_fields!(closure_fields, fields.cloud)
        compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
        solve_static_stability!(closure.static_stability, fields.buoyancy_flux, fields, closure_fields, closure, model, tracers, parameters)
        compute_applied_flux_conversions!(fields, closure_fields, closure, model, tracers)
        compute_tke_budget_flux!(fields, closure_fields, closure, model, tracers, true)
    end
    restore_closure_fields!(closure_fields, fields.cloud)
    return nothing
end

# After the conversions of the applied fluxes the stage uses: their budget, for the transport of TKE and as a diagnostic
compute_tke_budget!(fields::SubgridCloudFields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers) =
    compute_tke_budget_flux!(fields, closure_fields, closure, model, tracers, false)

#####
##### The third moment at the center k: the one diagnosed from the budget
#####

@inline center_third_moment_diagnosis(tm::TKEBudgetThirdMoment, i, j, k, d, cl, c³, ℓ, u, algebraic_arguments) =
    @inbounds cl.w′³ᵇ[i, j, k]

#####
##### The transport of TKE by the budget flux, in the stage tendency of ρe
#####

@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, distribution, fields) = zero(grid)
@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, ::TKEBudgetDistribution, fields) = @inbounds fields.cloud.Tᵉ[i, j, k]
