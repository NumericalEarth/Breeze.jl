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
##### The budget is formed at the end of each evaluation of the closure fields, from the conversions
##### of the applied fluxes of that evaluation (`cl.P`, `cl.B`: what the TKE equation receives), and the
##### distribution of the next evaluation reads the stored third moment: w′³ lags the budget by one
##### evaluation (one Runge–Kutta stage), because the buoyancy production needs the distribution.
##### The stored third moment relaxes toward the diagnosed one on the time ℂʷᵗ ℓ / √e, the relaxation
##### of the third-moment budget (∂t w′³ = ... - w′³ / τ₃, τ₃ ≈ 0.5 to 0.9 ℓ / √e in large-eddy
##### simulations), by backward Euler over a third of the last time step per evaluation; ℂʷᵗ = 0
##### stores the diagnosed value. The closure fields therefore carry memory: the stored w′³ is
##### state that `update_state!` does not rebuild from the prognostic fields.
#####
##### With ℂʷᵀ > 0 the same flux, with the sink shape e, is also a transport of TKE in the TKE
##### equation: ∂t e += ℂʷᵀ (uᵉ - (P + B)). It moves the share ℂʷᵀ of the production from where it is
##### produced to where e is, conserves the column's TKE exactly, and leaves the stage tendency
##### (1 - ℂʷᵀ)(P + B⁺) - ℂʷᵀ B⁻ + ℂʷᵀ uᵉ non-negative for 0 ≤ ℂʷᵀ ≤ 1 where uᵉ ≥ 0.
#####

"""
$(TYPEDEF)

The third moment of vertical velocity at the cell centers from the vertical integral of the
budget of turbulent kinetic energy,

    w′³ = r (ℂʷᵉ Fᵉ + ℂʷᵋ Fᵋ) + ℂʷᵇ ℓ B⁺ + ℂʷᵃ w′³ₐ,
    Fʸ(z) = (1 / ρ) ∫₀ᶻ ρ (P + B - uʸ) dz′,    uʸ = cʸ ωʸ,    ωᵉ = e,    ωᵋ = ε = 𝕊ᴰ e^{3/2} / ℓ,

where `P + B` is the shear and buoyancy production the TKE equation receives (the conversions of the
fluxes the closure applies), `B⁺ = max(B, 0)` (a buoyancy sink makes no skewness of its own),
`ℓ` the closure's mixing length, `r = 1 + ℂʷʳ αᴺ / (1 + |αᴺ|)` with
`αᴺ = ℓ² N² / e` a bounded function of the stratification number (`|ℂʷʳ| < 1` keeps it positive),
and `w′³ₐ` the third moment of `algebraic` ([`AlgebraicThirdMoment`](@ref), or `nothing` for none).
The amplitude `cʸ` of each sink is fixed by the budget so that `Fʸ` vanishes at the ground and at
the top of the column: one number for the column with `ℂʷᴸ = Inf`, or a profile exchanged between
neighbouring levels over the length `ℂʷᴸ ℓ` (one tridiagonal solve per sink shape and column). No
depth of a turbulent layer enters. The limiter in the `skewness` slot of [`DoubleGaussian`](@ref)
([`TanhSaturatedSkewness`](@ref), ...) keeps the result realizable.

The first three terms are stored and relax toward their diagnosed value on the time `ℂʷᵗ ℓ / √e`
(`ℂʷᵗ = 0`: no relaxation).

`Fᵋ` is the flux the closure's own dissipation leaves over: where the TKE equation is steady it
is the flux `-Kᵉ ∂z e` the closure applies to `e`. `Fᵉ` is the flux with one turnover rate for the
column. `ℂʷᵀ > 0` adds `ℂʷᵀ Fᵉ` to the flux of TKE in the TKE equation.

The initial values of `ℂʷᵉ`, `ℂʷᵋ` and `ℂʷᵇ` are a joint least-squares fit to coarse-grained
large-eddy simulations of BOMEX, ARM cumulus, ASTEX and DYCOMS (harness `notes/w3-proposal.md`).

$(TYPEDFIELDS)
"""
struct TKEBudgetThirdMoment{FT, A} <: SimplifiedHigherOrderClosures.AbstractSHOCComponent
    "Coefficient of the budget flux with the sink shape `e`"
    ℂʷᵉ :: FT
    "Coefficient of the budget flux with the sink shape `ε`"
    ℂʷᵋ :: FT
    "Coefficient of `ℓ B⁺`, the mixing length times the positive part of the buoyancy production"
    ℂʷᵇ :: FT
    "Exchange length of the sink amplitude in mixing lengths, `Λ = ℂʷᴸ ℓ`; `Inf` for one amplitude per column"
    ℂʷᴸ :: FT
    "Amplitude of the dependence of the ratio `r` on the stratification number, in (-1, 1)"
    ℂʷʳ :: FT
    "Relaxation time of the stored third moment in turnover times, `ℂʷᵗ ℓ / √e`; 0 for none"
    ℂʷᵗ :: FT
    "Coefficient of the third moment of `algebraic`"
    ℂʷᵃ :: FT
    "Share of the production that the budget flux with the sink shape `e` transports in the TKE equation, in [0, 1]"
    ℂʷᵀ :: FT
    "A local algebraic third moment ([`AlgebraicThirdMoment`](@ref)) added with the coefficient `ℂʷᵃ`, or `nothing`"
    algebraic :: A
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return a [`TKEBudgetThirdMoment`](@ref). The defaults exchange the sink amplitude over eight mixing
lengths and have a constant ratio, no algebraic member and no transport of TKE.
"""
function TKEBudgetThirdMoment(FT::DataType = Oceananigans.defaults.FloatType;
                              ℂʷᵉ = 0.12, ℂʷᵋ = 0.34, ℂʷᵇ = 0.6, ℂʷᴸ = 8, ℂʷʳ = 0, ℂʷᵗ = 0.6, ℂʷᵃ = 0, ℂʷᵀ = 0,
                              algebraic = nothing)
    abs(ℂʷʳ) < 1 || throw(ArgumentError("ℂʷʳ must lie in (-1, 1) for a positive ratio, got $(ℂʷʳ)."))
    0 <= ℂʷᵀ <= 1 || throw(ArgumentError("ℂʷᵀ must lie in [0, 1], got $(ℂʷᵀ)."))
    ℂʷᵗ >= 0 || throw(ArgumentError("ℂʷᵗ must not be negative, got $(ℂʷᵗ)."))
    ℂʷᴸ > 0 || throw(ArgumentError("ℂʷᴸ must be positive (Inf for one sink amplitude per column), got $(ℂʷᴸ)."))
    algebraic = isnothing(algebraic) ? nothing : convert_eltype(FT, algebraic)
    return TKEBudgetThirdMoment(map(c -> convert(FT, c), (ℂʷᵉ, ℂʷᵋ, ℂʷᵇ, ℂʷᴸ, ℂʷʳ, ℂʷᵗ, ℂʷᵃ, ℂʷᵀ))..., algebraic)
end

Adapt.@adapt_structure TKEBudgetThirdMoment

# `nothing` holds no scalars (the generic component functions would try to construct it)
SimplifiedHigherOrderClosures.map_scalars(f, ::Nothing) = nothing

const TKEBudgetDistribution = DoubleGaussian{<:Any, <:Any, <:TKEBudgetThirdMoment}

#####
##### Fields: the two sinks, the two fluxes, the stored third moment, the transport tendency of TKE and a work array
#####

third_moment_fields(grid, third_moment) = NamedTuple()

third_moment_fields(grid, ::TKEBudgetThirdMoment) =
    (Fᵉ = CenterField(grid), Fᵋ = CenterField(grid), uᵉ = CenterField(grid), uᵋ = CenterField(grid),
     w′³ᵇ = CenterField(grid), Tᵉ = CenterField(grid), budget_work = CenterField(grid))

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

@kernel function _compute_tke_budget_flux!(cl, grid, closure, closure_fields, e, velocities, ρ, Δt)
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

        # The third moment of the budget at the center, with ℓ, √e and N² as the distribution forms them there,
        # relaxed from its stored value on the time ℂʷᵗ ℓ / √e (backward Euler over Δt)
        ℓ = (ℓᵐ[i, j, k] + ℓᵐ[i, j, k+1]) / 2
        τ = ℓ / turbulent_velocityᶜᶜᶜ(i, j, k, grid, closure_ij, e)
        αᴺ = τ * τ * (N²[i, j, k] + N²[i, j, k+1]) / 2
        r = 1 + tm.ℂʷʳ * αᴺ / (1 + abs(αᴺ))
        w′³ = r * (tm.ℂʷᵉ * cl.Fᵉ[i, j, k] + tm.ℂʷᵋ * cl.Fᵋ[i, j, k]) + tm.ℂʷᵇ * ℓ * max(0, cl.B[i, j, k])
        γ = ifelse(isfinite(Δt), Δt / (tm.ℂʷᵗ * τ + Δt), one(Δt))
        cl.w′³ᵇ[i, j, k] += γ * (w′³ - cl.w′³ᵇ[i, j, k])
    end
end

compute_tke_budget_flux!(fields, distribution, closure_fields, closure, model, tracers) = nothing

# Within `update_state!`, after the conversions of the applied fluxes; the tracers hold their specific values
function compute_tke_budget_flux!(fields::SubgridCloudFields, ::TKEBudgetDistribution, closure_fields, closure, model, tracers)
    grid = model.grid
    # Three evaluations per time step; before the first step the stored third moment is the diagnosed one
    Δt = model.clock.last_Δt
    Δt = convert(eltype(grid), ifelse(isfinite(Δt) & (Δt > 0), Δt / 3, Inf))
    launch!(grid.architecture, grid, :xy, _compute_tke_budget_flux!, fields.cloud, grid, closure, closure_fields,
            tracers[TKE_NAME], model.velocities, total_density(model.dynamics), Δt)
    return nothing
end

#####
##### The third moment at the center k from the stored third moment of the budget
#####

@inline center_third_moment_diagnosis(tm::TKEBudgetThirdMoment, i, j, k, d, cl, c³, ℓ, u, algebraic_arguments) =
    @inbounds(cl.w′³ᵇ[i, j, k]) + tm.ℂʷᵃ * subgrid_third_moment(tm.algebraic, d, c³, algebraic_arguments...)

#####
##### The transport of TKE by the budget flux, in the stage tendency of ρe
#####

@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, distribution, fields) = zero(grid)
@inline tke_budget_transportᶜᶜᶜ(i, j, k, grid, ::TKEBudgetDistribution, fields) = @inbounds fields.cloud.Tᵉ[i, j, k]
