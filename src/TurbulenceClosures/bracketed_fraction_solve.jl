#####
##### A bracketed solve of the effective saturated fraction ℵ of `LiquidFluxFraction`, face by face
#####
##### The static stability of `SubgridCloudStaticStability(LiquidFluxFraction(...))` is
##### N² = N²ᶜ + ℵ ΔN², ΔN² = N²ˢ - N²ᶜ, with ℵ = w′qˡ′ / w′s′ of the distribution formed with that N².
##### Written as a scalar root per face, R(ℵ) = ℵ - ℵ̃(ℵ) on ℵ ∈ [0, 1], where ℵ̃ is the liquid flux
##### fraction of the distribution with the local mixing length and diffusivities of N²ᶜ + ℵ ΔN².
##### ℵ̃ is clamped to [0, 1], so R(0) ≤ 0 ≤ R(1) and [0, 1] brackets a root.
#####
##### Locality. ℵ̃ at a face is made a function of that face's ℵ alone:
#####   * the mixing length inside the solve is the local one, the blend of ℂˢ z and √e / N of the
#####     mixing length's blending; the column envelope of `GradientLimitedMixingLength` is applied once,
#####     after the solve;
#####   * the moments of the neighbouring faces, which the third moment at the face (k ± 1) and the
#####     assumed PDF at the two adjacent centers (interpolated from the faces below and above) read,
#####     are those stored by the distribution before the solve (explicit: a state approximation).
#####
##### The number of steps is a type parameter and every choice is an `ifelse`, so the solve is unrolled
##### when traced.
#####

using SpecialFunctions: erf

"""
    BracketedFractionSolve(iterations = 3; classification_tolerance = 1e-8)

The `stability_solver` of [`CloudBuoyancyFlux`](@ref) that solves the static stability of
[`LiquidFluxFraction`](@ref), `N² = N²ᶜ + ℵ ΔN²`, as a bracketed scalar root in the effective
saturated fraction `ℵ ∈ [0, 1]` at each interior face:

    R(ℵ) = ℵ - ℵ̃(N²ᶜ + ℵ ΔN²),    ΔN² = N²ˢ - N²ᶜ,

with `ℵ̃ = w′qˡ′ / w′s′` (clamped to `[0, 1]`) of the distribution formed at the face with the local
mixing length (the blend of `ℂˢ z` and `√e / N`), the stability functions, `Kᶜ` and the moment time
scale of that `N²`, the third moment at the face, and the assumed PDF at the two centers next to
it. The moments of the other faces are held at those of the distribution before the solve.

1. `ℵ̃` at both ends of the bracket: `R(0) ≤ 0 ≤ R(1)`. Where `ℵ̃ ≤ tol` at both ends the face is
   classified unsaturated (`ℵ = 0`); where `ℵ̃ ≥ 1 - tol` at both, saturated (`ℵ = 1`), with
   `tol = classification_tolerance`. These faces take no further step (their values are discarded).
2. Two first guesses: the regula falsi point of the ends, and the single-Gaussian fraction
   `Φ(s̄ / σₛ)`, with `σₛ² = τ Kᶜ (∂z s̄)²` at the regula falsi point (`σₛ² = (2 / ℂᵠ)(Sᶜ / Sᴰ) ℓ² (∂z s̄)²`
   with [`DissipationTimeScale`](@ref)). Both narrow the bracket; the one with the smaller `|R|` is kept.
3. `iterations` Illinois steps (regula falsi, halving the weight of an end kept twice), with
   bisection where a step would leave the bracket. Every iterate lies in `[0, 1]`.

After the solve `N² = N²ᶜ + ℵ ΔN²`; then the mixing length of the closure (with its column
envelope), the diffusivities and the distribution are formed once from it, as after the other
solvers. `solver_state` holds `ℵ`, the residual `|R|` after each step, the classification, the
`ℵ̃` of the two ends, and `ℵ̃` of the final distribution at the faces, whose difference from `ℵ` is
the inconsistency left by the local mixing length and the explicit neighbours.

With `lagged_envelope = true` the mixing length inside the solve is also bounded by the
stratification envelope `ℓˢ` of the column stored before the solve (lagged), blended as the
mixing length blends its lengths, instead of the local length alone.

`stability_iterations` and `stability_relaxation` of the [`CloudBuoyancyFlux`](@ref) are not used.
"""
struct BracketedFractionSolve{N, FT}
    classification_tolerance :: FT
    lagged_envelope :: Bool
    BracketedFractionSolve{N}(tol::FT, lagged_envelope::Bool = false) where {N, FT} = new{N, FT}(tol, lagged_envelope)
end

function BracketedFractionSolve(iterations::Integer = 3; classification_tolerance = 1e-8, lagged_envelope = false)
    iterations >= 0 || throw(ArgumentError("iterations must be ≥ 0, got $iterations."))
    return BracketedFractionSolve{Int(iterations)}(classification_tolerance, lagged_envelope)
end

Base.summary(b::BracketedFractionSolve{N}) where N =
    string("BracketedFractionSolve(", N, b.lagged_envelope ? ", lagged_envelope = true)" : ")")

const BracketedCloudBuoyancyFlux = CloudBuoyancyFlux{<:Any, <:Any, <:BracketedFractionSolve}

function build_solver_state(grid, ::CloudBuoyancyFlux{M, W, <:BracketedFractionSolve{N}}) where {M, W, N}
    residuals = Tuple(ZFaceField(grid) for _ in 1:N+1)
    return (ℵ = ZFaceField(grid), ℵ̃₀ = ZFaceField(grid), ℵ̃₁ = ZFaceField(grid), classified = ZFaceField(grid),
            initial_residuals = ZFaceField(grid), first_guess = ZFaceField(grid), slope = ZFaceField(grid),
            ℵ̃ᶠ = ZFaceField(grid), residuals)
end

#####
##### Pieces of the closure at one face, for a given N²
#####

@inline local_stability_functions(sf::ConstantStabilityFunctions, Ri) = (sf.ℂᵘ, sf.ℂᶜ, sf.ℂᴰ)
@inline local_stability_functions(sf::RationalStabilityFunction, Ri) =
    (momentum_stability_function(sf, Ri), tracer_stability_function(sf, Ri), dissipation_stability_function(sf, Ri))

# The squared shear of the Richardson number with the local mixing length `ℓ` (not the stored one)
@inline local_shear_squaredᶜᶜᶠ(i, j, k, grid, ::GridMeanShear, closure, velocities, u, ℓ) =
    shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)

@inline function local_shear_squaredᶜᶜᶠ(i, j, k, grid, shear::TurbulenceAugmentedShear, closure, velocities, u, ℓ)
    turbulent = ℓ > 0
    ℓ⁺ = ifelse(turbulent, ℓ, one(ℓ))
    S²ᵗ = shear.turbulence_shear_coefficient * (u / ℓ⁺)^2
    return shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v) + ifelse(turbulent, S²ᵗ, zero(S²ᵗ))
end

@inline local_moment_time_scale(ts, ℓ, u, N², Sᴰ) = moment_time_scale(ts, ℓ, u, N²)
@inline local_moment_time_scale(ts::EddyTurnoverTimeScale, ℓ, u, N², Sᴰ) = 2 * ℓ / (ts.ℂᵠ * u)
@inline local_moment_time_scale(ts::DissipationTimeScale, ℓ, u, N², Sᴰ) = 2 * ℓ / (ts.ℂᵠ * Sᴰ * u)

# SHOC's saturation excess of the mean state at a level, s = qᵗ - qˢ (1 + β qᵗ) / (1 + β qˢ)
@inline function mean_saturation_excess(sat, thermodynamics, c, θ, qᵗ, p)
    Tˡ = plume_liquid_water_temperature(sat, thermodynamics, c, θ, qᵗ, p)
    qˢ, β, _, _ = plume_saturation(sat, thermodynamics, c, Tˡ, qᵗ, p)
    return qᵗ - qˢ * (1 + β * qᵗ) / (1 + β * qˢ)
end

@inline standard_normal_cdf(x) = (1 + erf(x / sqrt(convert(typeof(x), 2)))) / 2

# The face's constant inputs, gathered once
@inline function fraction_face_inputs(i, j, k, grid, closure, d, m, cl, θ, qᵗ, e, p, thermodynamics)
    FT = eltype(grid)
    c = d.constants
    sm = d.second_moments
    eᵐⁱⁿ = closure.minimum_tke
    @inbounds begin
        Δz = Δzᶜᶜᶠ(i, j, k, grid)
        Δθ = θ[i, j, k] - θ[i, j, k-1]
        Δq = qᵗ[i, j, k] - qᵗ[i, j, k-1]
        ∂zθ = Δθ / Δz
        ∂zq = Δq / Δz
        gθᶠ = (cl.gθ[i, j, k] + cl.gθ[i, j, k-1]) / 2
        gAᶠ = (cl.gA[i, j, k] + cl.gA[i, j, k-1]) / 2
        N²ᶜ = gθᶠ * ∂zθ + gAᶠ * ∂zq
        c_q⁺, c_θ⁺, A⁺ = saturation_excess_coefficients(d.saturation, thermodynamics, c, θ[i, j, k], qᵗ[i, j, k], p[i, j, k])
        c_q⁻, c_θ⁻, A⁻ = saturation_excess_coefficients(d.saturation, thermodynamics, c, θ[i, j, k-1], qᵗ[i, j, k-1], p[i, j, k-1])
        ∂zs = (c_q⁺ + c_q⁻) / 2 * ∂zq - (c_θ⁺ + c_θ⁻) / 2 * ∂zθ
        ΔN² = gθᶠ * (A⁺ + A⁻) / 2 * ∂zs
        s̄ = (mean_saturation_excess(d.saturation, thermodynamics, c, θ[i, j, k], qᵗ[i, j, k], p[i, j, k]) +
             mean_saturation_excess(d.saturation, thermodynamics, c, θ[i, j, k-1], qᵗ[i, j, k-1], p[i, j, k-1])) / 2
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        w′²ᶠ = (vertical_velocity_variance(sm, max(e[i, j, k], eᵐⁱⁿ)) +
                vertical_velocity_variance(sm, max(e[i, j, k-1], eᵐⁱⁿ))) / 2
    end
    return (; Δz, Δθ, Δq, ∂zθ, ∂zq, N²ᶜ, ΔN², ∂zs, s̄, u, w′²ᶠ)
end

# The liquid flux fraction ℵ̃ at face k of the distribution formed with N² = N²ᶜ + ℵ ΔN² there
@inline function trial_liquid_flux_fraction(ℵ, i, j, k, grid, closure, d, m, cl, input, velocities, θ, qᵗ, e, p, thermodynamics, w, ℓᵉ)
    FT = eltype(grid)
    c = d.constants
    cp = d.clipping
    sm = d.second_moments
    eᵐⁱⁿ = closure.minimum_tke

    N² = input.N²ᶜ + ℵ * input.ΔN²
    u = input.u

    # The local mixing length
    ℓᵈ = closure.mixing_length.ℂˢ * height_above_bottomᶜᶜᶠ(i, j, k, grid)
    N²⁺ = max(N², zero(FT))
    ℓᵇ = ifelse(N²⁺ == 0, FT(Inf), u / sqrt(N²⁺))
    ℓ = blend(length_blending(closure.mixing_length), ℓᵈ, ℓᵇ)
    ℓ = ifelse(isnan(ℓ), ℓᵈ, ℓ)
    ℓ = blend(length_blending(closure.mixing_length), ℓ, ℓᵉ) # ℓᵉ = Inf without the lagged envelope

    # The stability functions and diffusivities of that N² and ℓ
    S² = local_shear_squaredᶜᶜᶠ(i, j, k, grid, closure.shear, closure, velocities, u, ℓ)
    Ri = richardson_number(N², S², closure.shear.minimum_shear_squared)
    Sᵘ, Sᶜ, Sᴰ = local_stability_functions(closure.stability_functions, Ri)
    K = ℓ * u
    Kᵘ = min(Sᵘ * K, closure.maximum_viscosity)
    Kᶜ = min(Sᶜ * K, closure.maximum_tracer_diffusivity)
    τ = local_moment_time_scale(d.time_scale, ℓ, u, N², Sᴰ)
    τ = ifelse(isfinite(τ), τ, zero(FT))

    # The second moments at the face
    θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′, _, _, _ =
        interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / input.Δz, input.Δθ, input.Δq,
                                 zero(FT), zero(FT), zero(FT))
    θˡ′², qᵗ′², θˡ′qᵗ′ = bound_flux_correlation(d.ℂʳ, input.w′²ᶠ, θˡ′², qᵗ′², θˡ′qᵗ′, w′θˡ′, w′qᵗ′)

    @inbounds begin
        # The third moment at the face, with the stored moments of the faces above and below
        w′²ᵐⁱⁿ = (convert(FT, 2) / 3) * c.mintke
        w′²ᶠ = clip_below(cp, (cl.w′²[i, j, k] + cl.w′²[i, j, k-1]) / 2, w′²ᵐⁱⁿ, c.mintke)
        θᶠ = (θ[i, j, k] + θ[i, j, k-1]) / 2
        thedz = 1 / Δzᶜᶜᶠ(i, j, k, grid)
        thedz2 = 1 / (Δzᶜᶜᶜ(i, j, k-1, grid) + Δzᶜᶜᶜ(i, j, k, grid))
        τ³ = third_moment_time_scale(d.time_scale, τ, ℓ, u, N²)
        τ³ = ifelse(isfinite(τ³), τ³, zero(FT))
        w′³ = subgrid_third_moment(d.third_moment, d, c, τ³, N², w′²ᶠ, θᶠ, thedz, thedz2,
                                   m.θˡ′²[i, j, k+1] - m.θˡ′²[i, j, k-1],
                                   m.w′θˡ′[i, j, k+1] - m.w′θˡ′[i, j, k-1],
                                   cl.w′²[i, j, k] - cl.w′²[i, j, k-1],
                                   max(e[i, j, k], eᵐⁱⁿ) - max(e[i, j, k-1], eᵐⁱⁿ),
                                   w′θˡ′)
        w′³ = limit_third_moment(d.skewness, cp, w′²ᶠ, w′³)

        trial = (w′³, θˡ′², w′θˡ′, θˡ′qᵗ′, w′qᵗ′, qᵗ′²)
        stored(n) = (m.w′³[i, j, n], m.θˡ′²[i, j, n], m.w′θˡ′[i, j, n], m.θˡ′qᵗ′[i, j, n], m.w′qᵗ′[i, j, n], m.qᵗ′²[i, j, n])

        # The assumed PDF at the center below (face k is its upper face) and above (its lower face)
        Bˡ⁻, C⁻ = center_liquid_buoyancy(i, j, k-1, grid, d, cp, c, cl, w, θ, qᵗ, p, thermodynamics, stored(k-1), trial)
        Bˡ⁺, C⁺ = center_liquid_buoyancy(i, j, k, grid, d, cp, c, cl, w, θ, qᵗ, p, thermodynamics, trial, stored(k+1))
    end

    Bᶠ = (Bˡ⁻ + Bˡ⁺) / 2
    resolved = (Kᶜ > sqrt(floatmin(FT))) & (abs(input.ΔN²) > floatmin(FT))
    ℵ̃ = ifelse(resolved, -Bᶠ / ifelse(resolved, Kᶜ * input.ΔN², one(FT)), zero(FT))
    return (; ℵ̃ = clamp(ℵ̃, zero(FT), one(FT)), raw = ℵ̃, τ, Kᶜ, C = (C⁻ + C⁺) / 2)
end

@inline function center_liquid_buoyancy(i, j, k, grid, d, cp, c, cl, w, θ, qᵗ, p, thermodynamics, moments_l, moments_u)
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())
    @inbounds begin
        zᶜ = znode(i, j, k, grid, ccc...)
        zˡ = znode(i, j, k, grid, ccf...)
        zᵘ = znode(i, j, k+1, grid, ccf...)
        wᶜ = (w[i, j, k] + w[i, j, k+1]) / 2
        θᵏ = θ[i, j, k]
        C, qˡ, _, w′qˡ′, _ = assumed_pdf(d.assumed_pdf, d.saturation, thermodynamics, cp, c, zᵘ, zˡ, zᶜ,
                                          p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], cl.w′²[i, j, k], moments_l, moments_u)
        _, A_c = buoyancy_flux_coefficients(d.saturation, thermodynamics, c, p[i, j, k], θᵏ, qᵗ[i, j, k], qˡ)
    end
    return c.ggr / θᵏ * A_c * w′qˡ′, C
end

#####
##### The solve
#####

# The regula falsi point of a bracket (a, Rₐ ≤ 0), (b, R_b ≥ 0); bisection where it is not inside
@inline function bracketed_step(a, Rₐ, b, R_b)
    FT = typeof(a)
    ΔR = R_b - Rₐ
    ok = ΔR > floatmin(FT)
    x = a - Rₐ * (b - a) / ifelse(ok, ΔR, one(FT))
    inside = ok & (x > a) & (x < b)
    return ifelse(inside, x, (a + b) / 2)
end

@kernel function _bracketed_fraction_solve!(N², state, solver::BracketedFractionSolve{N}, grid, closure, moments, cloud,
                                            velocities, θ, qᵗ, e, p, thermodynamics, ℓˢ) where N
    i, j, k = @index(Global, NTuple)
    FT = eltype(grid)
    Nz = size(grid, 3)
    interior_face = (k > 1) & (k ≤ Nz)
    kₛ = clamp(k, 2, max(Nz, 2)) # a valid face for the boundary threads, whose result is discarded

    closure_ij = getclosure(i, j, closure)
    d = closure_ij.subgrid_cloud_formulation.distribution
    w = velocities.w
    tol = convert(FT, solver.classification_tolerance)
    ℓᵉ = ifelse(solver.lagged_envelope, @inbounds(ℓˢ[i, j, kₛ]), FT(Inf))
    input = fraction_face_inputs(i, j, kₛ, grid, closure_ij, d, moments, cloud, θ, qᵗ, e, p, thermodynamics)
    ℵ̃(x) = trial_liquid_flux_fraction(x, i, j, kₛ, grid, closure_ij, d, moments, cloud, input, velocities,
                                      θ, qᵗ, e, p, thermodynamics, w, ℓᵉ)

    # The two ends
    t₀ = ℵ̃(zero(FT))
    t₁ = ℵ̃(one(FT))
    a, Rₐ = zero(FT), -t₀.ℵ̃          # R(0) = -ℵ̃(0) ≤ 0
    b, R_b = one(FT), one(FT) - t₁.ℵ̃ # R(1) = 1 - ℵ̃(1) ≥ 0
    unsaturated = (t₀.ℵ̃ ≤ tol) & (t₁.ℵ̃ ≤ tol)
    saturated = (t₀.ℵ̃ ≥ 1 - tol) & (t₁.ℵ̃ ≥ 1 - tol)

    # Two first guesses: the regula falsi point of the ends, and the single-Gaussian fraction
    x_rf = bracketed_step(a, Rₐ, b, R_b)
    t_rf = ℵ̃(x_rf)
    R_rf = x_rf - t_rf.ℵ̃
    σₛ = sqrt(max(t_rf.τ * t_rf.Kᶜ, zero(FT))) * abs(input.∂zs)
    x_g = clamp(standard_normal_cdf(input.s̄ / max(σₛ, floatmin(FT))), zero(FT), one(FT))
    t_g = ℵ̃(x_g)
    R_g = x_g - t_g.ℵ̃

    # Both narrow the bracket
    lower_rf = (R_rf ≤ 0) & (x_rf > a)
    upper_rf = (R_rf > 0) & (x_rf < b)
    a, Rₐ = ifelse(lower_rf, x_rf, a), ifelse(lower_rf, R_rf, Rₐ)
    b, R_b = ifelse(upper_rf, x_rf, b), ifelse(upper_rf, R_rf, R_b)
    lower_g = (R_g ≤ 0) & (x_g > a)
    upper_g = (R_g > 0) & (x_g < b)
    a, Rₐ = ifelse(lower_g, x_g, a), ifelse(lower_g, R_g, Rₐ)
    b, R_b = ifelse(upper_g, x_g, b), ifelse(upper_g, R_g, R_b)

    gaussian_better = abs(R_g) < abs(R_rf)
    x = ifelse(gaussian_better, x_g, x_rf)
    R = ifelse(gaussian_better, R_g, R_rf)
    x₋, ℵ̃₋ = ifelse(gaussian_better, x_rf, x_g), ifelse(gaussian_better, t_rf.ℵ̃, t_g.ℵ̃)

    @inbounds state.residuals[1][i, j, k] = abs(R)

    # Illinois steps; `side` is +1 when the last step replaced `a`, -1 when it replaced `b`
    Aₐ, A_b = Rₐ, R_b # the weights of the ends
    side = zero(FT)
    for n in 1:N
        x′ = bracketed_step(a, Aₐ, b, A_b)
        t = ℵ̃(x′)
        R′ = x′ - t.ℵ̃
        below = R′ ≤ 0
        # halve the weight of the end that is kept a second time
        A_b = ifelse(below & (side > 0), A_b / 2, A_b)
        Aₐ = ifelse(!below & (side < 0), Aₐ / 2, Aₐ)
        a, Aₐ = ifelse(below, x′, a), ifelse(below, R′, Aₐ)
        b, A_b = ifelse(below, b, x′), ifelse(below, A_b, R′)
        side = ifelse(below, one(FT), -one(FT))
        x₋, ℵ̃₋ = x, x - R
        x, R = x′, R′
        @inbounds state.residuals[n+1][i, j, k] = abs(R)
    end

    ℵ = ifelse(unsaturated, zero(FT), ifelse(saturated, one(FT), x))
    slope = ifelse(abs(x - x₋) > sqrt(eps(FT)), ((x - R) - ℵ̃₋) / ifelse(abs(x - x₋) > sqrt(eps(FT)), x - x₋, one(FT)), zero(FT))

    @inbounds begin
        N²[i, j, k] = ifelse(interior_face, input.N²ᶜ + ℵ * input.ΔN², N²[i, j, k])
        state.ℵ[i, j, k] = ℵ
        state.ℵ̃₀[i, j, k] = t₀.ℵ̃
        state.ℵ̃₁[i, j, k] = t₁.ℵ̃
        state.classified[i, j, k] = ifelse(unsaturated, -one(FT), ifelse(saturated, one(FT), zero(FT)))
        state.initial_residuals[i, j, k] = ifelse(gaussian_better, -abs(R_g), abs(R_rf)) # sign: which guess was kept
        state.first_guess[i, j, k] = ifelse(gaussian_better, x_g, x_rf)
        state.slope[i, j, k] = slope
    end
end

# ℵ̃ of the stored distribution at the faces, (N²ᵇ - N²ᶜ) / ΔN², after the solve
@kernel function _final_liquid_flux_fraction!(ℵ̃ᶠ, grid, closure, cloud, Kᶜ, θ, qᵗ, p, thermodynamics)
    i, j, k = @index(Global, NTuple)
    FT = eltype(grid)
    Nz = size(grid, 3)
    kₛ = clamp(k, 2, max(Nz, 2))
    d = getclosure(i, j, closure).subgrid_cloud_formulation.distribution
    @inbounds begin
        Δz = Δzᶜᶜᶠ(i, j, kₛ, grid)
        ∂zθ = (θ[i, j, kₛ] - θ[i, j, kₛ-1]) / Δz
        ∂zq = (qᵗ[i, j, kₛ] - qᵗ[i, j, kₛ-1]) / Δz
        gθᶠ = (cloud.gθ[i, j, kₛ] + cloud.gθ[i, j, kₛ-1]) / 2
        c_q⁺, c_θ⁺, A⁺ = saturation_excess_coefficients(d.saturation, thermodynamics, d.constants, θ[i, j, kₛ], qᵗ[i, j, kₛ], p[i, j, kₛ])
        c_q⁻, c_θ⁻, A⁻ = saturation_excess_coefficients(d.saturation, thermodynamics, d.constants, θ[i, j, kₛ-1], qᵗ[i, j, kₛ-1], p[i, j, kₛ-1])
        ΔN² = gθᶠ * (A⁺ + A⁻) / 2 * ((c_q⁺ + c_q⁻) / 2 * ∂zq - (c_θ⁺ + c_θ⁻) / 2 * ∂zθ)
        Bᶠ = (cloud.Bˡ[i, j, kₛ] + cloud.Bˡ[i, j, kₛ-1]) / 2
        Kᶜᵢ = Kᶜ[i, j, kₛ]
        resolved = (Kᶜᵢ > sqrt(floatmin(FT))) & (abs(ΔN²) > floatmin(FT))
        ℵ̃ = ifelse(resolved, -Bᶠ / ifelse(resolved, Kᶜᵢ * ΔN², one(FT)), zero(FT))
        ℵ̃ᶠ[i, j, k] = ifelse((k > 1) & (k ≤ Nz), ℵ̃, zero(FT))
    end
end

function cloud_buoyancy_stability!(buoyancy_flux::BracketedCloudBuoyancyFlux, fields, closure_fields, closure, model, tracers, parameters)
    grid = model.grid
    arch = grid.architecture
    s = distribution_state(model, nothing)
    state = fields.solver_state
    launch!(arch, grid, parameters, _bracketed_fraction_solve!, closure_fields.N², state, buoyancy_flux.stability_solver,
            grid, closure, fields.moments, fields.cloud, model.velocities, s.θ, s.qᵗ, s.e, s.p, s.thermodynamics, closure_fields.ℓˢ)
    # The column envelope of the mixing length, the diffusivities and the distribution, once
    compute_mixing_length!(closure_fields.ℓᵐ, closure_fields.ℓˢ, grid, closure, tracers[TKE_NAME], closure_fields.N²)
    launch!(arch, grid, parameters, _compute_tke_closure_fields!, closure_fields, grid, closure, model.velocities, tracers)
    compute_subgrid_distribution!(fields, closure_fields, closure, model, nothing)
    launch!(arch, grid, parameters, _final_liquid_flux_fraction!, state.ℵ̃ᶠ, grid, closure, fields.cloud, closure_fields.Kᶜ,
            s.θ, s.qᵗ, s.p, s.thermodynamics)
    return nothing
end
