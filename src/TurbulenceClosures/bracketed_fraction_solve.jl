#####
##### A bracketed solve of the effective saturated fraction aᴺ of `CondensateFluxFraction`, face by face
#####
##### The static stability of `SaturationWeightedStaticStability(CondensateFluxFraction(...))` is
##### N² = N²ᶜ + aᴺ ΔN², ΔN² = N²ˢ - N²ᶜ, with aᴺ = w′qˡ′ / w′s′ of the distribution formed with that N².
##### Written as a scalar problem per face for the unknown aᴺ ∈ [0, 1] with the residual R(aᴺ) = aᴺ - ãᴺ(aᴺ), where ãᴺ is
##### the liquid flux fraction of the distribution with the local mixing length and diffusivities of N²ᶜ + aᴺ ΔN².
##### ãᴺ is the ratio of two fluxes and is not bounded; the unknown is: it is a share of the flux of the saturation
##### excess, and N² lies between N²ᶜ and N²ˢ. The solution on the interval is the root of R where the residual changes
##### sign across it, R(0) < 0 < R(1), and otherwise the end out of which the residual points: aᴺ = 0 where R(0) ≥ 0
##### (even at the unsaturated stability the distribution condenses no part of the flux, ãᴺ(0) ≤ 0), and aᴺ = 1 where
##### R(0) < 0 and R(1) ≤ 0 (at the saturated stability it condenses all of it, ãᴺ(1) ≥ 1). No computed value is
##### clipped, and every iterate lies in the bracket.
#####
##### Locality. ãᴺ at a face is made a function of that face's aᴺ alone:
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

"""
    BracketedFractionSolve(iterations = 3; classification_tolerance = 1e-8, lagged_envelope = false)

The `solver` of [`TKEBasedTurbulenceClosure`](@ref) that solves the coupling of the static stability of
[`CondensateFluxFraction`](@ref), `N² = N²ᶜ + aᴺ ΔN²`, as a bracketed scalar root in the effective
saturated fraction `aᴺ ∈ [0, 1]` at each interior face:

    R(aᴺ) = aᴺ - ãᴺ(N²ᶜ + aᴺ ΔN²),    ΔN² = N²ˢ - N²ᶜ,

with `ãᴺ = w′qˡ′ / w′s′` of the distribution formed at the face with the local
mixing length (the blend of `ℂˢ z` and `√e / N`), the stability functions, `Kᶜ` and the moment time
scale of that `N²`, the third moment at the face, and the assumed PDF at the two centers next to
it. The moments of the other faces are held at those of the distribution before the solve.

1. `ãᴺ` at both ends of the interval. Where the residual does not change sign across it the solution is the end
   out of which it points: `aᴺ = 0` where `R(0) ≥ 0`, and `aᴺ = 1` where `R(0) < 0` and `R(1) ≤ 0` (`ãᴺ` is a ratio of
   fluxes and is not bounded; the unknown is). Where `ãᴺ ≤ tol` at both ends the face is also classified unsaturated
   (`aᴺ = 0`); where `ãᴺ ≥ 1 - tol` at both, saturated (`aᴺ = 1`), with `tol = classification_tolerance`. These faces
   take no further step (their values are discarded).
2. Two first guesses: the regula falsi point of the ends, and the fraction `(1 + t / √(1 + t²)) / 2` of the saturation
   excess in spread units, `t = s̄ / σₛ`, with `σₛ² = τ Kᶜ (∂z s̄)²` at the regula falsi point
   (`σₛ² = (2 / ℂᵠ)(𝕊ᶜ / 𝕊ᴰ) ℓ² (∂z s̄)²` with [`DissipationTimeScale`](@ref)). Both narrow the bracket; the one with the
   smaller `|R|` is kept.
3. `iterations` Illinois steps (regula falsi, halving the weight of an end kept twice), with
   bisection where a step would leave the bracket. Every iterate lies in `[0, 1]`.

After the solve `N² = N²ᶜ + aᴺ ΔN²`; then the mixing length of the closure (with its column
envelope), the diffusivities and the distribution are formed once from it, as after the other
solvers. `solver_state` holds `aᴺ`, the residual `|R|` after each step, the classification, and `ãᴺ` of
the final distribution at the faces, whose difference from `aᴺ` is the inconsistency left by the
local mixing length and the explicit neighbours.

With `lagged_envelope = true` the mixing length inside the solve is also bounded by the
stratification envelope `ℓˢ` of the column stored before the solve (lagged), blended as the
mixing length blends its lengths, instead of the local length alone.

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

coupling_iterations(::BracketedFractionSolve) = 0
coupling_relaxation(::BracketedFractionSolve) = 1
convert_eltype(::Type{FT}, s::BracketedFractionSolve) where FT = s
Base.show(io::IO, s::BracketedFractionSolve) = print(io, summary(s))

const BracketedAppliedFluxEnergetics = AppliedFluxEnergetics{<:Any, <:Any, <:BracketedFractionSolve}

function build_solver_state(grid, ::AppliedFluxEnergetics{M, W, <:BracketedFractionSolve{N}}) where {M, W, N}
    residuals = Tuple(ZFaceField(grid) for _ in 1:N+1)
    return (aᴺ = ZFaceField(grid), classified = ZFaceField(grid), ãᴺᶠ = ZFaceField(grid), residuals)
end

#####
##### Pieces of the closure at one face, for a given N²
#####

# The squared shear of the stability functions at the face
@inline local_shear_squaredᶜᶜᶠ(i, j, k, grid, ::GridMeanShear, velocities) =
    shearᶜᶜᶠ(i, j, k, grid, velocities.u, velocities.v)

@inline local_moment_time_scale(ts, ℓ, u, N², 𝕊ᴰ) = moment_time_scale(ts, ℓ, u, N²)
@inline local_moment_time_scale(ts::EddyTurnoverTimeScale, ℓ, u, N², 𝕊ᴰ) = 2 * ℓ / (ts.ℂᵠ * u)
@inline local_moment_time_scale(ts::DissipationTimeScale, ℓ, u, N², 𝕊ᴰ) = 2 * ℓ / (ts.ℂᵠ * 𝕊ᴰ * u)

# The face's constant inputs, gathered once
@inline function fraction_face_inputs(i, j, k, grid, closure, d, cl, s, θ, qᵗ, e, p, thermodynamics)
    sm = d.second_moments
    eᵐⁱⁿ = closure.minimum_tke
    @inbounds begin
        g = saturation_excess_gradientᶜᶜᶠ(i, j, k, grid, cl, s, θ, qᵗ, thermodynamics)
        b_qᵛᶠ = (cl.b_qᵛ[i, j, k] + cl.b_qᵛ[i, j, k-1]) / 2
        N²ᶜ = g.b_θᶠ * g.∂zθ + b_qᵛᶠ * g.∂zq
        ΔN² = g.bᶜᵛ * g.∂zs
        s̄ = (mean_saturation_excess(thermodynamics, i, j, k, grid, s, θ[i, j, k], qᵗ[i, j, k]) +
             mean_saturation_excess(thermodynamics, i, j, k-1, grid, s, θ[i, j, k-1], qᵗ[i, j, k-1])) / 2
        u = ℑzᵃᵃᶠ(i, j, k, grid, turbulent_velocityᶜᶜᶜ, closure, e)
        w′²ᶠ = (vertical_velocity_variance(sm, max(e[i, j, k], eᵐⁱⁿ)) +
                vertical_velocity_variance(sm, max(e[i, j, k-1], eᵐⁱⁿ))) / 2
    end
    return (; g.Δz, g.Δθ, g.Δq, N²ᶜ, ΔN², g.∂zs, s̄, u, w′²ᶠ)
end

# The liquid flux fraction ãᴺ at face k of the distribution formed with N² = N²ᶜ + aᴺ ΔN² there
@inline function trial_condensate_flux_fraction(aᴺ, i, j, k, grid, closure, d, m, cl, input, velocities, θ, qᵗ, e, p, thermodynamics, w, ℓᵉ)
    FT = grid_float_type(grid)
    sm = d.second_moments

    N² = input.N²ᶜ + aᴺ * input.ΔN²
    u = input.u

    # The local mixing length
    ℓᵈ = closure.mixing_length.ℂˢ * height_above_bottomᶜᶜᶠ(i, j, k, grid)
    N²⁺ = max(N², zero(FT))
    ℓᵇ = ifelse(N²⁺ == 0, FT(Inf), buoyancy_length_coefficient(closure.mixing_length) * u / sqrt(N²⁺))
    ℓ = blend(length_blending(closure.mixing_length), ℓᵈ, ℓᵇ)
    ℓ = ifelse(isnan(ℓ), ℓᵈ, ℓ)
    ℓ = blend(length_blending(closure.mixing_length), ℓ, ℓᵉ) # ℓᵉ = Inf without the lagged envelope

    # The stability functions and diffusivities of that N² and ℓ
    # with the time scale τ² = ℓ² / e of the local mixing length (not the stored one)
    S² = local_shear_squaredᶜᶜᶠ(i, j, k, grid, closure.shear, velocities)
    τ² = (ℓ / u)^2
    𝕊ᵘ, 𝕊ᶜ, _, 𝕊ᴰ = local_stability_functions(closure.stability_functions, N², S², closure.shear.minimum_shear_squared, τ²)
    K = ℓ * u
    Kᵘ = 𝕊ᵘ * K
    Kᶜ = 𝕊ᶜ * K
    τ = local_moment_time_scale(d.time_scale, ℓ, u, N², 𝕊ᴰ)
    # τ ∝ ℓ / u is 0 / 0 only where the turbulent velocity vanishes (`minimum_tke = 0` and e ≤ 0 at both cells of the
    # face), where the mixing length vanishes with it. The moments of still air are zero, which τ = 0 gives.
    τ = ifelse(isfinite(τ), τ, zero(FT))

    # The second moments at the face
    θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′, w′θˡⁱ′, w′qᵗ′, _, _, _ =
        interface_second_moments(sm, τ, Kᶜ, Kᵘ, 1 / input.Δz, input.Δθ, input.Δq,
                                 zero(FT), zero(FT), zero(FT))
    θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′ = bound_flux_correlation(d.ℂʳ, input.w′²ᶠ, θˡⁱ′², qᵗ′², θˡⁱ′qᵗ′, w′θˡⁱ′, w′qᵗ′)

    @inbounds begin
        # The second moments of the trial at face k and the stored ones of the faces below and above;
        # w′³ of each center is the stored one (it is diagnosed at the centers from the moments of the
        # distribution's own evaluation and held fixed across the trials of the face)
        trial = (θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
        stored(n) = @inbounds (m.θˡⁱ′²[i, j, n], m.w′θˡⁱ′[i, j, n], m.θˡⁱ′qᵗ′[i, j, n], m.w′qᵗ′[i, j, n], m.qᵗ′²[i, j, n])

        # The assumed PDF at the center below (face k is its upper face) and above (its lower face)
        Bˡ⁻ = center_liquid_buoyancy(i, j, k-1, grid, d, cl, w, θ, qᵗ, p, thermodynamics, stored(k-1), trial)
        Bˡ⁺ = center_liquid_buoyancy(i, j, k, grid, d, cl, w, θ, qᵗ, p, thermodynamics, trial, stored(k+1))
    end

    Bᶠ = (Bˡ⁻ + Bˡ⁺) / 2
    ãᴺ = condensate_flux_fraction(FT, Bᶠ, Kᶜ, input.ΔN²)
    return (; ãᴺ, τ, Kᶜ)
end

# ãᴺ = -Bᶠ / (Kᶜ ΔN²) of the latent buoyancy flux Bᶠ at a face, zero where the diffusivity or the latent stratification
# ΔN² vanishes
@inline function condensate_flux_fraction(FT, Bᶠ, Kᶜ, ΔN²)
    resolved = (Kᶜ > sqrt(floatmin(FT))) & (abs(ΔN²) > floatmin(FT))
    return ifelse(resolved, -Bᶠ / ifelse(resolved, Kᶜ * ΔN², one(FT)), zero(FT))
end

@inline function center_liquid_buoyancy(i, j, k, grid, d, cl, w, θ, qᵗ, p, thermodynamics, moments_below, moments_above)
    ccc = (Center(), Center(), Center())
    ccf = (Center(), Center(), Face())
    @inbounds begin
        zᶜ = znode(i, j, k, grid, ccc...)
        zˡ = znode(i, j, k, grid, ccf...)
        zᵘ = znode(i, j, k+1, grid, ccf...)
        wᶜ = (w[i, j, k] + w[i, j, k+1]) / 2
        θᵏ = θ[i, j, k]
        w′³ = cl.w′³[i, j, k]
        thermodynamicsᵏ = level_thermodynamics(thermodynamics, i, j, k)
        _, _, _, w′qˡ′, _ = assumed_pdf(d.assumed_pdf, thermodynamicsᵏ, zᵘ, zˡ, zᶜ,
                                          p[i, j, k], θᵏ, wᶜ, qᵗ[i, j, k], cl.w′²[i, j, k], (w′³, moments_below...), (w′³, moments_above...))
        # The buoyancy of the flux of liquid with the derivatives of the model's buoyancy, as the stored Bˡ of the
        # distribution has it
        Bˡ = (cl.b_qᶜ[i, j, k] - cl.b_qᵛ[i, j, k]) * w′qˡ′
    end
    return Bˡ
end

#####
##### The solve
#####

# The regula falsi point of a bracket (a, Rₐ ≤ 0), (b, R_b ≥ 0); bisection where it is not inside. The choice keeps the
# iterate of the root search inside its bracket: it is part of the algorithm, not a bound on a physical quantity.
@inline function bracketed_step(a, Rₐ, b, R_b)
    FT = typeof(a)
    ΔR = R_b - Rₐ
    ok = ΔR > floatmin(FT)
    x = a - Rₐ * (b - a) / ifelse(ok, ΔR, one(FT))
    inside = ok & (x > a) & (x < b)
    return ifelse(inside, x, (a + b) / 2)
end

@kernel function _bracketed_fraction_solve!(N², solver::BracketedFractionSolve{N}, grid, closure, velocities, e, closure_fields,
                                            s) where N
    i, j, k = @index(Global, NTuple)
    fields = closure_fields.subgrid_cloud
    state, moments, cloud = fields.solver_state, fields.moments, fields.cloud
    ℓˢ = closure_fields.ℓˢ
    θ, qᵗ, thermodynamics = distributed_water(s, grid)
    p = s.p
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    interior_face = (k > 1) & (k ≤ Nz)
    kₛ = clamp(k, 2, max(Nz, 2)) # an index, not a value: a valid face for the boundary threads, whose result is discarded

    closure_ij = getclosure(i, j, closure)
    d = closure_ij.subgrid_distribution
    w = velocities.w
    tol = convert(FT, solver.classification_tolerance)
    ℓᵉ = ifelse(solver.lagged_envelope, @inbounds(ℓˢ[i, j, kₛ]), FT(Inf))
    stored_input = fraction_face_inputs(i, j, kₛ, grid, closure_ij, d, cloud, s, θ, qᵗ, e, p, thermodynamics)
    # The latent response of the condensate outside the distributed water (ice and its liquid coating), with the
    # distribution and the eddy time before the solve (as the moments of the neighbouring faces)
    aʷ = face_condensate_flux_fraction(i, j, kₛ, cloud)
    X = latent_correctionᶜᶜᶠ(i, j, kₛ, grid, closure_ij, closure_fields, velocities, s, qᵗ, θ, e, thermodynamics, cloud,
                             aʷ).outside
    input = merge(stored_input, (; N²ᶜ = stored_input.N²ᶜ + X))
    ãᴺ(x) = trial_condensate_flux_fraction(x, i, j, kₛ, grid, closure_ij, d, moments, cloud, input, velocities,
                                      θ, qᵗ, e, p, thermodynamics, w, ℓᵉ)

    # The two ends
    t₀ = ãᴺ(zero(FT))
    t₁ = ãᴺ(one(FT))
    a, Rₐ = zero(FT), -t₀.ãᴺ          # R(0) = -ãᴺ(0)
    b, R_b = one(FT), one(FT) - t₁.ãᴺ # R(1) = 1 - ãᴺ(1)
    # The unknown lives on [0, 1]: where the residual does not change sign across the interval, the solution is the
    # end out of which it points
    lower_end = Rₐ ≥ 0
    upper_end = !lower_end & (R_b ≤ 0)
    unsaturated = lower_end | ((t₀.ãᴺ ≤ tol) & (t₁.ãᴺ ≤ tol))
    saturated = !unsaturated & (upper_end | ((t₀.ãᴺ ≥ 1 - tol) & (t₁.ãᴺ ≥ 1 - tol)))

    # Two first guesses: the regula falsi point of the ends, and the fraction (1 + t / √(1 + t²)) / 2 of the saturation
    # excess in spread units, t = s̄ / σₛ, written with the hypotenuse so that it is in [0, 1] for any s̄ and σₛ ≥ 0
    # (one half where both vanish: the smallest positive number keeps the quotient defined there)
    x_rf = bracketed_step(a, Rₐ, b, R_b)
    t_rf = ãᴺ(x_rf)
    R_rf = x_rf - t_rf.ãᴺ
    σₛ = sqrt(t_rf.τ * t_rf.Kᶜ) * abs(input.∂zs)
    x_g = (1 + input.s̄ / (hypot(input.s̄, σₛ) + floatmin(FT))) / 2
    t_g = ãᴺ(x_g)
    R_g = x_g - t_g.ãᴺ

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

    @inbounds state.residuals[1][i, j, k] = abs(R)

    # Illinois steps; `side` is +1 when the last step replaced `a`, -1 when it replaced `b`
    Aₐ, A_b = Rₐ, R_b # the weights of the ends
    side = zero(FT)
    for n in 1:N
        x′ = bracketed_step(a, Aₐ, b, A_b)
        t = ãᴺ(x′)
        R′ = x′ - t.ãᴺ
        below = R′ ≤ 0
        # halve the weight of the end that is kept a second time
        A_b = ifelse(below & (side > 0), A_b / 2, A_b)
        Aₐ = ifelse(!below & (side < 0), Aₐ / 2, Aₐ)
        a, Aₐ = ifelse(below, x′, a), ifelse(below, R′, Aₐ)
        b, A_b = ifelse(below, b, x′), ifelse(below, A_b, R′)
        side = ifelse(below, one(FT), -one(FT))
        x, R = x′, R′
        @inbounds state.residuals[n+1][i, j, k] = abs(R)
    end

    aᴺ = ifelse(unsaturated, zero(FT), ifelse(saturated, one(FT), x))

    @inbounds begin
        N²[i, j, k] = ifelse(interior_face, input.N²ᶜ + aᴺ * input.ΔN², N²[i, j, k])
        state.aᴺ[i, j, k] = aᴺ
        state.classified[i, j, k] = ifelse(unsaturated, -one(FT), ifelse(saturated, one(FT), zero(FT)))
    end
end

# ãᴺ of the stored distribution at the faces, (N²ᵇ - N²ᶜ) / ΔN², after the solve
@kernel function _final_condensate_flux_fraction!(ãᴺᶠ, grid, cloud, Kᶜ, s)
    i, j, k = @index(Global, NTuple)
    θ, qᵗ, thermodynamics = distributed_water(s, grid)
    p = s.p
    FT = grid_float_type(grid)
    Nz = size(grid, 3)
    kₛ = clamp(k, 2, max(Nz, 2))
    @inbounds begin
        g = saturation_excess_gradientᶜᶜᶠ(i, j, kₛ, grid, cloud, s, θ, qᵗ, thermodynamics)
        ΔN² = g.bᶜᵛ * g.∂zs
        Bᶠ = (cloud.Bˡ[i, j, kₛ] + cloud.Bˡ[i, j, kₛ-1]) / 2
        ãᴺ = condensate_flux_fraction(FT, Bᶠ, Kᶜ[i, j, kₛ], ΔN²)
        ãᴺᶠ[i, j, k] = ifelse((k > 1) & (k ≤ Nz), ãᴺ, zero(FT))
    end
end

function cloud_buoyancy_stability!(buoyancy_flux::BracketedAppliedFluxEnergetics, fields, closure_fields, closure, model, tracers, parameters)
    grid = model.grid
    arch = grid.architecture
    s = applied_flux_state(model)
    e = SpecificValue(model.tracers[TKE_NAME], nothing)
    launch!(arch, grid, parameters, _bracketed_fraction_solve!, closure_fields.N², buoyancy_flux.stability_solver, grid, closure,
            model.velocities, e, closure_fields, s)
    # The column envelope of the mixing length, the diffusivities and the distribution, once
    compute_mixing_length_and_diffusivities!(closure_fields, closure, model, tracers, parameters)
    compute_subgrid_distribution!(fields, closure_fields, closure, model)
    launch!(arch, grid, parameters, _final_condensate_flux_fraction!, fields.solver_state.ãᴺᶠ, grid, fields.cloud,
            closure_fields.Kᶜ, s)
    store_latent_correction_of_solved_stability!(fields, closure_fields, closure, model, tracers, parameters)
    return nothing
end
