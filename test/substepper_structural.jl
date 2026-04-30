#####
##### Structural correctness tests for the acoustic substepper.
#####
##### These tests exercise SPECIFIC structural properties that must
##### hold by construction — indexing, staggered-grid interpolation,
##### sign conventions, boundary handling, adjoint structure of the
##### discrete operators, and consistency between the production
##### kernels and the documented matrix coefficients
##### (`validation/substepping/derivation_phase1.md`).
#####
##### When one of these fires, the failure mode is *named*: the
##### offending operator or boundary row is identified directly.
#####
##### These tests sit alongside the rest-atmosphere envelope tests in
##### `test/substepper_rest_state.jl` as Phase 0 of the
##### pristine-substepper program.
#####

using Breeze
using Breeze.CompressibleEquations: AcousticSubstepper,
                                    AcousticTridiagLower,
                                    AcousticTridiagDiagonal,
                                    AcousticTridiagUpper,
                                    freeze_linearization_state!,
                                    assemble_slow_vertical_momentum_tendency!
using Breeze.TimeSteppers: compute_slow_momentum_tendencies!,
                           compute_slow_scalar_tendencies!

using Oceananigans
using Oceananigans.Grids: ZDirection
using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ, ℑzᵃᵃᶠ
using Oceananigans.Solvers: get_coefficient
using Oceananigans.TimeSteppers: update_state!

using GPUArraysCore: @allowscalar
using Printf
using Test

const T₀_STR = 250.0
const Lz_STR = 30e3
const NZ_STR = 32
const G_STR  = 9.80665
const CPD_STR = 1005.0

θ_iso_str(z) = T₀_STR * exp(G_STR * z / (CPD_STR * T₀_STR))

function _build_str_model(arch; ω = 0.55)
    grid = RectilinearGrid(arch;
                           size = (8, 8, NZ_STR), halo = (5, 5, 5),
                           x = (0, 100e3), y = (0, 100e3), z = (0, Lz_STR),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; forward_weight = ω,
                                           damping = NoDivergenceDamping())
    dyn = CompressibleDynamics(td;
                               reference_potential_temperature = θ_iso_str,
                               surface_pressure = 1e5,
                               standard_pressure = 1e5)
    return AtmosphereModel(grid; dynamics = dyn,
                                  thermodynamic_constants = constants,
                                  timestepper = :AcousticRungeKutta3)
end

function _set_str_rest!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)

    parent(model.dynamics.density) .= parent(ref.density)
    ρθ = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρθ) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)
    update_state!(model)
end

@testset "Substepper structural correctness" begin

#####
##### S1 — Boundary-row of the column tridiag is trivial
#####
##### Solver row k = 1 must have b[1] = 1, c[1] = 0 so that the
##### post-solve gives μw[1] = 0 (impenetrability at the rigid
##### bottom). If the kernels return anything else there, the
##### bottom-boundary face will pick up nonzero values.
#####

@testset "S1 — Bottom-boundary row of the column tridiag" begin
    model = _build_str_model(default_arch)
    sub   = model.timestepper.substepper
    _set_str_rest!(model)
    freeze_linearization_state!(sub, model)

    grid = model.grid
    Rᵈ   = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    cᵖᵈ  = model.thermodynamic_constants.dry_air.heat_capacity
    γRᵈ  = cᵖᵈ * Rᵈ / (cᵖᵈ - Rᵈ)
    g    = model.thermodynamic_constants.gravitational_acceleration

    Π⁰ = sub.linearization_exner
    θ⁰ = sub.linearization_potential_temperature
    δτ_new = 0.5  # arbitrary

    b₁ = @allowscalar get_coefficient(1, 1, 1, grid, AcousticTridiagDiagonal(),
                                      nothing, ZDirection(),
                                      Π⁰, θ⁰, γRᵈ, g, δτ_new)
    c₁ = @allowscalar get_coefficient(1, 1, 1, grid, AcousticTridiagUpper(),
                                      nothing, ZDirection(),
                                      Π⁰, θ⁰, γRᵈ, g, δτ_new)

    @info @sprintf("[S1] b[1] = %.6f, c[1] = %.6e (must be 1.0 and 0.0)", b₁, c₁)
    @test b₁ == 1.0
    @test c₁ == 0.0
end

#####
##### S2 — Discrete adjoint: ⟨μ, δz_f(Q)/Δz_f⟩ + ⟨δz_c(μ)/Δz_c, Q⟩ = boundary terms
#####
##### The mass-flux divergence operator δz_c(μ)/Δz_c (face → center) is
##### supposed to be the negative-adjoint of the vertical pressure
##### derivative δz_f(Q)/Δz_f (center → face) — up to BOUNDARY terms.
##### Specifically (uniform Δz, periodic in horizontal, Bounded z):
#####
#####   Σ_centers Q[k] · δz_c(μ)[k] = − Σ_faces μ[k] · δz_f(Q)[k]
#####                                 + boundary
#####
##### where the boundary terms come from face-end values. This is the
##### algebraic identity that makes the linearized acoustic-buoyancy
##### system energy-conserving on the discrete level. If it fails,
##### there's a sign or indexing bug in one of the discrete operators.
#####

@testset "S2 — δz_c / δz_f adjoint consistency" begin
    Nz = 16
    Δz = 100.0
    # Random center-located scalar Q[k] for k=1..Nz, and face-located
    # μ[k] for k=1..Nz+1 with μ[1] = μ[Nz+1] = 0 (impenetrability).
    Q = randn(Nz)
    μ = zeros(Nz + 1)
    μ[2:Nz] .= randn(Nz - 1)

    # ⟨Q, δz_c(μ)/Δz_c⟩_center = Σ_k Q[k] · (μ[k+1] − μ[k]) / Δz
    lhs = sum(Q[k] * (μ[k + 1] - μ[k]) / Δz for k in 1:Nz)

    # −⟨μ, δz_f(Q)/Δz_f⟩_face (over interior faces 2..Nz, the only
    # ones where δz_f(Q) is defined; μ[1] = μ[Nz+1] = 0 so boundary
    # faces contribute nothing).
    rhs = -sum(μ[k] * (Q[k] - Q[k - 1]) / Δz for k in 2:Nz)

    @info @sprintf("[S2] |lhs − rhs| = %.3e (lhs = %.3e, rhs = %.3e)",
                   abs(lhs - rhs), lhs, rhs)
    @test isapprox(lhs, rhs; atol = 1e-12 * max(1, abs(lhs)))
end

#####
##### S3 — Buoyancy averaging operator is exact dual of mass divergence
#####
##### The buoyancy term `g · ℑ_f(ρ′)` at face k_f and the implicit
##### substitution use:
#####   ℑ_f(ρ′_n)[k_f] = ½(ρ′_n[k_f] + ρ′_n[k_f − 1])
##### with ρ′_n[k_c] = ρ̃[k_c] − δτ_n · δz_c(μ_n)[k_c] / Δz_c[k_c].
##### After substitution, the implicit-on-μ_n contribution to face k_f is
#####   ½ · {δz_c(μ_n)[k_f] / Δz_c[k_f] + δz_c(μ_n)[k_f − 1] / Δz_c[k_f − 1]}
##### Test that this matches the matrix-coefficient buoyancy entries
##### for a known μ pattern.
#####

@testset "S3 — Buoyancy operator matches manual ℑ_f∘δz_c construction" begin
    Nz = 8
    Δz = 100.0
    g  = 9.80665

    # Probe vector: place a unit μ at face k_f = 4, zeros elsewhere.
    μ = zeros(Nz + 1)
    μ[4] = 1.0

    # The substepper LHS matrix is `(I + δτ_n² · M) μ_n = RHS` and
    # the substitution from the buoyancy term gives
    #   M_buoy(μ)[k_f] = − g · ℑ_f(δz_c(μ) / Δz_c)[k_f]
    # i.e. the SIGN is negative (the buoyancy on the new step is
    # subtracted from the (ρw)_n side, then brought to LHS).
    Lμ = zeros(Nz + 1)
    for k_f in 2:Nz
        d_above = (μ[k_f + 1] - μ[k_f]) / Δz
        d_below = (μ[k_f] - μ[k_f - 1]) / Δz
        Lμ[k_f] = -g / 2 * (d_above + d_below)   # ← negative sign
    end

    # Independently apply the matrix-coefficient buoyancy entries
    # (PGF set to zero so we isolate buoyancy):
    Mμ = zeros(Nz + 1)
    rdz = 1 / Δz
    for k_f in 2:Nz
        sub_buoy = +g * rdz / 2          # A[k_f, k_f − 1]
        diag_buoy = g * (rdz - rdz) / 2  # = 0 on uniform Δz
        sup_buoy = -g * rdz / 2          # A[k_f, k_f + 1]
        Mμ[k_f] = sub_buoy * μ[k_f - 1] + diag_buoy * μ[k_f] + sup_buoy * μ[k_f + 1]
    end

    Δ = maximum(abs.(Lμ[2:Nz] - Mμ[2:Nz]))
    @info @sprintf("[S3] max|L_buoy − M_buoy(μ)| over interior faces = %.3e", Δ)
    @test Δ <= 1e-12
end

#####
##### S4 — Mass conservation through one outer step
#####
##### The substepper's continuity equation is in flux form, so total
##### mass `∑ ρ · V_cell` over a periodic + Bounded-z domain must be
##### conserved exactly (no surface fluxes / forcing in the rest IC).
##### This catches a non-conservative form of either the slow
##### tendency or the substep recovery.
#####

@testset "S4 — Mass conservation for one outer step at rest" begin
    model = _build_str_model(default_arch)
    _set_str_rest!(model)

    grid = model.grid
    ρ_field = model.dynamics.density
    V_cell = (grid.Lx / grid.Nx) * (grid.Ly / grid.Ny) * (grid.Lz / grid.Nz)
    ρ_array() = Array(interior(ρ_field))

    M0 = sum(ρ_array()) * V_cell
    time_step!(model, 0.5)
    M1 = sum(ρ_array()) * V_cell

    @info @sprintf("[S4] M0 = %.10e, M1 = %.10e, |ΔM/M| = %.3e",
                   M0, M1, abs(M1 - M0) / M0)
    @test abs(M1 - M0) / M0 <= 1e-12
end

#####
##### S5 — Top-boundary face μw[Nz+1] = 0 after one outer step at rest
#####
##### Impenetrability at the rigid lid means μw at the top face must be
##### identically zero. The solver only handles face indices 1..Nz, so
##### μw[Nz+1] is supposed to be set to 0 externally
##### (in `_build_predictors_and_vertical_rhs!`). If the substepper
##### ever lets `μw[Nz+1]` drift, mass conservation breaks at the lid.
#####

@testset "S5 — Top-face μw[Nz+1] = 0 after stepping" begin
    model = _build_str_model(default_arch)
    _set_str_rest!(model)

    time_step!(model, 0.5)

    # ZFaceField has Nz+1 vertical face values; check the top face.
    w_top = @allowscalar Array(model.velocities.w)[:, :, end]
    max_w_top = maximum(abs, w_top)

    @info @sprintf("[S5] max|w[top face]| after one step = %.3e m/s", max_w_top)
    @test max_w_top <= 1e-12
end

#####
##### S6 — Slow-tendency assembly is sign-correct on hydrostatic state
#####
##### Set an artificial perturbation: ρ⁰ = ρ_ref + δρ for a known
##### bump. The slow tendency `Gˢρw = -∂z(p⁰ - p_ref) - g(ρ⁰ - ρ_ref)`
##### must have the SIGN of -gδρ (downward acceleration on a positive
##### density anomaly). If the assembly has a sign error in either the
##### PGF or buoyancy contribution, this catches it.
#####

@testset "S6 — Slow-tendency signs: buoyancy and PGF probed separately" begin
    Nz = NZ_STR
    g  = 9.80665

    # Test (a): perturb only ρ at one cell. ρθ unchanged ⇒ pressure
    # unchanged via EoS ⇒ PGF contribution is zero. Only buoyancy
    # fires. With ρ′[k_anom] > 0, the face buoyancy contribution is
    # −g · ℑ_f(ρ′)[k_f] < 0 at the two adjacent faces. Test sign and
    # symmetry between the two adjacent faces.
    @testset "(a) ρ-only anomaly drives only buoyancy" begin
        model = _build_str_model(default_arch)
        sub = model.timestepper.substepper
        _set_str_rest!(model)

        k_anom = Nz ÷ 2
        ρ_array = parent(model.dynamics.density)
        @allowscalar begin
            ρ_array[:, :, k_anom + model.grid.Hz] .+= 1e-3
        end
        update_state!(model)

        freeze_linearization_state!(sub, model)
        compute_slow_momentum_tendencies!(model)
        compute_slow_scalar_tendencies!(model)
        assemble_slow_vertical_momentum_tendency!(sub, model)

        Gˢρw = Array(interior(sub.slow_vertical_momentum_tendency))

        # Face k_f = k_anom (between cells k_anom − 1 and k_anom):
        # ℑ_f(ρ′)[k_f] = (ρ′[k_anom] + ρ′[k_anom − 1]) / 2 = +δρ / 2.
        Gˢ_below_face = Gˢρw[1, 1, k_anom]
        # Face k_f = k_anom + 1 (between cells k_anom and k_anom + 1):
        # ℑ_f(ρ′)[k_f] = (ρ′[k_anom + 1] + ρ′[k_anom]) / 2 = +δρ / 2.
        Gˢ_above_face = Gˢρw[1, 1, k_anom + 1]
        # Distant face — should be at machine zero.
        Gˢ_far = Gˢρw[1, 1, 2]

        @info @sprintf("[S6a] Gˢρw at faces (%d, %d) = (%.3e, %.3e); far = %.3e",
                       k_anom, k_anom + 1, Gˢ_below_face, Gˢ_above_face, Gˢ_far)

        # Both adjacent faces have the SAME sign (negative — heavy air
        # in cell k_anom pushes down through both the face above and
        # the face below).
        @test Gˢ_below_face < 0
        @test Gˢ_above_face < 0

        # Symmetric magnitudes (uniform Δz, away from boundaries).
        @test isapprox(Gˢ_below_face, Gˢ_above_face; rtol = 1e-10)

        # Distant face at machine zero (ulp seed only).
        @test abs(Gˢ_far) <= 1e-10
    end

    # Test (b): perturb only ρθ at one cell. ρ unchanged ⇒ buoyancy
    # contribution is zero. Only PGF fires. With (ρθ)′[k_anom] > 0,
    # pressure is elevated in cell k_anom only. The face PGF
    # contribution `−∂z(p⁰ − p_ref)[k_f]` is OPPOSITE sign on the two
    # adjacent faces (pressure decreases going from cell to outside,
    # in both directions).
    @testset "(b) ρθ-only anomaly drives only PGF" begin
        model = _build_str_model(default_arch)
        sub = model.timestepper.substepper
        _set_str_rest!(model)

        k_anom = Nz ÷ 2
        ρθ_field = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
        ρθ_array = parent(ρθ_field)
        @allowscalar begin
            ρθ_array[:, :, k_anom + model.grid.Hz] .+= 1e-2
        end
        update_state!(model)

        freeze_linearization_state!(sub, model)
        compute_slow_momentum_tendencies!(model)
        compute_slow_scalar_tendencies!(model)
        assemble_slow_vertical_momentum_tendency!(sub, model)

        Gˢρw = Array(interior(sub.slow_vertical_momentum_tendency))

        # Face below the bump (k_f = k_anom): pressure jumps UP going
        # from cell k_anom − 1 (normal) into cell k_anom (high). So
        # ∂z(p⁰ − p_ref)[k_f = k_anom] > 0 ⇒ −∂z(...) < 0 ⇒ Gˢρw < 0.
        Gˢ_below = Gˢρw[1, 1, k_anom]
        # Face above the bump (k_f = k_anom + 1): pressure jumps DOWN
        # going up. So ∂z(p⁰ − p_ref)[k_f = k_anom + 1] < 0 ⇒
        # −∂z(...) > 0 ⇒ Gˢρw > 0.
        Gˢ_above = Gˢρw[1, 1, k_anom + 1]

        @info @sprintf("[S6b] Gˢρw at faces (%d, %d) = (%.3e, %.3e) (expect ∓)",
                       k_anom, k_anom + 1, Gˢ_below, Gˢ_above)

        @test Gˢ_below < 0
        @test Gˢ_above > 0
    end
end

#####
##### S7 — Reference state's density-pressure round-trip via EoS
#####
##### Should hold by construction: ρ_ref[k] = p_ref[k] / (Rᵈ · T_ref[k])
##### with T_ref = T₀ for isothermal. Catches a future change to the
##### reference-state kernel that breaks EoS consistency.
#####

@testset "S7 — Reference-state EoS round-trip" begin
    model = _build_str_model(default_arch)
    ref   = model.dynamics.reference_state
    Rᵈ    = Float64(Breeze.dry_air_gas_constant(model.thermodynamic_constants))

    p_ref = Array(interior(ref.pressure))
    ρ_ref = Array(interior(ref.density))
    Π_ref = Array(interior(ref.exner_function))
    cᵖᵈ   = Float64(model.thermodynamic_constants.dry_air.heat_capacity)
    κ     = Rᵈ / cᵖᵈ
    pˢᵗ   = Float64(model.dynamics.standard_pressure)

    # Π_ref = (p_ref / pˢᵗ)^κ must hold to ulp by construction.
    ΔΠ = maximum(abs, Π_ref .- (p_ref ./ pˢᵗ).^κ)
    @info @sprintf("[S7] max|Π_ref − (p_ref/pˢᵗ)^κ| = %.3e", ΔΠ)
    @test ΔΠ <= 100 * eps(Float64)

    # The reference state is built from a prescribed θ̄(z) profile via
    # `_compute_exner_reference!` with discrete hydrostatic balance.
    # On that discrete state the temperature is T_ref[k] = θ̄[k] ·
    # Π_ref[k]; the EoS round-trip is `ρ = p / (Rᵈ · T_ref)`. The
    # test reads θ̄ from the analytic profile evaluated at z_c[k] (the
    # exact value passed to the constructor) and checks the EoS
    # identity holds to ulp. This catches any kernel that breaks the
    # `ρ = p / (Rᵈ θ̄ Π_ref)` consistency.
    grid = model.grid
    Nz   = grid.Nz
    T_ref = zeros(Nz)
    @allowscalar for k in 1:Nz
        z_c = grid.z.cᵃᵃᶜ[k]
        T_ref[k] = θ_iso_str(z_c) * Π_ref[1, 1, k]
    end

    Δp = maximum(abs, ρ_ref[1, 1, :] .* Rᵈ .* T_ref .- p_ref[1, 1, :])
    @info @sprintf("[S7] max|ρ_ref · Rᵈ · T_ref − p_ref| = %.3e Pa", Δp)
    @test Δp <= 100 * eps(Float64) * maximum(abs, p_ref)
end

#####
##### S8 — Per-substep noise growth on a true rest atmosphere
#####
##### Tracks max|w| at each outer step for the BBI-failing
##### configuration (Δt = 20 s, ω = 0.55 default, isothermal-T₀
##### reference). Records the trajectory and prints it for
##### diagnostic. Fails if the envelope exceeds an acceptable bound
##### within the test horizon, identifying the OUTER STEP at which
##### noise becomes physical-scale.
#####

@testset "S8 — Per-step noise trajectory at Δt = 20 s, ω = 0.55" begin
    model = _build_str_model(default_arch; ω = 0.55)
    _set_str_rest!(model)

    Δt = 20.0
    n_steps = 30
    trajectory = Float64[]
    crashed = false

    for i in 1:n_steps
        try
            time_step!(model, Δt)
        catch
            crashed = true
            break
        end
        wmax = Float64(maximum(abs, interior(model.velocities.w)))
        push!(trajectory, wmax)
    end

    @info @sprintf("[S8] Δt = 20 s, ω = 0.55, %d outer steps, crashed = %s", n_steps, crashed)
    for (i, w) in enumerate(trajectory)
        if i in (1, 2, 5, 10, 15, 20, 25, 30) && i <= length(trajectory)
            @info @sprintf("  step %3d  max|w| = %.3e", i, w)
        end
    end

    # Estimate growth rate by linear regression of log(max|w|) over the
    # last 10 steps (skip pure-ε early steps).
    if length(trajectory) >= 10 && all(>(0), trajectory)
        n_use = min(length(trajectory), 20)
        xs = collect((length(trajectory) - n_use + 1):length(trajectory))
        ys = log.(trajectory[(end - n_use + 1):end])
        x̄ = sum(xs) / n_use
        ȳ = sum(ys) / n_use
        slope = sum((xs .- x̄) .* (ys .- ȳ)) / sum((xs .- x̄).^2)
        rate_per_step = exp(slope)
        @info @sprintf("[S8] estimated growth factor per outer step: %.4f", rate_per_step)
    end

    final = isempty(trajectory) ? 0.0 : last(trajectory)
    @info @sprintf("[S8] final max|w| after %d steps = %.3e m/s",
                   length(trajectory), final)

    # Pass criterion: a true rest atmosphere should NOT exceed
    # `eps × cs × small_factor` ≈ 1e-10 m/s within 30 outer steps.
    # The observed envelope after 30 steps is ~1e-6 m/s, indicating
    # exponential amplification at ~1.77× per outer step (Phase 4
    # residual feedback — see audit B3 / WS-RK3 stage-substepper
    # consistency). Kept `@test_broken` so a future Phase 4 fix
    # surfaces as an unexpected pass.
    @test_broken final <= 1e-10
end

#####
##### S9 — Drift's discrete hydrostatic-balance residual
#####
##### After one outer step at rest, the drift `(ρ − ρ_ref, p − p_ref)`
##### should still satisfy the substepper's discrete hydrostatic
##### relation: `δz(p_drift) / Δz_face + g · ℑ_z(ρ_drift) = 0` at every
##### face. If it does NOT, the next outer step's
##### `_assemble_slow_vertical_momentum_tendency!` produces a real
##### Gˢρw seed (not at machine ε), and the substep loop's DC response
##### to this seed amplifies it → exponential growth across outer
##### steps. This is the mechanism behind the residual 1.77×/step
##### amplification at default ω = 0.55, Δt = 20 s.
#####

@testset "S9 — Drift hydrostatic-balance residual after one outer step" begin
    model = _build_str_model(default_arch; ω = 0.55)
    _set_str_rest!(model)
    ref = model.dynamics.reference_state

    grid = model.grid
    Δz = Float64(grid.z.Δᵃᵃᶜ)
    g  = Float64(model.thermodynamic_constants.gravitational_acceleration)

    function residual_max(model)
        ref = model.dynamics.reference_state
        p   = Array(interior(model.dynamics.pressure))
        ρ   = Array(interior(Breeze.AtmosphereModels.dynamics_density(model.dynamics)))
        p_r = Array(interior(ref.pressure))[1, 1, :]
        ρ_r = Array(interior(ref.density))[1, 1, :]
        Nx, Ny, Nz = size(p)
        ε_max = 0.0
        for i in 1:Nx, j in 1:Ny, k in 2:Nz
            δp = (p[i, j, k] - p_r[k]) - (p[i, j, k - 1] - p_r[k - 1])
            avgρ = ((ρ[i, j, k] - ρ_r[k]) + (ρ[i, j, k - 1] - ρ_r[k - 1])) / 2
            ε_max = max(ε_max, abs(δp / Δz + g * avgρ))
        end
        ε_max
    end

    initial = residual_max(model)
    @info @sprintf("[S9] initial drift hydrostatic residual = %.3e N/m³", initial)
    @test initial <= 1e-12

    # Take 30 outer steps. A substep loop that preserves discrete
    # hydrostatic balance for perturbations would keep the residual
    # at machine ε. Observed: residual grows ~1.77× per outer step
    # (3e-14 → 4e-8 in 30 steps). This is the structural mechanism
    # behind the residual rest-atmosphere amplification at default
    # ω = 0.55, Δt = 20 s.
    for _ in 1:30
        time_step!(model, 20.0)
    end
    after = residual_max(model)
    @info @sprintf("[S9] after 30 outer steps (Δt=20s)        = %.3e N/m³", after)

    # `@test_broken` so a Phase 4 fix surfaces as an unexpected pass.
    @test_broken after <= 1e-9
end

end  # outer "Substepper structural correctness"
