#####
##### Column-tridiag spectrum diagnostic for the acoustic substepper.
#####
##### Reconstructs (in pure Julia) the LHS matrix `A = I + δτ_new² M`
##### that the substepper's BatchedTridiagonalSolver applies per column
##### to update `μw`. We then study `eigvals(M)` — the spectrum of the
##### *operator* δτ_new factors out of.
#####
##### For the implicit half of off-centered Crank–Nicolson to be
##### stable for any `δτ_new`, M must have *non-negative real part on
##### every eigenvalue*. A negative-real-part eigenvalue means `A` is
##### singular at some `δτ_new` and the amplification matrix
##### leaves the unit disk — exactly the BBI report's hypothesis 1
##### / 2 (predictor / matrix weight mismatch, or buoyancy off-diagonal
##### sign asymmetry on stretched grids).
#####
##### This test is a diagnostic for the pristine substepper plan's S8
##### / F2 (validation/substepping/PRISTINE_SUBSTEPPER_PLAN.md) and is
##### the discrete analogue of the rest-atmosphere drift test (T4) but
##### at the matrix-spectrum level — when it goes red, T4 will be red
##### too, and the failure mode is localized to the column-matrix
##### coefficients.
#####
##### The matrix coefficients mirror exactly the `get_coefficient`
##### dispatch in `src/CompressibleEquations/acoustic_substepping.jl`
##### lines 509-561.
#####

using Breeze
using Breeze.CompressibleEquations: AcousticSubstepper,
                                    freeze_outer_step_state!

using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using GPUArraysCore: @allowscalar
using LinearAlgebra: eigvals
using Printf
using Test

const T₀_EIGS = 250.0
const Lz_EIGS = 30e3
const NZ_EIGS = 64
const G_EIGS  = 9.80665
const CPD_EIGS = 1005.0

θ_isothermal_eigs(z) = T₀_EIGS * exp(G_EIGS * z / (CPD_EIGS * T₀_EIGS))
θ_isothermal_xyz_eigs(x, y, z) = θ_isothermal_eigs(z)

function _build_eigs_model(arch)
    grid = RectilinearGrid(arch;
                           size = (8, 8, NZ_EIGS), halo = (5, 5, 5),
                           x = (0, 100e3), y = (0, 100e3), z = (0, Lz_EIGS),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    td = SplitExplicitTimeDiscretization(; forward_weight = 0.55,
                                           damping = NoDivergenceDamping())
    dyn = CompressibleDynamics(td; reference_potential_temperature = θ_isothermal_eigs)
    return AtmosphereModel(grid; dynamics = dyn,
                                  thermodynamic_constants = constants,
                                  timestepper = :AcousticRungeKutta3)
end

# Extract a single column's worth of Π⁰_face, θ⁰_face, Δz at centers and
# faces — everything the matrix-coefficient functions read. Move to CPU
# for eigvals.
function _extract_column(model, ifix, jfix)
    grid = model.grid
    sub  = model.timestepper.substepper
    Nz   = grid.Nz

    Π⁰_cpu = @allowscalar Array(parent(sub.outer_step_exner))
    θ⁰_cpu = @allowscalar Array(parent(sub.outer_step_potential_temperature))

    # Translate (i, j) into halo-shifted indices used by `parent`.
    Hx = grid.Hx; Hy = grid.Hy; Hz = grid.Hz
    icell = ifix + Hx
    jcell = jfix + Hy

    Π⁰_col = Float64[Π⁰_cpu[icell, jcell, k + Hz] for k in 1:Nz]
    θ⁰_col = Float64[θ⁰_cpu[icell, jcell, k + Hz] for k in 1:Nz]

    # For the regular grids we use here, vertical spacing is uniform.
    Δz = Float64(Lz_EIGS / Nz)
    return (Π⁰_col, θ⁰_col, Δz, Nz)
end

@inline _theta_face(θ⁰_col, k_face, Nz) =
    (k_face >= 2 && k_face <= Nz) ?
        (θ⁰_col[k_face] + θ⁰_col[k_face - 1]) / 2 :
        zero(eltype(θ⁰_col))

@inline _pi_face(Π⁰_col, k_face, Nz) =
    (k_face >= 2 && k_face <= Nz) ?
        (Π⁰_col[k_face] + Π⁰_col[k_face - 1]) / 2 :
        (k_face <= 1 ? Π⁰_col[1] : Π⁰_col[Nz])

# Build the *normalized* operator M such that A = I + δτ_new² M, using
# the same `get_coefficient` formulas as `AcousticTridiag{Lower,
# Diagonal, Upper}` in acoustic_substepping.jl.
#
# CRITICAL: the BatchedTridiagonalSolver works on rows 1..Nz (matching
# `Nx × Ny × Nz` scratch). Solver row k = 1 carries the bottom-boundary
# row (b = 1, μw[1] = 0); rows k = 2..Nz are interior faces 2..Nz.
# The TOP boundary face k_face = Nz + 1 is NOT a solver row — it's set
# to zero externally. Therefore the eigenvalue analysis covers the
# (Nz − 1) × (Nz − 1) interior block at solver rows 2..Nz, mapped to
# face indices k_face = 2..Nz.
function build_column_M(Π⁰_col, θ⁰_col, Δz, Nz, γRᵈ, g)
    Ni = Nz - 1                        # interior unknowns
    M = zeros(Float64, Ni, Ni)
    rdz = 1 / Δz                       # uniform-Δz grid

    for row in 1:Ni
        k_face = row + 1               # 2..Nz (interior faces only)
        Δz_face = Δz                   # uniform
        Π_face = _pi_face(Π⁰_col, k_face, Nz)

        θ_face = _theta_face(θ⁰_col, k_face, Nz)
        rdz_above = rdz; rdz_below = rdz
        pgf_diag  = γRᵈ * Π_face * θ_face * (rdz_above + rdz_below) / Δz_face
        buoy_diag = g * (rdz_above - rdz_below) / 2
        M[row, row] = pgf_diag + buoy_diag

        if row > 1                     # k_face > 2, sub-diagonal exists
            θ_below = _theta_face(θ⁰_col, k_face - 1, Nz)
            pgf  = -γRᵈ * Π_face * θ_below * rdz / Δz_face
            buoy = +g * rdz / 2
            M[row, row - 1] = pgf + buoy
        end

        if row < Ni                    # k_face < Nz, super-diagonal exists
            θ_above = _theta_face(θ⁰_col, k_face + 1, Nz)
            pgf  = -γRᵈ * Π_face * θ_above * rdz / Δz_face
            buoy = -g * rdz / 2
            M[row, row + 1] = pgf + buoy
        end
    end
    return M
end

@testset "Column tridiag spectrum at rest" begin
    model = _build_eigs_model(default_arch)
    ref   = model.dynamics.reference_state
    set!(model; θ = θ_isothermal_xyz_eigs, ρ = ref.density)
    update_state!(model)

    sub = model.timestepper.substepper
    freeze_outer_step_state!(sub, model)

    Rᵈ  = Float64(Breeze.dry_air_gas_constant(model.thermodynamic_constants))
    cᵖᵈ = Float64(model.thermodynamic_constants.dry_air.heat_capacity)
    γᵈ  = cᵖᵈ / (cᵖᵈ - Rᵈ)
    γRᵈ = γᵈ * Rᵈ
    g   = Float64(model.thermodynamic_constants.gravitational_acceleration)

    Π⁰_col, θ⁰_col, Δz, Nz = _extract_column(model, 1, 1)
    M = build_column_M(Π⁰_col, θ⁰_col, Δz, Nz, γRᵈ, g)
    λ = eigvals(M)

    re = real.(λ)
    im_max = maximum(abs, imag.(λ))
    re_min = minimum(re)
    re_max = maximum(re)

    @info @sprintf("[eigvals] Nz = %d, Δz = %.2f m, γRᵈ = %.4e, g = %.4f",
                   Nz, Δz, γRᵈ, g)
    @info @sprintf("[eigvals] real(λ) ∈ [%.4e, %.4e], max|imag(λ)| = %.4e",
                   re_min, re_max, im_max)

    # PRISTINE_SUBSTEPPER_PLAN.md S8 / F2: for the implicit half of
    # off-centered CN to be stable at every δτ_new, M = (A − I) / δτ_new²
    # must have non-negative real part on every eigenvalue. After
    # Phase 2 reference-state fix and Phase 1 algebra-closure fix,
    # the interior block (rows 2..Nz, mapped to faces 2..Nz, the
    # actual range solved by the BatchedTridiagonalSolver) of M
    # is positive-definite for the isothermal-T₀ reference. The
    # earlier "λ_min ≈ −8.79e−3" measurement was a TEST bug — it
    # included a spurious row k_face = Nz + 1 (outside the solver's
    # range) where `_theta_at_face` returns 0, producing a fake
    # zero-diagonal row.
    @test re_min >= -1e-6 * max(abs(re_min), abs(re_max))
    @test all(isfinite, re)
    @test isfinite(im_max)
end
