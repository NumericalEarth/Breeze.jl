#####
##### Verify that Oceananigans' BatchedTridiagonalSolver produces the same answer
##### as Breeze's column-kernel Thomas sweep on a tridiagonal system that has the
##### same structure as the MPAS acoustic vertical solve.
#####
##### This is a prerequisite for migrating the column kernel to use
##### BatchedTridiagonalSolver (refactor Step 10). It does NOT test the migration
##### itself; it only verifies that the algorithm + index conventions are
##### compatible, so that the migration can be done safely.
#####
##### Two key facts about BatchedTridiagonalSolver (from
##### Oceananigans.jl/src/Solvers/batched_tridiagonal_solver.jl):
#####
##### 1. The solver expects `a[k-1]` (lower) and `c[k-1]` (upper) in row k
#####    (Press et al. 1992 §2.4 indexing). The MPAS column kernel computes a_k
#####    and c_k _at face k_, so populating the solver's a/c arrays from MPAS
#####    coefficients requires shifting the index by one.
#####
##### 2. The solver elides the forward update with `ifelse` if `|β| < 10ε` to
#####    guard against non-diagonally-dominant systems. The MPAS acoustic
#####    tridiagonal has b_k ≥ 1, so this branch is never taken.
#####

using Test
using Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Solvers: BatchedTridiagonalSolver, solve!, get_coefficient
using Oceananigans.Grids: ZDirection
import Oceananigans.Solvers: get_coefficient

#####
##### Reference Thomas sweep — a stand-alone copy of the column-kernel sweep.
#####
##### The signature mirrors what the Breeze column kernel does internally for
##### one (i,j) column. It is purely a tridiagonal solver — no acoustic-specific
##### code. This lets us compare it head-to-head with BatchedTridiagonalSolver.
#####
##### Convention (matches the Breeze column kernel):
#####   - The unknowns are ϕ[2..N]; ϕ[1] is fixed by a Dirichlet condition (= 0).
#####   - Row k=2..N is:  a_k ϕ[k-1] + b_k ϕ[k] + c_k ϕ[k+1] = rhs[k]
#####   - At k=2: a_2 = 0  (no coupling to ϕ[1])
#####   - At k=N: c_N = 0  (no coupling to ϕ[N+1])
#####
##### γ_scratch is a per-face scratch buffer (same size as ϕ).
#####
function reference_mpas_thomas_sweep!(ϕ, a, b, c, rhs, γ_scratch, N)
    # Forward sweep: face k = 2..N
    for k in 2:N
        γ_prev = ifelse(k == 2, zero(eltype(ϕ)), γ_scratch[k - 1])
        α_k = 1 / (b[k] - a[k] * γ_prev)
        γ_scratch[k] = c[k] * α_k
        ϕ[k] = (rhs[k] - a[k] * ϕ[k - 1]) * α_k
    end
    # Backward sweep: face k = N-1..2
    for k in (N - 1):-1:2
        ϕ[k] = ϕ[k] - γ_scratch[k] * ϕ[k + 1]
    end
    return ϕ
end

#####
##### Helper: build a synthetic, diagonally-dominant tridiagonal that has the
##### same structural properties as the MPAS acoustic system.
##### - b > 0 and dominant
##### - a, c arbitrary sign
##### - boundary rows have a[2] = 0 and c[N] = 0
#####
function synthetic_acoustic_tridiagonal(N::Int; seed = 1)
    a   = zeros(N + 1)
    b   = zeros(N + 1)
    c   = zeros(N + 1)
    rhs = zeros(N + 1)
    # Use a deterministic seed-derived pattern so the test is reproducible.
    s = Float64(seed)
    for k in 2:N
        a[k]   = -0.10 - 0.01 * sin(s + k)
        c[k]   = -0.12 - 0.02 * cos(s + k)
        b[k]   =  1.50 + 0.05 * k                # always > |a| + |c|
        rhs[k] =  0.30 * sin(s + 0.5 * k) - 0.10 * k
    end
    a[2] = 0.0   # k=2: no coupling to ϕ[1]
    c[N] = 0.0   # k=N: no coupling to ϕ[N+1]
    return a, b, c, rhs
end

#####
##### Run BatchedTridiagonalSolver on the same system, accounting for its
##### a[k-1]/c[k-1] indexing convention. The solver acts on rows 1..N_solver
##### where N_solver = N - 1 (we drop the k=1 boundary row from the system,
##### since ϕ[1] = 0 is enforced separately).
#####
##### Mapping (face k → solver row k_s):
##### face k = 2  → solver row 1
##### face k = 3  → solver row 2
##### ...
##### face k = N  → solver row N - 1
#####
##### Solver coefficient arrays (with the index shift):
#####  solver_a[k_s]   = our a_k where k_s = k - 1, so solver_a[k-1] = a_k
#####                   But the solver USES `a[k_s - 1]` in row k_s, so:
#####                   solver row k_s reads solver_a[k_s - 1] = ?
#####  We want solver row k_s to use coefficient a_k where k = k_s + 1.
#####  So solver_a[k_s - 1] = a_{k_s + 1}, i.e., solver_a[m] = a_{m + 2} for m = 1..N_solver-1.
#####
function solve_with_batched(a_face, b_face, c_face, rhs_face, N)
    # Build a Breeze grid with Nz = N - 1 levels (the number of solver rows).
    N_solver = N - 1
    grid = RectilinearGrid(CPU(), size = (1, 1, N_solver), halo = (1, 1, 1),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))

    # Allocate the solver coefficient arrays. The convention is that
    #   row k_s reads:  solver_a[k_s - 1] (the lower) and solver_c[k_s - 1] (the upper)
    # So we shift the MPAS a/c by one in the solver direction.
    sa = zeros(1, 1, N_solver)
    sb = zeros(1, 1, N_solver)
    sc = zeros(1, 1, N_solver)
    rhs3 = zeros(1, 1, N_solver)

    # Diagonal and RHS:
    #   solver.b[k_s] = our b at face k = k_s + 1
    #   solver.rhs[k_s] = our rhs at face k = k_s + 1
    for k_s in 1:N_solver
        k = k_s + 1
        sb[1, 1, k_s]   = b_face[k]
        rhs3[1, 1, k_s] = rhs_face[k]
    end

    # Off-diagonals — Oceananigans uses ASYMMETRIC index conventions for a and c.
    # From the BatchedTridiagonalSolver docstring matrix at row k:
    #   row k reads lower entry labeled a^{k-1} → array index k-1   (shift)
    #   row k reads upper entry labeled c^k     → array index k     (no shift)
    # Translating to standard Thomas notation (a_k = lower at row k, c_k = upper at row k):
    #   standard a_k = solver.a[k - 1]    → solver.a[m] = standard a_{m+1}
    #   standard c_k = solver.c[k]        → solver.c[m] = standard c_m
    # Mapping face k → solver row k_s = k - 1, our standard at solver row k_s is face k = k_s + 1:
    #   solver.a[m] = standard a_{m+1} = our a_face at k = (m+1)+1 = m+2  (shift +2)
    #   solver.c[m] = standard c_m     = our c_face at k = m+1            (shift +1)
    # Both arrays cover m = 1..N_solver - 1; the last solver row has no upper, the first has no lower.
    for m in 1:N_solver - 1
        sa[1, 1, m] = a_face[m + 2]
        sc[1, 1, m] = c_face[m + 1]
    end

    scratch = zeros(1, 1, N_solver)
    solver = BatchedTridiagonalSolver(grid;
                                      lower_diagonal = sa,
                                      diagonal = sb,
                                      upper_diagonal = sc,
                                      scratch = scratch,
                                      tridiagonal_direction = ZDirection())

    ϕ_solver = zeros(1, 1, N_solver)
    solve!(ϕ_solver, solver, rhs3)

    # Map solver result back into face-indexed array.
    ϕ_face = zeros(N + 1)
    for k_s in 1:N_solver
        k = k_s + 1
        ϕ_face[k] = ϕ_solver[1, 1, k_s]
    end
    ϕ_face[1]     = 0.0
    ϕ_face[N + 1] = 0.0
    return ϕ_face
end

#####
##### Function-based coefficients (no allocated a/b/c arrays).
#####
##### `get_coefficient` is dispatched on the coefficient type. We define three
##### small wrapper types — one each for the lower, diagonal, and upper of the
##### face-system tridiagonal — that hold a 1D vector indexed by face. The
##### dispatch then looks up the value on the fly using the face index inferred
##### from the solver row index.
#####
##### This is a stand-in for the real acoustic-substepper case where the
##### lower/diagonal/upper would be inline functions of grid metrics, the
##### frozen reference state, and the scalar Δτᵋ — none of which need an
##### Nx × Ny × Nz array.
#####

struct LowerFaceCoeffs
    a_face::Vector{Float64}
end

struct DiagFaceCoeffs
    b_face::Vector{Float64}
end

struct UpperFaceCoeffs
    c_face::Vector{Float64}
end

# Mapping (matches solve_with_batched):
#   solver_a[m] = standard a_{m+1} = our a_face[m+2]   (m = 1..N_solver - 1)
#   solver_b[k_s]                  = our b_face[k_s+1] (k_s = 1..N_solver)
#   solver_c[m] = standard c_m     = our c_face[m+1]   (m = 1..N_solver - 1)
#
# Solver row index k_s comes through as the third argument of get_coefficient.
@inline get_coefficient(i, j, k, grid, coef::LowerFaceCoeffs, p, ::ZDirection, args...) =
    @inbounds coef.a_face[k + 2]
@inline get_coefficient(i, j, k, grid, coef::DiagFaceCoeffs, p, ::ZDirection, args...) =
    @inbounds coef.b_face[k + 1]
@inline get_coefficient(i, j, k, grid, coef::UpperFaceCoeffs, p, ::ZDirection, args...) =
    @inbounds coef.c_face[k + 1]

function solve_with_batched_functional(a_face, b_face, c_face, rhs_face, N)
    N_solver = N - 1
    grid = RectilinearGrid(CPU(), size = (1, 1, N_solver), halo = (1, 1, 1),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))

    sa = LowerFaceCoeffs(a_face)
    sb = DiagFaceCoeffs(b_face)
    sc = UpperFaceCoeffs(c_face)

    rhs3 = zeros(1, 1, N_solver)
    for k_s in 1:N_solver
        rhs3[1, 1, k_s] = rhs_face[k_s + 1]
    end

    scratch = zeros(1, 1, N_solver)
    solver = BatchedTridiagonalSolver(grid;
                                      lower_diagonal = sa,
                                      diagonal = sb,
                                      upper_diagonal = sc,
                                      scratch = scratch,
                                      tridiagonal_direction = ZDirection())

    ϕ_solver = zeros(1, 1, N_solver)
    solve!(ϕ_solver, solver, rhs3)

    ϕ_face = zeros(N + 1)
    for k_s in 1:N_solver
        ϕ_face[k_s + 1] = ϕ_solver[1, 1, k_s]
    end
    return ϕ_face
end

@testset "BatchedTridiagonalSolver vs MPAS Thomas sweep" begin
    @testset "Synthetic acoustic-style tridiagonal, N = $N" for N in (5, 10, 40)
        a, b, c, rhs = synthetic_acoustic_tridiagonal(N; seed = N)

        # Reference: Breeze column-kernel sweep
        ϕ_ref = zeros(N + 1)
        γ     = zeros(N + 1)
        reference_mpas_thomas_sweep!(ϕ_ref, a, b, c, rhs, γ, N)

        # BatchedTridiagonalSolver
        ϕ_solver = solve_with_batched(a, b, c, rhs, N)

        # The two solutions must agree to machine precision
        max_abs_diff = maximum(abs, ϕ_ref .- ϕ_solver)
        max_abs_ref  = maximum(abs, ϕ_ref)
        rel_diff     = max_abs_diff / max(max_abs_ref, eps())

        @test rel_diff < 1e-12

        # Sanity: residual of the original system should be small for both.
        function residual(ϕ, a, b, c, rhs, N)
            r = zeros(N + 1)
            for k in 2:N
                ϕ_km1 = k == 2 ? 0.0 : ϕ[k - 1]
                ϕ_kp1 = k == N ? 0.0 : ϕ[k + 1]
                r[k] = a[k] * ϕ_km1 + b[k] * ϕ[k] + c[k] * ϕ_kp1 - rhs[k]
            end
            return maximum(abs, r)
        end
        @test residual(ϕ_ref,    a, b, c, rhs, N) < 1e-12
        @test residual(ϕ_solver, a, b, c, rhs, N) < 1e-12
    end

    @testset "Function-based coefficients (no allocated arrays), N = $N" for N in (5, 10, 40)
        a, b, c, rhs = synthetic_acoustic_tridiagonal(N; seed = N + 100)

        ϕ_ref = zeros(N + 1)
        γ     = zeros(N + 1)
        reference_mpas_thomas_sweep!(ϕ_ref, a, b, c, rhs, γ, N)

        ϕ_solver_func = solve_with_batched_functional(a, b, c, rhs, N)

        rel_diff = maximum(abs, ϕ_ref .- ϕ_solver_func) /
                   max(maximum(abs, ϕ_ref), eps())
        @test rel_diff < 1e-12
    end

    @testset "Stress test: stronger off-diagonals (still diagonally dominant)" begin
        N = 20
        a, b, c, rhs = synthetic_acoustic_tridiagonal(N; seed = 7)
        # Push off-diagonals up while keeping b > |a| + |c|
        for k in 2:N
            a[k] *= 5
            c[k] *= 5
            b[k] += 1.5
        end
        a[2] = 0
        c[N] = 0

        ϕ_ref = zeros(N + 1)
        γ     = zeros(N + 1)
        reference_mpas_thomas_sweep!(ϕ_ref, a, b, c, rhs, γ, N)

        ϕ_solver = solve_with_batched(a, b, c, rhs, N)

        @test maximum(abs, ϕ_ref .- ϕ_solver) / maximum(abs, ϕ_ref) < 1e-12
    end
end
