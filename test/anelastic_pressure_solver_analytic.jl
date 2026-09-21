include(joinpath(@__DIR__, "setup.jl"))

using Test
using Breeze
using Oceananigans
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver
using Breeze.AnelasticEquations: AnelasticTridiagonalSolverFormulation, solve_for_anelastic_pressure!
using Statistics: mean

@testset "Anelastic pressure solver recovers analytic solution [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=48, z=(0, 1), topology=(Flat, Flat, Bounded))

    #=
    ρᵣ = z, and the predictor momentum ρw = z² - z³ (which vanishes at both boundaries).
    The anelastic pressure ϕ = p′/ρᵣ (kinematic) satisfies ∂z(ρᵣ ∂z ϕ) = ∂z(ρw) / Δt:

    ρw = z² - z³            ⟹ ∂z ρw = 2z - 3z²
    ϕ  = z²/2 - z³/3        ⟹ ∂z ϕ = z - z², so z ∂z ϕ = z² - z³
                           ⟹ ∂z(z ∂z ϕ) = 2z - 3z²

    so with Δt = 1 the discrete solve should recover ϕ = z²/2 - z³/3 (up to the mean, since the
    solve is defined only to within an additive constant by the homogeneous Neumann boundaries).

    We exercise the solver directly here — an `AtmosphereModel` on this `(Flat, Flat, Bounded)` grid
    runs in single-column mode, where the pressure solve is intentionally omitted (see
    `single_column_mode.jl`).
    =#

    # Reference density ρᵣ = z.
    ρᵣ = CenterField(grid)
    set!(ρᵣ, z -> z)
    fill_halo_regions!(ρᵣ)

    # Predictor momentum ρw = z² - z³.
    ρu = XFaceField(grid)
    ρv = YFaceField(grid)
    ρw = ZFaceField(grid)
    set!(ρw, z -> z^2 - z^3)
    fill_halo_regions!(ρw)

    # Solve for the kinematic pressure ϕ = p′/ρᵣ with Δt = 1.
    solver = FourierTridiagonalPoissonSolver(grid; tridiagonal_formulation=AnelasticTridiagonalSolverFormulation(ρᵣ))
    ϕ = CenterField(grid)
    solve_for_anelastic_pressure!(ϕ, solver, (ρu, ρv, ρw), 1)
    fill_halo_regions!(ϕ)

    # Test for zero mean (the solve is defined only up to an additive constant).
    atol = 10 * grid.Nz * eps(FT)
    @test mean(ϕ) ≈ 0 atol=atol

    # Test for the exact solution (mean removed to match the mean-zero numerical solution).
    ϕ_exact = CenterField(grid)
    set!(ϕ_exact, z -> z^2 / 2 - z^3 / 3 - 1 / 12)
    parent(ϕ_exact) .-= mean(ϕ_exact)

    @test isapprox(ϕ_exact, ϕ; rtol=1e-3)
end
