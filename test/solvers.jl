using Breeze
using Test

using Breeze.Solvers: newton_solve, secant_solve, materialize_solver
using Breeze.Thermodynamics: newton_hydrostatic_pressure

# Unit tests for the unified iterative solver abstraction (Breeze.Solvers): the solver
# types describe stopping rules that thermodynamic algorithms dispatch on — NewtonSolver
# and SecantSolver iterate to tolerance, FixedIterations unrolls a fixed trip count
# (Reactant/Enzyme-safe), and `nothing` returns the initial guess (closed form).

@testset "Solvers [$FT]" for FT in all_float_types()
    # r(x) = x² - 2 with root √2; Newton needs (r, r′), secant needs r only.
    residual_and_derivative(x) = (x^2 - 2, 2x)
    residual(x) = x^2 - 2
    x★ = sqrt(FT(2))
    x₀ = one(FT)

    @testset "newton_solve" begin
        @test newton_solve(residual_and_derivative, NewtonSolver(FT; reltol=1e-6, maxiter=20), x₀) ≈ x★
        @test newton_solve(residual_and_derivative, NewtonSolver(FT; reltol=0, abstol=1e-5, maxiter=20), x₀) ≈ x★
        @test newton_solve(residual_and_derivative, FixedIterations(20), x₀) ≈ x★

        # `nothing` returns the initial guess unmodified
        @test newton_solve(residual_and_derivative, nothing, x₀) == x₀

        # maxiter caps the iteration: zero iterations returns the initial guess
        @test newton_solve(residual_and_derivative, NewtonSolver(FT; maxiter=0), x₀) == x₀
        @test newton_solve(residual_and_derivative, FixedIterations(0), x₀) == x₀
    end

    @testset "secant_solve" begin
        x₁, x₂ = one(FT), FT(2)
        scale = one(FT)
        # A residual tolerance of 1e-5 bounds the root error by |r|/r′(√2) ≈ 3.5e-6
        @test secant_solve(residual, SecantSolver(FT; abstol=1e-5, maxiter=50), x₁, x₂, scale) ≈ x★  atol=FT(1e-5)
        @test secant_solve(residual, SecantSolver(FT; reltol=1e-5, abstol=0, maxiter=50), x₁, x₂, scale) ≈ x★  atol=FT(1e-5)
        @test secant_solve(residual, FixedIterations(20), x₁, x₂, scale) ≈ x★

        # A degenerate slope (r₂ = r₁) terminates instead of dividing by zero
        flat(x) = one(FT)
        @test isfinite(secant_solve(flat, SecantSolver(FT; abstol=1e-5, maxiter=50), x₁, x₂, scale))
        @test isfinite(secant_solve(flat, FixedIterations(5), x₁, x₂, scale))
    end

    @testset "default tolerances" begin
        # The defaults encode the tolerance conventions documented in Breeze.Solvers:
        # temperature solves use an absolute tolerance of 1e-4 K (see also the dewpoint
        # diagnostic, which opts into a relative tolerance against the vapor pressure).
        newton = NewtonSolver(FT)
        @test newton.reltol == 0
        @test newton.abstol == FT(1e-4)
        @test newton.maxiter == 8

        secant = SecantSolver(FT)
        @test secant.reltol == 0
        @test secant.abstol == FT(1e-4)
        @test secant.maxiter == 20
    end

    @testset "hydrostatic Newton convergence" begin
        # The discrete hydrostatic balance solved by `newton_hydrostatic_pressure` is
        # monotone in p, so Newton converges quadratically. Both call sites use
        # FixedIterations(5); verify that count reaches machine precision (matches a
        # heavily-iterated reference) across a representative range of layer thickness and θ.
        pˢᵗ = FT(1e5)
        g   = FT(9.81)
        Rᵐ  = FT(287)
        cᵖ  = FT(1005)
        κ   = Rᵐ / cᵖ

        for Δz in (FT(10), FT(100), FT(500)), θ in (FT(250), FT(300), FT(320))
            p⁻ = pˢᵗ
            ρ⁻ = p⁻ / (Rᵐ * θ * (p⁻ / pˢᵗ)^κ)
            # Continuous-Π initial guess, as used at the call sites.
            Π_init = (p⁻ / pˢᵗ)^κ - g * Δz / (cᵖ * θ)
            p_init = pˢᵗ * Π_init^(1 / κ)

            p5  = newton_hydrostatic_pressure(p⁻, ρ⁻, θ, Rᵐ, κ, Δz, pˢᵗ, g, p_init, FixedIterations(5))
            p30 = newton_hydrostatic_pressure(p⁻, ρ⁻, θ, Rᵐ, κ, Δz, pˢᵗ, g, p_init, FixedIterations(30))
            @test p5 ≈ p30 rtol=10 * eps(FT)
        end
    end

    @testset "materialize_solver" begin
        newton = materialize_solver(NewtonSolver(reltol=1e-8, abstol=0, maxiter=8), FT)
        @test newton isa NewtonSolver{FT}
        @test newton.maxiter == 8

        secant = materialize_solver(SecantSolver(abstol=1e-3), FT)
        @test secant isa SecantSolver{FT}

        @test materialize_solver(FixedIterations(2), FT) === FixedIterations(2)
        @test materialize_solver(nothing, FT) === nothing
    end
end
