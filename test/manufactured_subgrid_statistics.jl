include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "manufactured_subgrid_states.jl"))

#####
##### Manufactured-state consistency of the subgrid statistics.
#####
##### Fine columns whose per-level distribution of (w, θˡⁱ, qᵗ) is exactly the closure's double
##### Gaussian (see manufactured_subgrid_states.jl). "Truth" is the average over the columns of the
##### model's thermodynamics column by column; the analytic diagnostics are evaluated from the
##### sampled moments of the cell. The tolerances are the linearization floor measured with 4M
##### columns (harness notes/manufactured-statistics.md) plus about four sampling standard errors
##### at the N = 128² columns used here.
#####

using Test

const thermo = manufactured_thermodynamics()
const g = thermo.constants.gravitational_acceleration
const Lᵗ = 128

# Relative tolerances per level: (C and f⁺ absolute; the others relative to the truth)
const tolerances = Dict(
    "cumulus"       => (; C = 0.02,  qˡ = 0.04, excess = 0.05, w′qˡ′ = 0.06, ℵ = 0.08, w′b′ = 0.06, w′b′ˡⁱⁿ = 0.005, split = 0.06),
    "stratocumulus" => (; C = 0.02,  qˡ = 0.03, excess = 0.03, w′qˡ′ = 0.10, ℵ = 0.10, w′b′ = 0.20, w′b′ˡⁱⁿ = 0.01,  split = 0.04),
    "overcast"      => (; C = 0.01,  qˡ = 0.03, excess = 0.01, w′qˡ′ = 0.06, ℵ = 0.04, w′b′ = 0.06, w′b′ˡⁱⁿ = 0.005, split = 0.04),
    "clear"         => (; C = 0.006, qˡ = 0.30, excess = 0.30, w′qˡ′ = 0.35, ℵ = 0.35, w′b′ = 0.15, w′b′ˡⁱⁿ = 0.015, split = 0.15),
)

relerr(x, y) = abs(x - y) / abs(y)

@testset "Manufactured subgrid statistics: $(level.name)" for level in manufactured_levels
    dg = DoubleGaussianLevel(level, thermo)
    f = column_fields(sample_columns(dg, Lᵗ; seed = 17), dg.p, thermo)
    cell = cell_statistics(f, Lᵗ, 1, 1, dg.p, g)
    m, truth = cell.moments, cell.truth
    an = analytic_diagnostics(m, dg.p, thermo; w′qˡ′_given = truth.w′qˡ′)
    tol = tolerances[level.name]
    N = Lᵗ^2

    @testset "Moments: the double-Gaussian construction recovers the sampled moments" begin
        # The sampled moments are the prescribed ones, within sampling error
        @test m.w′² ≈ level.w′² rtol = 4 * sqrt(2 / N)
        @test m.w′³ / m.w′²^(3/2) ≈ level.Sk_w atol = 4 * sqrt(15 / N)
        @test m.θ′² ≈ level.σθ^2 rtol = 4 * sqrt(2 / N)
        @test m.q′² ≈ level.σq^2 rtol = 4 * sqrt(3 / N)
        @test m.w′θ′ / sqrt(m.w′² * m.θ′²) ≈ level.r_wθ atol = 4 / sqrt(N)
        @test m.w′q′ / sqrt(m.w′² * m.q′²) ≈ level.r_wq atol = 4 / sqrt(N)
        @test m.θ′q′ / sqrt(m.θ′² * m.q′²) ≈ level.r_θq atol = 4 / sqrt(N)
        @test truth.plume_fraction ≈ dg.a atol = 4 * sqrt(dg.a * (1 - dg.a) / N)

        # The closure's inversion of the sampled moments recovers the manufactured plumes
        P = an.plumes
        @test P.a ≈ dg.a atol = 0.01
        for i in 1:2
            @test P.w[i] ≈ dg.w[i] atol = 0.06 * sqrt(level.w′²)
            @test P.θ[i] ≈ dg.θ[i] atol = 0.06 * level.σθ
            @test P.q[i] ≈ dg.q[i] atol = 0.06 * level.σq
            @test P.σθ[i] ≈ dg.σθ[i] rtol = 0.05
            @test P.σq[i] ≈ dg.σq[i] rtol = 0.05
        end
        @test P.r ≈ dg.r atol = 0.03
    end

    @testset "Cloud fraction, condensate and the P3 condensation rate ωᶜ(μ + Γ)" begin
        @test an.C ≈ truth.C atol = tol.C
        @test an.f⁺ ≈ truth.f⁺ atol = tol.C
        @test relerr(an.qˡ, truth.qˡ) < tol.qˡ
        # μ̄ + Γ is the cell-mean positive vapor excess that SubgridCondensation condenses
        @test an.Γ ≥ 0
        @test relerr(an.excess, truth.excess) < tol.excess
    end

    @testset "Flux-weighted saturated fraction ℵ" begin
        @test 0 ≤ an.ℵ ≤ 1
        @test relerr(an.w′qˡ′, truth.w′qˡ′) < tol.w′qˡ′
        @test relerr(an.ℵ, truth.ℵ) < tol.ℵ
    end

    @testset "Buoyancy flux from the linearized coefficients" begin
        # The linear coefficients alone (sampled w′qˡ′): the linearization of b
        @test relerr(an.w′b′ˡⁱⁿ, truth.w′b′) < tol.w′b′ˡⁱⁿ
        # The closure's w′b′, with the w′qˡ′ of the distribution
        @test relerr(an.w′b′, truth.w′b′) < tol.w′b′
    end

    @testset "Plume condensate split Qᵢ" begin
        a, Q, qᵗ = an.plumes.a, an.Q, an.plumes.q
        for factor in (0.5, 1.0, 1.5), weights in (Q, (an.plumes.plumes[1].C, an.plumes.plumes[2].C))
            q̄ᶜˡ = factor * truth.qˡ
            qᶜˡ = plume_condensate_split(q̄ᶜˡ, a, weights, qᵗ)
            @test a * qᶜˡ[1] + (1 - a) * qᶜˡ[2] ≈ q̄ᶜˡ rtol = 1e-12   # grid mean recovered
            @test all(qᶜˡ .≥ 0)
            @test all(qᵗ .- qᶜˡ .≥ 0)                                 # non-negative vapor
        end
        # Degenerate: no plume condenses, the condensate goes to the moister plume
        @test plume_condensate_split(1e-5, a, (0.0, 0.0), qᵗ) == (qᵗ[1] ≥ qᵗ[2] ? (1e-5 / a, 0.0) : (0.0, 1e-5 / (1 - a)))
        # Against the per-plume equilibrium condensate of the columns, for the plume holding most
        # of it (the other's share is small and relatively less accurate)
        qᶜˡ = plume_condensate_split(truth.qˡ, a, Q, qᵗ)
        major = argmax((a * qᶜˡ[1], (1 - a) * qᶜˡ[2]))
        @test relerr(qᶜˡ[major], truth.plume_liquid[major]) < tol.split
    end
end

@testset "Self-similarity under coarse-graining" begin
    # Averaging by 2 in each direction keeps the distribution: cells of 32², 64² and 128² columns
    # of one field have the same mean truth and analytic error, with spread ∝ 1/√N
    level = manufactured_levels[2]
    dg = DoubleGaussianLevel(level, thermo)
    f = column_fields(sample_columns(dg, Lᵗ; seed = 5), dg.p, thermo)
    whole = cell_statistics(f, Lᵗ, 1, 1, dg.p, g).truth
    spreads = Float64[]
    for n in (32, 64)
        M = Lᵗ ÷ n
        cells = [cell_statistics(f, n, I, J, dg.p, g) for I in 1:M, J in 1:M]
        Cs = [c.truth.C for c in cells]
        qs = [c.truth.qˡ for c in cells]
        @test sum(Cs) / length(Cs) ≈ whole.C rtol = 1e-12                   # nested averages
        @test sum(qs) / length(qs) ≈ whole.qˡ rtol = 1e-12
        errors = [analytic_diagnostics(c.moments, dg.p, thermo).C - c.truth.C for c in cells]
        @test abs(sum(errors) / length(errors)) < 0.02
        push!(spreads, sqrt(sum(abs2, Cs .- whole.C) / length(Cs)))
    end
    # Quadrupling the columns halves the spread, within the noise of 4 × 4 and 2 × 2 cells
    @test 0.25 < spreads[2] / spreads[1] < 1.2
end
