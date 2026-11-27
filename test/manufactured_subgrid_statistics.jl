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

const thermodynamics = manufactured_thermodynamics()
const g = thermodynamics.constants.gravitational_acceleration
const Lᵗ = 128

# Relative tolerances per level: (aᶜ and f⁺ absolute; the others relative to the truth)
const tolerances = Dict(
    "cumulus"       => (; aᶜ = 0.02,  qˡ = 0.04, excess = 0.05, w′qˡ′ = 0.06, aʷ = 0.08, w′b′ = 0.06, w′b′ˡⁱⁿ = 0.005, split = 0.06),
    "stratocumulus" => (; aᶜ = 0.02,  qˡ = 0.03, excess = 0.03, w′qˡ′ = 0.10, aʷ = 0.10, w′b′ = 0.20, w′b′ˡⁱⁿ = 0.01,  split = 0.04),
    "overcast"      => (; aᶜ = 0.01,  qˡ = 0.03, excess = 0.01, w′qˡ′ = 0.06, aʷ = 0.04, w′b′ = 0.06, w′b′ˡⁱⁿ = 0.005, split = 0.04),
    "clear"         => (; aᶜ = 0.006, qˡ = 0.30, excess = 0.30, w′qˡ′ = 0.35, aʷ = 0.35, w′b′ = 0.15, w′b′ˡⁱⁿ = 0.015, split = 0.15),
)

relative_error(x, y) = abs(x - y) / abs(y)

@testset "Manufactured subgrid statistics: $(level.name)" for level in manufactured_levels
    mixture = DoubleGaussianLevel(level, thermodynamics)
    fields = column_fields(sample_columns(mixture, Lᵗ; seed = 17), mixture.p, thermodynamics)
    cell = cell_statistics(fields, Lᵗ, 1, 1, mixture.p, g)
    m, truth = cell.moments, cell.truth
    analytic = analytic_diagnostics(m, mixture.p, thermodynamics; w′qˡ′_given = truth.w′qˡ′)
    tolerance = tolerances[level.name]
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
        @test truth.updraft_fraction ≈ mixture.a atol = 4 * sqrt(mixture.a * (1 - mixture.a) / N)

        # The closure's inversion of the sampled moments recovers the manufactured components
        closure = analytic.mixture
        @test closure.a ≈ mixture.a atol = 0.01
        for i in 1:2
            @test closure.w[i] ≈ mixture.w[i] atol = 0.06 * sqrt(level.w′²)
            @test closure.θ[i] ≈ mixture.θ[i] atol = 0.06 * level.σθ
            @test closure.q[i] ≈ mixture.q[i] atol = 0.06 * level.σq
            @test closure.σθ[i] ≈ mixture.σθ[i] rtol = 0.05
            @test closure.σq[i] ≈ mixture.σq[i] rtol = 0.05
        end
        @test closure.r ≈ mixture.r atol = 0.03
    end

    @testset "Cloud fraction, condensate and the P3 condensation rate ωᶜ(μ + Γ)" begin
        @test analytic.aᶜ ≈ truth.aᶜ atol = tolerance.aᶜ
        @test analytic.f⁺ ≈ truth.f⁺ atol = tolerance.aᶜ
        @test relative_error(analytic.qˡ, truth.qˡ) < tolerance.qˡ
        # μ̄ + Γ is the cell-mean positive vapor excess that SubgridCondensation condenses
        @test analytic.Γ ≥ 0
        @test relative_error(analytic.excess, truth.excess) < tolerance.excess
    end

    @testset "Flux-weighted saturated fraction aʷ" begin
        @test 0 ≤ analytic.aʷ ≤ 1
        @test relative_error(analytic.w′qˡ′, truth.w′qˡ′) < tolerance.w′qˡ′
        @test relative_error(analytic.aʷ, truth.aʷ) < tolerance.aʷ
    end

    @testset "Buoyancy flux from the linearized coefficients" begin
        # The linear coefficients alone (sampled w′qˡ′): the linearization of b
        @test relative_error(analytic.w′b′ˡⁱⁿ, truth.w′b′) < tolerance.w′b′ˡⁱⁿ
        # The closure's w′b′, with the w′qˡ′ of the distribution
        @test relative_error(analytic.w′b′, truth.w′b′) < tolerance.w′b′
    end

    @testset "Component condensate split qᶜᵢ" begin
        a, qᶜ, qᵗ = analytic.mixture.a, analytic.qᶜ, analytic.mixture.q
        for factor in (0.5, 1.0, 1.5), weights in (qᶜ, (analytic.mixture.components[1].aᶜ, analytic.mixture.components[2].aᶜ))
            q̄ᶜˡ = factor * truth.qˡ
            qᶜˡ = component_condensate_split(q̄ᶜˡ, a, weights, qᵗ)
            @test a * qᶜˡ[1] + (1 - a) * qᶜˡ[2] ≈ q̄ᶜˡ rtol = 1e-12   # grid mean recovered
            @test all(qᶜˡ .≥ 0)
            @test all(qᵗ .- qᶜˡ .≥ 0)                                 # non-negative vapor
        end
        # Degenerate: no component condenses, the condensate goes to the moister component
        @test component_condensate_split(1e-5, a, (0.0, 0.0), qᵗ) == (qᵗ[1] ≥ qᵗ[2] ? (1e-5 / a, 0.0) : (0.0, 1e-5 / (1 - a)))
        # Against the per-component equilibrium condensate of the columns, for the component holding most
        # of it (the other's share is small and relatively less accurate)
        qᶜˡ = component_condensate_split(truth.qˡ, a, qᶜ, qᵗ)
        major = argmax((a * qᶜˡ[1], (1 - a) * qᶜˡ[2]))
        @test relative_error(qᶜˡ[major], truth.component_liquid[major]) < tolerance.split
    end
end

@testset "Self-similarity under coarse-graining" begin
    # Averaging by 2 in each direction keeps the distribution: cells of 32², 64² and 128² columns
    # of one field have the same mean truth and analytic error, with spread ∝ 1/√N
    level = manufactured_levels[2]
    mixture = DoubleGaussianLevel(level, thermodynamics)
    fields = column_fields(sample_columns(mixture, Lᵗ; seed = 5), mixture.p, thermodynamics)
    whole = cell_statistics(fields, Lᵗ, 1, 1, mixture.p, g).truth
    spreads = Float64[]
    for n in (32, 64)
        M = Lᵗ ÷ n
        cells = [cell_statistics(fields, n, I, J, mixture.p, g) for I in 1:M, J in 1:M]
        cloud_fractions = [cell.truth.aᶜ for cell in cells]
        liquids = [cell.truth.qˡ for cell in cells]
        @test sum(cloud_fractions) / length(cloud_fractions) ≈ whole.aᶜ rtol = 1e-12                   # nested averages
        @test sum(liquids) / length(liquids) ≈ whole.qˡ rtol = 1e-12
        errors = [analytic_diagnostics(cell.moments, mixture.p, thermodynamics).aᶜ - cell.truth.aᶜ for cell in cells]
        @test abs(sum(errors) / length(errors)) < 0.02
        push!(spreads, sqrt(sum(abs2, cloud_fractions .- whole.aᶜ) / length(cloud_fractions)))
    end
    # Quadrupling the columns halves the spread, within the noise of 4 × 4 and 2 × 2 cells
    @test 0.25 < spreads[2] / spreads[1] < 1.2
end
