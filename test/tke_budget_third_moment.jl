include(joinpath(@__DIR__, "setup.jl"))

#####
##### `TKEBudgetThirdMoment`: the third moment of the `DoubleGaussian` distribution from the vertical
##### integral of the TKE budget, and `TanhSaturatedSkewness`. The energy budget with its transport of
##### TKE is tested in tke_energy_budget.jl (the `budget_third_moment` paths).
#####

using Breeze
using Breeze.TurbulenceClosures: closure_parameters, with_parameters
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: limit_third_moment

using Oceananigans
using Test

column(f) = Array(interior(f))[1, 1, :]

function budget_third_moment_column(third_moment; Nz = 40, skewness = TanhSaturatedSkewness(), two_layers = false)
    grid = RectilinearGrid(CPU(); size=Nz, z=(0, 2400), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = PredictedParticlePropertiesMicrophysics(; thermodynamic_constants = constants)
    closure = TKEBasedTurbulenceClosure(; minimum_tke = 4e-4,
                                        static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                        subgrid_distribution = DoubleGaussian(; third_moment, skewness))
    boundary_conditions = (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 8e-3)),
                           ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 5.2e-5)),
                           ρu = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 0.08)))
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants,
                            microphysics, closure, boundary_conditions)
    θ₀(z) = z < 520 ? 298.7 : z < 1480 ? 298.7 + (z - 520) * 3.7 / 960 : 302.4 + (z - 1480) * 5.8 / 520
    qᵗ₀(z) = 1e-3 * (z < 520 ? 17.0 - z * 0.7 / 520 : z < 1480 ? 16.3 - (z - 520) * 5.6 / 960 : max(4.2, 10.7 - (z - 1480) * 6.5 / 520))
    # two layers: a second turbulent layer with shear above the inversion, separated from the first by still, stable air
    u₀(z) = two_layers && 1800 < z < 2200 ? -8.75 + 6 * sin(π * (z - 1800) / 400) : -8.75
    e₀(z) = two_layers ? (z < 1400 ? 0.6 * exp(-z / 500) : 1800 < z < 2200 ? 0.3 : 0.0) : 0.6 * exp(-z / 500)
    set!(model; θ=θ₀, qᵗ=qᵗ₀, u=u₀, ρe=e₀)
    return model, column(reference_state.density), fill(2400 / Nz, Nz)
end

@testset "TanhSaturatedSkewness" begin
    skewness = TanhSaturatedSkewness(maximum_skewness = 2)
    for variance in (0.0, 1e-8, 0.3), w³ in (-1e3, -0.1, 0.0, 1e-9, 0.2, 1e6)
        limited = limit_third_moment(skewness, nothing, variance, w³)
        @test abs(limited) <= 2 * variance^1.5
        @test sign(limited) == sign(w³) || limited == 0
        @test abs(limited) <= abs(w³)
    end
    # slope one at zero
    @test limit_third_moment(skewness, nothing, 0.3, 1e-6) ≈ 1e-6 rtol=1e-6
end

@testset "TKEBudgetThirdMoment (ℂʷᴸ = $ℂʷᴸ)" for ℂʷᴸ in (Inf, 8)
    third_moment = TKEBudgetThirdMoment(; ℂʷᴸ, ℂʷᵀ = 0.3)
    model, ρ, Δz = budget_third_moment_column(third_moment)
    @test :ℂʷᴸ ∈ keys(closure_parameters(model.closure)) || :ℂʷᵉ ∈ keys(closure_parameters(model.closure))
    @test with_parameters(model.closure; ℂʷᵉ = 0.2).subgrid_distribution.third_moment.ℂʷᵉ == 0.2
    for n in 1:120
        time_step!(model, 10)
    end
    cl = model.closure_fields.subgrid_cloud.cloud
    m = ρ .* Δz
    S = column(cl.P) .+ column(cl.B)
    scale = sum(m .* abs.(S))
    @test scale > 0
    # each sink balances the production of the column, so each flux returns to zero at the top
    @test abs(sum(m .* (S .- column(cl.uᵉ)))) < 1e-10 * scale
    @test abs(sum(m .* (S .- column(cl.uᵋ)))) < 1e-10 * scale
    # the transport conserves the TKE of the column
    @test abs(sum(m .* column(cl.Tᵉ))) < 1e-10 * sum(m .* abs.(column(cl.Tᵉ)))
    # surface-driven turbulence: an upward flux and a positive, bounded skewness in the layer
    @test all(column(cl.Fᵋ)[2:6] .> 0)
    Sk = column(cl.w′³) ./ column(cl.w′²) .^ 1.5
    @test all(abs.(Sk) .<= 2.5 * (1 + 1e-12)) # the hyperbolic tangent rounds to one where the argument is large
    @test all(Sk[2:6] .> 0)
    @test minimum(column(model.tracers.ρe)) >= 0
    # still, stable air above the layer: with the exchanged amplitude no flux reaches it, and the buoyancy sink
    # of the floored diffusivities makes no skewness
    # (not the two cells next to the layer, which its flux and its retreating top reach, nor the two below the
    # top of the domain, where the mixing length of the top face makes the relaxation time long)
    still = findall(<(1e-6), column(model.tracers.ρe) ./ ρ)[3:end-2]
    @test !isempty(still)
    @info "largest |skewness| in still air" ℂʷᴸ maximum(abs, Sk[still])
    isfinite(ℂʷᴸ) && @test maximum(abs, Sk[still]) < 1e-3
end

@testset "TKEBudgetThirdMoment: no flux across still air between two turbulent layers" begin
    flux_in_gap = map((Inf, 8)) do ℂʷᴸ
        model, ρ, Δz = budget_third_moment_column(TKEBudgetThirdMoment(; ℂʷᴸ); two_layers = true)
        for n in 1:6
            time_step!(model, 10)
        end
        cl = model.closure_fields.subgrid_cloud.cloud
        e = column(model.tracers.ρe) ./ ρ
        gap = findall(k -> e[k] < 1e-4 && 1500 < (k - 1/2) * Δz[k] < 1800, eachindex(e))
        @test !isempty(gap)
        maximum(abs, column(cl.Fᵋ)[gap]) / maximum(abs, column(cl.Fᵋ))
    end
    @info "relative flux in the gap (one amplitude, exchanged amplitude)" flux_in_gap
    # the still air keeps a production of order 1e-6 m² s⁻³ (the diffusivities are floored at `minimum_tke`),
    # so the flux there is small and not zero
    @test flux_in_gap[2] < 1e-3
    @test flux_in_gap[2] < flux_in_gap[1] / 10
end
