include(joinpath(@__DIR__, "setup.jl"))

#####
##### `TKEBudgetThirdMoment`: the third moment of the `DoubleGaussian` distribution from the vertical
##### integral of the TKE budget with its damped local third moment, and `TanhSaturatedSkewness`. The
##### energy budget with its transport of TKE is tested in tke_energy_budget.jl (the
##### `budget_third_moment` paths).
#####

using Breeze
using Breeze.TurbulenceClosures: closure_parameters, with_parameters, damped_local_third_moment, stratification_damping, softplus
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: limit_third_moment

using Oceananigans
using Oceananigans.TimeSteppers: update_state!
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
    names = keys(closure_parameters(model.closure))
    @test all(name -> name ∈ names, (:ℂʷᵉ, :ℂʷᵋ, :ℂʷᴸ, :ℂʷʳ, :ℂʷᵇ, :ℂʷᵍ, :ℂʷᴺ, :ℂʷᵀ))
    @test allunique(names)
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
    # top of the domain)
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

@testset "TKEBudgetThirdMoment: the damped local third moment" begin
    @test_throws ArgumentError TKEBudgetThirdMoment(ℂʷᴺ = -1)
    @test_throws ArgumentError TKEBudgetThirdMoment(passes = 0)

    # The damping is at least one, increases with the stratification number with a slope of at most ℂʷᴺ, and
    # is finite at any stratification number: the member has no pole
    @test_throws ArgumentError TKEBudgetThirdMoment(stratification_damping_width = 0)
    for ℂʷᴺ in (0.0, 1.0, 16.0), δ in (1.0, 0.1)
        αs = (-1e300, -1e6, -30.0, -1.0, 0.0, 0.5, 1.0, 30.0, 1e6, 1e300)
        D = map(α -> stratification_damping(ℂʷᴺ, δ, α), αs)
        @test all(isfinite, D)
        @test all(>=(1), D)
        @test issorted(D)
        @test all(abs(stratification_damping(ℂʷᴺ, δ, α + 1e-3) - stratification_damping(ℂʷᴺ, δ, α)) <= ℂʷᴺ * 1e-3 * (1 + 1e-8) for α in -5:0.1:5)
        @test stratification_damping(ℂʷᴺ, δ, 40.0) ≈ 1 + 40ℂʷᴺ
    end
    @test softplus(0.0) ≈ log(2)
    @test softplus(50.0) ≈ 50
    @test softplus(-800.0) == 0

    # On a column with w′² rising and then falling: bounded by the undamped member at every stratification
    # number, of the sign of the buoyant source where w′² is uniform, and of the sign opposite to ∂z w′² without one
    grid = RectilinearGrid(CPU(); size=8, z=(0, 800), topology=(Flat, Flat, Bounded))
    w′² = CenterField(grid)
    set!(w′², z -> 0.1 + 0.4 * sin(π * z / 800))
    damped = TKEBudgetThirdMoment(ℂʷᵇ = 0.6, ℂʷᵍ = 0.5, ℂʷᴺ = 4)
    undamped = TKEBudgetThirdMoment(ℂʷᵇ = 0.6, ℂʷᵍ = 0.5, ℂʷᴺ = 0)
    τ, u = 300.0, 0.5
    for k in 1:8, B in (-1e-3, 0.0, 2e-3), αᴺ in (-1e8, -2.0, 0.0, 1.0, 1e8)
        local_moment = damped_local_third_moment(1, 1, k, grid, damped, w′², τ, u, αᴺ, B)
        reference = damped_local_third_moment(1, 1, k, grid, undamped, w′², τ, u, αᴺ, B)
        @test isfinite(local_moment)
        @test abs(local_moment) <= abs(reference)
        @test sign(local_moment) == sign(reference)
    end
    gradient_only = TKEBudgetThirdMoment(ℂʷᵇ = 0, ℂʷᵍ = 1, ℂʷᴺ = 1)
    @test damped_local_third_moment(1, 1, 2, grid, gradient_only, w′², τ, u, 0.3, 1e-3) < 0 # w′² rises with height
    @test damped_local_third_moment(1, 1, 7, grid, gradient_only, w′², τ, u, 0.3, 1e-3) > 0 # w′² falls with height
    buoyant_only = TKEBudgetThirdMoment(ℂʷᵇ = 1, ℂʷᵍ = 0, ℂʷᴺ = 1)
    @test damped_local_third_moment(1, 1, 3, grid, buoyant_only, w′², τ, u, -0.3, 1e-3) > 0
    @test damped_local_third_moment(1, 1, 3, grid, buoyant_only, w′², τ, u, 0.3, -1e-3) == 0 # a buoyancy sink is no source

    # Through the limiter the skewness is strictly inside the bound whatever the coefficients
    model, ρ, Δz = budget_third_moment_column(TKEBudgetThirdMoment(ℂʷᵇ = 50, ℂʷᵍ = 50, ℂʷᴺ = 0); skewness = TanhSaturatedSkewness(maximum_skewness = 1.5))
    for n in 1:30
        time_step!(model, 10)
    end
    cl = model.closure_fields.subgrid_cloud.cloud
    @test maximum(abs, column(cl.w′³ᵇ)) > 0
    @test all(abs.(column(cl.w′³) ./ column(cl.w′²) .^ 1.5) .<= 1.5 * (1 + 1e-12))
end

@testset "TKEBudgetThirdMoment: the closure fields are a function of the prognostic state (passes = $passes)" for passes in (1, 2)
    third_moment = TKEBudgetThirdMoment(; passes, ℂʷᵍ = 0.2, ℂʷᴺ = 1, ℂʷᵀ = 0.3)
    model, ρ, Δz = budget_third_moment_column(third_moment)
    for n in 1:60
        time_step!(model, 10)
    end
    names = (:w′³, :w′³ᵇ, :aᶜ, :B, :P, :Fᵉ, :Fᵋ, :Tᵉ)
    snapshot(m) = (map(name -> column(getproperty(m.closure_fields.subgrid_cloud.cloud, name)), names)...,
                   Array(interior(m.closure_fields.Kᶜ))[1, 1, :], Array(interior(m.closure_fields.N²))[1, 1, :])
    first_evaluation = snapshot(model)
    @test maximum(abs, first_evaluation[1]) > 0

    # a second evaluation on the same state changes nothing
    update_state!(model)
    @test all(snapshot(model) .== first_evaluation)

    # a model built anew and given the prognostic state alone has the same closure fields: nothing but the
    # prognostic state needs to be checkpointed
    restarted, _, _ = budget_third_moment_column(third_moment)
    for (name, field) in pairs(Oceananigans.prognostic_fields(model))
        parent(Oceananigans.prognostic_fields(restarted)[name]) .= parent(field)
    end
    update_state!(restarted)
    @test all(snapshot(restarted) .== first_evaluation)
end
