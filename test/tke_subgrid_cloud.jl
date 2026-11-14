include(joinpath(@__DIR__, "setup.jl"))

#####
##### The subgrid distribution of `TKEBasedTurbulenceClosure`
#####

using Breeze
using Breeze.AtmosphereModels: ClosureCloud, ClosureCloudStatistics, NoSubgridCorrelation, subgrid_statisticsᶜᶜᶜ
using Breeze.TurbulenceClosures: SpecificValue, dissipation_stability_functionᶜᶜᶠ, tracer_stability_functionᶜᶜᶠ, SubgridCloudFields, tke_buoyancy_productionᶜᶜᶜ, buoyancy_productionᶜᶜᶠ, moment_time_scale, ℑbzᵃᵃᶜ
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: assumed_pdf, third_moment, limit_third_moment,
                                                              FixedConstantsLiquidSaturation, ModelThermodynamicsSaturation,
                                                              shoc_saturation_thermodynamics

using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Grids: znodes
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: step_closure_prognostics!
using Test

implicit_transport() = WENO(; order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(Float64; cfl=0.5))

column(field) = Array(interior(field))[1, 1, :]

# Kernel arithmetic on the device can differ from the host by rounding (fused multiply-add): exact on the CPU, close on the GPU
matches(a, b) = default_arch isa CPU ? a == b : isapprox(a, b; rtol=1e-10, atol=1e-15)

function tke_p3_column(distribution; subgrid = NoSubgridCorrelation(), arch = default_arch, static_stability = MoistStaticStability(),
                       shear = GridMeanShear(), solver = SinglePass())
    grid = RectilinearGrid(arch; size=32, z=(0, 1600), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=290)
    microphysics = PredictedParticlePropertiesMicrophysics(; subgrid,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(; static_stability, shear, subgrid_distribution = distribution, solver)
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics, closure,
                            scalar_advection=implicit_transport())

    # A well-mixed layer whose upper part is near saturation, under an inversion at 800 m, with shear
    θ₀(z) = ifelse(z < 800, 290.0, 290.0 + 8 * (1 - exp(-(z - 800) / 100)))
    qᵗ₀(z) = ifelse(z < 800, 10.0e-3, 4.0e-3)
    set!(model; θ=θ₀, qᵗ=qᵗ₀, u = z -> 5 * z / 1600, ρe=z -> 1.2 * 0.3 * exp(-z / 800))
    return model
end

state_columns(model) = (column(model.formulation.potential_temperature_density), column(model.moisture_density),
                        column(model.tracers.ρe), column(model.momentum.ρu), column(model.microphysical_fields.ρqᶜˡ))

non_precipitating_water(model) = column(model.moisture_density) .+ column(model.microphysical_fields.ρqᶜˡ)

# The saturated fraction of the static stability from the condensate flux of the distribution
const fraction_stability = SaturationWeightedStaticStability(CondensateFluxFraction(DryStaticStability()))
# SHOC's moment time scale, whose τ the structural checks below recompute (the default is DissipationTimeScale)
shoc_time_scale() = DoubleGaussian(; time_scale = StratificationLimitedMomentTimeScale())

@testset "DoubleGaussian construction" begin
    @test TKEBasedTurbulenceClosure().subgrid_distribution === nothing
    @test TKEBasedTurbulenceClosure().solver isa SinglePass
    d = DoubleGaussian()
    @test d.skewness.maximum_skewness == 0.85
    @test d.assumed_pdf isa EqualWidthDoubleGaussianPDF
    closure = TKEBasedTurbulenceClosure(Float32; subgrid_distribution = DoubleGaussian(Float64))
    @test closure.subgrid_distribution.second_moments.ℂᵛ₁ isa Float32
    @test closure.subgrid_distribution.time_scale.ℂᵠ === 1.3f0 # DissipationTimeScale is the default
    @test closure.subgrid_distribution.time_scale.third_moment_time_scale.ℂᴺ === 4.5f0
    @test occursin("subgrid_distribution", sprint(show, closure))
    @test occursin("solver: SinglePass()", sprint(show, closure))

    # The coefficients of the closure, as one flat view
    closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(), static_stability = fraction_stability)
    ℂ = closure_parameters(closure)
    @test all(name -> startswith(string(name), "ℂ") || name ∈ (:Ri⁻, :Ri⁺), keys(ℂ))
    @test (:ℂˢ, :ℂⁱ, :ℂᵘ⁰, :ℂᵠ, :ℂᵛ₁, :ℂʷ₁, :ℂʷ₂, :ℂᵍ₀) ⊆ keys(ℂ)
    changed = with_parameters(closure; ℂᵍ₀ = 0.35, ℂᵠ = 1.1, ℂᵘ⁰ = 0.2)
    @test changed.subgrid_distribution.assumed_pdf.ℂᵍ₀ == 0.35
    @test changed.subgrid_distribution.time_scale.ℂᵠ == 1.1
    @test changed.stability_functions.ℂᵘ⁰ == 0.2
    @test changed.static_stability === closure.static_stability
    @test closure_parameters(with_parameters(closure)) == ℂ
    @test keys(closure_parameters(TKEBasedTurbulenceClosure())) ⊆ keys(ℂ)
    @test_throws ArgumentError with_parameters(closure; not_a_coefficient = 1)
end

@testset "The default is the closure without a distribution" begin
    model = tke_p3_column(nothing)
    @test model.closure_fields.subgrid_cloud === nothing
    update_state!(model)
    grid = model.grid
    cf = model.closure_fields
    for k in 1:size(grid, 3)
        @test @allowscalar(tke_buoyancy_productionᶜᶜᶜ(1, 1, k, grid, cf)) ===
              @allowscalar(ℑbzᵃᵃᶜ(1, 1, k, grid, buoyancy_productionᶜᶜᶠ, cf.Kᶜ, cf.N²))
    end
end

@testset "The distribution's cloud" begin
    model = tke_p3_column(DoubleGaussian())
    @test model.closure_fields.subgrid_cloud isa SubgridCloudFields
    for _ in 1:4
        time_step!(model, 30)
    end
    cloud = model.closure_fields.subgrid_cloud.cloud
    @test maximum(column(cloud.aᶜ)) > 0
    @test all(0 .≤ column(cloud.aᶜ) .≤ 1)
    @test all(column(cloud.qˡ) .≥ 0)
    @test all(isfinite, column(cloud.w′θᵥ′))
end

@testset "The second and third moments are SHOC's down-gradient forms on the closure's ℓ, e, Kᶜ" begin
    model = tke_p3_column(shoc_time_scale())
    update_state!(model)
    grid = model.grid
    cf = model.closure_fields
    m = cf.subgrid_cloud.moments
    d = model.closure.subgrid_distribution
    ρ = column(model.dynamics.reference_state.density)
    θ = column(model.formulation.potential_temperature_density) ./ ρ
    qᵗ = (column(model.moisture_density) .+ column(model.microphysical_fields.ρqᶜˡ)) ./ ρ
    e = column(model.tracers.ρe) ./ ρ
    Kᶜ = column(cf.Kᶜ); ℓ = column(cf.ℓᵐ); N² = column(cf.N²)
    zc = znodes(grid, Center())
    Nz = size(grid, 3)
    eᵐⁱⁿ = model.closure.minimum_tke
    for k in 2:Nz
        Δz = zc[k] - zc[k-1]
        u = (sqrt(max(e[k], eᵐⁱⁿ)) + sqrt(max(e[k-1], eᵐⁱⁿ))) / 2
        τ = moment_time_scale(d.time_scale, ℓ[k], u, N²[k])
        @test column(m.τ)[k] ≈ τ rtol=1e-12
        @test column(m.θˡ′²)[k] ≈ τ * Kᶜ[k] * ((θ[k] - θ[k-1]) / Δz)^2 rtol=1e-12
        @test column(m.qᵗ′²)[k] ≈ τ * Kᶜ[k] * ((qᵗ[k] - qᵗ[k-1]) / Δz)^2 rtol=1e-12
        @test column(m.w′θˡ′)[k] ≈ -Kᶜ[k] * (θ[k] - θ[k-1]) / Δz rtol=1e-12
        @test column(m.w′qᵗ′)[k] ≈ -Kᶜ[k] * (qᵗ[k] - qᵗ[k-1]) / Δz rtol=1e-12
    end
    # The limited skewness
    w² = column(cf.subgrid_cloud.cloud.w′²)
    w³ = column(m.w′³)
    @test all(abs.(w³[2:Nz]) .≤ d.skewness.maximum_skewness .* ((w²[1:Nz-1] .+ w²[2:Nz]) ./ 2) .^ 1.5 .* (1 + 1e-12))
    @test any(!=(0), w³)
end

@testset "Variance-budget time scales: ρ² = 3 ℂᵠ 𝕊ᶜ 𝕊ᴰ / (4 ℂʷ₁) and 3 ℂᵠ 𝕊ᶜ / (4 ℂʷ₁)" begin
    for time_scale in (DissipationTimeScale(; ℂᵠ = 0.1), DissipationTimeScale(; ℂᵠ = 0.14), EddyTurnoverTimeScale(; ℂᵠ = 1))
        model = tke_p3_column(DoubleGaussian(; time_scale))
        update_state!(model)
        grid = model.grid
        cf = model.closure_fields
        m = cf.subgrid_cloud.moments
        closure = model.closure
        sf = closure.stability_functions
        ℂᵠ = time_scale.ℂᵠ
        dissipation = time_scale isa DissipationTimeScale
        ρ = column(model.dynamics.reference_state.density)
        e = column(model.tracers.ρe) ./ ρ
        ℓ = column(cf.ℓᵐ)
        Nz = size(grid, 3)
        eᵐⁱⁿ = closure.minimum_tke
        max_S = dissipation ? max(sf.ℂᶜ⁻ * sf.ℂᴰ⁻, sf.ℂᶜ⁰ * sf.ℂᴰ⁰, sf.ℂᶜ⁺ * sf.ℂᴰ⁺) : max(sf.ℂᶜ⁻, sf.ℂᶜ⁰, sf.ℂᶜ⁺)
        n = 0
        for k in 2:Nz
            u = (sqrt(max(e[k], eᵐⁱⁿ)) + sqrt(max(e[k-1], eᵐⁱⁿ))) / 2
            𝕊ᴰ = dissipation ? @allowscalar(dissipation_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N²)) : 1
            𝕊ᶜ = @allowscalar tracer_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N², nothing, nothing)
            @test column(m.τ)[k] ≈ 2 * ℓ[k] / (ℂᵠ * 𝕊ᴰ * u) rtol=1e-12
            θ² = column(m.θˡ′²)[k]
            if θ² > 0
                # With u² for e the squared correlation is a function of Ri alone, bounded by its largest value
                ρ² = column(m.w′θˡ′)[k]^2 / (θ² * (2 / 3) * u^2)
                @test ρ² ≈ 3 * ℂᵠ * 𝕊ᶜ * 𝕊ᴰ / 4 rtol=1e-8
                @test ρ² ≤ 3 * ℂᵠ * max_S / 4 * (1 + 1e-12)
                n += 1
            end
        end
        @test n > 0
        @test all(isfinite, column(m.w′³))
        @test all(isfinite, column(cf.subgrid_cloud.cloud.aᶜ))
    end
    @test DissipationTimeScale(Float32).ℂᵠ isa Float32
    @test EddyTurnoverTimeScale(Float32).ℂᵠ isa Float32
end

@testset "The cloud is SHOC's assumed PDF of the stored moments" begin
    for pdf in (DoubleGaussianPDF(), SmoothDoubleGaussianPDF())
        model = tke_p3_column(DoubleGaussian(; assumed_pdf = pdf))
        update_state!(model)
        grid = model.grid
        sc = model.closure_fields.subgrid_cloud
        m, cl = sc.moments, sc.cloud
        d = model.closure.subgrid_distribution
        ρ = column(model.dynamics.reference_state.density)
        θ = column(model.formulation.potential_temperature_density) ./ ρ
        qᵗ = (column(model.moisture_density) .+ column(model.microphysical_fields.ρqᶜˡ)) ./ ρ
        p = column(model.dynamics.reference_state.pressure)
        zc = znodes(grid, Center()); zf = znodes(grid, Face())
        moments(k) = (column(m.w′³)[k], column(m.θˡ′²)[k], column(m.w′θˡ′)[k], column(m.θˡ′qᵗ′)[k],
                      column(m.w′qᵗ′)[k], column(m.qᵗ′²)[k])
        for k in 1:size(grid, 3)
            C, qˡ, qˡ′², w′qˡ′, w′θᵥ′ = assumed_pdf(pdf, d.saturation, shoc_saturation_thermodynamics(model), d.clipping, d.constants,
                                                   zf[k+1], zf[k], zc[k], p[k], θ[k], 0.0, qᵗ[k], column(cl.w′²)[k],
                                                   moments(k), moments(k+1))
            @test matches(column(cl.aᶜ)[k], C)
            @test matches(column(cl.qˡ)[k], qˡ)
            @test matches(column(cl.w′qˡ′)[k], w′qˡ′)
            @test matches(column(cl.w′θᵥ′)[k], w′θᵥ′)
        end
        @test maximum(column(cl.aᶜ)) > 0
        @test maximum(column(cl.qˡ)) > 0
    end
end

@testset "The conversion of the applied fluxes drives the TKE equation" begin
    model = tke_p3_column(DoubleGaussian())
    update_state!(model)
    grid = model.grid
    cf = model.closure_fields
    B = column(cf.subgrid_cloud.cloud.B)
    for k in 1:size(grid, 3)
        @test @allowscalar(tke_buoyancy_productionᶜᶜᶜ(1, 1, k, grid, cf)) == B[k]
    end
    reference = tke_p3_column(nothing)
    for _ in 1:4
        time_step!(reference, 30)
        time_step!(model, 30)
    end
    @test column(model.tracers.ρe) != column(reference.tracers.ρe)
    @test all(column(model.tracers.ρe) .≥ 0)
    @test all(isfinite, column(model.tracers.ρe))

    # With the condensate flux fraction, N² is the stratification of the mixed variables
    # less the latent correction, N² = Σ b̄_χ δz χ - aᴺ (N²ᵇ - N²ˢ), with b_χ the stored derivatives
    model = tke_p3_column(DoubleGaussian(); static_stability = SaturationWeightedStaticStability(CondensateFluxFraction(MoistStaticStability())))
    update_state!(model)
    cf = model.closure_fields
    cl = cf.subgrid_cloud.cloud
    μ = model.microphysical_fields
    ρ = column(model.dynamics.reference_state.density)
    φ = hcat(column(model.formulation.potential_temperature_density) ./ ρ, column(model.moisture_density) ./ ρ,
             (column(μ.ρqᶜˡ) .+ column(μ.ρqʳ) .+ column(μ.ρqʷⁱ)) ./ ρ, column(μ.ρqⁱ) ./ ρ)
    b = hcat(column(cl.b_θ), column(cl.b_qᵛ), column(cl.b_qˡ), column(cl.b_qⁱ))
    zc = znodes(model.grid, Center())
    for k in 2:size(model.grid, 3)
        N²ᵇ = sum((b[k, s] + b[k-1, s]) / 2 * (φ[k, s] - φ[k-1, s]) for s in 1:4) / (zc[k] - zc[k-1])
        @test column(cf.N²)[k] ≈ N²ᵇ - column(cl.ΔN²ᴸ)[k] rtol=1e-10 atol=1e-14
    end
    @test all(0 .≤ column(cl.aᴺ) .≤ 1)
    @test all(isfinite, column(cf.N²))
    @test all(isfinite, column(cf.Kᶜ))
    for _ in 1:4
        time_step!(model, 30)
    end
    @test all(isfinite, column(model.tracers.ρe))
end

@testset "The closure owns the cloud of a P3 column; water and energy are conserved" begin
    model = tke_p3_column(DoubleGaussian(); subgrid = ClosureCloud())
    μ = model.microphysical_fields
    ρ = column(model.dynamics.reference_state.density)
    ρθ = model.formulation.potential_temperature_density

    water₀ = sum(non_precipitating_water(model) .+ column(μ.ρqʳ))
    for _ in 1:5
        time_step!(model, 60)
    end
    sc = model.closure_fields.subgrid_cloud
    @test maximum(column(sc.cloud.qˡ)) > 1e-6
    @test 0 < maximum(column(sc.cloud.aᶜ)) ≤ 1
    stats = @allowscalar subgrid_statisticsᶜᶜᶜ(1, 1, 20, model.grid, ClosureCloud(), model.closure, model.closure_fields)
    @test stats == ClosureCloudStatistics(column(sc.cloud.aᶜ)[20], column(sc.cloud.qˡ)[20])

    # Total water of a closed column with rain
    water = sum(non_precipitating_water(model) .+ column(μ.ρqʳ))
    @test water ≈ water₀ rtol=1e-12
    @test all(column(model.moisture_density) .≥ 0)

    # The step after the stages repartitions vapor and cloud liquid to the liquid of the
    # distribution of the end-of-step state, conserving their sum and ∫ρθ exactly
    heat = sum(column(ρθ))
    water = sum(non_precipitating_water(model))
    step_closure_prognostics!(model.closure_fields, model.closure, model, 60)
    @test sum(column(ρθ)) == heat
    @test sum(non_precipitating_water(model)) ≈ water rtol=1e-14
    qˡ = column(sc.cloud.qˡ)
    qᵗ = non_precipitating_water(model) ./ ρ
    @test column(μ.ρqᶜˡ) ≈ ρ .* min.(max.(qˡ, 0), max.(qᵗ, 0)) rtol=1e-14

    # Without ClosureCloud the closure leaves the cloud liquid to the microphysics
    model = tke_p3_column(DoubleGaussian(); subgrid = NoSubgridCorrelation())
    set!(model.microphysical_fields.ρqᶜˡ, z -> ifelse(600 < z < 800, 1.2e-4, 0.0))
    update_state!(model)
    cloud_liquid = column(model.microphysical_fields.ρqᶜˡ)
    step_closure_prognostics!(model.closure_fields, model.closure, model, 60)
    @test column(model.microphysical_fields.ρqᶜˡ) == cloud_liquid
end

@testset "The distribution with the saturation of the model" begin
    # The distribution of this closure saturates with the model's thermodynamics only; the fixed
    # constants of shoc.F90 are SimplifiedHigherOrderClosure's
    @test DoubleGaussian().saturation isa ModelThermodynamicsSaturation
    @test_throws ArgumentError DoubleGaussian(; saturation = FixedConstantsLiquidSaturation())
    for saturation in (ModelThermodynamicsSaturation(),)
        model = tke_p3_column(DoubleGaussian(; saturation); subgrid = ClosureCloud(), static_stability = fraction_stability)
        μ = model.microphysical_fields
        water₀ = sum(non_precipitating_water(model) .+ column(μ.ρqʳ))
        for _ in 1:10
            time_step!(model, 60)
        end
        @test sum(non_precipitating_water(model) .+ column(μ.ρqʳ)) ≈ water₀ rtol=1e-12
        @test all(isfinite, column(model.tracers.ρe))
        @test maximum(column(μ.ρqᶜˡ)) > 0
    end
end

@testset "Damped fixed-point iteration of the static stability and the saturated fraction" begin
    @test DampedFixedPoint().iterations == 1
    @test Breeze.TurbulenceClosures.stability_iterations(Breeze.TurbulenceClosures.AppliedFluxEnergetics(DampedFixedPoint(; iterations = 3))) == 3
    @test Breeze.TurbulenceClosures.stability_iterations(Breeze.TurbulenceClosures.AppliedFluxEnergetics(SinglePass())) == 0
    @test_throws ArgumentError DampedFixedPoint(; iterations = -1)
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(), solver = DampedFixedPoint())
    @test DampedFixedPoint(; iterations = 3, relaxation = 0.5).relaxation == 0.5
    @test_throws ArgumentError DampedFixedPoint(; relaxation = 0)
    @test_throws ArgumentError DampedFixedPoint(; relaxation = 1.5)
    relaxed = TKEBasedTurbulenceClosure(Float32; subgrid_distribution = DoubleGaussian(Float64), static_stability = fraction_stability,
                                        solver = DampedFixedPoint(; iterations = 2, relaxation = 0.3))
    @test relaxed.solver.relaxation === 0.3f0

    closure_columns(model) = (cf = model.closure_fields; sc = cf.subgrid_cloud;
                              map(column, (cf.N², cf.ℓᵐ, cf.Kᶜ, sc.moments..., sc.cloud...)))
    build(n; relaxation = 1) = tke_p3_column(DoubleGaussian(); subgrid = ClosureCloud(), static_stability = fraction_stability,
                                             solver = DampedFixedPoint(; iterations = n, relaxation))

    reference = build(0)
    default = tke_p3_column(DoubleGaussian(); subgrid = ClosureCloud(), static_stability = fraction_stability)
    for m in (reference, default), _ in 1:3
        time_step!(m, 30)
    end
    @test closure_columns(default) == closure_columns(reference)

    # ω = 1 is the unrelaxed iteration
    unrelaxed = build(2)
    relaxed = build(2; relaxation = 0.5)
    for m in (unrelaxed, relaxed), _ in 1:3
        time_step!(m, 30)
    end
    @test closure_columns(relaxed) != closure_columns(unrelaxed)
    @test all(isfinite, column(relaxed.closure_fields.Kᶜ))

    for n in (1, 2)
        model = build(n)
        for _ in 1:3
            time_step!(model, 30)
        end
        @test closure_columns(model) != closure_columns(reference)
        @test all(isfinite, column(model.closure_fields.Kᶜ))
        @test all(isfinite, column(model.tracers.ρe))
        @test all(0 .≤ column(model.closure_fields.subgrid_cloud.cloud.aᶜ) .≤ 1)

        # The TKE buoyancy production is the stored B of the last distribution
        update_state!(model)
        cf = model.closure_fields
        B = column(cf.subgrid_cloud.cloud.B)
        @test all(@allowscalar(tke_buoyancy_productionᶜᶜᶜ(1, 1, k, model.grid, cf)) == B[k] for k in 1:size(model.grid, 3))
    end
end

@testset "Secant and bisection solves of the static stability and the saturated fraction" begin
    @test BisectionIteration().final_interpolation
    @test occursin("SecantIteration", summary(SecantIteration()))
    build(solver) = tke_p3_column(DoubleGaussian(); subgrid = ClosureCloud(), static_stability = fraction_stability, solver)
    for solver in (SecantIteration(; iterations = 4), BisectionIteration(; iterations = 4),
                   BisectionIteration(; iterations = 4, final_interpolation = false))
        model = build(solver)
        @test model.closure_fields.subgrid_cloud.solver_state.x₋ isa Field
        for _ in 1:3
            time_step!(model, 30)
        end
        cf = model.closure_fields
        @test all(isfinite, column(cf.Kᶜ))
        @test all(isfinite, column(cf.N²))
        @test all(isfinite, column(model.tracers.ρe))
        @test all(0 .≤ column(cf.subgrid_cloud.cloud.aᶜ) .≤ 1)
        # N² stays in the bracket of the conserved and the saturated stratification
        L, U = column(cf.subgrid_cloud.solver_state.lower), column(cf.subgrid_cloud.solver_state.upper)
        N² = column(cf.N²)
        @test all(L[k] - 1e-12 ≤ N²[k] ≤ U[k] + 1e-12 for k in 2:length(N²) - 1)
    end
    @test build(DampedFixedPoint(; iterations = 2)).closure_fields.subgrid_cloud.solver_state === nothing
end

@testset "Lagged stability" begin
    model = tke_p3_column(DoubleGaussian(); subgrid = ClosureCloud(), static_stability = fraction_stability, solver = LaggedStability())
    N²₋ = model.closure_fields.subgrid_cloud.solver_state.N²₋
    update_state!(model)
    @test column(N²₋) == column(model.closure_fields.N²)
    for _ in 1:3
        time_step!(model, 30)
    end
    @test all(isfinite, column(model.closure_fields.Kᶜ))
    @test all(isfinite, column(model.tracers.ρe))
end

@testset "One N² path: the condensate flux fraction of the distribution is the saturated fraction of the static stability" begin
    closure = TKEBasedTurbulenceClosure(; static_stability = fraction_stability, subgrid_distribution = DoubleGaussian())
    @test closure.static_stability isa SaturationWeightedStaticStability{<:CondensateFluxFraction}
    @test closure.static_stability.fraction.initial_stability isa DryStaticStability
    @test sprint(show, closure.static_stability) == "SaturationWeightedStaticStability(CondensateFluxFraction(DryStaticStability))"
    @test CondensateFluxFraction() == CondensateFluxFraction(DryStaticStability())
    # Without the fraction the static stability is left alone
    @test TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian()).static_stability isa MoistStaticStability
    # The fraction needs the distribution
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
end

@testset "DissipationTimeScale reads the Richardson number of the closure's shear" begin
    time_scale = DissipationTimeScale(; ℂᵠ = 0.14)
    τ = Dict()
    for ℂˢᵗ in (0, 0.3)
        model = tke_p3_column(DoubleGaussian(; time_scale); shear = TurbulenceAugmentedShear(; turbulence_shear_coefficient = ℂˢᵗ))
        update_state!(model)
        grid = model.grid
        cf = model.closure_fields
        closure = model.closure
        e = SpecificValue(model.tracers.ρe, model.dynamics.reference_state.density)
        ρ = column(model.dynamics.reference_state.density)
        eᶜ = column(model.tracers.ρe) ./ ρ
        ℓ = column(cf.ℓᵐ)
        τ[ℂˢᵗ] = column(cf.subgrid_cloud.moments.τ)
        for k in 2:size(grid, 3)
            u = (sqrt(max(eᶜ[k], closure.minimum_tke)) + sqrt(max(eᶜ[k-1], closure.minimum_tke))) / 2
            𝕊ᴰ = @allowscalar dissipation_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N², e, cf.ℓᵐ)
            @test τ[ℂˢᵗ][k] ≈ 2 * ℓ[k] / (time_scale.ℂᵠ * 𝕊ᴰ * u) rtol=1e-10
        end
    end
    @test τ[0] != τ[0.3]
end

@testset "Bracketed solve of the effective saturated fraction aᴺ" begin
    @test summary(BracketedFractionSolve(2)) == "BracketedFractionSolve(2)"
    @test_throws ArgumentError BracketedFractionSolve(-1)
    @test summary(BracketedFractionSolve(1; lagged_envelope = true)) == "BracketedFractionSolve(1, lagged_envelope = true)"
    for (n, lagged_envelope) in ((0, false), (2, false), (2, true))
        model = tke_p3_column(DoubleGaussian(); subgrid = ClosureCloud(), static_stability = fraction_stability,
                              solver = BracketedFractionSolve(n; lagged_envelope))
        @test model.closure.static_stability isa SaturationWeightedStaticStability{<:CondensateFluxFraction}
        state = model.closure_fields.subgrid_cloud.solver_state
        @test length(state.residuals) == n + 1
        for _ in 1:3
            time_step!(model, 30)
        end
        Nz = size(model.grid, 3)
        aᴺ = column(state.aᴺ)[2:Nz]
        @test all(x -> 0 ≤ x ≤ 1, aᴺ)
        @test all(x -> 0 ≤ x ≤ 1, column(state.ãᴺ₀)[2:Nz]) && all(x -> 0 ≤ x ≤ 1, column(state.ãᴺ₁)[2:Nz])
        classified = column(state.classified)[2:Nz]
        @test all(x -> x ∈ (-1, 0, 1), classified)
        @test all(aᴺ[classified .== -1] .== 0) && all(aᴺ[classified .== 1] .== 1)
        @test all(r -> all(isfinite, column(r)) && all(≥(0), column(r)), state.residuals)
        @test all(isfinite, column(state.ãᴺᶠ))
        @test all(x -> 0 ≤ x ≤ 1, column(model.closure_fields.subgrid_cloud.cloud.aᶜ))
        @test all(isfinite, column(model.closure_fields.Kᶜ))
        @test all(isfinite, column(model.tracers.ρe))
    end
end

@testset "The within-component share of w′² as a function of the skewness" begin
    using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: within_component_share
    pdf = EqualWidthDoubleGaussianPDF()
    # zero slopes: the constant, exactly
    @test all(within_component_share(pdf, Sk) === pdf.ℂᵍ₀ for Sk in (-3.0, -0.5, 0.0, 0.7, 2.5))
    for (ℂᵍˢ, ℂᵍˢˢ) in ((0.3, 0.0), (0.0, 0.2), (-0.4, 0.15))
        pdf = EqualWidthDoubleGaussianPDF(; ℂᵍˢ, ℂᵍˢˢ)
        @test within_component_share(pdf, 0.0) ≈ pdf.ℂᵍ₀ rtol=1e-14
        for Sk in (-2.0, -0.5, 0.0, 0.8, 2.0)
            c = within_component_share(pdf, Sk)
            @test 0 < c < 1
            # the two equal-spread components of the closure reproduce w′² = 1 and the skewness
            a = (1 - Sk / sqrt(4 * (1 - c)^3 + Sk^2)) / 2
            w₁ = sqrt((1 - a) / a) * sqrt(1 - c)
            w₂ = -sqrt(a / (1 - a)) * sqrt(1 - c)
            @test a * w₁ + (1 - a) * w₂ ≈ 0 atol=1e-12
            @test a * (w₁^2 + c) + (1 - a) * (w₂^2 + c) ≈ 1 rtol=1e-12
            @test a * (w₁^3 + 3w₁ * c) + (1 - a) * (w₂^3 + 3w₂ * c) ≈ Sk rtol=1e-10 atol=1e-12
        end
    end
end
