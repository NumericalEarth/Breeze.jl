include(joinpath(@__DIR__, "setup.jl"))

#####
##### The subgrid cloud formulation of `TKEBasedTurbulenceClosure`
#####

using Breeze
using Breeze.AtmosphereModels: ClosureCloud, ClosureCloudStatistics, NoSubgridCorrelation, subgrid_statisticsᶜᶜᶜ
using Breeze.TurbulenceClosures: dissipation_stability_functionᶜᶜᶠ, tracer_stability_functionᶜᶜᶠ, SubgridCloudFields, tke_buoyancy_productionᶜᶜᶜ, buoyancy_productionᶜᶜᶠ, moment_time_scale, ℑbzᵃᵃᶜ
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: assumed_pdf, third_moment, limit_third_moment,
                                                              FixedConstantsLiquidSaturation

using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.Grids: znodes
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: step_closure_prognostics!
using Test

implicit_transport() = WENO(; order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(Float64; cfl=0.5))

column(field) = Array(interior(field))[1, 1, :]

function tke_p3_column(formulation; subgrid = NoSubgridCorrelation(), arch = default_arch, static_stability = MoistStaticStability())
    grid = RectilinearGrid(arch; size=32, z=(0, 1600), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=290)
    microphysics = PredictedParticlePropertiesMicrophysics(; subgrid,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(; static_stability, subgrid_cloud_formulation = formulation)
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

all_parts(; kw...) = SubgridCloudFormulation(; kw...)
distribution_only() = SubgridCloudFormulation(; buoyancy_flux = nothing, cloud = nothing)
with_buoyancy_flux(; stability = false) = SubgridCloudFormulation(; buoyancy_flux = CloudBuoyancyFlux(; stability), cloud = nothing)
with_cloud_ownership() = SubgridCloudFormulation(; buoyancy_flux = nothing)

@testset "SubgridCloudFormulation construction" begin
    @test TKEBasedTurbulenceClosure().subgrid_cloud_formulation === nothing
    f = SubgridCloudFormulation()
    @test f.distribution isa SubgridDistribution
    @test f.buoyancy_flux isa CloudBuoyancyFlux
    @test f.cloud isa ClosureOwnedCloud
    @test f.buoyancy_flux.stability
    @test f.distribution.skewness.maximum_skewness == 0.85
    @test_throws ArgumentError SubgridCloudFormulation(; distribution = nothing)
    @test_throws ArgumentError SubgridCloudFormulation(; distribution = nothing, cloud = nothing)
    @test SubgridCloudFormulation(; distribution = nothing, buoyancy_flux = nothing, cloud = nothing).distribution === nothing
    closure = TKEBasedTurbulenceClosure(Float32; subgrid_cloud_formulation = SubgridCloudFormulation(Float64))
    @test closure.subgrid_cloud_formulation.distribution.second_moments.ℂᵛ₁ isa Float32
    @test closure.subgrid_cloud_formulation.distribution.time_scale.ℂᴺ === 4.5f0
    @test occursin("subgrid_cloud_formulation", sprint(show, closure))
end

@testset "The default is the closure without the formulation" begin
    model = tke_p3_column(nothing)
    @test model.closure_fields.subgrid_cloud === nothing
    update_state!(model)
    grid = model.grid
    cf = model.closure_fields
    for k in 1:size(grid, 3)
        @test tke_buoyancy_productionᶜᶜᶜ(1, 1, k, grid, cf) ===
              ℑbzᵃᵃᶜ(1, 1, k, grid, buoyancy_productionᶜᶜᶠ, cf.Kᶜ, cf.N²)
    end
end

@testset "The distribution alone changes nothing" begin
    reference = tke_p3_column(nothing)
    model = tke_p3_column(distribution_only())
    @test model.closure_fields.subgrid_cloud isa SubgridCloudFields
    for _ in 1:4
        time_step!(reference, 30)
        time_step!(model, 30)
    end
    @test all(state_columns(model) .== state_columns(reference))
    cloud = model.closure_fields.subgrid_cloud.cloud
    @test maximum(column(cloud.C)) > 0
    @test all(0 .≤ column(cloud.C) .≤ 1)
    @test all(column(cloud.qˡ) .≥ 0)
    @test all(isfinite, column(cloud.w′θᵥ′))
end

@testset "The second and third moments are SHOC's down-gradient forms on the closure's ℓ, e, Kᶜ" begin
    model = tke_p3_column(distribution_only())
    update_state!(model)
    grid = model.grid
    cf = model.closure_fields
    m = cf.subgrid_cloud.moments
    d = model.closure.subgrid_cloud_formulation.distribution
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

@testset "Variance-budget time scales: ρ² = 3 ℂᵠ Sᶜ Sᴰ / (4 ℂʷ₁) and 3 ℂᵠ Sᶜ / (4 ℂʷ₁)" begin
    for time_scale in (DissipationTimeScale(; ℂᵠ = 0.1), DissipationTimeScale(; ℂᵠ = 0.14), EddyTurnoverTimeScale(; ℂᵠ = 1))
        formulation = SubgridCloudFormulation(; distribution = SubgridDistribution(; time_scale), buoyancy_flux = nothing, cloud = nothing)
        model = tke_p3_column(formulation)
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
            Sᴰ = dissipation ? dissipation_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N²) : 1
            Sᶜ = tracer_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N²)
            @test column(m.τ)[k] ≈ 2 * ℓ[k] / (ℂᵠ * Sᴰ * u) rtol=1e-12
            θ² = column(m.θˡ′²)[k]
            if θ² > 0
                # With u² for e the squared correlation is a function of Ri alone, bounded by its largest value
                ρ² = column(m.w′θˡ′)[k]^2 / (θ² * (2 / 3) * u^2)
                @test ρ² ≈ 3 * ℂᵠ * Sᶜ * Sᴰ / 4 rtol=1e-8
                @test ρ² ≤ 3 * ℂᵠ * max_S / 4 * (1 + 1e-12)
                n += 1
            end
        end
        @test n > 0
        @test all(isfinite, column(m.w′³))
        @test all(isfinite, column(cf.subgrid_cloud.cloud.C))
    end
    @test DissipationTimeScale(Float32).ℂᵠ isa Float32
    @test EddyTurnoverTimeScale(Float32).ℂᵠ isa Float32
end

@testset "The cloud is SHOC's assumed PDF of the stored moments" begin
    for pdf in (DoubleGaussianPDF(), SmoothDoubleGaussianPDF())
        formulation = SubgridCloudFormulation(; distribution = SubgridDistribution(; assumed_pdf = pdf),
                                                buoyancy_flux = nothing, cloud = nothing)
        model = tke_p3_column(formulation)
        update_state!(model)
        grid = model.grid
        sc = model.closure_fields.subgrid_cloud
        m, cl = sc.moments, sc.cloud
        d = model.closure.subgrid_cloud_formulation.distribution
        ρ = column(model.dynamics.reference_state.density)
        θ = column(model.formulation.potential_temperature_density) ./ ρ
        qᵗ = (column(model.moisture_density) .+ column(model.microphysical_fields.ρqᶜˡ)) ./ ρ
        p = column(model.dynamics.reference_state.pressure)
        zc = znodes(grid, Center()); zf = znodes(grid, Face())
        moments(k) = (column(m.w′³)[k], column(m.θˡ′²)[k], column(m.w′θˡ′)[k], column(m.θˡ′qᵗ′)[k],
                      column(m.w′qᵗ′)[k], column(m.qᵗ′²)[k])
        for k in 1:size(grid, 3)
            C, qˡ, qˡ′², w′qˡ′, w′θᵥ′ = assumed_pdf(pdf, FixedConstantsLiquidSaturation(), nothing, d.clipping, d.constants,
                                                   zf[k+1], zf[k], zc[k], p[k], θ[k], 0.0, qᵗ[k], column(cl.w′²)[k],
                                                   moments(k), moments(k+1))
            @test column(cl.C)[k] == C
            @test column(cl.qˡ)[k] == qˡ
            @test column(cl.w′qˡ′)[k] == w′qˡ′
            @test column(cl.w′θᵥ′)[k] == w′θᵥ′
            @test column(cl.B)[k] ≈ d.constants.ggr / d.constants.basetemp * w′θᵥ′
        end
        @test maximum(column(cl.C)) > 0
        @test maximum(column(cl.qˡ)) > 0
    end
end

@testset "The cloud buoyancy flux drives the TKE equation" begin
    model = tke_p3_column(with_buoyancy_flux())
    update_state!(model)
    grid = model.grid
    cf = model.closure_fields
    B = column(cf.subgrid_cloud.cloud.B)
    for k in 1:size(grid, 3)
        @test tke_buoyancy_productionᶜᶜᶜ(1, 1, k, grid, cf) == B[k]
    end
    reference = tke_p3_column(nothing)
    for _ in 1:4
        time_step!(reference, 30)
        time_step!(model, 30)
    end
    @test column(model.tracers.ρe) != column(reference.tracers.ρe)
    @test all(column(model.tracers.ρe) .≥ 0)
    @test all(isfinite, column(model.tracers.ρe))

    # Optionally the stability of the diffusivities is that of the cloud buoyancy flux:
    # (g / θ)(∂z θˡ + A_q ∂z qᵗ) - Bˡ / Kᶜ, with Bˡ the part carried by the liquid flux
    model = tke_p3_column(with_buoyancy_flux(; stability = true))
    update_state!(model)
    cf = model.closure_fields
    cl = cf.subgrid_cloud.cloud
    ρ = column(model.dynamics.reference_state.density)
    θ = column(model.formulation.potential_temperature_density) ./ ρ
    qᵗ = (column(model.moisture_density) .+ column(model.microphysical_fields.ρqᶜˡ)) ./ ρ
    zc = znodes(model.grid, Center())
    # Kᶜ of the first pass, from which Bˡ / Kᶜ was formed, is overwritten by the second; check where Bˡ = 0
    gθ, gA, Bˡ = column(cl.gθ), column(cl.gA), column(cl.Bˡ)
    for k in 2:size(model.grid, 3)
        Bˡ[k] == 0 && Bˡ[k-1] == 0 || continue
        N²ᵇ = (gθ[k] + gθ[k-1]) / 2 * (θ[k] - θ[k-1]) / (zc[k] - zc[k-1]) + (gA[k] + gA[k-1]) / 2 * (qᵗ[k] - qᵗ[k-1]) / (zc[k] - zc[k-1])
        @test column(cf.N²)[k] ≈ N²ᵇ rtol=1e-12
    end
    @test any(!=(0), Bˡ)
    @test all(isfinite, column(cf.N²))
    @test all(isfinite, column(cf.Kᶜ))
    for _ in 1:4
        time_step!(model, 30)
    end
    @test all(isfinite, column(model.tracers.ρe))
end

@testset "The closure owns the cloud of a P3 column; water and energy are conserved" begin
    model = tke_p3_column(with_cloud_ownership(); subgrid = ClosureCloud())
    μ = model.microphysical_fields
    ρ = column(model.dynamics.reference_state.density)
    ρθ = model.formulation.potential_temperature_density

    water₀ = sum(non_precipitating_water(model) .+ column(μ.ρqʳ))
    for _ in 1:5
        time_step!(model, 60)
    end
    sc = model.closure_fields.subgrid_cloud
    @test maximum(column(sc.cloud.qˡ)) > 1e-6
    @test 0 < maximum(column(sc.cloud.C)) ≤ 1
    stats = subgrid_statisticsᶜᶜᶜ(1, 1, 20, model.grid, ClosureCloud(), model.closure, model.closure_fields)
    @test stats == ClosureCloudStatistics(column(sc.cloud.C)[20], column(sc.cloud.qˡ)[20])

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
    model = tke_p3_column(with_cloud_ownership(); subgrid = NoSubgridCorrelation())
    set!(model.microphysical_fields.ρqᶜˡ, z -> ifelse(600 < z < 800, 1.2e-4, 0.0))
    update_state!(model)
    cloud_liquid = column(model.microphysical_fields.ρqᶜˡ)
    step_closure_prognostics!(model.closure_fields, model.closure, model, 60)
    @test column(model.microphysical_fields.ρqᶜˡ) == cloud_liquid
end

@testset "All three parts, with the saturation of the model" begin
    for saturation in (FixedConstantsLiquidSaturation(), ModelThermodynamicsSaturation())
        formulation = SubgridCloudFormulation(; distribution = SubgridDistribution(; saturation))
        model = tke_p3_column(formulation; subgrid = ClosureCloud(), static_stability = DryStaticStability())
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

@testset "Iterating the distribution and the cloud buoyancy stability" begin
    @test CloudBuoyancyFlux() isa CloudBuoyancyFlux{0}
    @test CloudBuoyancyFlux(true) isa CloudBuoyancyFlux{0}
    @test Breeze.TurbulenceClosures.stability_iterations(CloudBuoyancyFlux(; stability_iterations = 3)) == 3
    @test_throws ArgumentError CloudBuoyancyFlux(; stability_iterations = -1)
    @test_throws ArgumentError CloudBuoyancyFlux(; stability = false, stability_iterations = 1)
    @test CloudBuoyancyFlux().stability_relaxation == 1
    @test CloudBuoyancyFlux(; stability_iterations = 3, stability_relaxation = 0.5).stability_relaxation == 0.5
    @test_throws ArgumentError CloudBuoyancyFlux(; stability_relaxation = 0)
    @test_throws ArgumentError CloudBuoyancyFlux(; stability_relaxation = 1.5)
    relaxed = TKEBasedTurbulenceClosure(Float32; subgrid_cloud_formulation =
        SubgridCloudFormulation(Float64; buoyancy_flux = CloudBuoyancyFlux(; stability_iterations = 2, stability_relaxation = 0.3)))
    @test relaxed.subgrid_cloud_formulation.buoyancy_flux.stability_relaxation === 0.3f0

    iterated(n) = SubgridCloudFormulation(; buoyancy_flux = CloudBuoyancyFlux(; stability_iterations = n))
    closure_columns(model) = (cf = model.closure_fields; sc = cf.subgrid_cloud;
                              map(column, (cf.N², cf.ℓᵐ, cf.Kᶜ, sc.moments..., sc.cloud...)))
    build(n) = tke_p3_column(iterated(n); subgrid = ClosureCloud(), static_stability = DryStaticStability())

    reference = build(0)
    default = tke_p3_column(SubgridCloudFormulation(); subgrid = ClosureCloud(), static_stability = DryStaticStability())
    for m in (reference, default), _ in 1:3
        time_step!(m, 30)
    end
    @test closure_columns(default) == closure_columns(reference)

    # ω = 1 is the unrelaxed iteration
    unrelaxed = build(2)
    explicit_one = tke_p3_column(SubgridCloudFormulation(; buoyancy_flux = CloudBuoyancyFlux(; stability_iterations = 2, stability_relaxation = 1.0));
                                 subgrid = ClosureCloud(), static_stability = DryStaticStability())
    relaxed = tke_p3_column(SubgridCloudFormulation(; buoyancy_flux = CloudBuoyancyFlux(; stability_iterations = 2, stability_relaxation = 0.5));
                            subgrid = ClosureCloud(), static_stability = DryStaticStability())
    for m in (unrelaxed, explicit_one, relaxed), _ in 1:3
        time_step!(m, 30)
    end
    @test closure_columns(explicit_one) == closure_columns(unrelaxed)
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
        @test all(0 .≤ column(model.closure_fields.subgrid_cloud.cloud.C) .≤ 1)

        # The TKE buoyancy production is the stored B of the last distribution
        update_state!(model)
        cf = model.closure_fields
        B = column(cf.subgrid_cloud.cloud.B)
        @test all(tke_buoyancy_productionᶜᶜᶜ(1, 1, k, model.grid, cf) == B[k] for k in 1:size(model.grid, 3))
    end
end

@testset "Secant and bisection solves of the cloud buoyancy stability" begin
    @test CloudBuoyancyFlux().stability_solver isa FixedPointIteration
    @test BisectionIteration().final_interpolation
    @test occursin("SecantIteration", summary(CloudBuoyancyFlux(; stability_solver = SecantIteration())))
    solved(solver, n) = SubgridCloudFormulation(; buoyancy_flux = CloudBuoyancyFlux(; stability_iterations = n, stability_solver = solver))
    build(f) = tke_p3_column(f; subgrid = ClosureCloud(), static_stability = DryStaticStability())
    for solver in (SecantIteration(), BisectionIteration(), BisectionIteration(; final_interpolation = false))
        model = build(solved(solver, 4))
        @test model.closure_fields.subgrid_cloud.solver_state.x₋ isa Field
        for _ in 1:3
            time_step!(model, 30)
        end
        cf = model.closure_fields
        @test all(isfinite, column(cf.Kᶜ))
        @test all(isfinite, column(cf.N²))
        @test all(isfinite, column(model.tracers.ρe))
        @test all(0 .≤ column(cf.subgrid_cloud.cloud.C) .≤ 1)
        # N² stays in the bracket of the conserved and the saturated stratification
        L, U = column(cf.subgrid_cloud.solver_state.lower), column(cf.subgrid_cloud.solver_state.upper)
        N² = column(cf.N²)
        @test all(L[k] - 1e-12 ≤ N²[k] ≤ U[k] + 1e-12 for k in 2:length(N²) - 1)
    end
    @test build(solved(FixedPointIteration(), 2)).closure_fields.subgrid_cloud.solver_state === nothing
end

@testset "Lagged stability" begin
    f = SubgridCloudFormulation(; buoyancy_flux = CloudBuoyancyFlux(; stability_solver = LaggedStability()))
    model = tke_p3_column(f; subgrid = ClosureCloud(), static_stability = DryStaticStability())
    N²₋ = model.closure_fields.subgrid_cloud.solver_state.N²₋
    update_state!(model)
    @test column(N²₋) == column(model.closure_fields.N²)
    for _ in 1:3
        time_step!(model, 30)
    end
    @test all(isfinite, column(model.closure_fields.Kᶜ))
    @test all(isfinite, column(model.tracers.ρe))
end
