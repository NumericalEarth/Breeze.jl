include(joinpath(@__DIR__, "setup.jl"))

#####
##### The subgrid distribution of `TKEBasedTurbulenceClosure`
#####

using Breeze
using Breeze.Microphysics.PredictedParticleProperties: GridMeanCondensation, SubgridCondensation
using Breeze.TurbulenceClosures: SpecificValue, dissipation_stability_functionᶜᶜᶠ, tracer_stability_functionᶜᶜᶠ, SubgridCloudFields, tke_buoyancy_productionᶜᶜᶜ, buoyancy_productionᶜᶜᶠ, moment_time_scale, ℑbzᵃᵃᶜ
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: representable_skewness
using Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures: assumed_pdf, third_moment, limit_third_moment, linearly_interpolate,
                                                              distribution_thermodynamics
using Breeze.AtmosphereModels.Diagnostics: condensate_phase_equilibrium

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

function tke_p3_column(distribution; condensation = GridMeanCondensation(), arch = default_arch, static_stability = MoistStaticStability(),
                       stability_functions = TurbulentRichardsonStabilityFunction(), solver = SinglePass(),
                       scalar_advection = implicit_transport(), model_kwargs...)
    grid = RectilinearGrid(arch; size=32, z=(0, 1600), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=290)
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(; static_stability, stability_functions, subgrid_distribution = distribution, solver)
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics, closure,
                            scalar_advection, model_kwargs...)

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
    @test d.assumed_pdf isa EqualWidthDoubleGaussianPDF
    # The bound of the limiter is the largest skewness the PDF represents: that of its two components of equal width at
    # the least weight of the updraft component, (1 - 2a)(1 - ℂᵍ)^{3/2} / √(a (1 - a)) with a = 0.01 and ℂᵍ = 0.4
    @test d.skewness.maximum_skewness == representable_skewness(d.assumed_pdf)
    @test d.skewness.maximum_skewness ≈ 0.98 * 0.6^1.5 / sqrt(0.01 * 0.99) rtol=1e-14
    @test 4.57 < d.skewness.maximum_skewness < 4.58
    # At the bound the PDF's updraft component has its least weight, and the bound follows the PDF given
    for (a, ℂᵍ₀) in ((0.01, 0.4), (0.05, 0.4), (0.02, 0.6), (0.1, 0.25))
        pdf = EqualWidthDoubleGaussianPDF(; ℂᵍ₀, minimum_updraft_fraction = a, maximum_updraft_fraction = 1 - a)
        Sk = representable_skewness(pdf)
        @test (1 - Sk / sqrt(4 * (1 - ℂᵍ₀)^3 + Sk^2)) / 2 ≈ a rtol=1e-12
        @test DoubleGaussian(; assumed_pdf = pdf).skewness.maximum_skewness == Sk
    end
    # the less extreme of the two bounds on the weight decides
    @test representable_skewness(EqualWidthDoubleGaussianPDF(; minimum_updraft_fraction = 0.01, maximum_updraft_fraction = 0.9)) ≈
          representable_skewness(EqualWidthDoubleGaussianPDF(; minimum_updraft_fraction = 0.1, maximum_updraft_fraction = 0.9))
    # SHOC's PDFs have components of equal width in w too
    @test representable_skewness(DoubleGaussianPDF()) == representable_skewness(EqualWidthDoubleGaussianPDF())
    @test representable_skewness(SmoothDoubleGaussianPDF()) == representable_skewness(EqualWidthDoubleGaussianPDF())
    @test DoubleGaussian(; skewness = PolynomialSaturatedSkewness(maximum_skewness = 2)).skewness.maximum_skewness == 2
    @test DoubleGaussian(Float32).skewness.maximum_skewness isa Float32
    closure = TKEBasedTurbulenceClosure(Float32; subgrid_distribution = DoubleGaussian(Float64))
    @test closure.subgrid_distribution.second_moments.ℂᵛ₁ isa Float32
    @test closure.subgrid_distribution.time_scale.ℂᵠ === 1.3f0 # DissipationTimeScale is the default
    @test closure.subgrid_distribution.time_scale.third_moment_time_scale.ℂᴺ === 4.5f0
    @test occursin("subgrid_distribution", sprint(show, closure))
    @test occursin("solver: SinglePass()", sprint(show, closure))

    # The coefficients of the closure, as one flat view
    closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(), static_stability = fraction_stability)
    ℂ = closure_parameters(closure)
    @test all(name -> startswith(string(name), "ℂ") || startswith(string(name), "ϑ") || name ∈ (:Ri⁻, :Ri⁺), keys(ℂ))
    @test (:ℂˢ, :ℂⁱ, :ℂᵘ⁰, :ℂᵠ, :ℂᵛ₁, :ℂʷ₁, :ℂʷᵉ, :ℂʷᵋ, :ℂʷᵇ, :ℂᵍ₀) ⊆ keys(ℂ)
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
    @test all(isfinite, column(cloud.w′θᵛ′))
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
        @test column(m.θˡⁱ′²)[k] ≈ τ * Kᶜ[k] * ((θ[k] - θ[k-1]) / Δz)^2 rtol=1e-12
        @test column(m.qᵗ′²)[k] ≈ τ * Kᶜ[k] * ((qᵗ[k] - qᵗ[k-1]) / Δz)^2 rtol=1e-12
        @test column(m.w′θˡⁱ′)[k] ≈ -Kᶜ[k] * (θ[k] - θ[k-1]) / Δz rtol=1e-12
        @test column(m.w′qᵗ′)[k] ≈ -Kᶜ[k] * (qᵗ[k] - qᵗ[k-1]) / Δz rtol=1e-12
    end
    # The skewness at the centers, limited with the center w′²
    w² = column(cf.subgrid_cloud.cloud.w′²)
    w³ = column(cf.subgrid_cloud.cloud.w′³)
    @test all(abs.(w³) .≤ d.skewness.maximum_skewness .* w² .^ 1.5 .* (1 + 1e-12))
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
        eᶠ = SpecificValue(model.tracers.ρe, model.dynamics.reference_state.density)
        Nz = size(grid, 3)
        eᵐⁱⁿ = closure.minimum_tke
        # The largest value over Riᵗ (the quadratic in ln 𝕊 may peak between its three values)
        products = [Breeze.TurbulenceClosures.tracer_stability_function(sf, αᴺ, 0.0) *
                    (dissipation ? Breeze.TurbulenceClosures.dissipation_stability_function(sf, αᴺ, 0.0) : 1)
                    for αᴺ in sf.ℂᵅ .* sinh.(range(-20, 20, length = 4001))]
        max_S = maximum(products)
        n = 0
        for k in 2:Nz
            u = (sqrt(max(e[k], eᵐⁱⁿ)) + sqrt(max(e[k-1], eᵐⁱⁿ))) / 2
            𝕊ᴰ = dissipation ? @allowscalar(dissipation_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N², eᶠ, cf.ℓᵐ)) : 1
            𝕊ᶜ = @allowscalar tracer_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N², eᶠ, cf.ℓᵐ)
            @test column(m.τ)[k] ≈ 2 * ℓ[k] / (ℂᵠ * 𝕊ᴰ * u) rtol=1e-12
            θ² = column(m.θˡⁱ′²)[k]
            if θ² > 0
                # With u² for e the squared correlation is a function of Ri alone, bounded by its largest value
                ρ² = column(m.w′θˡⁱ′)[k]^2 / (θ² * (2 / 3) * u^2)
                @test ρ² ≈ 3 * ℂᵠ * 𝕊ᶜ * 𝕊ᴰ / 4 rtol=1e-8
                @test ρ² ≤ 3 * ℂᵠ * max_S / 4 * (1 + 1e-12)
                n += 1
            end
        end
        @test n > 0
        @test all(isfinite, column(cf.subgrid_cloud.cloud.w′³))
        @test all(isfinite, column(cf.subgrid_cloud.cloud.aᶜ))
    end
    @test DissipationTimeScale(Float32).ℂᵠ isa Float32
    @test EddyTurnoverTimeScale(Float32).ℂᵠ isa Float32
end

@testset "The cloud is SHOC's assumed PDF of the moments co-located at the centers" begin
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
        moments(k) = (column(m.θˡⁱ′²)[k], column(m.w′θˡⁱ′)[k], column(m.θˡⁱ′qᵗ′)[k], column(m.w′qᵗ′)[k], column(m.qᵗ′²)[k])
        floors = (0.0, -99999999.99, -99999999.99, -99999999.99, 0.0)
        for k in 1:size(grid, 3)
            # The second moments interpolated to the center, with the third moment diagnosed there
            mᶜ = map((u, l, f) -> max(f, linearly_interpolate(zf[k+1], zf[k], u, l, zc[k])), moments(k+1), moments(k), floors)
            w′²ᶜ = column(cl.w′²)[k]
            w′³ᶜ = column(cl.w′³)[k]
            @test abs(w′³ᶜ) ≤ d.skewness.maximum_skewness * w′²ᶜ^1.5 * (1 + 1e-12)
            mᶜ = (w′³ᶜ, mᶜ...)
            C, qˡ, qˡ′², w′qˡ′, w′θᵛ′ = assumed_pdf(pdf, distribution_thermodynamics(model),
                                                   zf[k+1], zf[k], zc[k], p[k], θ[k], 0.0, qᵗ[k], w′²ᶜ, mᶜ, mᶜ)
            @test matches(column(cl.aᶜ)[k], C)
            @test matches(column(cl.qˡ)[k], qˡ)
            @test matches(column(cl.w′qˡ′)[k], w′qˡ′)
            @test matches(column(cl.w′θᵛ′)[k], w′θᵛ′)
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

@testset "With P3 the closure leaves the cloud liquid to the microphysics" begin
    model = tke_p3_column(DoubleGaussian())
    set!(model.microphysical_fields.ρqᶜˡ, z -> ifelse(600 < z < 800, 1.2e-4, 0.0))
    update_state!(model)
    cloud_liquid = column(model.microphysical_fields.ρqᶜˡ)
    step_closure_prognostics!(model.closure_fields, model.closure, model, 60)
    @test column(model.microphysical_fields.ρqᶜˡ) == cloud_liquid
end

@testset "The distribution condenses with the thermodynamics and the condensate phase of the model" begin
    # The distribution of this closure condenses on the phase of the condensate in the water it distributes: with P3,
    # the vapor and the cloud liquid, whose condensate is liquid (P3's ice is not distributed)
    for distribution in (DoubleGaussian(),)
        model = tke_p3_column(distribution; static_stability = fraction_stability)
        thermodynamics = distribution_thermodynamics(model)
        @test thermodynamics.constants === model.thermodynamic_constants
        @test thermodynamics.phase_equilibrium === condensate_phase_equilibrium(model.microphysics)
        @test thermodynamics.phase_equilibrium isa WarmPhaseEquilibrium
        μ = model.microphysical_fields
        water₀ = sum(non_precipitating_water(model) .+ column(μ.ρqʳ))
        for _ in 1:10
            time_step!(model, 60)
        end
        @test sum(non_precipitating_water(model) .+ column(μ.ρqʳ)) ≈ water₀ rtol=1e-12
        @test all(isfinite, column(model.tracers.ρe))
        @test all(isfinite, column(μ.ρqᶜˡ))
    end
end

@testset "Damped fixed-point iteration of the static stability and the saturated fraction" begin
    @test DampedFixedPoint().iterations == 1
    @test Breeze.TurbulenceClosures.stability_iterations(Breeze.TurbulenceClosures.AppliedFluxEnergetics(DampedFixedPoint(; iterations = 3))) == 3
    @test Breeze.TurbulenceClosures.stability_iterations(Breeze.TurbulenceClosures.AppliedFluxEnergetics(SinglePass())) == 0
    @test_throws ArgumentError DampedFixedPoint(; iterations = -1)
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(), static_stability = MoistStaticStability(),
                                                         solver = DampedFixedPoint())
    @test DampedFixedPoint(; iterations = 3, relaxation = 0.5).relaxation == 0.5
    @test_throws ArgumentError DampedFixedPoint(; relaxation = 0)
    @test_throws ArgumentError DampedFixedPoint(; relaxation = 1.5)
    relaxed = TKEBasedTurbulenceClosure(Float32; subgrid_distribution = DoubleGaussian(Float64), static_stability = fraction_stability,
                                        solver = DampedFixedPoint(; iterations = 2, relaxation = 0.3))
    @test relaxed.solver.relaxation === 0.3f0

    closure_columns(model) = (cf = model.closure_fields; sc = cf.subgrid_cloud;
                              map(column, (cf.N², cf.ℓᵐ, cf.Kᶜ, sc.moments..., sc.cloud...)))
    build(n; relaxation = 1) = tke_p3_column(DoubleGaussian(); static_stability = fraction_stability,
                                             solver = DampedFixedPoint(; iterations = n, relaxation))

    reference = build(0)
    default = tke_p3_column(DoubleGaussian(); static_stability = fraction_stability)
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
    build(solver) = tke_p3_column(DoubleGaussian(); static_stability = fraction_stability, solver)
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
    model = tke_p3_column(DoubleGaussian(); static_stability = fraction_stability, solver = LaggedStability())
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
    # With a distribution and no static stability given, the fraction is the default; without one, MoistStaticStability
    @test TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian()).static_stability ==
          SaturationWeightedStaticStability(CondensateFluxFraction())
    @test TKEBasedTurbulenceClosure().static_stability isa MoistStaticStability
    # The fraction needs the distribution
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
end

@testset "DissipationTimeScale reads the stability functions of the closure" begin
    time_scale = DissipationTimeScale(; ℂᵠ = 0.14)
    τ = Dict()
    for (name, stability_functions) in ((:stratification, TurbulentRichardsonStabilityFunction()), (:richardson, RationalStabilityFunction()))
        model = tke_p3_column(DoubleGaussian(; time_scale); stability_functions)
        update_state!(model)
        grid = model.grid
        cf = model.closure_fields
        closure = model.closure
        e = SpecificValue(model.tracers.ρe, model.dynamics.reference_state.density)
        ρ = column(model.dynamics.reference_state.density)
        eᶜ = column(model.tracers.ρe) ./ ρ
        ℓ = column(cf.ℓᵐ)
        τ[name] = column(cf.subgrid_cloud.moments.τ)
        for k in 2:size(grid, 3)
            u = (sqrt(max(eᶜ[k], closure.minimum_tke)) + sqrt(max(eᶜ[k-1], closure.minimum_tke))) / 2
            𝕊ᴰ = @allowscalar dissipation_stability_functionᶜᶜᶠ(1, 1, k, grid, closure, model.velocities, cf.N², e, cf.ℓᵐ)
            @test τ[name][k] ≈ 2 * ℓ[k] / (time_scale.ℂᵠ * 𝕊ᴰ * u) rtol=1e-10
        end
    end
    @test τ[:stratification] != τ[:richardson]
end

@testset "Bracketed solve of the effective saturated fraction aᴺ" begin
    @test summary(BracketedFractionSolve(2)) == "BracketedFractionSolve(2)"
    @test_throws ArgumentError BracketedFractionSolve(-1)
    @test summary(BracketedFractionSolve(1; lagged_envelope = true)) == "BracketedFractionSolve(1, lagged_envelope = true)"
    for (n, lagged_envelope, stability_functions) in ((0, false, TurbulentRichardsonStabilityFunction()),
                                                     (2, false, TurbulentRichardsonStabilityFunction()),
                                                     (2, true, TurbulentRichardsonStabilityFunction()),
                                                     (2, false, RationalStabilityFunction()))
        model = tke_p3_column(DoubleGaussian(); static_stability = fraction_stability, stability_functions,
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

#####
##### The ratios the distribution supplies are formed where it lives: the condensate flux fraction aʷ at the
##### centers (averaged to the faces), the skewness from w′³ and w′² at the centers. With aʷ formed at the faces
##### from the condensate flux averaged there over the raw face flux of the saturation excess, the diffusivity
##### and the 2Δz of the gradients pass to aʷ, and the diffusivity of a cloudy layer settles in a 2Δz
##### checkerboard (Kᶜ alternating by an order of magnitude); formed at the centers they cancel.
#####

# A BOMEX-like column with surface fluxes, its cloud layer moistened to saturation
function saturated_layer_column()
    grid = RectilinearGrid(default_arch; size=40, z=(0, 2400), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = PredictedParticlePropertiesMicrophysics(; negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    closure = TKEBasedTurbulenceClosure(; minimum_tke = 4e-4, subgrid_distribution = DoubleGaussian(),
                                        static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()))
    boundary_conditions = (ρθ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 8e-3)),
                           ρqᵗ = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 5.2e-5)),
                           ρu = FieldBoundaryConditions(bottom = FluxBoundaryCondition(1.17 * 0.08)),
                           ρv = FieldBoundaryConditions(bottom = FluxBoundaryCondition(-1.17 * 0.02)))
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants,
                            microphysics, closure, boundary_conditions)
    θ₀(z) = z < 520 ? 298.7 : z < 1480 ? 298.7 + (z - 520) * 3.7 / 960 : 302.4 + (z - 1480) * 5.8 / 520
    qᵗ₀(z) = 1e-3 * (z < 520 ? 17.0 - z * 0.7 / 520 : z < 1480 ? 16.3 - (z - 520) * 5.6 / 960 : max(4.2, 10.7 - (z - 1480) * 6.5 / 520))
    set!(model; θ=θ₀, qᵗ = z -> qᵗ₀(z) + (500 < z < 1100 ? 2.5e-3 : 0.0),
         u = z -> z < 700 ? -8.75 : -8.75 + (z - 700) * 1.8e-3, ρe = z -> 0.6 * exp(-z / 500))
    return model
end

# The 2Δz content of K over the turbulent faces: Σ|δ²K| / (4 Σ|K|), 1 for a pure 2Δz mode
checkerboard(K) = sum(abs, K[1:end-2] .- 2 .* K[2:end-1] .+ K[3:end]) / (4 * sum(abs, K[2:end-1]))

@testset "A seeded 2Δz perturbation leaves no checkerboard of Kᶜ in a cloudy column" begin
    model = saturated_layer_column()
    for _ in 1:50
        time_step!(model, 2)
    end
    @test maximum(column(model.closure_fields.subgrid_cloud.cloud.aᶜ)) > 0

    # 0.01 K of θ in the 2Δz mode
    Nz = size(model.grid, 3)
    ρ = column(model.dynamics.reference_state.density)
    ρθ = column(model.formulation.potential_temperature_density)
    set!(model.formulation.potential_temperature_density, reshape(ρθ .+ 0.01 .* ρ .* [(-1.0)^k for k in 1:Nz], 1, 1, Nz))
    update_state!(model)

    for _ in 1:20
        time_step!(model, 2)
        Kᶜ = column(model.closure_fields.Kᶜ)[2:Nz]
        turbulent = Kᶜ[Kᶜ .> 1]
        @test length(turbulent) ≥ 20 # half the column turbulent (20 faces with the default stability functions)
        @test checkerboard(turbulent) < 0.2 # 0.75 with aʷ formed at the faces
    end
    @test all(isfinite, column(model.closure_fields.Kᶜ))
    aʷ = column(model.closure_fields.subgrid_cloud.cloud.aʷ)
    @test all(a -> 0 ≤ a ≤ 1, aʷ)
end

#####
##### Rain that falls many cells per time step
#####

# Rain seeded in the cloud layer falls at about 6 m s⁻¹: a Courant number of 35 at Δt = 300 s on this grid.
# Rain is transported by the adaptive vertically implicit discretization, which moves what exceeds its
# Courant limit in the tridiagonal solve. With rain left to an explicit scheme (the default `Centered`
# advection of `AtmosphereModel`) each stage removes many times the content of a cell: rain reaches
# -3 × 10⁻² kg m⁻³ and below, the temperature diagnosed from θˡⁱ and the condensate follows it, and the saturation
# vapor pressure of a component of the distribution throws a `DomainError` within the first two steps.
rain_transport() = (ρqʳ = WENO(; order=5, bounds=(0.0, 1.0), time_discretization=AdaptiveVerticallyImplicitDiscretization(Float64; cfl=0.25)),
                    ρnʳ = implicit_transport())

@testset "Rain at a Courant number of 35 leaves the column with the closure intact [$timestepper]" for timestepper in
        (:SSPRungeKutta3, :MultirateInfinitesimalStep)
    model = tke_p3_column(DoubleGaussian(); condensation = SubgridCondensation(TurbulenceClosureStatistics()),
                          static_stability = fraction_stability, scalar_advection = rain_transport(), timestepper)
    set!(model; qʳ = z -> ifelse(500 < z < 800, 2e-4, 0.0), nʳ = z -> ifelse(500 < z < 800, 1e3, 0.0))
    μ = model.microphysical_fields
    Δt = 300
    Δz = 1600 / 32
    @test maximum(abs, column(μ.wʳ)) * Δt / Δz > 30

    T⁰ = column(model.temperature)
    ρqʳ⁰ = maximum(column(μ.ρqʳ))
    minimum_rain = Inf
    for _ in 1:12
        time_step!(model, Δt)
        minimum_rain = min(minimum_rain, minimum(column(μ.ρqʳ)))
    end

    T = column(model.temperature)
    @test all(isfinite, T)
    @test maximum(abs, T .- T⁰) < 5 # more than 10 K within two steps with rain transported explicitly
    @test all(isfinite, column(model.closure_fields.Kᶜ))
    # The TKE stays non-negative under SSP-RK3. Under `:SSPMIS43` the extrapolating stage leaves a decaying TKE
    # with the opposite sign at the level of the underflow (-1.3 × 10⁻¹⁵⁶ J m⁻³ here, at one cell above the
    # turbulent layer): bounded by the floor of e in the buoyancy sink rate, √floatmin.
    tke_bound = timestepper === :SSPRungeKutta3 ? 0.0 : sqrt(floatmin(Float64))
    @test all(e -> e ≥ -tke_bound, column(model.tracers.ρe))

    # Every stage of SSP-RK3 is a convex combination of states, and the solves keep their sign. Stage 2 of
    # `:SSPMIS43` extrapolates, V₂ = -0.05 Uⁿ + 1.05 Y₁ + ⋯, so where a solve empties a cell it leaves a
    # twentieth of the rain the cell held with the opposite sign.
    bound = timestepper === :SSPRungeKutta3 ? 1e-12 * ρqʳ⁰ : 0.1 * ρqʳ⁰
    @test minimum_rain > -bound
end

@testset "ScalarVarianceTransport: the steady budget of the scalar second moments with transport" begin
    moments(model) = (m = model.closure_fields.subgrid_cloud.moments; Array.(interior.((m.θˡⁱ′², m.qᵗ′², m.θˡⁱ′qᵗ′))))
    local_model = tke_p3_column(DoubleGaussian())
    update_state!(local_model)
    θ₀, q₀, θq₀ = moments(local_model)

    # With no diffusivity the budget is the local diagnosis
    none = tke_p3_column(DoubleGaussian(; variance_transport = ScalarVarianceTransport(; ℂᵛᵗ = 0)))
    update_state!(none)
    θ, q, θq = moments(none)
    @test θ ≈ θ₀ && q ≈ q₀ && θq ≈ θq₀

    @test DoubleGaussian(Float32; variance_transport = ScalarVarianceTransport()).variance_transport.ℂᵛᵗ isa Float32
    @test_throws ArgumentError ScalarVarianceTransport(; ℂᵛᵗ = -1)

    for third in (false, true)
        model = tke_p3_column(DoubleGaussian(; variance_transport = ScalarVarianceTransport(; ℂᵛᵗ = 1, component_third_moment = third)))
        update_state!(model)
        θ, q, θq = moments(model)
        @test all(isfinite, θ) && all(isfinite, q) && all(isfinite, θq)
        @test θ != θ₀
        if !third # the M-matrix: non-negative variances and a correlation no larger than 1
            @test minimum(θ) ≥ 0 && minimum(q) ≥ 0
            @test all(θq .^ 2 .≤ θ .* q .* (1 + 1e-12) .+ 1e-30)
        end
        for _ in 1:3
            time_step!(model, 10)
        end
        @test all(isfinite, column(model.tracers.ρe))
    end
end

@testset "VarianceStabilityFunctions: θˡⁱ′² = 𝕊ᵛ₁ ℓ² (∂z θˡⁱ)², qᵗ′² = 𝕊ᵛ₂ ℓ² (∂z qᵗ)², θˡⁱ′qᵗ′ = 𝕏ᵛ √(𝕊ᵛ₁ 𝕊ᵛ₂) ℓ² ∂z θˡⁱ ∂z qᵗ" begin
    moments(model) = (m = model.closure_fields.subgrid_cloud.moments; Array.(interior.((m.θˡⁱ′², m.qᵗ′², m.θˡⁱ′qᵗ′, m.w′θˡⁱ′, m.w′qᵗ′, m.τ))))
    sf = TurbulentRichardsonStabilityFunction()

    @test DoubleGaussian().variance_stability_functions === nothing
    @test DoubleGaussian(Float32; variance_stability_functions = VarianceStabilityFunctions()).variance_stability_functions.ℂᵛ₁⁰ isa Float32
    @test_throws ArgumentError VarianceStabilityFunctions(; ℂᵛ₁⁺ = 0)
    @test_throws ArgumentError VarianceStabilityFunctions(; ℂᵛ₃ᵐ = 1.1)
    @test_throws ArgumentError TKEBasedTurbulenceClosure(; stability_functions = RationalStabilityFunction(),
        subgrid_distribution = DoubleGaussian(; variance_stability_functions = VarianceStabilityFunctions()))
    @test_throws ArgumentError VarianceStabilityFunctions(sf, DissipationTimeScale(), DowngradientSecondMoments(; ℂᵛ₃ = 2))

    # The defaults are those that reproduce the default distribution
    d = DoubleGaussian()
    vs = VarianceStabilityFunctions(sf, d.time_scale, d.second_moments)
    @test all(name -> getfield(vs, name) ≈ getfield(VarianceStabilityFunctions(), name), fieldnames(VarianceStabilityFunctions))

    # The coefficients are those of the closure, calibratable
    closure = TKEBasedTurbulenceClosure(; subgrid_distribution = DoubleGaussian(; variance_stability_functions = vs))
    names = (:ℂᵛ₁⁻, :ℂᵛ₁⁰, :ℂᵛ₁⁺, :ℂᵛ₂⁻, :ℂᵛ₂⁰, :ℂᵛ₂⁺, :ℂᵛ₃⁻, :ℂᵛ₃ᵐ, :ℂᵛ₃⁺)
    @test all(name -> name ∈ keys(closure_parameters(closure)), names)
    @test with_parameters(closure; ℂᵛ₂⁺ = 0.7).subgrid_distribution.variance_stability_functions.ℂᵛ₂⁺ == 0.7

    # Reproduce the down-gradient diagnosis ℂᵛ τ Kᶜ ∂z a ∂z b to rounding, alone, with the flux-correlation
    # bound and with the transport of the variances
    sm = DowngradientSecondMoments(; ℂᵛ₁ = 1.5, ℂᵛ₂ = 0.7, ℂᵛ₃ = 0.8)
    for time_scale in (DissipationTimeScale(; ℂᵠ = 1.3), EddyTurnoverTimeScale(; ℂᵠ = 1)),
        options in ((;), (; ℂʳ = 0.5), (; variance_transport = ScalarVarianceTransport()))
        reference = tke_p3_column(DoubleGaussian(; time_scale, second_moments = sm, options...))
        reproduced = tke_p3_column(DoubleGaussian(; time_scale, second_moments = sm, options...,
                                                  variance_stability_functions = VarianceStabilityFunctions(sf, time_scale, sm)))
        update_state!(reference)
        update_state!(reproduced)
        for (a, b) in zip(moments(reference), moments(reproduced))
            @test isapprox(a, b; rtol = 1e-12, atol = 1e-30)
        end
        @test column(reference.closure_fields.subgrid_cloud.cloud.aᶜ) ≈ column(reproduced.closure_fields.subgrid_cloud.cloud.aᶜ) rtol=1e-10
    end

    # Realizability: |θˡⁱ′qᵗ′| ≤ √(θˡⁱ′² qᵗ′²) for any coefficients within their intervals, and the correlation follows 𝕏ᵛ
    for (ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺) in ((1, 1, 1), (-1, -1, -1), (0.9, -1, 0.3), (-0.5, 1, 1), (0, 0, 0))
        vs = VarianceStabilityFunctions(; ℂᵛ₁⁺ = 2, ℂᵛ₂⁻ = 0.1, ℂᵛ₃⁻, ℂᵛ₃ᵐ, ℂᵛ₃⁺)
        model = tke_p3_column(DoubleGaussian(; variance_stability_functions = vs))
        update_state!(model)
        θ, q, θq = moments(model)
        @test minimum(θ) ≥ 0 && minimum(q) ≥ 0
        @test all(θq .^ 2 .≤ θ .* q .* (1 + 1e-12) .+ 1e-30)
        for _ in 1:3
            time_step!(model, 10)
        end
        @test all(isfinite, column(model.tracers.ρe))
    end
    for v in range(-1, 1, length = 41)
        _, _, 𝕏ᵛ = Breeze.TurbulenceClosures.variance_stability_functions(VarianceStabilityFunctions(; ℂᵛ₃⁻ = -1, ℂᵛ₃ᵐ = 1, ℂᵛ₃⁺ = -1), v)
        @test -1 ≤ 𝕏ᵛ ≤ 1
    end

    # Larger stable variance: larger variances where the column is stratified, the fluxes unchanged
    base = tke_p3_column(DoubleGaussian(; variance_stability_functions = VarianceStabilityFunctions()))
    larger = tke_p3_column(DoubleGaussian(; variance_stability_functions = VarianceStabilityFunctions(; ℂᵛ₁⁺ = 1.2, ℂᵛ₂⁺ = 1.2)))
    update_state!(base)
    update_state!(larger)
    θ₀, _, _, wθ₀ = moments(base)
    θ₁, _, _, wθ₁ = moments(larger)
    @test all(θ₁ .≥ θ₀ .* (1 - 1e-12)) && any(θ₁ .> θ₀ .* 1.01)
    @test wθ₁ == wθ₀
end

@testset "SkewedScalarDoubleGaussianPDF: the mixture has the moments it is given and the skewness the third moment implies" begin
    model = tke_p3_column(DoubleGaussian())
    thermodynamics = distribution_thermodynamics(model)
    components = Breeze.TurbulenceClosures.SimplifiedHigherOrderClosures.skewed_scalar_components
    p = 9.0e4; θ̄ = 293.0; q̄ = 11.0e-3; w′² = 0.4
    θˡⁱ′² = 0.04; qᵗ′² = 4e-7; θˡⁱ′qᵗ′ = -0.5 * sqrt(θˡⁱ′² * qᵗ′²); w′θˡⁱ′ = -0.02 * sqrt(w′² * θˡⁱ′²); w′qᵗ′ = 0.45 * sqrt(w′² * qᵗ′²)
    for pdf in (SkewedScalarDoubleGaussianPDF(), SkewedScalarDoubleGaussianPDF(; ℂᵍᵏ = 2, ℂᵍᶜ = 0.3)), w′³ in (0.0, 0.3, -0.2)
        m = components(pdf, thermodynamics, p, θ̄, q̄, w′², w′³, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
        a = m.a
        @test 0 < a < 1
        # the means and the covariance of θˡⁱ and qᵗ
        @test a * m.θ₁ + (1 - a) * m.θ₂ ≈ θ̄ rtol=1e-14
        @test a * m.q₁ + (1 - a) * m.q₂ ≈ q̄ rtol=1e-14
        @test m.θˡⁱ′²ᵢ + a * (1 - a) * (m.θ₁ - m.θ₂)^2 ≈ θˡⁱ′² rtol=1e-10
        @test m.qᵗ′²ᵢ + a * (1 - a) * (m.q₁ - m.q₂)^2 ≈ qᵗ′² rtol=1e-10
        @test m.θˡⁱ′qᵗ′ᵢ + a * (1 - a) * (m.θ₁ - m.θ₂) * (m.q₁ - m.q₂) ≈ θˡⁱ′qᵗ′ rtol=1e-10 atol=1e-20
        @test m.θˡⁱ′²ᵢ * m.qᵗ′²ᵢ ≥ m.θˡⁱ′qᵗ′ᵢ^2 # positive semidefinite within the components
        # the skewness of the saturation excess s = c_q q - c_θ θ of the mixture is the one the third moment implies,
        # ℂᵍᵏ Sk_w r̂³ / (1 - ℂᵍ₀)³, with the within-component share ℂᵍᶜ of its variance
        c_θ, c_q = m.c_θ, m.c_q
        s₁ = c_q * (m.q₁ - q̄) - c_θ * (m.θ₁ - θ̄); s₂ = c_q * (m.q₂ - q̄) - c_θ * (m.θ₂ - θ̄)
        σₛ² = c_θ^2 * θˡⁱ′² + c_q^2 * qᵗ′² - 2c_θ * c_q * θˡⁱ′qᵗ′
        @test a * (1 - a) * (s₁ - s₂)^2 ≈ (1 - pdf.ℂᵍᶜ) * σₛ² rtol=1e-10
        r = (c_q * w′qᵗ′ - c_θ * w′θˡⁱ′) / sqrt(w′² * σₛ²); r̂ = r / (1 + r^4)^(1/4)
        Sk_s = (a * s₁^3 + (1 - a) * s₂^3) / σₛ²^1.5
        @test Sk_s ≈ pdf.ℂᵍᵏ * w′³ / w′²^1.5 * r̂^3 / (1 - pdf.ℂᵍ₀)^3 rtol=1e-9 atol=1e-14
        # positive skewness of w and an upward flux of the excess: the moist component is the narrow tail
        w′³ > 0 && @test a < 1 / 2 && s₁ > 0
        w′³ == 0 && @test a == 1 / 2
    end

    # without skewness, as the within-component share tends to one, the components tend to the mean (the single Gaussian);
    # with skewness the share between the components carries it, and ℂᵍᶜ = 1 is excluded
    pdf = SkewedScalarDoubleGaussianPDF(; ℂᵍᶜ = 1 - 1e-8)
    m = components(pdf, thermodynamics, p, θ̄, q̄, w′², 0.0, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
    @test isapprox(m.θ₁, θ̄; atol = 1e-3) && isapprox(m.θ₂, θ̄; atol = 1e-3)
    @test_throws ArgumentError SkewedScalarDoubleGaussianPDF(; ℂᵍᶜ = 1)
    # a skewed mixture of a subsaturated mean has more cloud than the unskewed one
    q̄ˢ = 7.0e-3
    moments = (0.3, θˡⁱ′², w′θˡⁱ′, θˡⁱ′qᵗ′, w′qᵗ′, qᵗ′²)
    C₀, qˡ₀ = assumed_pdf(SkewedScalarDoubleGaussianPDF(), thermodynamics, 1.0, 0.0, 0.5, p, θ̄, 0.0, q̄ˢ, w′², (0.0, moments[2:end]...), (0.0, moments[2:end]...))
    Cᵏ, qˡᵏ = assumed_pdf(SkewedScalarDoubleGaussianPDF(), thermodynamics, 1.0, 0.0, 0.5, p, θ̄, 0.0, q̄ˢ, w′², moments, moments)
    @test Cᵏ > C₀ > 0 && qˡᵏ > qˡ₀ > 0

    # in a column: the distribution's cloud is the assumed PDF of the moments at the centers, finite and positive
    for distribution in (DoubleGaussian(; assumed_pdf = SkewedScalarDoubleGaussianPDF()),
                         DoubleGaussian(; assumed_pdf = SkewedScalarDoubleGaussianPDF(), third_moment = TKEBudgetThirdMoment(location = Face())))
        model = tke_p3_column(distribution; static_stability = fraction_stability)
        update_state!(model)
        cl = model.closure_fields.subgrid_cloud.cloud
        @test all(isfinite, column(cl.aᶜ)) && all(isfinite, column(cl.qˡ′²)) && all(isfinite, column(cl.w′qˡ′))
        @test all(x -> 0 ≤ x ≤ 1, column(cl.aᶜ)) && maximum(column(cl.aᶜ)) > 0
        for n in 1:3
            time_step!(model, 10)
        end
        @test all(isfinite, column(model.tracers.ρe))
    end
end

@testset "LiquidFluxVarianceRelation: the latent part of the buoyancy production from w′qˡ′ = ℂʳˡ √(w′² qˡ′²)" begin
    for static_stability in (MoistStaticStability(), fraction_stability)
        relation = LiquidFluxVarianceRelation(; ℂʳˡ = 0.62)
        model = tke_p3_column(DoubleGaussian(; liquid_flux = relation); static_stability)
        update_state!(model)
        cl = model.closure_fields.subgrid_cloud.cloud
        Nz = size(model.grid, 3)
        ρ = column(model.dynamics.reference_state.density)
        w′² = column(cl.w′²); qˡ′² = column(cl.qˡ′²)
        bᶜᵛ = column(cl.b_qᶜ) .- column(cl.b_qᵛ)
        for k in 2:Nz
            expected = (bᶜᵛ[k] + bᶜᵛ[k-1]) / 2 * 0.62 * sqrt((w′²[k] + w′²[k-1]) / 2 * (qˡ′²[k] + qˡ′²[k-1]) / 2)
            @test column(cl.Bˡᶠ)[k] ≈ expected rtol=1e-12 atol=1e-30
        end
        @test column(cl.Bˡᶠ)[1] == 0 && column(cl.Bˡᶠ)[Nz+1] == 0
        @test maximum(column(cl.Bˡᶠ)) > 0
        # with liquid in the layer the latent part is positive where the relation's flux dominates
        @test maximum(column(cl.Bᴸ)) > 0
        @test with_parameters(model.closure; ℂʳˡ = 0.3).subgrid_distribution.liquid_flux == LiquidFluxVarianceRelation(; ℂʳˡ = 0.3)
    end
    @test_throws ArgumentError LiquidFluxVarianceRelation(; ℂʳˡ = 1.5)
end
