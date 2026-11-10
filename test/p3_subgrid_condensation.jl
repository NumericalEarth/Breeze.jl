include(joinpath(@__DIR__, "setup.jl"))

#####
##### Subgrid (PDF) condensation in P3: closed forms, the Dirac identity, and a long-step column
#####

using Breeze
using Breeze.AtmosphereModels: DiracStatistics, GaussianStatistics, DoubleGaussianStatistics,
                               GaussianSubgridCorrelation, NoSubgridCorrelation
using Breeze.Microphysics: gaussian_cloud_fraction, gaussian_equilibrium_condensate, subgrid_cloud,
                           normal_cumulative_distribution

using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using GPUArraysCore: @allowscalar
using Test
using Enzyme: Enzyme

# Midpoint quadrature of ∫ f(s) N(s̄, σ²) ds over ±10σ.
function gaussian_quadrature(f, s̄, σ; N = 200_000)
    Δ = 20σ / N
    total = 0.0
    for n in 1:N
        s = s̄ - 10σ + (n - 1/2) * Δ
        total += f(s) * exp(-(s - s̄)^2 / 2σ^2) / sqrt(2π * σ^2) * Δ
    end
    return total
end

@testset "Gaussian cloud relations" begin
    σ = 4e-4
    for s̄ in (-1e-3, -2e-4, 0.0, 3e-4, 1e-3)
        C = gaussian_cloud_fraction(s̄, σ)
        q⁺ = gaussian_equilibrium_condensate(s̄, σ)
        @test C ≈ gaussian_quadrature(s -> s > 0, s̄, σ) rtol=1e-5 atol=1e-9
        @test q⁺ ≈ gaussian_quadrature(s -> max(0, s), s̄, σ) rtol=1e-5 atol=1e-12
    end

    # σ → 0 is all-or-nothing condensation
    for s̄ in (-1e-3, -1e-5, 1e-5, 1e-3)
        @test gaussian_cloud_fraction(s̄, 1e-12) == (s̄ > 0)
        @test gaussian_equilibrium_condensate(s̄, 1e-12) ≈ max(0, s̄) atol=1e-15
    end

    @test normal_cumulative_distribution(1.0) ≈ 0.8413447460685429 rtol=1e-6

    # A single-component mixture is the single Gaussian; mixture weights sum the components.
    a, b = 0.4, 1e-3
    𝒢 = GaussianStatistics(1e-8, 0.01, -5e-6)
    single = subgrid_cloud(𝒢, -2e-4, a, b, 1e-4)
    mixture = subgrid_cloud(DoubleGaussianStatistics(0.0, 0.0, 0.0, 𝒢, 𝒢), -2e-4, a, b, 1e-4)
    @test mixture.C ≈ single.C
    @test mixture.q⁺ ≈ single.q⁺

    # Negative qᵗ-θˡⁱ correlation widens σₛ and raises the subsaturated cloud fraction.
    uncorrelated = subgrid_cloud(GaussianStatistics(1e-8, 0.01, 0.0), -2e-4, a, b, 1e-5)
    @test single.C > uncorrelated.C
end

column_grid(arch = default_arch) = RectilinearGrid(arch; size=32, z=(0, 1600), topology=(Flat, Flat, Bounded))

# Long steps need the sedimentation and closure transport inside the adaptive vertically implicit solve
implicit_transport() = WENO(; order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(Float64; cfl=0.5))

function p3_column_model(subgrid; grid = column_grid(), sink_limiting_timescale = 10,
                         closure = TKEBasedTurbulenceClosure(), scalar_advection = nothing)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=290)
    process_rates = Breeze.Microphysics.PredictedParticleProperties.ProcessRate(Float64; sink_limiting_timescale)
    microphysics = PredictedParticlePropertiesMicrophysics(; subgrid, process_rates)
    advection = isnothing(scalar_advection) ? (;) : (; scalar_advection)
    return AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics, closure, advection...)
end

# A well-mixed layer under an inversion at 800 m, with a near-saturated, slightly subsaturated
# cloud layer at its top.
θ₀(z) = ifelse(z < 800, 290.0, 290.0 + 8 * (1 - exp(-(z - 800) / 100)))
qᵗ₀(z) = ifelse(z < 800, 9.0e-3, 4.0e-3)
qᶜˡ₀(z) = ifelse(600 < z < 800, 1.0e-4, 0.0)

function initialize!(model)
    tke = isnothing(model.closure) ? (;) : (; ρe=z -> 1.2 * 1e-2)
    set!(model; θ=θ₀, qᵗ=qᵗ₀, ρqᶜˡ=z -> 1.2 * qᶜˡ₀(z), tke...)
    return model
end

@testset "Dirac statistics are the grid-mean P3 rates" begin
    model = initialize!(p3_column_model(NoSubgridCorrelation()))
    time_step!(model, 1)
    reference = initialize!(p3_column_model(NoSubgridCorrelation()))
    time_step!(reference, 1)
    for name in (:ρqᵛ, :ρqᶜˡ, :ρqʳ, :ρθ)
        a = @allowscalar Array(interior(Oceananigans.fields(model)[name]))
        b = @allowscalar Array(interior(Oceananigans.fields(reference)[name]))
        @test a == b
    end
end

@testset "Gaussian P3 column at Δt = 900 s" begin
    Δt = 900
    condensation = ExponentialRateCondensateFormation(1 / 120, Δt)
    subgrid = GaussianSubgridCorrelation(Float64; condensation, minimum_saturation_deficit_spread=2.5e-4)
    # P3's sink limiters act over the model step; σₛ from the floor (no closure), so the inversion stays put.
    # Δt is not reduced: sedimentation is in the adaptive vertically implicit transport.
    model = initialize!(p3_column_model(subgrid; sink_limiting_timescale=Δt, closure=nothing,
                                        scalar_advection=implicit_transport()))
    grid_mean = initialize!(p3_column_model(NoSubgridCorrelation(); sink_limiting_timescale=Δt, closure=nothing,
                                            scalar_advection=implicit_transport()))

    for n in 1:24 # six hours
        time_step!(model, Δt)
        time_step!(grid_mean, Δt)
    end

    μ = model.microphysical_fields
    for name in (:ρqᶜˡ, :ρqʳ)
        values = @allowscalar Array(interior(μ[name]))
        @test all(isfinite, values)
        @test minimum(values) ≥ -1e-12
    end
    ρqᵛ = @allowscalar Array(interior(model.moisture_density))
    @test all(isfinite, ρqᵛ) && minimum(ρqᵛ) > 0

    # The near-saturated cloud layer keeps more liquid under the Gaussian distribution than at the grid mean.
    gaussian_liquid = sum(@allowscalar Array(interior(μ.ρqᶜˡ)))
    grid_mean_liquid = sum(@allowscalar Array(interior(grid_mean.microphysical_fields.ρqᶜˡ)))
    @info "column liquid after 6 h at Δt = 900 s" gaussian_liquid grid_mean_liquid
    @test gaussian_liquid > grid_mean_liquid
    @test maximum(@allowscalar Array(interior(μ.ρqᶜˡ))) > 1e-5
end

@testset "TKE closure supplies the subgrid variance" begin
    Δt = 900
    condensation = ExponentialRateCondensateFormation(1 / 120, Δt)
    subgrid = GaussianSubgridCorrelation(Float64; condensation)
    model = initialize!(p3_column_model(subgrid; sink_limiting_timescale=Δt, scalar_advection=implicit_transport()))
    time_step!(model, Δt)

    # Statistics at the inversion are non-zero and come from the TKE mixing length and Δz terms
    𝒮 = @allowscalar Breeze.AtmosphereModels.subgrid_statisticsᶜᶜᶜ(1, 1, 16, model.grid, subgrid, model.closure,
                                                                  model.closure_fields,
                                                                  Breeze.Microphysics.PredictedParticleProperties.p3_subgrid_state(subgrid, model))
    @test 𝒮 isa GaussianStatistics
    @test 𝒮.σ²ᵗ > 0 && 𝒮.σ²ᶿ > 0
    @test 𝒮.cᵗᶿ < 0 # qᵗ falls and θ rises across the inversion

    for _ in 1:3
        time_step!(model, Δt)
    end
    for name in (:ρqᶜˡ, :ρqʳ)
        values = @allowscalar Array(interior(model.microphysical_fields[name]))
        @test all(isfinite, values) && minimum(values) ≥ -1e-12
    end
end

@testset "Cloud fraction diagnostic C = Φ(Q₁)" begin
    PPP = Breeze.Microphysics.PredictedParticleProperties
    Δt = 900
    condensation = ExponentialRateCondensateFormation(1 / 120, Δt)
    subgrid = GaussianSubgridCorrelation(Float64; condensation)
    model = initialize!(p3_column_model(subgrid; sink_limiting_timescale=Δt, scalar_advection=implicit_transport()))
    time_step!(model, Δt)

    operation = cloud_fraction(model)
    C = Array(interior(compute!(Field(operation))))[:]
    @test all(isfinite, C)
    @test all(0 .≤ C .≤ 1)
    @test 0 < maximum(C) && minimum(C) < 1

    # Recompute Q₁ = s̄ / σₛ from the state and the closure statistics, cell by cell
    kernel = operation.kernel_function
    Nz = size(model.grid, 3)
    for k in 1:Nz
        cell = @allowscalar PPP.p3_cell_saturation_deficit(1, 1, k, model.grid, kernel)
        σₛ = Breeze.Microphysics.saturation_deficit_spread(cell.𝒮, cell.a, cell.b, cell.σ₀)
        @test σₛ > 0
        @test C[k] ≈ normal_cumulative_distribution(cell.s̄ / σₛ)
        @test C[k] == PPP.subgrid_cloud(cell.𝒮, cell.s̄, cell.a, cell.b, cell.σ₀).C
    end

    # σₛ = 0: saturated cells are cloudy, unsaturated cells clear
    @test PPP.subgrid_cloud_fraction(GaussianStatistics(0.0, 0.0, 0.0), 1e-4, 1.0, 1.0, 0.0) == 1
    @test PPP.subgrid_cloud_fraction(GaussianStatistics(0.0, 0.0, 0.0), -1e-4, 1.0, 1.0, 0.0) == 0
    @test PPP.subgrid_cloud_fraction(GaussianStatistics(0.0, 0.0, 0.0), 0.0, 1.0, 1.0, 0.0) == 0

    # Without a subgrid distribution C is the 0/1 saturation indicator
    grid_mean = initialize!(p3_column_model(NoSubgridCorrelation(); closure=nothing))
    time_step!(grid_mean, 1)
    C₀ = Array(interior(compute!(Field(cloud_fraction(grid_mean)))))[:]
    @test all(c -> c == 0 || c == 1, C₀)
end

@testset "Subgrid statistics take the float type of their values" begin
    # Mixed argument types promote, so a merge never sees two float types
    @test GaussianStatistics(1f-8, 0.01, -1f-5) isa GaussianStatistics{Float64}
    updraft = GaussianStatistics{Float32}()
    environment = GaussianStatistics(1e-8, 0.01, -1e-5)
    𝒮 = DoubleGaussianStatistics(0.1f0, 1f-3, -0.5f0, updraft, environment)
    @test 𝒮 isa DoubleGaussianStatistics{Float64}
    @test 𝒮.environment === environment
    merged = Breeze.AtmosphereModels.merge_statistics(GaussianStatistics(1f-8, 1f-2, 0f0), 𝒮)
    @test merged isa DoubleGaussianStatistics{Float64}
    @test merged.environment.σ²ᵗ ≈ 2e-8

    # The fallback statistics read the float type from the state, not the grid
    grid = RectilinearGrid(CPU(); size=4, z=(0, 1), topology=(Flat, Flat, Bounded))
    subgrid = GaussianSubgridCorrelation(Float64)
    state = (; θ = fill(290f0, 1, 1, 4), qᵗ = fill(1f-2, 1, 1, 4))
    AM = Breeze.AtmosphereModels
    @test AM.subgrid_statisticsᶜᶜᶜ(1, 1, 2, grid, subgrid, nothing, nothing, state) === GaussianStatistics{Float32}()
    @test AM.subgrid_statisticsᶜᶜᶜ(1, 1, 2, grid, subgrid, (), (), state) === GaussianStatistics{Float32}()
    @test AM.subgrid_statisticsᶜᶜᶜ(1, 1, 2, grid, AM.NoSubgridCorrelation(), nothing, nothing, state) === DiracStatistics{Float32}()
    @test AM.subgrid_statisticsᶜᶜᶜ(1, 1, 2, grid, AM.NoSubgridCorrelation(), nothing, nothing, nothing) === DiracStatistics{Float64}()
end

@testset "Relative saturation-deficit spread floor σ₀ = max(σ₀, c₀ qᵛ⁺)" begin
    PPP = Breeze.Microphysics.PredictedParticleProperties
    default = GaussianSubgridCorrelation(Float64)
    relative = GaussianSubgridCorrelation(Float64; relative_minimum_saturation_deficit_spread=0.05)
    @test default.relative_minimum_saturation_deficit_spread === nothing
    @test relative.relative_minimum_saturation_deficit_spread === 0.05
    @test GaussianSubgridCorrelation(Float32; relative_minimum_saturation_deficit_spread=0.05).relative_minimum_saturation_deficit_spread === 0.05f0

    # Off by default: the dimensional floor, capped at qᵛ⁺ / 10
    @test PPP.saturation_deficit_spread_floor(default, 0.015) === 1e-4
    @test PPP.saturation_deficit_spread_floor(default, 5e-4) === 5e-5
    # On: the larger of the two floors
    @test PPP.saturation_deficit_spread_floor(relative, 0.015) ≈ 0.05 * 0.015
    @test PPP.saturation_deficit_spread_floor(relative, 1e-3) ≈ 1e-4
    @test PPP.saturation_deficit_spread_floor(NoSubgridCorrelation(), 0.015) === 0.0

    # At 92% relative humidity the relative floor yields cloud that the dimensional floor does not
    qᵛ⁺ = 0.015
    s̄ = -0.08qᵛ⁺
    𝒮 = GaussianStatistics{Float64}()
    C₀ = subgrid_cloud(𝒮, s̄, 1.0, 0.0, PPP.saturation_deficit_spread_floor(default, qᵛ⁺)).C
    C₁ = subgrid_cloud(𝒮, s̄, 1.0, 0.0, PPP.saturation_deficit_spread_floor(relative, qᵛ⁺)).C
    @test C₀ < 1e-10
    @test C₁ ≈ normal_cumulative_distribution(-0.08 / 0.05)
end

@testset "Zero spread (σ₀ = 0, zero variance): all-or-nothing limit, no NaN, finite gradients" begin
    𝒮 = GaussianStatistics{Float64}()
    for s̄ in (2e-4, -2e-4)
        cloud = subgrid_cloud(𝒮, s̄, 0.4, 1e-3, 0.0)
        @test isfinite(cloud.C) && isfinite(cloud.q⁺)
        @test cloud.C == (s̄ > 0 ? 1 : 0)
        @test cloud.q⁺ ≈ max(0, s̄) atol=1e-18
    end
    @test subgrid_cloud(𝒮, 0.0, 0.4, 1e-3, 0.0).C == 0
    cloud = subgrid_cloud(GaussianStatistics{Float32}(), 2f-4, 0.4f0, 1f-3, 0f0)
    @test cloud.C == 1 && cloud.q⁺ ≈ 2f-4

    # The guard never binds at the default floor
    @test Breeze.Microphysics.saturation_deficit_spread(𝒮, 0.4, 1e-3, 1e-4) === 1e-4

    # Derivatives with respect to the mean and the variance stay finite
    q(s̄, σ²ᵗ) = subgrid_cloud(GaussianStatistics(σ²ᵗ, 0.0, 0.0), s̄, 0.4, 1e-3, 0.0).q⁺
    ∂s̄ = only(Enzyme.autodiff(Enzyme.Forward, s̄ -> q(s̄, 0.0), Enzyme.Duplicated(2e-4, 1.0)))
    ∂σ² = only(Enzyme.autodiff(Enzyme.Forward, σ² -> q(2e-4, σ²), Enzyme.Duplicated(0.0, 1.0)))
    @test isfinite(∂s̄) && ∂s̄ ≈ 1
    @test isfinite(∂σ²)
end

#####
##### The condensation increment is differentiable in the stiff limit Δt·rate ≫ 1
#####

using Breeze.Microphysics.PredictedParticleProperties: compute_p3_process_rates, P3MicrophysicalState,
                                                        tendency_ρqᶜˡ, tendency_ρqᵛ, ProcessRate
using Breeze.Microphysics: phase_relaxation_rate
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface

# One P3 cell with a Gaussian subgrid distribution: the ρqᶜˡ and ρqᵛ tendencies as functions of the
# state (qᵛ, qᶜˡ, T, σ²ᵗ, σ²ᶿ) and the condensation rate.
function p3_subgrid_cell(rate, Δt; phase_relaxation_coefficient = nothing)
    condensation = ExponentialRateCondensateFormation(rate, Δt)
    subgrid = GaussianSubgridCorrelation(Float64; condensation, phase_relaxation_coefficient)
    process_rates = ProcessRate(Float64; sink_limiting_timescale=Δt)
    return PredictedParticlePropertiesMicrophysics(; subgrid, process_rates)
end

function condensation_increment(x, p3, constants)
    qᵛ, qᶜˡ, T, σ²ᵗ, σ²ᶿ = x
    ρ = 1.0
    q = MoistureMassFractions(qᵛ, qᶜˡ, 0.0)
    𝒰 = with_temperature(LiquidIcePotentialTemperatureState(0.0, q, 1e5, 85e3), T, constants)
    ℳ = P3MicrophysicalState(qᶜˡ, 2e8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    𝒮 = GaussianStatistics(σ²ᵗ, σ²ᶿ, -sqrt(σ²ᵗ * σ²ᶿ) / 2)
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing, T, 0.0, 0.0, 𝒮)
    return (tendency_ρqᶜˡ(rates, ρ), tendency_ρqᵛ(rates, ρ))
end

reverse_gradient(f, x) = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.Reverse), Enzyme.Const(f),
                                         Enzyme.Active, Enzyme.Active(x))[1][1]
central_difference(f, x, h) = (f(x * (1 + h)) - f(x * (1 - h))) / (2x * h)

@testset "Subgrid condensation increment: Enzyme reverse against finite differences" begin
    Δt = 900.0
    constants = ThermodynamicConstants(Float64)
    T = 280.0
    qᵛ⁺ = saturation_specific_humidity(T, 1.0, constants, PlanarLiquidSurface())
    for α in (nothing, 1.0), x_rate in (0.1, 1.0, 10.0, 90.0, 1000.0)
        p3 = p3_subgrid_cell(x_rate / Δt, Δt; phase_relaxation_coefficient=α)
        for (qᵛ, qᶜˡ) in ((qᵛ⁺ - 2e-4, 1e-4), (qᵛ⁺ + 1e-5, 1e-4), (qᵛ⁺ - 5e-4, 0.0))
            x₀ = (qᵛ, qᶜˡ, T, 1e-8, 0.02)
            for output in 1:2, j in 1:5
                x₀[j] == 0 && continue
                f = v -> condensation_increment(ntuple(k -> k == j ? v : x₀[k], 5), p3, constants)[output]
                ad = reverse_gradient(f, x₀[j])
                fd = central_difference(f, x₀[j], 1e-5)
                scale = max(abs(fd), 1e-12 * abs(f(x₀[j]) / x₀[j]))
                @test isfinite(ad)
                @test abs(ad - fd) ≤ 1e-3 * scale
            end
        end
    end
end

@testset "Phase-relaxation timescale τ = 1 / (α 4π Dᵛ N r̄)" begin
    Dᵛ, N, ρ, ρʷ, r_act = 2.5e-5, 2e8, 1.0, 1e3, 1e-6
    τ(qˡ) = 1 / phase_relaxation_rate(1.0, Dᵛ, N, qˡ, ρ, ρʷ, r_act)
    # The activation radius bounds τ as the liquid vanishes
    @test τ(0.0) ≈ 1 / (4π * Dᵛ * N * r_act)
    @test τ(-1e-5) == τ(0.0)
    @test τ(1e-10) == τ(0.0)
    # Typical cloudy cells: a few seconds
    @test 1 < τ(1e-3) < τ(1e-4) < τ(0.0) < 20
    @test τ(1e-3) ≈ 1 / (4π * Dᵛ * N * cbrt(3ρ * 1e-3 / (4π * ρʷ * N)))
    # α scales the rate
    @test phase_relaxation_rate(2.0, Dᵛ, N, 1e-4, ρ, ρʷ, r_act) ≈ 2 / τ(1e-4)
    # The derivative at qˡ = 0 is finite (the floor sits inside the cube root)
    ∂τ = only(Enzyme.autodiff(Enzyme.Forward, τ, Enzyme.Duplicated(0.0, 1.0)))
    @test ∂τ == 0

    # The default (no coefficient) keeps the constant timescale; α = 1 replaces it in the model
    default = GaussianSubgridCorrelation(Float64)
    @test default.phase_relaxation_coefficient === nothing
    @test default.activation_radius == 1e-6
    opt_in = GaussianSubgridCorrelation(Float32; phase_relaxation_coefficient=0.5, activation_radius=2e-6)
    @test opt_in.phase_relaxation_coefficient === 0.5f0
    @test opt_in.activation_radius === 2f-6

    Δt = 60.0
    constants = ThermodynamicConstants(Float64)
    T = 280.0
    qᵛ⁺ = saturation_specific_humidity(T, 1.0, constants, PlanarLiquidSurface())
    x = (qᵛ⁺ + 1e-5, 1e-4, T, 1e-8, 0.02)
    constant = condensation_increment(x, p3_subgrid_cell(1 / 10, Δt), constants)
    physical = condensation_increment(x, p3_subgrid_cell(1 / 10, Δt; phase_relaxation_coefficient=1.0), constants)
    @test all(isfinite, physical)
    @test physical[1] != constant[1]
    @test physical[1] > 0
    @test physical[2] < 0 # vapor is consumed
end
