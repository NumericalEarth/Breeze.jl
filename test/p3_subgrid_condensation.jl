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
