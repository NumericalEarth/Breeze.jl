include(joinpath(@__DIR__, "setup.jl"))

#####
##### SHOC-owned cloud: `ClosureCloud` couples the subgrid cloud of
##### `SimplifiedHigherOrderClosure` to the model through P3
#####

using Breeze
using Breeze.AtmosphereModels: ClosureCloud, ClosureCloudStatistics, DiracStatistics, NoSubgridCorrelation,
                               MicrophysicalProcessRate, subgrid_correlation, subgrid_statisticsᶜᶜᶜ

using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: step_closure_prognostics!
using Test

# Rain falls several cells per step
implicit_transport() = WENO(; order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(Float64; cfl=0.5))

function shoc_p3_column(subgrid; arch = default_arch)
    grid = RectilinearGrid(arch; size=32, z=(0, 1600), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=290)
    # Without P3's negative-moisture correction, whose species borrowing in `update_state!` moves
    # traces of water between cloud liquid and the other species after the SHOC step
    microphysics = PredictedParticlePropertiesMicrophysics(; subgrid,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics,
                            closure=SimplifiedHigherOrderClosure(), scalar_advection=implicit_transport())

    # A well-mixed layer whose upper part is saturated in the mean, under an inversion at 800 m
    θ₀(z) = ifelse(z < 800, 290.0, 290.0 + 8 * (1 - exp(-(z - 800) / 100)))
    qᵗ₀(z) = ifelse(z < 800, 10.0e-3, 4.0e-3)
    set!(model; θ=θ₀, qᵗ=qᵗ₀, ρe=z -> 1.2 * 0.3)
    return model
end

column(field) = Array(interior(field))[1, 1, :]

function non_precipitating_water(model)
    μ = model.microphysical_fields
    return column(model.moisture_density) .+ column(μ.ρqᶜˡ)
end

@testset "ClosureCloud interface" begin
    @test ClosureCloud(Float64).autoconversion_enhancement == 1
    @test ClosureCloud(Float32; autoconversion_enhancement=3).autoconversion_enhancement === 3f0
    @test subgrid_correlation(nothing) isa NoSubgridCorrelation

    grid = RectilinearGrid(CPU(); size=4, z=(0, 1), topology=(Flat, Flat, Bounded))
    @test subgrid_statisticsᶜᶜᶜ(1, 1, 1, grid, NoSubgridCorrelation(), nothing, nothing) isa DiracStatistics
    no_cloud = subgrid_statisticsᶜᶜᶜ(1, 1, 1, grid, ClosureCloud(), nothing, nothing)
    @test no_cloud == ClosureCloudStatistics(0.0, 0.0)
end

@testset "SHOC owns the cloud of a P3 column" begin
    model = shoc_p3_column(ClosureCloud())
    @test subgrid_correlation(model.microphysics) isa ClosureCloud

    μ = model.microphysical_fields
    diagnostics = model.closure_fields.diagnostics
    ρ = column(model.dynamics.reference_state.density)
    ρθ = model.formulation.potential_temperature_density

    water₀ = sum(non_precipitating_water(model) .+ column(μ.ρqʳ))

    for _ in 1:5
        time_step!(model, 60)
    end

    ql = column(diagnostics.shoc_ql)
    qw = non_precipitating_water(model) ./ ρ
    @test maximum(ql) > 1e-5
    @test 0 < maximum(column(diagnostics.shoc_cldfrac)) ≤ 1

    # The cloud liquid of the model is SHOC's, the vapor is the rest
    @test column(μ.ρqᶜˡ) ≈ ρ .* min.(max.(ql, 0), max.(qw, 0)) rtol=1e-13
    @test all(column(model.moisture_density) .≥ 0)

    # P3 neither condenses nor evaporates cloud liquid, and forms rain from SHOC's cloud
    condensation = Field(MicrophysicalProcessRate(model, :condensation))
    autoconversion = Field(MicrophysicalProcessRate(model, :autoconversion))
    compute!(condensation)
    compute!(autoconversion)
    @test all(column(condensation) .== 0)
    @test maximum(column(autoconversion)) > 0
    @test all(column(autoconversion)[ql .== 0] .== 0)

    # Total water of a closed column
    water = sum(non_precipitating_water(model) .+ column(μ.ρqʳ))
    @test water ≈ water₀ rtol=1e-12
    @test all(isfinite, column(μ.ρqʳ))
    @test all(isfinite, column(model.temperature))

    # One SHOC step alone: it repartitions vapor and cloud liquid exactly, and conserves their
    # sum and ∫ρθ. (Over whole time steps ∫ρθ is not an invariant: sedimenting rain carries its
    # energy, not its ρθ, between pressure levels.)
    heat = sum(column(ρθ))
    water = sum(non_precipitating_water(model))
    step_closure_prognostics!(model.closure_fields, model.closure, model, 60)
    @test sum(column(ρθ)) ≈ heat rtol=1e-13
    @test sum(non_precipitating_water(model)) ≈ water rtol=1e-13
    ql = column(diagnostics.shoc_ql)
    qw = non_precipitating_water(model) ./ ρ
    @test column(μ.ρqᶜˡ) ≈ ρ .* min.(max.(ql, 0), max.(qw, 0)) rtol=1e-14
end

@testset "Without ClosureCloud SHOC leaves the cloud liquid alone" begin
    model = shoc_p3_column(NoSubgridCorrelation())
    μ = model.microphysical_fields
    set!(μ.ρqᶜˡ, z -> ifelse(600 < z < 800, 1.2e-4, 0.0))
    update_state!(model)
    cloud_liquid = column(μ.ρqᶜˡ)
    step_closure_prognostics!(model.closure_fields, model.closure, model, 60)
    @test column(μ.ρqᶜˡ) == cloud_liquid
end
