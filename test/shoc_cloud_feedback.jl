include(joinpath(@__DIR__, "setup.jl"))

#####
##### SHOC with P3: P3 condenses its own cloud liquid, and SHOC mixes it with total water
#####

using Breeze
using Breeze.AtmosphereModels: DiracStatistics, NoSubgridCorrelation,
                               MicrophysicalProcessRate, subgrid_correlation, subgrid_statisticsᶜᶜᶜ

using Breeze.Microphysics.PredictedParticleProperties: GridMeanCondensation
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: step_closure_prognostics!
using Test

# Rain falls several cells per step
implicit_transport() = WENO(; order=5, time_discretization=AdaptiveVerticallyImplicitDiscretization(Float64; cfl=0.5))

function shoc_p3_column(condensation = GridMeanCondensation(); arch = default_arch, boundary_conditions = NamedTuple())
    grid = RectilinearGrid(arch; size=32, z=(0, 1600), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=290)
    # Without P3's negative-moisture correction, whose species borrowing in `update_state!` moves
    # traces of water between cloud liquid and the other species after the SHOC step
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation,
                                                           negative_moisture_correction = nothing,
                                                           precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), microphysics, boundary_conditions,
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

@testset "Subgrid statistics interface" begin
    @test subgrid_correlation(nothing) isa NoSubgridCorrelation
    grid = RectilinearGrid(CPU(); size=4, z=(0, 1), topology=(Flat, Flat, Bounded))
    @test subgrid_statisticsᶜᶜᶜ(1, 1, 1, grid, NoSubgridCorrelation(), nothing, nothing) isa DiracStatistics
end

@testset "With P3, SHOC mixes total water and the cloud liquid" begin
    Δt = 60
    cloud(z) = ifelse(600 < z < 800, 1.2e-4, 0.0)

    # The same total water, held partly as cloud liquid and all as vapor
    model = shoc_p3_column()
    μ = model.microphysical_fields
    set!(μ.ρqᶜˡ, cloud)
    set!(model.moisture_density, Array(interior(model.moisture_density)) .- Array(interior(μ.ρqᶜˡ)))
    update_state!(model)

    all_vapor = shoc_p3_column()
    set!(all_vapor.moisture_density, reshape(non_precipitating_water(model), 1, 1, :))
    update_state!(all_vapor)
    @test non_precipitating_water(all_vapor) ≈ non_precipitating_water(model)

    water = non_precipitating_water(model)
    cloud_liquid = column(μ.ρqᶜˡ)
    vapor = column(model.moisture_density)

    step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)
    step_closure_prognostics!(all_vapor.closure_fields, all_vapor.closure, all_vapor, Δt)

    # The cloud liquid is diffused, and each species' column water is conserved (no surface flux)
    @test column(μ.ρqᶜˡ) != cloud_liquid
    @test maximum(column(μ.ρqᶜˡ)) < maximum(cloud_liquid)
    @test sum(column(μ.ρqᶜˡ)) ≈ sum(cloud_liquid) rtol=1e-12
    @test sum(column(model.moisture_density)) ≈ sum(vapor) rtol=1e-12
    @test all(column(model.moisture_density) .>= 0)

    # Total water is mixed as SHOC mixes it when all of it is vapor
    @test non_precipitating_water(model) ≈ non_precipitating_water(all_vapor) rtol=1e-12
    @test sum(non_precipitating_water(model)) ≈ sum(water) rtol=1e-12
end
