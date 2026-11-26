include(joinpath(@__DIR__, "setup.jl"))

#####
##### SHOC with P3: P3 condenses its own cloud liquid, which the closure diffuses with Kᶜ like every scalar, and
##### may condense over the closure's double Gaussian (TurbulenceClosureStatistics)
#####

using Breeze

using Breeze.Microphysics.PredictedParticleProperties: GridMeanCondensation, SubgridCondensation
using Breeze.AtmosphereModels: supersaturation_statistics, supersaturation_statistics_model_fields, saturated_fraction
using Breeze.TurbulenceClosures: TurbulenceClosureStatistics
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Advection: AdaptiveVerticallyImplicitDiscretization
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.TimeSteppers: update_state!
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

@testset "With P3, SHOC diffuses the cloud liquid in the stages" begin
    Δt = 30
    cloud(z) = ifelse(600 < z < 800, 1.2e-4, 0.0)
    model = shoc_p3_column()
    μ = model.microphysical_fields
    set!(μ.ρqᶜˡ, cloud)
    set!(model.moisture_density, Array(interior(model.moisture_density)) .- Array(interior(μ.ρqᶜˡ)))
    update_state!(model)
    cloud_liquid = column(μ.ρqᶜˡ)
    for _ in 1:5
        time_step!(model, Δt)
    end
    @test all(isfinite, column(model.moisture_density))
    @test all(isfinite, column(μ.ρqᶜˡ))
    @test all(column(model.moisture_density) .>= 0)
    @test all(column(μ.ρqᶜˡ) .>= 0)
    @test column(μ.ρqᶜˡ) != cloud_liquid
    @test all(≥(0), column(model.tracers.ρe))
end

@testset "TurbulenceClosureStatistics: the saturated fraction is SHOC's cloud fraction" begin
    model = shoc_p3_column(SubgridCondensation(TurbulenceClosureStatistics()))
    time_step!(model, 10)
    grid, fields = model.grid, model.closure_fields
    model_fields = supersaturation_statistics_model_fields(model)
    f⁺ = @allowscalar [saturated_fraction(supersaturation_statistics(TurbulenceClosureStatistics(), 1, 1, k, grid,
                                                                       model.closure, fields, model_fields)) for k in 1:size(grid, 3)]
    @test f⁺ == column(fields.subgrid_cloud.cloud.aᶜ)
    @test maximum(f⁺) > 0
end
