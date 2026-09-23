include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant — terrain-following LatitudeLongitudeGrid
#####
#
# Oceananigans' Reactant extension moves a LatitudeLongitudeGrid to ReactantState field by
# field; BreezeReactantExt teaches it the terrain-following vertical coordinate and both
# terrain formulations. Check the transfer for each formulation, then that a compressible
# model on the LinearDecay grid compiles and runs one forward time step.

using Breeze
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.Architectures: CPU, ReactantState, architecture, on_architecture
using Oceananigans.Fields: interior
using Reactant
using Test
using CUDA

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

mountain(λ, φ) = 200 * exp(-((λ - 180)^2 + φ^2) / 20^2)

# Build on the CPU and materialize the terrain there, then move to ReactantState: the route
# `LatitudeLongitudeGrid(::ReactantState, ...)` itself takes, with the terrain filled in between.
function terrain_grid(formulation; N = (8, 6, 4), Lz = 10_000.0)
    z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = N[3] + 1)); formulation)
    cpu_grid = LatitudeLongitudeGrid(CPU(); size = N, halo = (5, 5, 5),
                                     longitude = (0, 360), latitude = (-60, 60), z = z_faces,
                                     topology = (Periodic, Bounded, Bounded))
    materialize_terrain!(cpu_grid, mountain)
    return on_architecture(ReactantState(), cpu_grid)
end

on_reactant(a) = architecture(a) isa ReactantState

@testset "Reactant — terrain-following LatitudeLongitudeGrid" begin
    @testset "LinearDecay transfer" begin
        grid = terrain_grid(LinearDecay())
        @test architecture(grid) isa ReactantState
        @test grid.z isa TerrainFollowingVerticalDiscretization
        @test grid.z.formulation isa LinearDecay
        @test on_reactant(grid.z.cᵃᵃᶠ)
        @test on_reactant(grid.z.formulation.h)
        @test on_reactant(grid.z.formulation.∂x_h)
        @test on_reactant(grid.z.formulation.∂y_h)
        @test grid.z.formulation.z_top isa Number
    end

    @testset "TwoLevelDecay transfer" begin
        grid = terrain_grid(TwoLevelDecay(large_scale_height = 8000, small_scale_height = 2000))
        @test grid.z.formulation isa TwoLevelDecay
        @test on_reactant(grid.z.formulation.h₁)
        @test on_reactant(grid.z.formulation.h₂)
        @test on_reactant(grid.z.formulation.∂x_h₁)
        @test on_reactant(grid.z.formulation.∂y_h₂)
        @test on_reactant(grid.z.formulation.basis.b₁ᶜ)
        @test on_reactant(grid.z.formulation.basis.∂b₂ᶠ)
    end

    @testset "CompressibleDynamics forward step" begin
        grid = terrain_grid(LinearDecay())
        model = AtmosphereModel(grid; dynamics = CompressibleDynamics(), advection = Centered(order = 2))
        @test model.dynamics.terrain_metrics isa TerrainMetrics

        set!(model; ρ = 1.0, ρθ = 300.0)
        Δt = 2.0
        step! = Reactant.@compile raise = true raise_first = true sync = true time_step!(model, Δt)
        step!(model, Δt)

        ρθ = Array(interior(prognostic_fields(model).ρθ))
        @test all(isfinite, ρθ)
    end
end
