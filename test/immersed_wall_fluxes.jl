include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.TimeSteppers: compute_flux_bc_tendencies!
using Test

# Uniform flow over an immersed floor that fills the two lowest of eight cells
function wall_flux_tendencies(boundary_conditions)
    underlying = RectilinearGrid(size=(4, 4, 8), x=(0, 1), y=(0, 1), z=(0, 2),
                                 topology=(Periodic, Periodic, Bounded))
    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(0.5))
    model = AtmosphereModel(grid; boundary_conditions)
    set!(model; θ=300, u=5)
    compute_flux_bc_tendencies!(model)
    return model, model.timestepper.Gⁿ
end

@testset "Immersed wall fluxes on a flat floor" begin
    z₀, θʷ = 0.01, 310
    boundary_conditions = (ρu = FieldBoundaryConditions(immersed=ImmersedWallDrag(roughness_length=z₀)),
                           ρθ = FieldBoundaryConditions(immersed=ImmersedWallHeatFlux(wall_temperature=θʷ, roughness_length=z₀)))
    model, G = wall_flux_tendencies(boundary_conditions)
    _, G₀ = wall_flux_tendencies(NamedTuple())

    ΔGρu = interior(G.ρu) .- interior(G₀.ρu)
    ΔGρθ = interior(G.ρθ) .- interior(G₀.ρθ)

    κ, Δz = 0.4, 0.25
    d = Δz / 2
    Cᴰ = (κ / log(d / z₀))^2
    Cᴴ = κ^2 / (log(d / z₀) * log(d / (z₀ / 10)))
    ρu = interior(model.momentum.ρu)[1, 1, 3]
    ρ = interior(model.dynamics.reference_state.density)[1, 1, 3]

    # Drag decelerates and a warm wall heats the first fluid cell, and nothing else
    @test all(ΔGρu[:, :, 3] .≈ - Cᴰ * 5 * ρu / Δz)
    @test all(ΔGρθ[:, :, 3] .≈ ρ * Cᴴ * 5 * (θʷ - 300) / Δz)
    @test all(ΔGρu[:, :, 4:end] .== 0)
    @test all(ΔGρθ[:, :, 4:end] .== 0)

    time_step!(model, 0.01)
    @test all(isfinite, interior(model.momentum.ρu))
    @test maximum(interior(model.velocities.u)[:, :, 3]) < 5
end
