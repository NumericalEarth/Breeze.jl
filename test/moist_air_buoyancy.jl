using Breeze
using Oceananigans
using Test

@testset "NonhydrostaticModel with MoistAirBuoyancy" begin
    grid = RectilinearGrid(size=(8, 8, 8), x=(0, 400), y=(0, 400), z=(0, 400))
    buoyancy = MoistAirBuoyancy(grid; reference_potential_temperature=300)
    model = NonhydrostaticModel(; grid, buoyancy, tracers = (:θ, :qᵗ))

    θ₀ = reference_constants.potential_temperature
    Δθ = 2
    Lz = grid.Lz

    θᵢ(x, y, z) = θ₀ + Δθ * z / Lz
    set!(model; θ = θᵢ, q = 0)

    # Can time-step
    success = try
        time_step!(model, 1e-2)
        true
    catch
        false
    end

    @test success
end
