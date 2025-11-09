using Breeze
using Oceananigans
using Test

@testset "NonhydrostaticModel with MoistAirBuoyancy" begin
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 400), y=(0, 400), z=(0, 400))
    buoyancy = MoistAirBuoyancy(grid; reference_potential_temperature=300)
    model = NonhydrostaticModel(; grid, buoyancy, tracers = (:θ, :qᵗ))

    θ₀ = buoyancy.reference_state.potential_temperature
    Δθ = 2
    Lz = grid.Lz

    θᵢ(x, y, z) = θ₀ + Δθ * z / Lz
    set!(model; θ = θᵢ, qᵗ = 0)

    # Can time-step
    success = try
        time_step!(model, 1e-2)
        true
    catch
        false
    end

    @test success
end
