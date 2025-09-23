include(joinpath(@__DIR__, "runtests_setup.jl"))

@testset "NonhydrostaticModel with MoistAirBuoyancy" begin
    reference_state = ReferenceState(potential_temperature=300)
    buoyancy = MoistAirBuoyancy(; reference_state)

    grid = RectilinearGrid(size=(8, 8, 8), x=(0, 400), y=(0, 400), z=(0, 400))
    model = NonhydrostaticModel(; grid, buoyancy, tracers = (:θ, :q))

    θ₀ = reference_state.potential_temperature
    Δθ = 2
    Lz = grid.Lz

    θᵢ(x, y, z) = θ₀ + Δθ * z / Lz
    set!(model; θ = θᵢ, q = 0)

    success = try
        time_step!(model, 1e-2)
        true
    catch
        false
    end

    @test success
end
