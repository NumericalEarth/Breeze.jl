using Breeze
using Oceananigans
using Oceananigans: interior
using Test
using Base: Returns

function setup_forcing_model(FT)
    grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))
    model = AtmosphereModel(grid)
    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀)
    return model
end

increment_tolerance(::Type{Float32}) = 1f-5
increment_tolerance(::Type{Float64}) = 1e-10

@testset "AtmosphereModel forcing increments prognostic fields" begin
    for FT in (Float32, Float64)
        Δt = 3
        forcing = Returns(one(FT))

        @testset "Energy forcing ($FT)" begin
            model = setup_forcing_model(FT)
            model.forcing = merge(model.forcing, (; ρe=forcing))

            ρe_before = deepcopy(model.energy_density)
            time_step!(model, Δt)
            ρe_after = deepcopy(model.energy_density)

            expected = ρe_before + Δt
            diff = maximum(abs, ρe_after - expected)
            @test diff ≤ increment_tolerance(FT)
        end

        @testset "Moisture forcing ($FT)" begin
            model = setup_forcing_model(FT)
            model.forcing = merge(model.forcing, (; ρqᵗ=forcing))

            ρq_before = deepcopy(model.moisture_density)
            time_step!(model, Δt)
            ρq_after = deepcopy(model.moisture_density)

            expected = ρq_before + Δt
            diff = maximum(abs, ρq_after - expected)
            @test diff ≤ increment_tolerance(FT)
        end
    end
end
