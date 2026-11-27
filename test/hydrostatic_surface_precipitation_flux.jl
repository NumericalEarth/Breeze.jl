using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Breeze.HydrostaticEquations: surface_precipitation_flux
using Breeze.Microphysics: InstantaneousPrecipitation, MixedPhaseEquilibrium
using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization

#####
##### The surface precipitation flux of a time step is what that step added to the ledger
#####

function cold_precipitating_columns(; Nz = 16, top_pressure = 2.5e4, p₀ = 1e5)
    η = range(1, top_pressure / p₀, length = Nz + 1)
    B = [max(0, (e - 0.4) / 0.6) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0; A[end] = top_pressure
    grid = RectilinearGrid(CPU(), Float64; size=(4, Nz), x=(0, 4e5), y=0, z = PressureLevelVerticalDiscretization(A, B),
                           topology=(Periodic, Flat, Bounded))
    microphysics = InstantaneousPrecipitation(Float64; equilibrium = MixedPhaseEquilibrium(Float64))
    model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(), microphysics,
                            formulation = moist_formulation(), momentum_advection = VectorInvariant())
    # Columns from warm to cold at the ground, moist and supersaturated aloft, so that some rain and some snow
    θᵢ(x, z) = 258 + 12 * x / 4e5 + 3e-3 * z
    set!(model; pˢ = p₀, θ = θᵢ, qᵗ = (x, z) -> 0.01 * exp(-z / 2500), u = 10)
    return model
end

@testset "Surface precipitation flux of 0M mixed-phase hydrostatic steps" begin
    model = cold_precipitating_columns()
    rain = model.dynamics.moisture.ledger.surface_rain
    flux = surface_precipitation_flux(model.dynamics)
    Δt = 300.0
    steps = 6

    total_rain = 0.0
    total_snow = 0.0
    for _ in 1:steps
        before = Array(interior(rain))
        time_step!(model, Δt)
        increment = Array(interior(rain)) .- before
        total = Array(interior(flux.total))
        frozen = Array(interior(flux.frozen))

        @test all(isfinite, total)
        @test all(abs.(total .* Δt .- increment) .≤ 1e-12 .* max.(abs.(increment), eps()))
        @test all(0 .≤ frozen .≤ total)
        total_rain += sum(increment)
        total_snow += sum(frozen) * Δt
    end

    @info "Surface precipitation over $steps steps: $total_rain kg m⁻² of which frozen $total_snow kg m⁻² (summed over columns)"
    @test total_rain > 0
    @test total_snow > 0
end
