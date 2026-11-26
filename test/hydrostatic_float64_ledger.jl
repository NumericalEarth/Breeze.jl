using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Breeze.AtmosphereModels: dynamics_restart_state, restore_dynamics_restart_state!, precipitation_mass_change!
using Breeze.Microphysics: InstantaneousPrecipitation, MixedPhaseEquilibrium
using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization

#####
##### The accumulated totals of the precipitation ledger are Float64 on a Float32 grid
#####
#
# After years of integration a column holds thousands of kg m⁻² of rain, where the spacing of
# Float32 is comparable to the rain of one time step. The ledger keeps its totals in Float64,
# so that many small increments added to a large total sum to what a Float64 sum gives.

function float32_pressure_level_column(; Nz = 16, top_pressure = 2.5f4, p₀ = 1f5)
    η = range(1, top_pressure / p₀, length = Nz + 1)
    B = [max(0, (e - 0.4f0) / 0.6f0) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0; A[end] = top_pressure
    grid = RectilinearGrid(CPU(), Float32; size=Nz, x=0, y=0, z = PressureLevelVerticalDiscretization(A, B),
                           topology=(Flat, Flat, Bounded))
    microphysics = InstantaneousPrecipitation(Float32; equilibrium = MixedPhaseEquilibrium(Float32))
    model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(), microphysics,
                            formulation = moist_formulation(), momentum_advection = VectorInvariant())
    set!(model; pˢ = p₀, θ = z -> 262 + 3e-3 * z, qᵗ = z -> 0.008 * exp(-z / 2500), u = z -> 10 + 2e-3 * z)
    return model
end

@testset "Float64 precipitation ledger on a Float32 grid" begin
    model = float32_pressure_level_column()
    ledger = model.dynamics.moisture.ledger

    @test eltype(model.grid) == Float32
    for name in (:surface_rain, :precipitation_energy, :precipitation_kinetic_energy, :rain_frictional_heating)
        @test eltype(ledger[name]) == Float64
    end
    @test eltype(ledger.maximum_sedimentation_courant) == Float32

    # Light rain, 0.06 kg m⁻² per step, falling onto a column that already holds 6000 kg m⁻²
    total₀ = 6000
    set!(ledger.surface_rain, total₀)
    P, Δt = 1f-4, 600f0
    surface_precipitation = Field{Center, Center, Nothing}(model.grid)
    set!(surface_precipitation, P)
    g = model.thermodynamic_constants.gravitational_acceleration
    increment = g * Δt * P / g # as the kernel computes it, in Float32

    steps = 500
    for _ in 1:steps
        precipitation_mass_change!(model.dynamics, model, Δt, surface_precipitation)
    end

    exact = total₀ + steps * Float64(increment)
    float32_total = foldl((total, _) -> total + increment, 1:steps; init=Float32(total₀))
    float32_error = abs(float32_total - exact)
    float64_error = abs(ledger.surface_rain[1, 1, 1] - exact)
    @info "Float64 ledger: error $float64_error kg m⁻²; a Float32 total would err by $float32_error kg m⁻²"

    @test float64_error ≤ 1e-9 * steps * increment
    @test float32_error > 1e-4 * steps * increment # the case does exercise the Float32 rounding

    # A checkpoint written before the ledger was Float64 holds it in Float32: the restore converts it
    state = deepcopy(dynamics_restart_state(model.dynamics)) # on the CPU the state aliases the fields
    float32_ledger = map(state.ledger) do saved
        merge(saved, (; data = Float32.(saved.data)))
    end
    rain = ledger.surface_rain[1, 1, 1]
    saved_rain = Float32(rain)
    saved_energy = Float32(ledger.precipitation_energy[1, 1, 1])
    set!(ledger.surface_rain, 0)
    set!(ledger.precipitation_energy, 0)
    restore_dynamics_restart_state!(model.dynamics, merge(state, (; ledger = float32_ledger)))

    @test eltype(ledger.surface_rain) == Float64
    @test ledger.surface_rain[1, 1, 1] == saved_rain
    @test ledger.precipitation_energy[1, 1, 1] == saved_energy

    # And a Float64 checkpoint restores the totals exactly
    set!(ledger.surface_rain, 0)
    restore_dynamics_restart_state!(model.dynamics, state)
    @test ledger.surface_rain[1, 1, 1] == rain
end
