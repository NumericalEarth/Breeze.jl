include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant: ecCKD radiative transfer inside a traced window
#####
#
# A traced window of a clear-sky column with scheduled ecCKD radiation, compiled with loop raising
# (as gradient compilation is), must match the eager window.

using Breeze
using Breeze.TimeSteppers: prepare_first_time_step!
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Test
using CUDA

using ClimaComms
using NCDatasets
using NumericalRadiation: NumericalRadiation

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

function ecckd_column(arch)
    grid = RectilinearGrid(arch; size=8, z=(0, 4000), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
    radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                       surface_temperature = 300, surface_emissivity = 0.98,
                                       surface_albedo = 0.1, solar_position = FixedCosineZenith(0.5),
                                       schedule = IterationInterval(2))
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants,
                            advection=nothing, radiation)
    set!(model; θ = z -> 300 + 4e-3 * z, qᵛ = z -> 0.015 * exp(-z / 2500))
    return model
end

function ecckd_window!(model, Δt, steps)
    prepare_first_time_step!(model, Δt)
    @trace mincut=true checkpointing=false track_numbers=false for _ in 1:steps
        time_step!(model, Δt)
    end
    return nothing
end

column_state(model) = map(f -> Array(interior(f)), (model.temperature, model.radiation.flux_divergence,
                                                    model.radiation.upwelling_longwave_flux))

@testset "ecCKD radiation in a traced window" begin
    Δt, steps = 30.0, 5

    eager = ecckd_column(CPU())
    for _ in 1:steps
        time_step!(eager, Δt)
    end

    model = ecckd_column(ReactantState())
    Reactant.@jit raise=true raise_first=true ecckd_window!(model, Δt, steps)

    for (traced, expected) in zip(column_state(model), column_state(eager))
        @test all(isfinite, traced)
        @test traced ≈ expected rtol=1e-9
    end
end
