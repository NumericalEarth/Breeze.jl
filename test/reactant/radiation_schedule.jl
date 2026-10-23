include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant: scheduled radiation inside traced time stepping, and Kessler microphysics
#####
#
# `update_radiation!` is a traced conditional under Reactant (BreezeReactantExt), which needs a
# stateless `IterationInterval`; `validate_schedule` rejects other schedules when the model is
# built. A stub radiation model, whose update depends on the temperature and counts itself, checks
# that a traced window updates radiation as often and with the same result as an eager one, with
# Kessler cloud and rain (sedimentation substeps) and a traced first-step preparation.

using Breeze
using Breeze.AtmosphereModels: validate_schedule, _update_radiation!
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula
using Oceananigans
using Oceananigans.Architectures: ReactantState, architecture, on_architecture
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: launch!
using Oceananigans.Utils.KernelAbstractions: @kernel, @index
using Reactant
using Reactant: @trace
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

struct CountingRadiation{S, F, C}
    schedule :: S
    flux_divergence :: F
    updates :: C
end

function Breeze.AtmosphereModels._update_radiation!(radiation::CountingRadiation, model)
    launch!(architecture(model.grid), model.grid, :xyz, _temperature_heating!,
            radiation.flux_divergence, model.temperature)
    radiation.updates .+= 1
    return nothing
end

@kernel function _temperature_heating!(F, T)
    i, j, k = @index(Global, NTuple)
    @inbounds F[i, j, k] = 1e-2 * (T[i, j, k] - 285)
end

function scheduled_radiation_column(arch, schedule)
    grid = RectilinearGrid(arch; size=16, z=(0, 2400), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    radiation = CountingRadiation(schedule, CenterField(grid), on_architecture(arch, zeros(1)))
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants,
                            microphysics=DCMIP2016KesslerMicrophysics(maximum_sedimentation_substeps=4),
                            advection=nothing, radiation)
    set!(model; θ = z -> 299 + 3e-3 * z, qᵛ = z -> 0.016 * exp(-z / 2000),
                qᶜˡ = z -> 6e-4 * exp(-((z - 1200) / 200)^2), qʳ = z -> 2e-3 * exp(-((z - 900) / 300)^2))
    return model
end

# Reactant skips the eager first-step preparation; trace it, with the initial fluxes
function scheduled_window!(model, Δt, steps)
    model.clock.last_Δt = Δt
    update_state!(model)
    _update_radiation!(model.radiation, model)
    update_state!(model)
    @trace mincut=true checkpointing=false track_numbers=false for _ in 1:steps
        time_step!(model, Δt)
    end
    return nothing
end

radiation_updates(model) = Int(Array(model.radiation.updates)[1])
column_state(model) = map(f -> Array(interior(f)), (model.temperature, model.microphysical_fields.qʳ,
                                                    model.radiation.flux_divergence))

@testset "Radiation schedules under Reactant" begin
    @testset "validate_schedule" begin
        @test validate_schedule(TimeInterval(600), CPU()) isa TimeInterval
        @test validate_schedule(IterationInterval(80), ReactantState()) isa IterationInterval
        @test_throws ArgumentError validate_schedule(TimeInterval(600), ReactantState())
        # Model construction fails before anything is traced
        @test_throws ArgumentError scheduled_radiation_column(ReactantState(), TimeInterval(600))
    end

    @testset "Traced window matches eager (Δt = $Δt)" for (Δt, steps) in ((7.5, 12), (30.0, 3))
        schedule = IterationInterval(4)
        eager = scheduled_radiation_column(CPU(), schedule)
        before = radiation_updates(eager)
        for _ in 1:steps
            time_step!(eager, Δt)
        end
        eager_updates = radiation_updates(eager) - before
        @test eager_updates == 1 + count(n -> n % 4 == 0, 1:steps-1) # initialization, then once per scheduled iteration

        model = scheduled_radiation_column(ReactantState(), schedule)
        before = radiation_updates(model)
        Reactant.@jit scheduled_window!(model, Δt, steps)
        @test radiation_updates(model) - before == eager_updates
        for (traced, expected) in zip(column_state(model), column_state(eager))
            @test traced ≈ expected rtol=1e-9
        end
    end
end
