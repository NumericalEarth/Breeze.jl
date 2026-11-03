include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant: the static sedimentation substep bound of DCMIP2016 Kessler
#####
#
# Under Reactant the Kessler kernel loops `maximum_sedimentation_substeps` times (reverse mode cannot
# cache a data-dependent loop). Substeps beyond those a step needs must leave the state unchanged,
# so a traced step at or below the bound equals the eager step, water and energy included; a step
# that needs more must not be truncated: its update is NaN, and `sedimentation_substeps` records how
# many it needed.

using Breeze
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!
using Reactant
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

function rainy_column(arch, maximum_sedimentation_substeps)
    grid = RectilinearGrid(arch; size=16, z=(0, 2400), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
    reference_state = ReferenceState(grid, constants; base_pressure=101500, potential_temperature=299)
    microphysics = DCMIP2016KesslerMicrophysics(; maximum_sedimentation_substeps)
    model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), thermodynamic_constants=constants,
                            microphysics, advection=nothing)
    set!(model; θ = z -> 299 + 3e-3 * z, qᵛ = z -> 0.016 * exp(-z / 2000),
                qᶜˡ = z -> 6e-4 * exp(-((z - 1200) / 200)^2), qʳ = z -> 2e-3 * exp(-((z - 900) / 300)^2))
    return model
end

# One step with the eager first-step preparation, which Reactant skips
function prepared_step!(model, Δt)
    model.clock.last_Δt = Δt
    update_state!(model)
    time_step!(model, Δt)
    return nothing
end

needed_substeps(model) = Int(maximum(Array(interior(model.microphysical_fields.sedimentation_substeps))))

function column_budgets(model)
    ρ = Array(interior(model.dynamics.reference_state.density))
    μ = model.microphysical_fields
    water = sum(ρ .* (Array(interior(μ.qᵛ)) .+ Array(interior(μ.qᶜˡ)) .+ Array(interior(μ.qʳ))))
    potential_temperature = sum(Array(interior(model.formulation.potential_temperature_density)))
    return (; water, potential_temperature)
end

@testset "Kessler sedimentation substep bound under Reactant" begin
    @test_throws ArgumentError rainy_column(ReactantState(), 0) # no bound on Reactant

    Δt = 30.0
    eager = rainy_column(CPU(), 0)
    prepared_step!(eager, Δt)
    N = needed_substeps(eager)
    @test N ≥ 2 # the case exercises the bound
    expected = column_budgets(eager)
    θ = Array(interior(eager.temperature))

    @testset "bound $(bound) for $N needed substeps" for bound in (N + 1, N, N - 1)
        model = rainy_column(ReactantState(), bound)
        Reactant.@jit prepared_step!(model, Δt)
        @test needed_substeps(model) == N # recorded, whatever the bound
        traced = column_budgets(model)
        if bound ≥ N
            @test Array(interior(model.temperature)) ≈ θ rtol=1e-12
            @test traced.water ≈ expected.water rtol=1e-12
            @test traced.potential_temperature ≈ expected.potential_temperature rtol=1e-12
        else # rejected, not truncated
            @test isnan(traced.water) && isnan(traced.potential_temperature)
        end
    end
end
