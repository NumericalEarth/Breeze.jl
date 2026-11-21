include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### A Reactant forward with per-column closure coefficients as inputs: one compiled program serves
##### any ensemble, written in place with `set_per_column_coefficients!`, and matches the eager CPU run.
#####

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: ColumnEnsembleSize
using Reactant
using Reactant: @trace
using GPUArraysCore: @allowscalar
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

function column_tke(model, Δt, Nsteps)
    set!(model; θ = z -> 290 + 3e-3 * max(z - 800, 0), qᵗ=0.012, u=5, ρe=0.1)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return interior(model.tracers.ρe)
end

ensemble_closures(ℂⁱs, ℂᴰ⁰s) = reshape([TKEBasedTurbulenceClosure(mixing_length = GradientLimitedMixingLength(ℂˢ=1.3, ℂⁱ=ℂⁱ),
                                                                   stability_functions = TurbulentRichardsonStabilityFunction(ℂᴰ⁰=ℂᴰ⁰))
                                         for (ℂⁱ, ℂᴰ⁰) in zip(ℂⁱs, ℂᴰ⁰s)], length(ℂⁱs), 1)

function column_model(arch, closures)
    grid = RectilinearGrid(arch; size=ColumnEnsembleSize(Nz=8, ensemble=(size(closures, 1), 1), Hz=3),
                           z=(0, 2000), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(0.1)))
    return AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), boundary_conditions,
                           microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()),
                           closure = per_column_closure(grid, closures))
end

@testset "reactant_per_column_coefficients" begin
    Δt, Ns = 1.0, 3
    first_ensemble = ensemble_closures((1.2, 1.6, 2.0), (2.0, 2.5, 3.0))
    second_ensemble = ensemble_closures((2.4, 0.9, 1.4), (3.5, 1.8, 2.2))

    model = column_model(ReactantState(), first_ensemble)
    @test model.closure.mixing_length.ℂⁱ.values isa Reactant.ConcreteRArray
    compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true column_tke(model, Δt, Ns)

    for closures in (first_ensemble, second_ensemble)
        set_per_column_coefficients!(model.closure, closures)
        traced = Array(@with_stack_size compiled(model, Δt, Ns))
        eager = Array(column_tke(column_model(CPU(), closures), Δt, Ns))
        @test isapprox(traced, eager; rtol=1e-10)
        @test traced[1, 1, :] != traced[3, 1, :]
    end
end
