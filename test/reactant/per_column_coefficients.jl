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
using Breeze.Microphysics.PredictedParticleProperties: SubgridCondensation, AllProcesses
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

function column_tke(model, Δt, Nsteps)
    set!(model; θ = z -> 290 + 3e-3 * max(z - 800, 0), qᵗ=0.012, u=5, ρe=0.1)
    # The eager first `time_step!` prepares itself (`clock.last_Δt = Δt`, then the tendencies), which a
    # traced loop cannot: without this the traced first stage would apply the tendencies of `set!`,
    # built before the stage interval is known (P3's explicit rates in place of its implicit relaxation).
    Breeze.TimeSteppers.prepare_first_time_step!(model, Δt)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return interior(model.tracers.ρe)
end

ensemble_closures(ℂⁱs, ℂᴰ⁰s) = reshape([TKEBasedTurbulenceClosure(mixing_length = GradientLimitedMixingLength(ℂˢ=1.3, ℂⁱ=ℂⁱ),
                                                                   stability_functions = TurbulentRichardsonStabilityFunction(ℂᴰ⁰=ℂᴰ⁰))
                                         for (ℂⁱ, ℂᴰ⁰) in zip(ℂⁱs, ℂᴰ⁰s)], length(ℂⁱs), 1)

function column_model(arch, closures; microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium()))
    grid = RectilinearGrid(arch; size=ColumnEnsembleSize(Nz=8, ensemble=(size(closures, 1), 1), Hz=3),
                           z=(0, 2000), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    boundary_conditions = (; ρθ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(0.1)))
    return AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state), boundary_conditions,
                           microphysics, closure = per_column_closure(grid, closures))
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

# The DoubleGaussian with VarianceStabilityFunctions and ℂᵐ varying from column to column, and P3 reading the
# distribution in every process (SubgridCondensation with AllProcesses)
updated_closure(ℂᵛ₁⁰, ℂᵛ₃ᵐ, ℂᵐ) =
    TKEBasedTurbulenceClosure(static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                              subgrid_distribution = DoubleGaussian(; ℂᵐ, variance_transport = ScalarVarianceTransport(),
                                  variance_stability_functions = VarianceStabilityFunctions(; ℂᵛ₁⁰, ℂᵛ₂⁰ = ℂᵛ₁⁰, ℂᵛ₃ᵐ)))
updated_closures(values) = reshape([updated_closure(v...) for v in values], length(values), 1)

@testset "reactant_per_column_coefficients with VarianceStabilityFunctions, ℂᵐ and P3 AllProcesses" begin
    Δt, Ns = 1.0, 3
    first_ensemble = updated_closures(((0.5, 0.9, 0.5), (0.7, 0.6, 1.0), (0.9, 0.3, 2.0)))
    second_ensemble = updated_closures(((1.1, -0.2, 3.0), (0.6, 0.8, 0.7), (0.8, 0.5, 1.5)))
    condensation = SubgridCondensation(TurbulenceClosureStatistics(); processes = AllProcesses())
    microphysics = PredictedParticlePropertiesMicrophysics(; condensation)

    model = column_model(ReactantState(), first_ensemble; microphysics)
    @test model.closure.subgrid_distribution.ℂᵐ.values isa Reactant.ConcreteRArray
    @test model.closure.subgrid_distribution.variance_stability_functions.ℂᵛ₃ᵐ.values isa Reactant.ConcreteRArray
    # The closure and P3 keep state that `set!` does not reset (lagged stability, the stage start of the implicit
    # relaxation, ...), so each ensemble runs on a copy of the model as built, as a calibration runs each member
    template = deepcopy(model)
    compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true column_tke(model, Δt, Ns)

    for closures in (first_ensemble, second_ensemble)
        member = deepcopy(template)
        set_per_column_coefficients!(member.closure, closures)
        traced = Array(@with_stack_size compiled(member, Δt, Ns))
        eager = Array(column_tke(column_model(CPU(), closures; microphysics), Δt, Ns))
        @test all(isfinite, traced)
        @test isapprox(traced, eager; rtol=1e-10)
        @test traced[1, 1, :] != traced[3, 1, :]
    end
end
