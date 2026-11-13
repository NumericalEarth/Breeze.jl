include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Split-explicit time step with a traced Δt. When Δt is passed to a compiled `time_step!` as a
##### device number, the acoustic substepper must not convert it to a plain float: with a fixed
##### substep count Δt only sets the substep size, and with `substeps = nothing` the substep count
##### itself is computed from Δt and becomes a traced integer. Checks that both compile and match
##### the same step compiled with Δt as a constant.
#####

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: time_step!
using Reactant
using Test
using CUDA

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

function split_explicit_model(substeps)
    grid = RectilinearGrid(ReactantState(); size=(8, 8, 8), halo=(5, 5, 5), extent=(1e3, 1e3, 1e3),
                           topology=(Periodic, Periodic, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(; substeps);
                                    reference_potential_temperature=300)
    model = AtmosphereModel(grid; dynamics)
    set_initial_state!(model) = set!(model; ρ=model.dynamics.reference_state.density, u=1,
                                     θ=(x, y, z) -> 300 + 0.1 * sin(2π * x / 1e3))
    Reactant.@jit set_initial_state!(model)
    return model
end

@testset "Split-explicit time step with a traced Δt [substeps = $substeps]" for substeps in (12, nothing)
    Δt = 0.5

    constant_model = split_explicit_model(substeps)
    Reactant.@jit raise=true time_step!(constant_model, Δt)

    traced_model = split_explicit_model(substeps)
    Reactant.@jit raise=true time_step!(traced_model, Reactant.ConcreteRNumber(Δt))

    @test Reactant.to_number(traced_model.clock.time) ≈ Δt

    Ψᶜ = Oceananigans.fields(constant_model)
    Ψᵗ = Oceananigans.fields(traced_model)
    for name in keys(Ψᶜ)
        @test isapprox(Array(interior(Ψᶜ[name])), Array(interior(Ψᵗ[name])); rtol=1e-8, atol=sqrt(eps(Float64)))
    end
end
