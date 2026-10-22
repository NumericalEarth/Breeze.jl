include(joinpath(dirname(@__DIR__), "setup.jl"))

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Test

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

warm_and_rebalance!(model) = set!(model; θ = 302, compute_reference_state = true)

@testset "Reactant reference state reset [$FT]" for FT in test_float_types()
    reference_model(arch) = AtmosphereModel(
        RectilinearGrid(arch, FT; size = (4, 4, 8), x = (0, 4e3), y = (0, 4e3), z = (0, 4e3));
        dynamics = CompressibleDynamics(; reference_potential_temperature = 300),
    )

    model = reference_model(CPU())
    warm_and_rebalance!(model)

    r_model = reference_model(ReactantState())
    compiled_initialize! = Reactant.@compile sync = true initialize!(r_model)
    compiled_initialize!(r_model)
    compiled_reset! = Reactant.@compile sync = true warm_and_rebalance!(r_model)
    compiled_reset!(r_model)

    reference(m) = m.dynamics.reference_state
    @test Array(interior(reference(r_model).pressure)) ≈ Array(interior(reference(model).pressure))
    @test Array(interior(reference(r_model).density)) ≈ Array(interior(reference(model).density))
end
