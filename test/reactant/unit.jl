include(joinpath(dirname(@__DIR__), "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: has_condensate
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Test

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

# A compressible model's density is zero until set, and θ is diagnosed from ρθ / ρ; without `ρ` the
# mean state, and so the whole reset reference, would be NaN on every architecture.
warm_and_rebalance!(model) = set!(model; ρ = 1, θ = 302, compute_reference_state = true)

@testset "Reactant reference state reset [$FT]" for FT in test_float_types()
    reference_model(arch) = AtmosphereModel(
        RectilinearGrid(arch, FT; size = (4, 4, 8), x = (0, 4e3), y = (0, 4e3), z = (0, 4e3));
        dynamics = CompressibleDynamics(; reference_potential_temperature = 300),
    )

    model = reference_model(CPU())
    warm_and_rebalance!(model)

    r_model = reference_model(ReactantState())
    compiled_reset! = Reactant.@compile sync = true warm_and_rebalance!(r_model)
    compiled_reset!(r_model)

    reference(m) = m.dynamics.reference_state
    @test all(isfinite, interior(reference(model).pressure))
    @test Array(interior(reference(r_model).pressure)) ≈ Array(interior(reference(model).pressure))
    @test Array(interior(reference(r_model).density)) ≈ Array(interior(reference(model).density))
end

# `has_condensate` guards `set!(model; ρ = HydrostaticallyBalancedDensity())`. It reduces each condensate field outside a
# compiled program, where a view of a concrete Reactant array on a GPU cannot be reduced in place.
@testset "Reactant has_condensate and HydrostaticallyBalancedDensity [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(ReactantState(), FT; size = (4, 4, 8), x = (0, 4e3), y = (0, 4e3), z = (0, 8e3),
                           topology = (Periodic, Periodic, Bounded))
    # Reactant requires a static sedimentation substep bound; this test never steps
    microphysics = DCMIP2016KesslerMicrophysics(FT; maximum_sedimentation_substeps = 1)
    thermodynamic_constants = ThermodynamicConstants(FT; saturation_vapor_pressure = TetensFormula(FT))
    model = AtmosphereModel(grid; microphysics, thermodynamic_constants, dynamics = CompressibleDynamics())

    @test !has_condensate(model)
    set!(model; θ = 300, ρ = HydrostaticallyBalancedDensity())
    @test all(>(0), Array(interior(model.dynamics.total_density)))

    set!(model; θ = 300, qᵗ = FT(0.01), qʳ = FT(0.001))
    @test has_condensate(model)
    @test_throws ArgumentError set!(model; ρ = HydrostaticallyBalancedDensity())
end
