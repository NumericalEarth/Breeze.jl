include(joinpath(dirname(@__DIR__), "setup.jl"))

using Breeze
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Logging: with_logger, NullLogger
using Test

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

function initialize_total_water!(model, total_water)
    FT = eltype(model.grid)
    set!(model; ρ=FT(1), θ=FT(300), qᵗ=total_water, qᶜˡ=FT(0.001), qʳ=FT(0.002))
    return nothing
end

function initialized_water_mass(model, total_water)
    initialize_total_water!(model, total_water)
    return sum(interior(model.moisture_density))
end

function initialization_gradient!(model, dmodel, total_water, dtotal_water)
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, initialized_water_mass, Enzyme.Active,
                    Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(total_water, dtotal_water))
    return dtotal_water
end

@testset "Reactant total-water initialization [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(ReactantState(), FT; size=4, z=(0, 100), topology=(Flat, Flat, Bounded))
    microphysics = DCMIP2016KesslerMicrophysics(FT)
    thermodynamic_constants = ThermodynamicConstants(FT; saturation_vapor_pressure=TetensFormula(FT))
    model = AtmosphereModel(grid; microphysics, thermodynamic_constants, dynamics=CompressibleDynamics())
    total_water = CenterField(grid)
    set!(total_water, FT(0.01))

    initialize_total_water!(model, total_water)
    @test all(isapprox.(Array(interior(model.moisture_density)), FT(0.007); rtol=20eps(FT)))

    # The same compiled initializer must read the current input values at each call.
    compiled = Reactant.@compile sync=true initialize_total_water!(model, total_water)
    for water in FT.((0.01, 0.02))
        set!(total_water, water)
        compiled(model, total_water)
        vapor = Array(interior(model.moisture_density))
        @test all(isapprox.(vapor, water - FT(0.003); rtol=20eps(FT)))
        @test all(isapprox.(Array(interior(model.dynamics.total_density)), FT(1); rtol=20eps(FT)))
    end

    # The Boolean runtime check must not prevent differentiation through initialization.
    dmodel = Enzyme.make_zero(model)
    dtotal_water = CenterField(grid)
    compiled_gradient = Reactant.@compile raise=true raise_first=true sync=true initialization_gradient!(
        model, dmodel, total_water, dtotal_water)
    gradient = @with_stack_size compiled_gradient(model, dmodel, total_water, dtotal_water)
    @test all(isapprox.(Array(interior(gradient)), FT(1); rtol=50eps(FT)))

    # This value changes after compilation, so rejection must happen at execution time.
    # The callback logs the ArgumentError, which Reactant reports as an execution error.
    set!(total_water, FT(0))
    @test_throws ArgumentError initialize_total_water!(model, total_water)
    # A failed compiled call can consume donated buffers, so run it last.
    with_logger(NullLogger()) do
        @test_throws Reactant.XLA.ReactantInternalError compiled(model, total_water)
    end
end
