include(joinpath(@__DIR__, "setup.jl"))

using Test
using Breeze
using Oceananigans.BoundaryConditions: BoundaryCondition

# This test checks that setting up a BulkDrag with PolynomialCoefficient throws an error when it is misconfigured
#   i.e. surface_temperature kwarg is missing
# This test *only* checks that an error is thrown when BulkDrag is misconfigured, and that an error is not thrown when it is correctly configured
@testset "Bulk drag boundary condition errors when misconfigured" begin
    # Test that an ArgumentError is thrown when surface_temperature is not supplied
    @test_throws ArgumentError BulkDrag(coefficient=PolynomialCoefficient())

    # Test that no error is thrown when surface temperature is supplied
    #   This just checks that a BulkDrag is returned, which will not happen if an error is thrown
    @test BulkDrag(coefficient=PolynomialCoefficient(), surface_temperature=295.0) isa BoundaryCondition
end

@testset "Surface vapor flux carries its enthalpy into ρs [$FT]" for FT in test_float_types()
    # A column without closure or microphysics: evaporated vapor at the temperature of the lowest cell
    # changes ρs by (cᵖᵛ - cᵖᵈ) T₁ Jᵛ (the vapor displaces dry air at fixed density), so the static energy
    # and potential temperature formulations keep the same temperature
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=8, z=(0, 800), topology=(Flat, Flat, Bounded))
    J = 1e-4
    T₁ = map((:StaticEnergy, :LiquidIcePotentialTemperature)) do formulation
        reference_state = ReferenceState(grid; potential_temperature=300)
        model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state), formulation,
                                boundary_conditions=(; ρqᵗ = FieldBoundaryConditions(bottom=FluxBoundaryCondition(J))))
        set!(model; θ=300, qᵗ=0.005)
        for _ in 1:20
            time_step!(model, 10)
        end
        @allowscalar model.temperature[1, 1, 1]
    end
    @test abs(T₁[1] - T₁[2]) < 2e-3 # without the vapor enthalpy, ρs cools the cell by ≈ 0.04 K
    Oceananigans.defaults.FloatType = Float64
end
