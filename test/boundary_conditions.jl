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
