using Breeze
using Test

@testset "PiecewiseStretchedDiscretization" begin
    @testset "Uniform grid" begin
        z = PiecewiseStretchedDiscretization(z=[0, 1000], Δz=[50, 50])
        Nz = length(z) - 1
        Δz = diff(z.faces)

        @test z[1] == 0
        @test z[end] == 1000
        @test Nz == 20
        @test all(Δz .≈ 50)
    end

    @testset "Two uniform regions with transition" begin
        z = PiecewiseStretchedDiscretization(
            z  = [0, 1000, 3500, 28000],
            Δz = [62.5, 62.5, 2000, 2000])

        Δz = diff(z.faces)

        # Starts and ends at correct positions
        @test z[1] == 0
        @test z[end] == 28000

        # First region: uniform 62.5 m from 0 to 1000
        first_region = Δz[z[1:end-1] .< 1000 - 1e-6]
        @test all(first_region .≈ 62.5)

        # 1000 m is an exact face position
        @test 1000.0 ∈ z.faces

        # Last region: uniform 2000 m (except possibly a shorter final cell)
        last_region_start = findfirst(f -> f ≥ 3500 - 1e-6, z.faces)
        last_region_Δz = Δz[last_region_start:end]
        @test all(d -> d ≤ 2000 + 1e-6, last_region_Δz)

        # Transition region: spacings increase monotonically except for
        # a possible short final cell at the breakpoint boundary
        transition_mask = (z[1:end-1] .≥ 1000 - 1e-6) .& (z[1:end-1] .< 3500 - 1e-6)
        transition_Δz = Δz[transition_mask]
        @test issorted(transition_Δz[1:end-1])
    end

    @testset "Multiple transitions (GATE-like)" begin
        z = PiecewiseStretchedDiscretization(
            z  = [0, 1275, 5100, 18000, 27000],
            Δz = [50, 50, 100, 100, 300])

        Δz = diff(z.faces)
        Nz = length(z) - 1

        @test z[1] == 0
        @test z[end] == 27000

        # All spacings are positive
        @test all(Δz .> 0)

        # All spacings are bounded above by the maximum input spacing
        @test maximum(Δz) ≤ 300 + 1e-6

        # All breakpoints are face positions
        for zb in [0, 1275, 5100, 18000, 27000]
            @test any(f -> abs(f - zb) < 1e-6, z.faces)
        end

        # Uniform region in troposphere (5100 to 18000): all ≈ 100 m
        tropo_start = findfirst(f -> f ≥ 5100 - 1e-6, z.faces)
        tropo_end = findlast(f -> f ≤ 18000 + 1e-6, z.faces) - 1
        tropo_Δz = Δz[tropo_start:tropo_end]
        @test all(d -> abs(d - 100) < 1e-6, tropo_Δz)
    end

    @testset "AbstractVector interface" begin
        z = PiecewiseStretchedDiscretization(z=[0, 100], Δz=[10, 10])

        # length works
        @test length(z) == 11

        # Indexing works
        @test z[1] == 0
        @test z[11] == 100

        # Behaves as AbstractVector
        @test z isa AbstractVector{Float64}
    end

    @testset "show method" begin
        z = PiecewiseStretchedDiscretization(z=[0, 100], Δz=[10, 10])
        str = sprint(show, MIME"text/plain"(), z)
        @test occursin("PiecewiseStretchedDiscretization", str)
        @test occursin("10 cells", str)
    end

    @testset "Input validation" begin
        # Mismatched lengths
        @test_throws ArgumentError PiecewiseStretchedDiscretization(z=[0, 100], Δz=[10])

        # Too few breakpoints
        @test_throws ArgumentError PiecewiseStretchedDiscretization(z=[0], Δz=[10])

        # Unsorted breakpoints
        @test_throws ArgumentError PiecewiseStretchedDiscretization(z=[100, 0], Δz=[10, 10])

        # Non-positive spacing
        @test_throws ArgumentError PiecewiseStretchedDiscretization(z=[0, 100], Δz=[10, -5])
        @test_throws ArgumentError PiecewiseStretchedDiscretization(z=[0, 100], Δz=[0, 10])
    end

    @testset "Passes directly to RectilinearGrid" begin
        z = PiecewiseStretchedDiscretization(
            z  = [0, 500, 1000],
            Δz = [50, 50, 100])

        Nz = length(z) - 1
        grid = RectilinearGrid(CPU();
                               size = (4, 4, Nz),
                               x = (0, 400),
                               y = (0, 400),
                               z,
                               topology = (Periodic, Periodic, Bounded))

        @test grid.Nz == Nz
        @test grid.Lz ≈ 1000
    end
end
