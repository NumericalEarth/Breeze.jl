include(joinpath(@__DIR__, "setup.jl"))

#####
##### Tests for piecewise_linear_regrid!: conservative, second-order regridding of column profiles in z
#####

using Breeze
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Test

column(arch, FT, z) = RectilinearGrid(arch, FT; size=length(z) - 1, z, topology=(Flat, Flat, Bounded))
cell_means(f, z) = [f(z[k], z[k+1]) for k in 1:length(z)-1]

@testset "piecewise_linear_regrid!" begin
    arch = default_arch
    FT = Float64

    # A stretched source column and a target whose faces do not align with it
    zˢ = vcat(collect(0.0:20:1000), collect(1050.0:50:3000))
    zᵗ = vcat(collect(0.0:70:2800), [3000])

    @testset "Linear profiles map exactly" begin
        source = CenterField(column(arch, FT, zˢ))
        set!(source, reshape(cell_means((a, b) -> 2 * (a + b) / 2 - 300, zˢ), 1, 1, :))
        target = piecewise_linear_regrid!(CenterField(column(arch, FT, zᵗ)), source)
        expected = cell_means((a, b) -> 2 * (a + b) / 2 - 300, zᵗ)
        @test maximum(abs, Array(interior(target, 1, 1, :)) .- expected) < 1e-10

        # Oceananigans' piecewise-constant regrid! places a coarse cell's mean at the centroid of its sources
        constant = regrid!(CenterField(column(arch, FT, zᵗ)), source)
        @test maximum(abs, Array(interior(constant, 1, 1, :)) .- expected) > 1
    end

    @testset "Conservation and agreement with regrid! on nested cells" begin
        source = CenterField(column(arch, FT, zˢ))
        values = [sin(k / 3) + 2 for k in 1:length(zˢ)-1]
        set!(source, reshape(values, 1, 1, :))
        target = piecewise_linear_regrid!(CenterField(column(arch, FT, zᵗ)), source)
        integral(field, z) = sum(Array(interior(field, 1, 1, :)) .* diff(z))
        @test integral(target, zᵗ) ≈ integral(source, zˢ) rtol=1e-13

        # Target cells that are unions of whole source cells receive the source means, as with regrid!
        nested = vcat(collect(0.0:100:1000), collect(1200.0:200:3000))
        a = piecewise_linear_regrid!(CenterField(column(arch, FT, nested)), source)
        b = regrid!(CenterField(column(arch, FT, nested)), source)
        @test Array(interior(a)) ≈ Array(interior(b)) rtol=1e-13
    end

    @testset "Nonnegative limiting" begin
        # A cloud layer: zero, then a jump; the centred slope would drive the edge cells negative
        source = CenterField(column(arch, FT, zˢ))
        values = [1000 < (zˢ[k] + zˢ[k+1]) / 2 < 1500 ? 1e-3 : 0.0 for k in 1:length(zˢ)-1]
        set!(source, reshape(values, 1, 1, :))
        unlimited = piecewise_linear_regrid!(CenterField(column(arch, FT, zᵗ)), source)
        limited = piecewise_linear_regrid!(CenterField(column(arch, FT, zᵗ)), source; nonnegative=true)
        @test minimum(Array(interior(unlimited))) < 0
        @test minimum(Array(interior(limited))) ≥ 0
        integral(field) = sum(Array(interior(field, 1, 1, :)) .* diff(zᵗ))
        @test integral(limited) ≈ sum(values .* diff(zˢ)) rtol=1e-12
    end

    @testset "Column ensembles" begin
        Nᶜ = 3
        ensemble(z) = RectilinearGrid(arch, FT; size=ColumnEnsembleSize(Nz=length(z) - 1, ensemble=(Nᶜ, 1)), z,
                                      topology=(Flat, Flat, Bounded))
        source = CenterField(ensemble(zˢ))
        slopes = [1, -2, 0.5]
        set!(source, cat([reshape(cell_means((a, b) -> s * (a + b) / 2, zˢ), 1, 1, :) for s in slopes]...; dims=1))
        target = piecewise_linear_regrid!(CenterField(ensemble(zᵗ)), source)
        for (i, s) in enumerate(slopes)
            @test Array(interior(target))[i, 1, :] ≈ s .* cell_means((a, b) -> (a + b) / 2, zᵗ)
        end

        # A single-column source fills every target column
        single = CenterField(column(arch, FT, zˢ))
        set!(single, reshape(cell_means((a, b) -> (a + b) / 2, zˢ), 1, 1, :))
        broadcast = piecewise_linear_regrid!(CenterField(ensemble(zᵗ)), single)
        @test all(i -> Array(interior(broadcast))[i, 1, :] ≈ cell_means((a, b) -> (a + b) / 2, zᵗ), 1:Nᶜ)
    end

    @testset "Errors" begin
        source = CenterField(column(arch, FT, zˢ))
        @test_throws ArgumentError piecewise_linear_regrid!(CenterField(column(arch, FT, [0, 1000, 4000])), source)
        @test_throws ArgumentError piecewise_linear_regrid!(ZFaceField(column(arch, FT, zᵗ)), source)
    end
end
