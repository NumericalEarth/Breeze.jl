include(joinpath(@__DIR__, "setup.jl"))

#####
##### Column extension above the grid top, ecCKD optics types, and the standard atmosphere
#####
##### None of these need NumericalRadiation: they are the backend-agnostic pieces that the
##### NumericalRadiation extension consumes.
#####

using Breeze
using Breeze.AtmosphereModels: standard_specific_humidity, geometric_stretching_ratio, column_extension_faces
using Oceananigans
using Oceananigans.Units
using Test

@testset "Standard atmosphere temperature" begin
    # U.S. Standard Atmosphere 1976 layer bases
    @test standard_atmosphere_temperature(0) == 288.15
    @test standard_atmosphere_temperature(11e3) == 216.65
    @test standard_atmosphere_temperature(20e3) == 216.65
    @test standard_atmosphere_temperature(32e3) == 228.65
    @test standard_atmosphere_temperature(47e3) == 270.65
    @test standard_atmosphere_temperature(50e3) == 270.65
    @test standard_atmosphere_temperature(51e3) == 270.65
    @test standard_atmosphere_temperature(71e3) == 214.65
    @test standard_atmosphere_temperature(84852) == 186.946

    # Lapse rates inside the layers
    @test standard_atmosphere_temperature(1e3) ≈ 288.15 - 6.5
    @test standard_atmosphere_temperature(25e3) ≈ 216.65 + 5
    @test standard_atmosphere_temperature(40e3) ≈ 228.65 + 2.8 * 8
    @test standard_atmosphere_temperature(61e3) ≈ 270.65 - 2.8 * 10
    @test standard_atmosphere_temperature(80e3) ≈ 214.65 - 2 * 9

    # Edge hold outside the tabulated range
    @test standard_atmosphere_temperature(100e3) == 186.946
    @test standard_atmosphere_temperature(-100) == 288.15

    # Continuous and piecewise linear: no jumps between 0 and 90 km
    z = 0:100:90e3
    T = standard_atmosphere_temperature.(z)
    @test maximum(abs, diff(T)) < 0.7

    @testset "Float type [$(FT)]" for FT in all_float_types()
        @test @inferred(standard_atmosphere_temperature(FT(11e3))) isa FT
        @test standard_atmosphere_temperature(FT(11e3)) ≈ 216.65
        @test @inferred(standard_specific_humidity(FT(1e3))) isa FT
    end
end

@testset "Standard specific humidity" begin
    @test standard_specific_humidity(0) == 1e-2
    @test standard_specific_humidity(2.5e3) ≈ 1e-2 * exp(-1)
    @test standard_specific_humidity(50e3) == 3e-6
    z = 0:100:65e3
    q = standard_specific_humidity.(z)
    @test issorted(q, rev=true)
    @test all(q .>= 3e-6)
end

@testset "ColumnExtension [$(FT)]" for FT in all_float_types()
    Oceananigans.defaults.FloatType = FT
    extension = ColumnExtension()
    @test extension isa ColumnExtension{FT}
    @test extension.top == 65e3
    @test extension.layers == 40
    @test extension.blending_height == 1e3
    @test extension.temperature === standard_atmosphere_temperature
    @test extension.specific_humidity === standard_specific_humidity
    @test isnothing(extension.ozone_mole_fraction)

    ozone(z) = 1e-6
    custom = ColumnExtension(FT; top = 80kilometers, layers = 60, blending_height = 0,
                             temperature = z -> 250, specific_humidity = z -> 1e-5, ozone_mole_fraction = ozone)
    @test custom.top isa FT
    @test custom.top == 80e3
    @test custom.layers == 60
    @test custom.blending_height == 0
    @test custom.temperature(1e3) == 250
    @test custom.ozone_mole_fraction === ozone

    constant_ozone = ColumnExtension(FT; ozone_mole_fraction = 1e-6)
    @test constant_ozone.ozone_mole_fraction isa FT
    @test constant_ozone.ozone_mole_fraction ≈ 1e-6

    @test_throws ArgumentError ColumnExtension(FT; layers = 0)
    @test_throws ArgumentError ColumnExtension(FT; blending_height = -1)

    @testset "show" begin
        str = sprint(show, extension)
        @test occursin("ColumnExtension{$FT}", str)
        @test occursin("├── top: 65000.0 m", str)
        @test occursin("├── layers: 40", str)
        @test occursin("├── blending_height: 1000.0 m", str)
        @test occursin("├── temperature: standard_atmosphere_temperature", str)
        @test occursin("├── specific_humidity: standard_specific_humidity", str)
        @test occursin("└── ozone_mole_fraction: nothing", str)
        @test occursin("└── ozone_mole_fraction: 1.0e-6", sprint(show, constant_ozone))
    end
end

Oceananigans.defaults.FloatType = Float64

@testset "Geometric stretching" begin
    # 1 + 2 + 4 = 7
    @test geometric_stretching_ratio(1, 7, 3) ≈ 2 rtol=1e-14
    # Uniform layers
    @test geometric_stretching_ratio(2.0, 8.0, 4) ≈ 1 rtol=1e-14
    # Shrinking layers: 1 + r + r² = 1.5 ⇒ r = (√3 - 1) / 2
    @test geometric_stretching_ratio(1.0, 1.5, 3) ≈ (sqrt(3) - 1) / 2 rtol=1e-14
    # A single layer spans the depth whatever the ratio
    @test geometric_stretching_ratio(1.0, 50.0, 1) == 1
    # No stretching reaches a depth shallower than the first layer
    @test_throws ArgumentError geometric_stretching_ratio(10.0, 5.0, 3)
    @test geometric_stretching_ratio(10f0, 100f0, 3) isa Float32

    @testset "Extension faces [$(FT)]" for FT in all_float_types()
        extension = ColumnExtension(FT; top = 65e3, layers = 40)
        z_top = FT(3e3)
        Δz_top = FT(100)
        faces = column_extension_faces(extension, z_top, Δz_top)

        @test faces isa Vector{FT}
        @test length(faces) == 41
        @test faces[1] == z_top
        @test faces[end] == extension.top   # lands exactly on `top`
        Δz = diff(faces)
        @test all(Δz .> 0)                  # monotone
        @test Δz[1] == Δz_top               # first layer as thick as the grid's top layer
        @test all(Δz[2:end] .> Δz[1:end-1]) # layers grow with height
        # Geometric: every thickness is Δz_top rᵐ, the last one absorbing the rounding of the ratio
        r = geometric_stretching_ratio(Δz_top, extension.top - z_top, extension.layers)
        @test all(isapprox.(Δz, Δz_top .* r .^ (0:extension.layers-1), rtol = 200 * eps(FT)))

        # Layer count and the top face agree with `top` and `layers`
        for layers in (1, 2, 5), top in (3.2e3, 10e3, 80e3)
            extension = ColumnExtension(FT; top, layers)
            faces = column_extension_faces(extension, z_top, Δz_top)
            @test length(faces) == layers + 1
            @test faces[end] == FT(top)
            @test issorted(faces)
        end

        # A grid that already reaches `top` gets no extension layers
        faces = column_extension_faces(ColumnExtension(FT; top = 3e3), z_top, Δz_top)
        @test faces == [z_top]
        faces = column_extension_faces(ColumnExtension(FT; top = 2e3), z_top, Δz_top)
        @test faces == [z_top]
    end
end

@testset "ecCKD optics types" begin
    optics = EcCKDOptics()
    @test optics.gas_model == :climate_32x32
    @test isnothing(optics.clouds)

    tables = CloudScatteringTables()
    @test tables.liquid == :mie_droplet
    @test tables.ice == :baum_general_habit_mixture
    @test sprint(show, tables) == "CloudScatteringTables(liquid=:mie_droplet, ice=:baum_general_habit_mixture)"

    custom_tables = CloudScatteringTables(liquid = "liquid.nc", ice = "ice.nc")
    @test sprint(show, custom_tables) == "CloudScatteringTables(liquid=\"liquid.nc\", ice=\"ice.nc\")"

    optics = EcCKDOptics(:climate_64x64, clouds = tables)
    @test optics.gas_model == :climate_64x64
    @test optics.clouds === tables
    str = sprint(show, optics)
    @test occursin("├── gas_model: :climate_64x64", str)
    @test occursin("└── clouds: CloudScatteringTables(liquid=:mie_droplet", str)

    paths = (longwave = "lw.nc", shortwave = "sw.nc")
    str = sprint(show, EcCKDOptics(paths))
    @test occursin("├── gas_model: (longwave = \"lw.nc\", shortwave = \"sw.nc\")", str)

    @testset "Constructor without the extension" begin
        grid = RectilinearGrid(default_arch; size=4, x=0, y=45, z=(0, 10kilometers), topology=(Flat, Flat, Bounded))
        constants = ThermodynamicConstants()
        err = try
            RadiativeTransferModel(grid, EcCKDOptics(), constants; surface_temperature = 300, surface_albedo = 0.1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("EcCKDOptics", err.msg)
        @test occursin("NumericalRadiation", err.msg)
        @test occursin("NCDatasets", err.msg)
    end
end
