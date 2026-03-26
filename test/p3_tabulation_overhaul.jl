using Test
using Adapt: Adapt
using Oceananigans.Architectures: CPU
using Oceananigans.Utils: TabulatedFunction1D, TabulatedFunction4D
import Oceananigans.Architectures: on_architecture
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.Microphysics.PredictedParticleProperties:
    IceProperties, NullP3LookupTables, P3TabulationParameters,
    LookupTable1Parameters, LookupTable2Parameters, LookupTable3Parameters

struct MockArchitecture end

on_architecture(::MockArchitecture, a::Array) = a

@testset "P3 uses Oceananigans generic tabulated function types" begin
    p3 = PredictedParticlePropertiesMicrophysics()
    p3_tab = tabulate(p3, CPU();
        number_of_mass_points = 4,
        number_of_rime_fraction_points = 2,
        number_of_liquid_fraction_points = 2,
        number_of_rime_density_points = 3,
        number_of_quadrature_points = 12)

    @test p3_tab.rain.velocity_mass isa TabulatedFunction1D
    @test p3_tab.ice.fall_speed.mass_weighted isa TabulatedFunction4D
end

@testset "P3 tabulated function adapters preserve singleton axes" begin
    f1 = TabulatedFunction1D(x -> 3x, MockArchitecture(), Float64;
        x_range = (2.0, 4.0),
        x_points = 1)

    @test f1.table == [6.0]
    @test f1(999.0) ≈ 6.0

    f4 = TabulatedFunction4D((x, y, z, w) -> x + y + z + w, MockArchitecture(), Float64;
        x_range = (0.0, 1.0),
        y_range = (2.0, 3.0),
        z_range = (5.0, 6.0),
        w_range = (8.0, 9.0),
        x_points = 2,
        y_points = 1,
        z_points = 1,
        w_points = 1)

    @test size(f4.table) == (2, 1, 1, 1)
    @test all(isfinite, f4.table)
    @test f4(0.75, 42.0, 43.0, 44.0) ≈ 15.75
end

@testset "P3 lookup family parameters and storage" begin
    params = P3TabulationParameters(Float64)
    @test params.lookup_table_1 isa LookupTable1Parameters
    @test params.lookup_table_2 isa LookupTable2Parameters
    @test params.lookup_table_3 isa LookupTable3Parameters
    @test params.lookup_table_3.number_of_znorm_points == 80

    ice = IceProperties(Float64)
    @test ice.lookup_tables isa NullP3LookupTables
end

@testset "P3 tabulation parameter overrides propagate to nested families" begin
    params = P3TabulationParameters(Float64; number_of_mass_points = 4)
    @test params.lookup_table_1.number_of_mass_points == 4
end

@testset "IceProperties reconstruction preserves lookup_tables" begin
    lookup_tables = :sentinel_lookup_tables
    ice = IceProperties(Float64; lookup_tables)

    @test on_architecture(CPU(), ice).lookup_tables === lookup_tables
    @test Adapt.adapt(identity, ice).lookup_tables === lookup_tables
end
