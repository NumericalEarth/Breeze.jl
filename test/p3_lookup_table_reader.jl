include(joinpath(@__DIR__, "setup.jl"))

using Test
using Pkg.Artifacts: ensure_artifact_installed
using Oceananigans
using Oceananigans.Utils: TabulatedFunction, TabulatedFunction1D
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.Microphysics.PredictedParticleProperties:
    ice_terminal_velocities,
    rain_terminal_velocities

@testset "Rime-density-indexed table transforms" begin
    # Create a 4D table that returns its 4th argument (rime density index).
    identity_4th = (x1, x2, x3, x4) -> Float64(x4)
    table = TabulatedFunction(identity_4th, CPU(), Float64;
                              range=((0.0, 1.0), (0.0, 1.0),
                                     (0.0, 1.0), (1.0, 5.0)),
                              points=(2, 2, 2, 5))
    wrapped = RimeDensityIndexedTable4D(table)

    # Physical rime densities should map to table indices
    # rho=50 -> index 1, rho=250 -> index 2, rho=450 -> index 3,
    # rho=650 -> index 4, rho=900 -> index 5
    @test wrapped(0.5, 0.5, 0.5, 50.0) ≈ 1.0
    @test wrapped(0.5, 0.5, 0.5, 250.0) ≈ 2.0
    @test wrapped(0.5, 0.5, 0.5, 450.0) ≈ 3.0
    @test wrapped(0.5, 0.5, 0.5, 650.0) ≈ 4.0
    @test wrapped(0.5, 0.5, 0.5, 900.0) ≈ 5.0
    # Intermediate value
    @test wrapped(0.5, 0.5, 0.5, 150.0) ≈ 1.5
end

const _lookup_table_dir = ensure_artifact_installed("P3_lookup_tables", joinpath(dirname(@__DIR__), "Artifacts.toml"))

@testset "Read P3 lookup tables (2momI)" begin
    p3 = read_lookup_tables(_lookup_table_dir; FT=Float64)

    tables = p3.ice.lookup_tables
    @test tables isa P3LookupTables
    @test tables.ice_integrals isa P3IceIntegralsTable
    @test tables.rain_ice_collection isa P3RainIceCollectionTable
    @test p3.water_density == 1000
    @test p3.process_rates.liquid_water_density == 1000

    @test p3.ice.fall_speed.mass_weighted isa RimeDensityIndexedTable4D
    @test p3.ice.deposition.ventilation isa RimeDensityIndexedTable4D
    @test tables.rain_ice_collection.mass isa RimeDensityIndexedTable5D
    @test size(p3.ice.fall_speed.mass_weighted.table.table) == (50, 4, 4, 5)
    @test size(tables.rain_ice_collection.mass.table.table) == (50, 30, 4, 4, 5)

    # Spot-check first row of 2momI: i_rhor=1, i_Fr=1, i_Fl=1, i_Qnorm=1
    # uns = 0.15624E-03, ums = 0.35587E-03
    uns = p3.ice.fall_speed.number_weighted(-14.807, 0.0, 0.0, 50.0)
    ums = p3.ice.fall_speed.mass_weighted(-14.807, 0.0, 0.0, 50.0)
    @test uns ≈ 0.15624e-03 rtol=1e-3
    @test ums ≈ 0.35587e-03 rtol=1e-3

    # Rain 1D tables should be populated
    @test p3.rain.velocity_mass isa TabulatedFunction
    @test p3.rain.velocity_number isa TabulatedFunction
    @test p3.rain.evaporation isa TabulatedFunction
end

@testset "Rain tables are computed, not read from file" begin
    p3 = read_lookup_tables(_lookup_table_dir)

    @test p3.rain.velocity_mass isa TabulatedFunction1D
    @test p3.rain.velocity_number isa TabulatedFunction1D

    log_lambda = 3.5
    @test p3.rain.velocity_mass(log_lambda) > 0
end

@testset "PredictedParticlePropertiesMicrophysics constructor with lookup tables" begin
    # Test constructor interface
    p3 = PredictedParticlePropertiesMicrophysics()
    @test p3 isa PredictedParticlePropertiesMicrophysics
    @test p3.ice.lookup_tables isa P3LookupTables
end

@testset "Process rates with table-loaded ice integrals" begin
    p3 = PredictedParticlePropertiesMicrophysics()

    FT = Float64
    qⁱ = FT(1e-4)    # ice mass mixing ratio
    nⁱ = FT(1e5)     # ice number
    qʳ = FT(1e-4)    # rain mass mixing ratio
    nʳ = FT(1e5)     # rain number
    Fᶠ = FT(0.5)     # rime fraction
    ρᶠ = FT(400.0)   # rime density
    Fˡ = FT(0.0)     # liquid fraction
    ρ  = FT(0.8)     # air density

    # Ice terminal velocities should be physical
    vⁱ = ice_terminal_velocities(p3, qⁱ, nⁱ, Fᶠ, ρᶠ, ρ; Fˡ)
    @test 0 < vⁱ.mass_weighted < 50
    @test 0 < vⁱ.number_weighted < 50

    # Rain terminal velocity
    vʳ = rain_terminal_velocities(p3, qʳ, nʳ, ρ)
    @test 0 < vʳ.mass_weighted < 20
    @test 0 < vʳ.number_weighted < 20
end
