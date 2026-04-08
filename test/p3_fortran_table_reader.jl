using Test
using Oceananigans
using Oceananigans.Utils: TabulatedFunction
using Breeze.Microphysics.PredictedParticleProperties

@testset "FortranTabulatedFunction5D rime density transform" begin
    # Create a 5D table that returns its 4th argument (rime density index)
    identity_4th = (x1, x2, x3, x4, x5) -> Float64(x4)
    table = TabulatedFunction(identity_4th, CPU(), Float64;
                              range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (1.0, 5.0), (0.0, 20.0)),
                              points=(2, 2, 2, 5, 2))
    wrapped = FortranTabulatedFunction5D(table)

    # Physical rime densities should map to Fortran indices
    # rho=50 -> index 1, rho=250 -> index 2, rho=450 -> index 3,
    # rho=650 -> index 4, rho=900 -> index 5
    @test wrapped(0.5, 0.5, 0.5, 50.0, 10.0) ≈ 1.0
    @test wrapped(0.5, 0.5, 0.5, 250.0, 10.0) ≈ 2.0
    @test wrapped(0.5, 0.5, 0.5, 450.0, 10.0) ≈ 3.0
    @test wrapped(0.5, 0.5, 0.5, 650.0, 10.0) ≈ 4.0
    @test wrapped(0.5, 0.5, 0.5, 900.0, 10.0) ≈ 5.0
    # Intermediate value
    @test wrapped(0.5, 0.5, 0.5, 150.0, 10.0) ≈ 1.5
end

@testset "Read Fortran lookup tables (3momI)" begin
    table_dir = expanduser("~/Aeolus/P3-microphysics/lookup_tables")
    isdir(table_dir) || error("Fortran tables not found at $table_dir")

    p3 = read_fortran_lookup_tables(table_dir; FT=Float64)

    tables = p3.ice.lookup_tables
    @test tables isa P3LookupTables
    @test tables.table_1 isa P3LookupTable1
    @test tables.table_2 isa P3LookupTable2
    @test tables.table_3 isa P3LookupTable3

    @test p3.ice.fall_speed.mass_weighted isa FortranTabulatedFunction5D
    @test p3.ice.deposition.ventilation isa FortranTabulatedFunction5D
    @test tables.table_2.mass isa FortranTabulatedFunction6D
    @test tables.table_3.shape isa FortranTabulatedFunction3
    @test tables.table_3.slope === nothing

    # Spot-check first data row of 3momI Table 1
    # At first grid point: log_m ≈ LOG_MASS_MIN, Fr=0, Fl=0, rho=50, mu=0
    # uns = 0.29834E-03, ums = 0.28204E-02
    log_m_min = 11 * 0.1 * log10(800) - 18
    uns = p3.ice.fall_speed.number_weighted(log_m_min, 0.0, 0.0, 50.0, 0.0)
    ums = p3.ice.fall_speed.mass_weighted(log_m_min, 0.0, 0.0, 50.0, 0.0)
    @test uns ≈ 0.29834e-03 rtol=1e-3
    @test ums ≈ 0.28204e-02 rtol=1e-3

    # Rain 1D tables should be populated
    @test p3.rain.velocity_mass isa TabulatedFunction
    @test p3.rain.velocity_number isa TabulatedFunction
    @test p3.rain.evaporation isa TabulatedFunction
end
