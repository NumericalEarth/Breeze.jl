using Test
using Oceananigans
using Oceananigans.Utils: TabulatedFunction, TabulatedFunction1D
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.Microphysics.PredictedParticleProperties:
    ice_terminal_velocities,
    rain_terminal_velocity_mass_weighted

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

const _fortran_table_dir = expanduser("~/Aeolus/P3-microphysics/lookup_tables")
const _has_fortran_tables = isdir(_fortran_table_dir)

@testset "Read Fortran lookup tables (3momI)" skip=!_has_fortran_tables begin
    p3 = read_fortran_lookup_tables(_fortran_table_dir; FT=Float64)

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

@testset "Read Fortran lookup tables (2momI)" skip=!_has_fortran_tables begin
    p3 = read_fortran_lookup_tables(_fortran_table_dir; three_moment_ice=false)

    tables = p3.ice.lookup_tables
    @test tables.table_1 isa P3LookupTable1
    @test tables.table_2 isa P3LookupTable2
    @test tables.table_3 === nothing

    # Spot-check first row of 2momI: i_rhor=1, i_Fr=1, i_Fl=1, i_Qnorm=1
    # uns = 0.15624E-03, ums = 0.35587E-03
    uns = p3.ice.fall_speed.number_weighted(-14.807, 0.0, 0.0, 50.0, 0.0)
    ums = p3.ice.fall_speed.mass_weighted(-14.807, 0.0, 0.0, 50.0, 0.0)
    @test uns ≈ 0.15624e-03 rtol=1e-3
    @test ums ≈ 0.35587e-03 rtol=1e-3

    # 2momI: reflectivity_weighted and sixth_moment should be nothing
    @test p3.ice.fall_speed.reflectivity_weighted === nothing
    @test p3.ice.sixth_moment.rime === nothing
end

@testset "Rain tables are computed (not from Fortran)" skip=!_has_fortran_tables begin
    p3 = read_fortran_lookup_tables(_fortran_table_dir)

    @test p3.rain.velocity_mass isa TabulatedFunction1D
    @test p3.rain.velocity_number isa TabulatedFunction1D

    log_lambda = 3.5
    @test p3.rain.velocity_mass(log_lambda) > 0
end

@testset "PredictedParticlePropertiesMicrophysics constructor with Fortran tables" skip=!_has_fortran_tables begin
    # Test constructor interface
    p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=_fortran_table_dir)
    @test p3 isa PredictedParticlePropertiesMicrophysics
    @test p3.ice.lookup_tables isa P3LookupTables

    # Test 2momI override
    p3_2mom = PredictedParticlePropertiesMicrophysics(;
        lookup_tables=_fortran_table_dir, three_moment_ice=false)
    @test p3_2mom.ice.lookup_tables.table_3 === nothing
end

@testset "Process rates with Fortran-loaded tables" skip=!_has_fortran_tables begin
    p3 = PredictedParticlePropertiesMicrophysics(; lookup_tables=_fortran_table_dir)

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
    vt_rain = rain_terminal_velocity_mass_weighted(p3, qʳ, nʳ, ρ)
    @test 0 < vt_rain < 20
end
