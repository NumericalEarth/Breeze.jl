include(joinpath(@__DIR__, "setup.jl"))

using Test
using Pkg.Artifacts: ensure_artifact_installed
using Oceananigans
using Oceananigans.Utils: TabulatedFunction, TabulatedFunction1D
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.Microphysics.PredictedParticleProperties:
    evaluate_at,
    ice_terminal_velocities,
    prepare_interpolation,
    rain_terminal_velocities

@testset "Rime-density-indexed table transforms" begin
    # Tables that return whichever argument the rime-density index occupies: the last of
    # four for the ice tables, the last of five for the ice-rain tables. The wrappers
    # place the index at different slots, so both need exercising.
    unit_axis = (0.0, 1.0)
    index_axis = (1.0, 5.0)
    table_4d = TabulatedFunction((x1, x2, x3, x4) -> Float64(x4), CPU(), Float64;
                                 range=(unit_axis, unit_axis, unit_axis, index_axis),
                                 points=(2, 2, 2, 5))
    table_5d = TabulatedFunction((x1, x2, x3, x4, x5) -> Float64(x5), CPU(), Float64;
                                 range=(unit_axis, unit_axis, unit_axis, unit_axis,
                                        index_axis),
                                 points=(2, 2, 2, 2, 5))
    wrapped_4d = RimeDensityIndexedTable4D(table_4d)
    wrapped_5d = RimeDensityIndexedTable5D(table_5d)

    # The tabulated rime densities {50, 250, 450, 650, 900} kg/m³ map onto their indices
    # 1..5, and 150 kg/m³ lands halfway between the first two.
    for (rime_density, index) in ((50.0, 1.0), (250.0, 2.0), (450.0, 3.0),
                                 (650.0, 4.0), (900.0, 5.0), (150.0, 1.5))
        @test wrapped_4d(0.5, 0.5, 0.5, rime_density) ≈ index
        @test wrapped_5d(0.5, 0.5, 0.5, 0.5, rime_density) ≈ index
    end

    # Process code reads these tables through the prepared-index path, which applies the
    # transform on its own rather than by way of the call operator above.
    prepared_4d = prepare_interpolation(wrapped_4d, 0.5, 0.5, 0.5, 150.0)
    prepared_5d = prepare_interpolation(wrapped_5d, 0.5, 0.5, 0.5, 0.5, 150.0)
    @test evaluate_at(wrapped_4d, prepared_4d) ≈ 1.5
    @test evaluate_at(wrapped_5d, prepared_5d) ≈ 1.5
end

const _lookup_table_dir = ensure_artifact_installed("P3_lookup_tables", joinpath(dirname(@__DIR__), "Artifacts.toml"))

@testset "Read P3 lookup tables (2momI)" begin
    p3 = read_lookup_tables(_lookup_table_dir; FT=Float64)

    @test p3.ice.lambda_limiter isa IceLambdaLimiter
    @test p3.ice.ice_rain isa IceRainCollection
    @test p3.process_rates.liquid_water_density == 1000

    @test p3.ice.fall_speed.mass_weighted isa RimeDensityIndexedTable4D
    @test p3.ice.deposition.ventilation isa RimeDensityIndexedTable4D
    @test p3.ice.ice_rain.mass isa RimeDensityIndexedTable5D
    @test size(p3.ice.fall_speed.mass_weighted.table.table) == (50, 4, 4, 5)
    @test size(p3.ice.ice_rain.mass.table.table) == (50, 30, 4, 4, 5)

    # Spot-check the first row of 2momI (i_rhor=1, i_Fr=1, i_Fl=1, i_Qnorm=1), where the
    # table gives uns = 0.15624E-03 and ums = 0.35587E-03. Every coordinate below sits on
    # its axis origin -- ρᶠ = 50 kg/m³ is index 1 -- so the interpolation weights are 1 and
    # 0 and the stored entry comes back unrounded. Reading the origin off the axis rather
    # than writing a number below it keeps that true if the tabulated range moves.
    log_m_first = p3.ice.fall_speed.number_weighted.table.range[1][1]
    uns = p3.ice.fall_speed.number_weighted(log_m_first, 0.0, 0.0, 50.0)
    ums = p3.ice.fall_speed.mass_weighted(log_m_first, 0.0, 0.0, 50.0)
    @test uns == 0.15624e-03
    @test ums == 0.35587e-03

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

# The rain quadrature runs before a scheme exists, so `read_lookup_tables` hands it the
# resolved `process_rates.floors`. Nothing else would catch that handoff being dropped:
# at the default divisor (1e-30) the floor cannot bind, so the tables would look correct.
@testset "Configured floors reach the rain tabulation" begin
    FT = Float64

    # The mass-weighted denominator is ∫D³exp(-λD)dD ≈ 6/λ⁴, about 6e-22 at the top of
    # the tabulated range. A divisor above that binds; the default cannot. The resulting
    # velocities are unphysical on purpose: this asserts plumbing, not physics.
    raised = NumericalFloors(FT; divisor = 1e-15)
    raised_p3 = PredictedParticlePropertiesMicrophysics(
        FT; process_rates = ProcessRateParameters(FT; floors = raised))
    default_p3 = PredictedParticlePropertiesMicrophysics(FT)

    @test raised_p3.process_rates.floors.divisor == FT(1e-15)

    # Low λ: the integral is far above either floor, so the tables must agree.
    @test raised_p3.rain.velocity_mass(FT(2.5)) == default_p3.rain.velocity_mass(FT(2.5))

    # High λ: the mass-weighted denominator (≈ 6/λ⁴ ≈ 6e-22 here) falls below the raised
    # floor, so that velocity has to drop.
    @test raised_p3.rain.velocity_mass(FT(5.5)) < default_p3.rain.velocity_mass(FT(5.5))

    # The number-weighted denominator is ∫exp(-λD)dD ≈ 1/λ ≈ 3e-6, still far above the
    # raised floor, so it must be untouched. Asserting the asymmetry catches a floor
    # applied where it does not belong.
    @test raised_p3.rain.velocity_number(FT(5.5)) == default_p3.rain.velocity_number(FT(5.5))
end

@testset "PredictedParticlePropertiesMicrophysics constructor with lookup tables" begin
    # Test constructor interface
    p3 = PredictedParticlePropertiesMicrophysics()
    @test p3 isa PredictedParticlePropertiesMicrophysics
    @test p3.ice.ice_rain.number isa RimeDensityIndexedTable5D
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
