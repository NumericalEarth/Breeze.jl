using Test
using Adapt: Adapt
using Oceananigans.Architectures: CPU
using Oceananigans.Utils: TabulatedFunction1D, TabulatedFunction4D
import Oceananigans.Architectures: on_architecture
import Breeze
using Breeze.Microphysics.PredictedParticleProperties
using Breeze.Microphysics.PredictedParticleProperties:
    IceProperties, NullP3LookupTables, P3LookupTable1, P3LookupTable2,
    P3LookupTable3, P3LookupTables, P3TabulationParameters, LookupTable1Parameters,
    LookupTable2Parameters, LookupTable3Parameters

struct MockArchitecture end
struct TransferArchitecture end
struct ShiftArrays end

on_architecture(::MockArchitecture, a::Array) = a
on_architecture(::TransferArchitecture, a::Array) = a .+ 1
Adapt.adapt_storage(::ShiftArrays, a::Array) = a .+ 1

@testset "Tabulation entry points still materialize P3 tables" begin
    p3 = PredictedParticlePropertiesMicrophysics()
    p3_tab = tabulate(p3, CPU();
        number_of_mass_points = 4,
        number_of_rime_fraction_points = 2,
        number_of_liquid_fraction_points = 2,
        number_of_rime_density_points = 3,
        number_of_quadrature_points = 12)

    @test p3_tab.ice.lookup_tables isa P3LookupTables
    @test p3_tab.ice.lookup_tables.table_1 isa P3LookupTable1
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
    params = P3TabulationParameters(Float64;
        number_of_mass_points = 4,
        shape_parameter_override = 3)
    @test params.lookup_table_1.number_of_mass_points == 4
    @test params.lookup_table_1.shape_parameter_override == 3
end

@testset "IceProperties reconstruction preserves lookup_tables" begin
    lookup_tables = P3LookupTables(
        P3LookupTable1([1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]),
        P3LookupTable2([8.0], [9.0], [10.0]),
        P3LookupTable3([11.0], [12.0], [13.0]))
    ice = IceProperties(Float64; lookup_tables)

    transferred = on_architecture(TransferArchitecture(), ice)
    adapted = Adapt.adapt(ShiftArrays(), ice)

    @test transferred.lookup_tables.table_1.fall_speed == [2.0]
    @test transferred.lookup_tables.table_2.mass == [9.0]
    @test transferred.lookup_tables.table_3.shape == [12.0]

    @test adapted.lookup_tables.table_1.fall_speed == [2.0]
    @test adapted.lookup_tables.table_2.mass == [9.0]
    @test adapted.lookup_tables.table_3.shape == [12.0]
end

using Breeze.Microphysics.PredictedParticleProperties:
    ThreeMomentClosure, solve_shape_parameter, shape_parameter_lookup, slope_parameter_lookup

@testset "lookupTable_3 reproduces three-moment closure reference" begin
    p3 = tabulate(PredictedParticlePropertiesMicrophysics(), CPU();
        number_of_mass_points = 8,
        number_of_rime_fraction_points = 4,
        number_of_liquid_fraction_points = 4,
        number_of_rime_density_points = 5,
        number_of_quadrature_points = 24)

    table3 = p3.ice.lookup_tables.table_3
    L_ice = 1e-5
    N_ice = 2e4
    Z_ice = 5e-14
    Fᶠ = 0.35
    Fˡ = 0.25
    ρᶠ = 450.0

    μ_lookup = shape_parameter_lookup(table3, L_ice, N_ice, Z_ice, Fᶠ, Fˡ, ρᶠ)
    λ_lookup = slope_parameter_lookup(table3, L_ice, N_ice, Z_ice, Fᶠ, Fˡ, ρᶠ)
    μ_ref = solve_shape_parameter(L_ice, N_ice, Z_ice, Fᶠ, ρᶠ; closure = ThreeMomentClosure())

    @test μ_lookup ≈ μ_ref rtol = 5e-2
    @test λ_lookup > 0
end

using Oceananigans.Utils: TabulatedFunction5D
using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState, PhasePartition, ThermodynamicConstants
using Breeze.Microphysics.PredictedParticleProperties: P3MicrophysicalState, compute_p3_process_rates

@testset "lookupTable_1 and lookupTable_2 missing families are materialized" begin
    p3 = tabulate(PredictedParticlePropertiesMicrophysics(), CPU();
        number_of_mass_points = 8,
        number_of_rime_fraction_points = 4,
        number_of_liquid_fraction_points = 4,
        number_of_rime_density_points = 5,
        number_of_quadrature_points = 24)

    @test p3.ice.lookup_tables.table_3.shape isa TabulatedFunction5D
    @test p3.ice.lookup_tables.table_2.mass isa TabulatedFunction5D
    @test p3.ice.lookup_tables.table_2.number isa TabulatedFunction5D

    FT = Float64
    q = Breeze.Thermodynamics.MoistureMassFractions(FT(0.010), FT(0), FT(0))
    𝒰 = Breeze.Thermodynamics.LiquidIcePotentialTemperatureState(FT(263), q, FT(100000), FT(85000))
    ℳ = P3MicrophysicalState(FT(0), FT(1e-5), FT(1e4), FT(1e-4), FT(1e5), FT(2e-5), FT(5e-8), FT(1e-10), FT(0))
    rates = compute_p3_process_rates(
        p3,
        FT(1.0),
        ℳ,
        𝒰,
        Breeze.Thermodynamics.ThermodynamicConstants())

    @test all(isfinite(getfield(rates, name)) for name in fieldnames(typeof(rates)))
end
