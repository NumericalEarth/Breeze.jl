using Test
using Breeze
using CloudMicrophysics
using CloudMicrophysics.Microphysics1M: get_n0, lambda_inverse
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using GPUArraysCore: @allowscalar
using Oceananigans

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

#####
##### Number concentration diagnostic
#####

@testset "NumberConcentration: one-moment rain [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    qʳ_value = FT(1e-3)
    set!(model; θ=300, qᵗ=FT(0.020), qᶜˡ=FT(0), qʳ=qʳ_value)

    op = number_concentration(model, :rain)
    @test op isa Oceananigans.AbstractOperations.KernelFunctionOperation

    ρnʳ = Field(op)
    compute!(ρnʳ)
    @test all(isfinite.(interior(ρnʳ)))

    # Compare with n0 · λ⁻¹ computed directly from the scheme's DSD.
    rain = microphysics.categories.rain
    ρ = @allowscalar reference_state.density[1, 1, 1]
    q = qʳ_value
    n0 = get_n0(rain.pdf, q, ρ)
    λ⁻¹ = lambda_inverse(rain.pdf, rain.mass, q, ρ)
    expected = n0 * λ⁻¹

    @test @allowscalar isapprox(ρnʳ[1, 1, 1], expected, rtol=100eps(FT))

    # NumberConcentrationField convenience wrapper.
    ρnʳ_field = NumberConcentrationField(model, :rain)
    @test ρnʳ_field isa Field
    compute!(ρnʳ_field)
    @test @allowscalar isapprox(ρnʳ_field[1, 1, 1], expected, rtol=100eps(FT))
end

@testset "NumberConcentration: one-moment snow (mixed phase) [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    qˢ_value = FT(5e-4)
    set!(model; θ=260, qᵗ=FT(0.005), qˢ=qˢ_value)

    op = number_concentration(model, :snow)
    @test op isa Oceananigans.AbstractOperations.KernelFunctionOperation

    ρnˢ = Field(op)
    compute!(ρnˢ)
    @test all(isfinite.(interior(ρnˢ)))

    # Snow's n₀ depends on (q, ρ), so the closed-form rain expression cannot
    # substitute. Compare against CloudMicrophysics directly.
    snow = microphysics.categories.snow
    ρ = @allowscalar reference_state.density[1, 1, 1]
    q = qˢ_value
    n0 = get_n0(snow.pdf, q, ρ)
    λ⁻¹ = lambda_inverse(snow.pdf, snow.mass, q, ρ)
    expected = n0 * λ⁻¹

    @test @allowscalar isapprox(ρnˢ[1, 1, 1], expected, rtol=100eps(FT))
end

@testset "NumberConcentration: unsupported species returns nothing [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Warm-phase 1-mom carries rain but not snow, hail, or graupel.
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    @test number_concentration(model, :hail) === nothing
    @test number_concentration(model, :graupel) === nothing
    @test number_concentration(model, :snow) === nothing
    @test NumberConcentrationField(model, :hail) === nothing
end

@testset "NumberConcentration: two-moment returns prognostic field [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    set!(model; θ=300, qᵗ=FT(0.015))

    @test number_concentration(model, :rain) === model.microphysical_fields.ρnʳ
    @test number_concentration(model, :cloud_liquid) === model.microphysical_fields.ρnᶜˡ
    @test number_concentration(model, :snow) === nothing
end

@testset "NumberConcentration: SaturationAdjustment errors [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; microphysics)

    @test_throws ErrorException number_concentration(model, :rain)
    @test_throws ErrorException NumberConcentrationField(model, :rain)
end
