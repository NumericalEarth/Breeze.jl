include(joinpath(@__DIR__, "setup.jl"))

using Test
using Breeze
using Breeze.Thermodynamics: dry_air_gas_constant, adiabatic_hydrostatic_pressure,
                             mixture_gas_constant, MoistureMassFractions
using Breeze.AtmosphereModels: standard_pressure
using Oceananigans
using Oceananigans.Operators: Δzᶜᶜᶜ
using GPUArraysCore: @allowscalar

@testset "Potential temperature diagnostics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), extent=(100, 100, 1000))
    model = AtmosphereModel(grid)
    set!(model, θ=300, qᵗ=0.01)

    # Test PotentialTemperature (mixture)
    θ = PotentialTemperature(model)
    @test θ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θ_field = Field(θ)
    @test all(isfinite.(interior(θ_field)))
    # Potential temperature should be in a reasonable range
    @test all(interior(θ_field) .> 290)
    @test all(interior(θ_field) .< 310)

    # Test density flavor
    θ_density = PotentialTemperature(model, :density)
    θ_density_field = Field(θ_density)
    @test all(isfinite.(interior(θ_density_field)))
    @test all(interior(θ_density_field) .> 0)

    # Test LiquidIcePotentialTemperature
    θˡⁱ = LiquidIcePotentialTemperature(model)
    @test θˡⁱ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θˡⁱ_field = Field(θˡⁱ)
    @test all(isfinite.(interior(θˡⁱ_field)))
    # Liquid-ice potential temperature should match what we set (θ=300)
    @test all(interior(θˡⁱ_field) .≈ 300)

    # Test VirtualPotentialTemperature
    θᵛ = VirtualPotentialTemperature(model)
    @test θᵛ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θᵛ_field = Field(θᵛ)
    @test all(isfinite.(interior(θᵛ_field)))
    # Virtual potential temperature should be larger than liquid-ice when moisture is present
    @test all(interior(θᵛ_field) .> interior(θˡⁱ_field))

    # Test density flavor
    θᵛ_density = VirtualPotentialTemperature(model, :density)
    θᵛ_density_field = Field(θᵛ_density)
    @test all(isfinite.(interior(θᵛ_density_field)))
    @test all(interior(θᵛ_density_field) .> 0)

    # Test EquivalentPotentialTemperature
    θᵉ = EquivalentPotentialTemperature(model)
    @test θᵉ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θᵉ_field = Field(θᵉ)
    @test all(isfinite.(interior(θᵉ_field)))
    # Equivalent potential temperature should be larger than liquid-ice when moisture is present
    @test all(interior(θᵉ_field) .> interior(θˡⁱ_field))

    # Test density flavor
    θᵉ_density = EquivalentPotentialTemperature(model, :density)
    θᵉ_density_field = Field(θᵉ_density)
    @test all(isfinite.(interior(θᵉ_density_field)))
    @test all(interior(θᵉ_density_field) .> 0)

    # Test StabilityEquivalentPotentialTemperature
    θᵇ = StabilityEquivalentPotentialTemperature(model)
    @test θᵇ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θᵇ_field = Field(θᵇ)
    @test all(isfinite.(interior(θᵇ_field)))
    # Stability-equivalent potential temperature should be ≥ equivalent
    # (equal when no liquid water is present, i.e., qˡ = 0)
    @test all(interior(θᵇ_field) .≥ interior(θᵉ_field))

    # Test density flavor
    θᵇ_density = StabilityEquivalentPotentialTemperature(model, :density)
    θᵇ_density_field = Field(θᵇ_density)
    @test all(isfinite.(interior(θᵇ_density_field)))
    @test all(interior(θᵇ_density_field) .> 0)
end

# Regression test for #659 / PR #656: the definition of virtual potential temperature.
@testset "Virtual potential temperature buoyancy formulation [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    Nz = 8
    grid = RectilinearGrid(default_arch; size=(2, 2, Nz), x=(0, 1_000), y=(0, 1_000), z=(0, 5_000))

    constants = ThermodynamicConstants()
    p₀ = FT(101325)
    θ₀ = FT(300)
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)

    set!(model; θ=θ₀, qᵗ=FT(0.01))

    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    pˢᵗ = standard_pressure(dynamics)

    qᵛ_field = specific_humidity(model)
    T_field = model.temperature
    pᵣ_field = dynamics.reference_state.pressure

    θᵥ_diagnostic = Field(VirtualPotentialTemperature(model))

    @allowscalar for k in 1:Nz
        T_k  = T_field[1, 1, k]
        qᵛ_k = qᵛ_field[1, 1, k]
        pᵣ_k = pᵣ_field[1, 1, k]
        Rᵐ_k = mixture_gas_constant(MoistureMassFractions(qᵛ_k), constants)

        # θᵥ = T (Rᵐ / Rᵈ) (pˢᵗ / pᵣ)^(Rᵈ / cᵖᵈ) — dry exponent
        θᵥ_expected = T_k * (Rᵐ_k / Rᵈ) * (pˢᵗ / pᵣ_k)^(Rᵈ / cᵖᵈ)

        θᵥ_kernel = Breeze.AtmosphereModels.virtual_potential_temperature(
            1, 1, k, grid, constants, dynamics, T_field, qᵛ_field)

        @test θᵥ_kernel ≈ θᵥ_expected rtol = 100eps(FT)
        @test θᵥ_diagnostic[1, 1, k] ≈ θᵥ_expected rtol = 100eps(FT)
    end
end

@testset "Static energy diagnostics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), extent=(100, 100, 1000))
    model = AtmosphereModel(grid)
    set!(model, θ=300, qᵗ=0.01)

    # Test StaticEnergy
    s = StaticEnergy(model)
    @test s isa Oceananigans.AbstractOperations.KernelFunctionOperation
    s_field = Field(s)
    @test all(isfinite.(interior(s_field)))
    @test all(interior(s_field) .> 0)

    # Test density flavor
    s_density = StaticEnergy(model, :density)
    s_density_field = Field(s_density)
    @test all(isfinite.(interior(s_density_field)))
    @test all(interior(s_density_field) .> 0)
end

@testset "Relative humidity diagnostics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), extent=(100, 100, 1000))
    microphysics = SaturationAdjustment()
    model = AtmosphereModel(grid; microphysics)

    # Test with subsaturated conditions (low moisture)
    set!(model, θ=300, qᵗ=0.005)
    RH = RelativeHumidity(model)
    @test RH isa Oceananigans.AbstractOperations.KernelFunctionOperation
    RH_field = Field(RH)
    @test all(isfinite.(interior(RH_field)))
    # Relative humidity should be between 0 and 1 for subsaturated conditions
    @test all(interior(RH_field) .≥ 0)
    @test all(interior(RH_field) .≤ 1)

    # With low moisture, should be subsaturated (RH < 1)
    @test all(interior(RH_field) .< 1)

    # Test with saturated conditions (high moisture)
    set!(model, θ=300, qᵗ=0.03)  # High moisture to ensure saturation
    RH_saturated = RelativeHumidityField(model)
    # For saturated conditions with saturation adjustment, RH should be very close to 1
    # where there is condensate
    qˡ = model.microphysical_fields.qˡ
    @allowscalar begin
        for k in 1:8
            if qˡ[1, 1, k] > 0  # If there's condensate, should be saturated
                @test RH_saturated[1, 1, k] ≈ 1 rtol=FT(1e-3)
            end
        end
    end
end

@testset "Supersaturation diagnostics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), extent=(100, 100, 1000))
    model = AtmosphereModel(grid; microphysics=SaturationAdjustment())

    # Subsaturated air: the supersaturation is negative everywhere, and is exactly one less than
    # the relative humidity
    set!(model, θ=300, qᵗ=0.005)
    𝒮 = Supersaturation(model)
    @test 𝒮 isa Oceananigans.AbstractOperations.AbstractOperation
    𝒮_field = SupersaturationField(model)
    ℋ_field = RelativeHumidityField(model)
    @test all(isfinite, 𝒮_field)
    @test all(<(0), 𝒮_field)
    @test maximum(abs, 𝒮_field - ℋ_field + 1) < eps(FT)

    # Moist enough to condense: saturation adjustment pins the supersaturation to zero wherever it
    # makes condensate, and leaves the rest of the column subsaturated
    set!(model, θ=300, qᵗ=0.03)
    𝒮_saturated = SupersaturationField(model)
    @test maximum(model.microphysical_fields.qˡ) > 0
    @test abs(maximum(𝒮_saturated)) < FT(1e-3)
    @test minimum(𝒮_saturated) < 0
end
