using Test
using Breeze
using Breeze.Thermodynamics: dry_air_gas_constant, adiabatic_hydrostatic_pressure
using Oceananigans
using Oceananigans.Operators: Δzᶜᶜᶜ
using GPUArraysCore: @allowscalar

include("test_utils.jl")

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
    # Stability-equivalent potential temperature should be >= equivalent
    # (equal when no liquid water is present, i.e., qˡ = 0)
    @test all(interior(θᵇ_field) .>= interior(θᵉ_field))

    # Test density flavor
    θᵇ_density = StabilityEquivalentPotentialTemperature(model, :density)
    θᵇ_density_field = Field(θᵇ_density)
    @test all(isfinite.(interior(θᵇ_density_field)))
    @test all(interior(θᵇ_density_field) .> 0)
end

@testset "Static energy diagnostics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), extent=(100, 100, 1000))
    model = AtmosphereModel(grid)
    set!(model, θ=300, qᵗ=0.01)

    # Test StaticEnergy
    e = StaticEnergy(model)
    @test e isa Oceananigans.AbstractOperations.KernelFunctionOperation
    e_field = Field(e)
    @test all(isfinite.(interior(e_field)))
    @test all(interior(e_field) .> 0)

    # Test density flavor
    e_density = StaticEnergy(model, :density)
    e_density_field = Field(e_density)
    @test all(isfinite.(interior(e_density_field)))
    @test all(interior(e_density_field) .> 0)
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
    @test all(interior(RH_field) .>= 0)
    @test all(interior(RH_field) .<= 1)

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

@testset "Hydrostatic pressure computation [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 20), x=(0, 1000), y=(0, 1000), z=(0, 10000))
    constants = ThermodynamicConstants()

    p₀ = FT(101325) # surface pressure, Pa
    pˢᵗ = FT(1e5) # standard pressure for potential temperature, Pa
    θ₀ = 288 # K
    reference_state = ReferenceState(grid, constants, surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)

    # Set up isothermal atmosphere: T = T₀ = constant
    # For constant T, we need: θ = T₀ * (pˢᵗ/pᵣ)^(Rᵈ/cᵖᵈ) using the standard pressure
    T₀ = θ₀
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    θ_field = CenterField(grid)
    set!(θ_field, (x, y, z) -> begin
        pᵣ_z = adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants)
        T₀ * (pˢᵗ / pᵣ_z)^(Rᵈ / cᵖᵈ)
    end)

    set!(model; θ = θ_field)

    # Verify temperature is approximately constant
    T_interior = interior(model.temperature)
    max_rel_error = @allowscalar maximum(abs.((T_interior .- T₀) ./ T₀))
    @test max_rel_error < FT(1e-5)

    # Compute hydrostatic pressure
    ph = Breeze.AtmosphereModels.compute_hydrostatic_pressure!(CenterField(grid), model)

    # Expected cell-mean pressure for isothermal atmosphere:
    # p_mean = p_interface_bottom * (H / Δz) * (1 - exp(-Δz / H))
    # where H = Rᵈ * T₀ / g is the scale height
    p_expected = CenterField(grid)
    H = Rᵈ * T₀ / g

    @allowscalar begin
        p_interface_bottom = p₀
        for k in 1:grid.Nz
            Δz = Δzᶜᶜᶜ(1, 1, k, grid)
            p_expected[1, 1, k] = p_interface_bottom * (H / Δz) * (1 - exp(-Δz / H))
            p_interface_bottom = exp(-Δz / H) * p_interface_bottom
        end
    end

    @test ph ≈ p_expected
end

