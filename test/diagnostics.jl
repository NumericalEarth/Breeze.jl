using Test
using Breeze
using Breeze.Thermodynamics: dry_air_gas_constant, adiabatic_hydrostatic_pressure
using Oceananigans
using Oceananigans.Operators: Δzᶜᶜᶜ
using GPUArraysCore: @allowscalar

@testset "Potential temperature diagnostics [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 8), extent=(100, 100, 1000))
    model = AtmosphereModel(grid)
    set!(model, θ=300, qᵗ=0.01)

    # Test DryPotentialTemperature
    θ = DryPotentialTemperature(model)
    @test θ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θ_field = Field(θ)
    @test all(isfinite.(interior(θ_field)))
    # Dry potential temperature should be in a reasonable range
    @test all(interior(θ_field) .> 290)
    @test all(interior(θ_field) .< 310)

    # Test VirtualPotentialTemperature
    θᵛ = VirtualPotentialTemperature(model)
    @test θᵛ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θᵛ_field = Field(θᵛ)
    @test all(isfinite.(interior(θᵛ_field)))
    # Virtual potential temperature should be larger than dry when moisture is present
    @test all(interior(θᵛ_field) .> interior(θ_field))

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
    # Equivalent potential temperature should be larger than dry when moisture is present
    @test all(interior(θᵉ_field) .> interior(θ_field))

    # Test density flavor
    θᵉ_density = EquivalentPotentialTemperature(model, :density)
    θᵉ_density_field = Field(θᵉ_density)
    @test all(isfinite.(interior(θᵉ_density_field)))
    @test all(interior(θᵉ_density_field) .> 0)

    # Test LiquidIcePotentialTemperature
    θˡⁱ = LiquidIcePotentialTemperature(model)
    @test θˡⁱ isa Oceananigans.AbstractOperations.KernelFunctionOperation
    θˡⁱ_field = Field(θˡⁱ)
    @test all(isfinite.(interior(θˡⁱ_field)))
    # Liquid-ice potential temperature should match what we set (θ=300)
    @test all(interior(θˡⁱ_field) .≈ 300)
end

@testset "Static energy diagnostics [$(FT)]" for FT in (Float32, Float64)
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

@testset "Hydrostatic pressure computation [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 20), x=(0, 1000), y=(0, 1000), z=(0, 10000))
    constants = ThermodynamicConstants()

    p₀, θ₀ = 101325, 288
    reference_state = ReferenceState(grid, constants, base_pressure=p₀, potential_temperature=θ₀)
    formulation = AnelasticFormulation(reference_state)
    model = AtmosphereModel(grid; thermodynamic_constants=constants, formulation)

    # Set up isothermal atmosphere: T = T₀ = constant
    # For constant T, we need: θ = T₀ * (p₀/pᵣ)^(Rᵈ/cᵖᵈ)
    T₀ = θ₀
    Rᵈ = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    g = constants.gravitational_acceleration

    θ_field = CenterField(grid)
    set!(θ_field, (x, y, z) -> begin
        pᵣ_z = adiabatic_hydrostatic_pressure(z, p₀, θ₀, constants)
        T₀ * (p₀ / pᵣ_z)^(Rᵈ / cᵖᵈ)
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

