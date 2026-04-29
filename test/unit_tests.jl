#####
##### Consolidated unit tests for fast-running tests
#####
##### These tests verify basic construction and simple functionality.
##### They are grouped together to reduce compilation overhead.
#####

using Breeze
using Oceananigans
using Test

#####
##### AnelasticDynamics
#####

using Breeze: ReferenceState, AnelasticDynamics
using Breeze.AtmosphereModels: materialize_dynamics, default_dynamics
using Breeze.AtmosphereModels: mean_pressure, pressure_anomaly, total_pressure
using Breeze.AtmosphereModels: dynamics_density, dynamics_pressure

@testset "AnelasticDynamics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))
    constants = ThermodynamicConstants()

    @testset "Constructor with ReferenceState" begin
        reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        dynamics = AnelasticDynamics(reference_state)

        @test dynamics isa AnelasticDynamics
        @test dynamics.reference_state === reference_state
        @test dynamics.pressure_anomaly === nothing  # Not materialized yet
    end

    @testset "default_dynamics" begin
        dynamics = default_dynamics(grid, constants)

        @test dynamics isa AnelasticDynamics
        @test dynamics.reference_state isa ReferenceState
        @test dynamics.pressure_anomaly === nothing
    end

    @testset "materialize_dynamics" begin
        reference_state = ReferenceState(grid, constants)
        dynamics_stub = AnelasticDynamics(reference_state)
        boundary_conditions = NamedTuple()

        dynamics = materialize_dynamics(dynamics_stub, grid, boundary_conditions, constants)

        @test dynamics isa AnelasticDynamics
        @test dynamics.reference_state === reference_state
        @test dynamics.pressure_anomaly isa Field  # Now materialized
    end

    @testset "Pressure utilities" begin
        reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
        dynamics_stub = AnelasticDynamics(reference_state)
        dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple(), constants)

        # Test mean_pressure
        p̄ = mean_pressure(dynamics)
        @test p̄ === reference_state.pressure

        # Test pressure_anomaly (returns an AbstractOperation)
        p′ = pressure_anomaly(dynamics)
        @test p′ isa Oceananigans.AbstractOperations.AbstractOperation

        # Test total_pressure (returns an AbstractOperation)
        p = total_pressure(dynamics)
        @test p isa Oceananigans.AbstractOperations.AbstractOperation
    end
end

#####
##### CompressibleDynamics
#####

using Breeze: CompressibleDynamics

@testset "CompressibleDynamics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    @testset "Constructor" begin
        dynamics = CompressibleDynamics()
        @test dynamics isa CompressibleDynamics
        @test dynamics.density === nothing  # Not materialized yet
        @test dynamics.standard_pressure == 1e5
        @test dynamics.surface_pressure == 101325
    end

    @testset "materialize_dynamics" begin
        dynamics_stub = CompressibleDynamics()
        constants = ThermodynamicConstants()
        dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple(), constants)

        @test dynamics isa CompressibleDynamics
        @test dynamics.density isa Field
        @test dynamics.pressure isa Field
        @test dynamics_density(dynamics) === dynamics.density
        @test dynamics_pressure(dynamics) === dynamics.pressure
    end
end

#####
##### ThermodynamicFormulations
#####

using Breeze: StaticEnergyFormulation, LiquidIcePotentialTemperatureFormulation
using Breeze.AtmosphereModels: materialize_formulation
using Breeze.AtmosphereModels: prognostic_thermodynamic_field_names
using Breeze.AtmosphereModels: additional_thermodynamic_field_names
using Breeze.AtmosphereModels: thermodynamic_density_name, thermodynamic_density

@testset "ThermodynamicFormulations [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)
    dynamics_stub = AnelasticDynamics(reference_state)
    dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple(), constants)

    # Boundary conditions needed for materialization (must pass grid to respect topology)
    ccc = (Center(), Center(), Center())
    boundary_conditions = (ρθ = FieldBoundaryConditions(grid, ccc), ρe = FieldBoundaryConditions(grid, ccc))

    @testset "LiquidIcePotentialTemperature field naming (Symbol)" begin
        @test prognostic_thermodynamic_field_names(:LiquidIcePotentialTemperature) == (:ρθ,)
        @test additional_thermodynamic_field_names(:LiquidIcePotentialTemperature) == (:θ,)
        @test thermodynamic_density_name(:LiquidIcePotentialTemperature) == :ρθ
    end

    @testset "StaticEnergy field naming (Symbol)" begin
        @test prognostic_thermodynamic_field_names(:StaticEnergy) == (:ρe,)
        @test additional_thermodynamic_field_names(:StaticEnergy) == (:e,)
        @test thermodynamic_density_name(:StaticEnergy) == :ρe
    end

    @testset "materialize_formulation(:LiquidIcePotentialTemperature)" begin
        formulation = materialize_formulation(:LiquidIcePotentialTemperature, dynamics, grid, boundary_conditions)

        @test formulation isa LiquidIcePotentialTemperatureFormulation
        @test formulation.potential_temperature_density isa Field
        @test formulation.potential_temperature isa Field

        # Test struct methods
        @test prognostic_thermodynamic_field_names(formulation) == (:ρθ,)
        @test additional_thermodynamic_field_names(formulation) == (:θ,)
        @test thermodynamic_density_name(formulation) == :ρθ
        @test thermodynamic_density(formulation) === formulation.potential_temperature_density
    end

    @testset "materialize_formulation(:StaticEnergy)" begin
        formulation = materialize_formulation(:StaticEnergy, dynamics, grid, boundary_conditions)

        @test formulation isa StaticEnergyFormulation
        @test formulation.energy_density isa Field
        @test formulation.specific_energy isa Field

        # Test struct methods
        @test prognostic_thermodynamic_field_names(formulation) == (:ρe,)
        @test additional_thermodynamic_field_names(formulation) == (:e,)
        @test thermodynamic_density_name(formulation) == :ρe
        @test thermodynamic_density(formulation) === formulation.energy_density
    end

    @testset "Oceananigans.fields and prognostic_fields" begin
        θ_formulation = materialize_formulation(:LiquidIcePotentialTemperature, dynamics, grid, boundary_conditions)
        e_formulation = materialize_formulation(:StaticEnergy, dynamics, grid, boundary_conditions)

        # LiquidIcePotentialTemperature
        @test haskey(Oceananigans.fields(θ_formulation), :θ)
        @test haskey(Oceananigans.prognostic_fields(θ_formulation), :ρθ)

        # StaticEnergy
        @test haskey(Oceananigans.fields(e_formulation), :e)
        @test haskey(Oceananigans.prognostic_fields(e_formulation), :ρe)
    end
end

#####
##### BulkMicrophysics construction
#####

@testset "BulkMicrophysics construction [$(FT)]" for FT in test_float_types()
    # Test default construction
    bμp_default = BulkMicrophysics(FT)
    @test bμp_default.cloud_formation isa SaturationAdjustment
    @test bμp_default.categories === nothing
    @test bμp_default isa BulkMicrophysics{<:SaturationAdjustment, Nothing}

    # Test construction with explicit clouds scheme
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    bμp_warm = BulkMicrophysics(; cloud_formation)
    @test bμp_warm.cloud_formation === cloud_formation
    @test bμp_warm.categories === nothing

    # Test construction with mixed-phase equilibrium
    cloud_formation_mixed = SaturationAdjustment(; equilibrium=MixedPhaseEquilibrium(FT))
    bμp_mixed = BulkMicrophysics(; cloud_formation=cloud_formation_mixed)
    @test bμp_mixed.cloud_formation === cloud_formation_mixed
    @test bμp_mixed.categories === nothing
end

#####
##### Basic thermodynamics
#####

using Breeze.Thermodynamics:
    MoistureMassFractions,
    StaticEnergyState,
    temperature,
    mixture_heat_capacity,
    temperature_from_potential_temperature,
    potential_temperature_from_temperature

@testset "Thermodynamics" begin
    thermo = ThermodynamicConstants()

    # Test Saturation specific humidity calculation
    T = 293.15  # 20°C
    ρ = 1.2     # kg/m³
    q★ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, thermo.liquid)
    @test q★ > 0
end

@testset "StaticEnergyState [$(FT)]" for FT in test_float_types()
    T = FT(253.15)
    p = FT(101325)
    z = FT(1000)
    thermo = ThermodynamicConstants(FT)

    # Reduced parameter sweep for faster testing (was 6×7×7 = 294, now 3×3×3 = 27)
    for qᵛ in (5e-3, 1.5e-2, 3e-2), qˡ in (0, 1.5e-4, 3e-4), qⁱ in (0, 1.5e-4, 3e-4)
        qᵛ = convert(FT, qᵛ)
        qˡ = convert(FT, qˡ)
        qⁱ = convert(FT, qⁱ)
        q = MoistureMassFractions(qᵛ, qˡ, qⁱ)
        cᵖᵐ = mixture_heat_capacity(q, thermo)
        g = thermo.gravitational_acceleration
        ℒˡᵣ = thermo.liquid.reference_latent_heat
        ℒⁱᵣ = thermo.ice.reference_latent_heat
        e = cᵖᵐ * T + g * z - ℒˡᵣ * qˡ - ℒⁱᵣ * qⁱ

        # Test with saturation adjustment
        𝒰 = StaticEnergyState(e, q, z, p)
        T★ = temperature(𝒰, thermo)
        @test T★ ≈ T
    end
end

@testset "Potential temperature convenience functions [$(FT)]" for FT in test_float_types()
    thermo = ThermodynamicConstants(FT)
    p = FT(101325)
    pˢᵗ = FT(1e5)
    T = FT(290)

    θ = potential_temperature_from_temperature(T, p, pˢᵗ, thermo)
    θ_default = potential_temperature_from_temperature(T, p, thermo)
    θ_integer_temperature = potential_temperature_from_temperature(290, p, pˢᵗ, thermo)

    @test θ != T
    @test temperature_from_potential_temperature(θ, p, pˢᵗ, thermo) ≈ T
    @test θ_default isa FT
    @test temperature_from_potential_temperature(θ_default, p, thermo) isa FT
    @test temperature_from_potential_temperature(θ_default, p, thermo) ≈ T
    @test θ_integer_temperature isa FT
    @test θ_integer_temperature ≈ θ
end

#####
##### Saturation vapor pressure
#####

using Breeze.Thermodynamics:
    TetensFormula,
    saturation_vapor_pressure,
    PlanarLiquidSurface,
    PlanarIceSurface,
    PlanarMixedPhaseSurface,
    absolute_zero_latent_heat,
    specific_heat_difference,
    vapor_gas_constant

function reference_mixed_surface_pressure(T, thermo, λ)
    ℒˡ₀ = absolute_zero_latent_heat(thermo, thermo.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(thermo, thermo.ice)
    Δcˡ = specific_heat_difference(thermo, thermo.liquid)
    Δcⁱ = specific_heat_difference(thermo, thermo.ice)

    ℒ₀ = λ * ℒˡ₀ + (one(λ) - λ) * ℒⁱ₀
    Δcᵝ = λ * Δcˡ + (one(λ) - λ) * Δcⁱ

    Tᵗʳ = thermo.triple_point_temperature
    pᵗʳ = thermo.triple_point_pressure
    Rᵛ = vapor_gas_constant(thermo)

    return pᵗʳ * (T / Tᵗʳ)^(Δcᵝ / Rᵛ) * exp((one(T) / Tᵗʳ - one(T) / T) * ℒ₀ / Rᵛ)
end

@testset "Saturation vapor pressure surfaces [$FT]" for FT in test_float_types()
    thermo = ThermodynamicConstants(FT)
    Tᵗʳ = thermo.triple_point_temperature
    temperatures = FT.((Tᵗʳ * FT(0.9), Tᵗʳ, Tᵗʳ * FT(1.1)))

    liquid_surface = PlanarLiquidSurface()
    ice_surface = PlanarIceSurface()
    rtol = FT === Float64 ? 1e-12 : FT(1e-5)

    @testset "Planar homogeneous surfaces" begin
        for T in temperatures
            pˡ = saturation_vapor_pressure(T, thermo, thermo.liquid)
            pⁱ = saturation_vapor_pressure(T, thermo, thermo.ice)

            @test saturation_vapor_pressure(T, thermo, liquid_surface) ≈ pˡ rtol=rtol
            @test saturation_vapor_pressure(T, thermo, ice_surface) ≈ pⁱ rtol=rtol
        end
    end

    @testset "Planar mixed-phase surfaces" begin
        for λ in (zero(FT), FT(0.5), one(FT))  # Reduced from 5 to 3 values
            surface = PlanarMixedPhaseSurface(λ)
            for T in temperatures
                p_surface = saturation_vapor_pressure(T, thermo, surface)
                p_reference = reference_mixed_surface_pressure(T, thermo, λ)

                @test p_surface ≈ p_reference rtol=rtol
            end
        end
    end
end

@testset "Tetens formula saturation vapor pressure [$FT]" for FT in test_float_types()
    tetens = TetensFormula()
    thermo = ThermodynamicConstants(; saturation_vapor_pressure=tetens)
    rtol = FT === Float64 ? eps(FT) : FT(1e-5)

    # Test at reference temperature (273.15 K): should return reference pressure
    Tᵣ = FT(273.15)
    pᵛ⁺ᵣ= saturation_vapor_pressure(Tᵣ, thermo, PlanarLiquidSurface())
    @test pᵛ⁺ᵣ≈ FT(610) rtol=rtol

    # Test monotonicity: pressure increases with temperature (liquid)
    T_warm = FT(300)
    T_cold = FT(250)
    pᵛ⁺_warm = saturation_vapor_pressure(T_warm, thermo, PlanarLiquidSurface())
    pᵛ⁺_cold = saturation_vapor_pressure(T_cold, thermo, PlanarLiquidSurface())
    @test pᵛ⁺_warm > pᵛ⁺ᵣ> pᵛ⁺_cold

    # Test ice surface at reference temperature
    pⁱ_ref = saturation_vapor_pressure(Tᵣ, thermo, PlanarIceSurface())
    @test pⁱ_ref ≈ FT(610) rtol=rtol

    # Test monotonicity for ice
    pⁱ_warm = saturation_vapor_pressure(T_warm, thermo, PlanarIceSurface())
    pⁱ_cold = saturation_vapor_pressure(T_cold, thermo, PlanarIceSurface())
    @test pⁱ_warm > pⁱ_ref > pⁱ_cold

    # Verify analytic expressions for liquid
    pᵣ = FT(610)
    aˡ = FT(17.27)
    δTˡ = FT(35.85)
    T_test = FT(288)
    expected_liquid = pᵣ * exp(aˡ * (T_test - Tᵣ) / (T_test - δTˡ))
    @test saturation_vapor_pressure(T_test, thermo, PlanarLiquidSurface()) ≈ expected_liquid rtol=rtol

    # Verify analytic expressions for ice
    aⁱ = FT(21.875)
    δTⁱ = FT(7.65)
    expected_ice = pᵣ * exp(aⁱ * (T_test - Tᵣ) / (T_test - δTⁱ))
    @test saturation_vapor_pressure(T_test, thermo, PlanarIceSurface()) ≈ expected_ice rtol=rtol

    # Test mixed-phase surface: linear interpolation between liquid and ice
    for λ in (FT(0), FT(0.5), FT(1))
        surface = PlanarMixedPhaseSurface(λ)
        pˡ = saturation_vapor_pressure(T_test, thermo, PlanarLiquidSurface())
        pⁱ = saturation_vapor_pressure(T_test, thermo, PlanarIceSurface())
        expected_mixed = λ * pˡ + (1 - λ) * pⁱ
        @test saturation_vapor_pressure(T_test, thermo, surface) ≈ expected_mixed rtol=rtol
    end
end

@testset "Tetens vs Clausius-Clapeyron comparison [$FT]" for FT in test_float_types()
    tetens = TetensFormula(FT)
    thermo_tetens = ThermodynamicConstants(FT; saturation_vapor_pressure=tetens)
    thermo_cc = ThermodynamicConstants(FT) # Default is Clausius-Clapeyron

    # Both formulas should agree reasonably well in the typical atmospheric range
    temperatures = FT.((260, 285, 300))  # Reduced from 4 to 3 temperatures

    for T in temperatures
        pˡ_tetens = saturation_vapor_pressure(T, thermo_tetens, PlanarLiquidSurface())
        pˡ_cc = saturation_vapor_pressure(T, thermo_cc, PlanarLiquidSurface())
        @test pˡ_tetens ≈ pˡ_cc rtol=FT(0.05)

        pⁱ_tetens = saturation_vapor_pressure(T, thermo_tetens, PlanarIceSurface())
        pⁱ_cc = saturation_vapor_pressure(T, thermo_cc, PlanarIceSurface())
        @test pⁱ_tetens ≈ pⁱ_cc rtol=FT(0.05)
    end
end

#####
##### BackgroundAtmosphere
#####

using Breeze.AtmosphereModels: BackgroundAtmosphere,
                               materialize_background_atmosphere,
                               radiation_flux_divergence,
                               _vmr_string

@testset "BackgroundAtmosphere" begin
    @testset "Default constructor" begin
        atm = BackgroundAtmosphere()
        @test atm.N₂ ≈ 0.78084
        @test atm.O₂ ≈ 0.20946
        @test atm.CO₂ ≈ 420e-6
        @test atm.CH₄ ≈ 1.8e-6
        @test atm.N₂O ≈ 330e-9
        @test atm.O₃ == 0.0
        @test atm.CFC₁₁ == 0.0
    end

    @testset "Custom constructor" begin
        atm = BackgroundAtmosphere(CO₂ = 400e-6, O₃ = 30e-9)
        @test atm.CO₂ ≈ 400e-6
        @test atm.O₃ ≈ 30e-9
        @test atm.N₂ ≈ 0.78084  # default preserved
    end

    @testset "Function-based O₃" begin
        ozone(z) = 30e-9 * (1 + z / 10000)
        atm = BackgroundAtmosphere(O₃ = ozone)
        @test atm.O₃ === ozone
    end

    @testset "_vmr_string" begin
        @test _vmr_string(0.0) === nothing
        @test _vmr_string(0.78084) == "0.78084"
        @test _vmr_string(420e-6) == "420.0 ppm"
        @test _vmr_string(330e-9) == "330.0 ppb"
        @test _vmr_string(1e-12) == "1.0e-12"
        # Non-number fallback
        f(z) = z
        @test _vmr_string(f) isa String
    end

    @testset "show method" begin
        atm = BackgroundAtmosphere(CO₂ = 400e-6, CH₄ = 1.8e-6, O₃ = 0.0)
        s = sprint(show, atm)
        @test occursin("BackgroundAtmosphere", s)
        @test occursin("active gases", s)
        @test occursin("CO₂", s)
        @test !occursin("O₃", s)  # O₃ = 0, should be hidden

        # With function O₃
        atm2 = BackgroundAtmosphere(O₃ = z -> 30e-9)
        s2 = sprint(show, atm2)
        @test occursin("O₃", s2)
    end

    @testset "materialize_background_atmosphere [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(default_arch; size=8, z=(0, 10000),
                               topology=(Flat, Flat, Bounded))

        # Constant O₃
        atm = BackgroundAtmosphere(CO₂ = 400e-6, O₃ = 30e-9)
        matm = materialize_background_atmosphere(atm, grid)
        @test matm.CO₂ isa FT
        @test matm.CO₂ ≈ FT(400e-6)

        # Function O₃
        ozone(z) = 30e-9 * (1 + z / 10000)
        atm2 = BackgroundAtmosphere(O₃ = ozone)
        matm2 = materialize_background_atmosphere(atm2, grid)
        @test matm2.O₃ isa Oceananigans.Fields.AbstractField

        # Nothing atmosphere
        @test materialize_background_atmosphere(nothing, grid) === nothing
    end
end

#####
##### radiation_flux_divergence accessors
#####

@testset "radiation_flux_divergence" begin
    @test radiation_flux_divergence(nothing) === nothing

    # Inline Nothing accessor
    grid = RectilinearGrid(default_arch; size=4, z=(0, 100), topology=(Flat, Flat, Bounded))
    @test radiation_flux_divergence(1, 1, 1, grid, nothing) == zero(eltype(grid))
end
