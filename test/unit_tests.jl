include(joinpath(@__DIR__, "setup.jl"))
include(joinpath(@__DIR__, "supposition_setup.jl"))

#####
##### Consolidated unit tests for fast-running tests
#####
##### These tests verify basic construction and simple functionality.
##### They are grouped together to reduce compilation overhead.
#####

using Breeze
using Oceananigans
using Test
using Adapt: adapt

#####
##### AnelasticDynamics
#####

using Breeze: ReferenceState, AnelasticDynamics, total_density
using Breeze.AtmosphereModels: materialize_dynamics, default_dynamics
using Breeze.AtmosphereModels: dynamics_pressure, pressure_anomaly, total_pressure
using Breeze.AtmosphereModels: dynamics_density

@testset "AnelasticDynamics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))
    constants = ThermodynamicConstants()

    @testset "Constructor with ReferenceState" begin
        reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
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
        reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=300)
        dynamics_stub = AnelasticDynamics(reference_state)
        dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple(), constants)

        # Test dynamics_pressure
        p̄ = dynamics_pressure(dynamics)
        @test p̄ === reference_state.pressure
        @test total_density(dynamics) === reference_state.density

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
using Breeze.Thermodynamics: pressure_balanced_density

@testset "CompressibleDynamics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 8), x=(0, 100), y=(0, 100), z=(0, 1000))

    @testset "Constructor" begin
        dynamics = CompressibleDynamics()
        @test dynamics isa CompressibleDynamics
        @test dynamics.dry_density === nothing  # Not materialized yet
        @test dynamics.standard_pressure == 1e5
        @test dynamics.base_pressure == 101325
    end

    @testset "materialize_dynamics" begin
        dynamics_stub = CompressibleDynamics()
        constants = ThermodynamicConstants()
        dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple(), constants)

        @test dynamics isa CompressibleDynamics
        @test dynamics.dry_density isa Field
        @test dynamics.pressure isa Field
        @test dynamics_density(dynamics) === dynamics.dry_density
        @test dynamics_pressure(dynamics) === dynamics.pressure
        @test total_density(dynamics) === dynamics.total_density
    end

    @testset "materialize_dynamics seeds pressure" begin
        base_pressure = FT(100000)
        constants = ThermodynamicConstants(FT)

        dynamics_stub = CompressibleDynamics(; base_pressure, reference_state=nothing)
        dynamics = materialize_dynamics(dynamics_stub, grid, NamedTuple(), constants)
        @test all(Array(interior(dynamics.pressure)) .== base_pressure)

        automatic_stub = CompressibleDynamics(; base_pressure)
        automatic_dynamics = materialize_dynamics(automatic_stub, grid, NamedTuple(), constants)
        pressure = Array(interior(automatic_dynamics.pressure))
        reference_pressure = Array(interior(automatic_dynamics.reference_state.pressure))
        @test all(pressure .== reference_pressure)
    end
end

@testset "Pressure-balanced density [$(FT)]" for FT in test_float_types()
    ρ_background = FT(1.1)
    θ_background = FT(300)
    θ_initial = FT(303)

    ρ_initial = pressure_balanced_density(ρ_background, θ_background, θ_initial)

    @test ρ_initial < ρ_background
    @test ρ_initial * θ_initial ≈ ρ_background * θ_background
end

@testset "Moist CompressibleDynamics reference state [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(default_arch;
                           size = (8, 8, 8),
                           halo = (5, 5, 5),
                           x = (0, 100),
                           y = (0, 100),
                           z = (0, 1000))

    constants = ThermodynamicConstants(FT)
    base_pressure = FT(100000)
    standard_pressure = FT(100000)
    θ_reference(z) = FT(300) + FT(0.01) * z
    qᵛ_reference(z) = FT(0.012) * exp(-z / FT(1000))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=2);
                                    base_pressure,
                                    standard_pressure,
                                    reference_potential_temperature = θ_reference,
                                    reference_vapor_mass_fraction = qᵛ_reference)

    model = AtmosphereModel(grid; dynamics,
                            thermodynamic_constants = constants)

    reference_state = model.dynamics.reference_state
    θ_column = Field{Nothing, Nothing, Center}(grid)
    set!(θ_column, θ_reference)

    qᵛ_column = Field{Nothing, Nothing, Center}(grid)
    set!(qᵛ_column, qᵛ_reference)

    set!(model, θ=θ_column, qᵛ=qᵛ_column, ρ=reference_state.density)

    pressure_error = maximum(abs, interior(model.dynamics.pressure) .-
                                  interior(reference_state.pressure))
    pressure_scale = maximum(abs, interior(reference_state.pressure))

    @test pressure_error <= 100 * eps(FT) * pressure_scale

    θ_perturbed = CenterField(grid)
    set!(θ_perturbed, (x, y, z) -> θ_reference(z) + FT(1))

    ρ_balanced = CenterField(grid)
    set!(ρ_balanced, pressure_balanced_density(reference_state.density, θ_column, θ_perturbed))
    set!(model, θ=θ_perturbed, qᵛ=qᵛ_column, ρ=ρ_balanced)

    balanced_pressure_error = maximum(abs, interior(model.dynamics.pressure) .-
                                           interior(reference_state.pressure))

    @test balanced_pressure_error <= 100 * eps(FT) * pressure_scale
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
    boundary_conditions = (ρθ = FieldBoundaryConditions(grid, ccc), ρs = FieldBoundaryConditions(grid, ccc))

    @testset "LiquidIcePotentialTemperature field naming (Symbol)" begin
        @test prognostic_thermodynamic_field_names(:LiquidIcePotentialTemperature) == (:ρθ,)
        @test additional_thermodynamic_field_names(:LiquidIcePotentialTemperature) == (:θ,)
        @test thermodynamic_density_name(:LiquidIcePotentialTemperature) == :ρθ
    end

    @testset "StaticEnergy field naming (Symbol)" begin
        @test prognostic_thermodynamic_field_names(:StaticEnergy) == (:ρs,)
        @test additional_thermodynamic_field_names(:StaticEnergy) == (:s,)
        @test thermodynamic_density_name(:StaticEnergy) == :ρs
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
        @test prognostic_thermodynamic_field_names(formulation) == (:ρs,)
        @test additional_thermodynamic_field_names(formulation) == (:s,)
        @test thermodynamic_density_name(formulation) == :ρs
        @test thermodynamic_density(formulation) === formulation.energy_density
    end

    @testset "Oceananigans.fields and prognostic_fields" begin
        θ_formulation = materialize_formulation(:LiquidIcePotentialTemperature, dynamics, grid, boundary_conditions)
        s_formulation = materialize_formulation(:StaticEnergy, dynamics, grid, boundary_conditions)

        # LiquidIcePotentialTemperature
        @test haskey(Oceananigans.fields(θ_formulation), :θ)
        @test haskey(Oceananigans.prognostic_fields(θ_formulation), :ρθ)

        # StaticEnergy
        @test haskey(Oceananigans.fields(s_formulation), :s)
        @test haskey(Oceananigans.prognostic_fields(s_formulation), :ρs)
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
    MoistureMixingRatio,
    StaticEnergyState,
    LiquidIcePotentialTemperatureState,
    temperature,
    density,
    exner_function,
    mixture_gas_constant,
    mixture_heat_capacity,
    total_specific_moisture,
    temperature_from_potential_temperature,
    potential_temperature_from_temperature

@testset "Thermodynamics" begin
    thermo = ThermodynamicConstants()
    @test thermo.liquid.density == 1000
    @test thermo.ice.density == 917
    @test occursin("density=1000.0", sprint(show, thermo.liquid))

    adapted_thermo = adapt(CPU(), ThermodynamicConstants(Float32))
    @test adapted_thermo.liquid.density === Float32(1000)
    @test adapted_thermo.ice.density === Float32(917)

    # Test Saturation specific humidity calculation
    T = 293.15  # 20°C
    ρ = 1.2     # kg/m³
    q★ = Breeze.Thermodynamics.saturation_specific_humidity(T, ρ, thermo, thermo.liquid)
    @test q★ > 0
end

@testset "StaticEnergyState [$(FT)]" for FT in all_float_types()
    thermo = ThermodynamicConstants(FT)
    g = thermo.gravitational_acceleration
    ℒˡᵣ = thermo.liquid.reference_latent_heat
    ℒⁱᵣ = thermo.ice.reference_latent_heat

    # temperature(::StaticEnergyState) is the closed-form inverse of
    # s = cᵖᵐ T + g z − ℒˡᵣ qˡ − ℒⁱᵣ qⁱ, so the round trip holds to rounding for any state.
    @breeze_check function static_energy_temperature_round_trip(T = spstn_temperatures(FT),
                                                                p = spstn_pressures(FT),
                                                                z = spstn_heights(FT),
                                                                q = spstn_mass_fractions(FT))
        cᵖᵐ = mixture_heat_capacity(q, thermo)
        s = cᵖᵐ * T + g * z - ℒˡᵣ * q.liquid - ℒⁱᵣ * q.ice
        T★ = temperature(StaticEnergyState(s, q, z, p), thermo)
        return isapprox(T★, T; rtol = spstn_rounding_rtol(FT))
    end
end

@testset "Potential temperature convenience functions [$(FT)]" for FT in all_float_types()
    thermo = ThermodynamicConstants(FT)
    p = FT(101325)
    pˢᵗ = FT(1e5)
    T = FT(290)

    θ = potential_temperature_from_temperature(T, p, pˢᵗ, thermo)
    θ_default = potential_temperature_from_temperature(T, p, thermo)
    θ_integer_temperature = potential_temperature_from_temperature(290, p, pˢᵗ, thermo)

    @test θ != T
    @test θ_default isa FT
    @test temperature_from_potential_temperature(θ_default, p, thermo) isa FT
    @test temperature_from_potential_temperature(θ_default, p, thermo) ≈ T
    @test θ_integer_temperature isa FT
    @test θ_integer_temperature ≈ θ

    # θ ↔ T are an exact inverse pair through the Exner function, and θ ≥ T exactly when p ≤ pˢᵗ.
    @breeze_check function potential_temperature_round_trip(T = spstn_temperatures(FT),
                                                            p = spstn_pressures(FT),
                                                            pˢᵗ = spstn_pressures(FT; lo=9e4, hi=1.1e5))
        θ = potential_temperature_from_temperature(T, p, pˢᵗ, thermo)
        T★ = temperature_from_potential_temperature(θ, p, pˢᵗ, thermo)
        ordering_ok = abs(p - pˢᵗ) < 100 || (θ >= T) == (p <= pˢᵗ)
        return isapprox(T★, T; rtol = spstn_rounding_rtol(FT)) && ordering_ok
    end
end

@testset "Thermodynamic identities [$(FT)]" for FT in all_float_types()
    thermo = ThermodynamicConstants(FT)
    rtol = spstn_rounding_rtol(FT)

    # Equation of state: density(T, p, q) = p / (Rᵐ T), so ρ Rᵐ T recovers p.
    @breeze_check function ideal_gas_law_closes(T = spstn_temperatures(FT),
                                                p = spstn_pressures(FT),
                                                q = spstn_mass_fractions(FT))
        ρ = density(T, p, q, thermo)
        Rᵐ = mixture_gas_constant(q, thermo)
        return isapprox(ρ * Rᵐ * T, p; rtol)
    end

    # Mass fractions ↔ mixing ratios is a bijection that preserves the mixture properties.
    @breeze_check function mixing_ratio_round_trip(q = spstn_mass_fractions(FT; total_max=0.1))
        r = MoistureMixingRatio(q)
        q★ = MoistureMassFractions(r)
        return isapprox(q★.vapor, q.vapor; rtol) &&
               isapprox(q★.liquid, q.liquid; rtol) &&
               isapprox(q★.ice, q.ice; rtol) &&
               isapprox(total_specific_moisture(r), total_specific_moisture(q); rtol) &&
               isapprox(mixture_gas_constant(r, thermo), mixture_gas_constant(q, thermo); rtol) &&
               isapprox(mixture_heat_capacity(r, thermo), mixture_heat_capacity(q, thermo); rtol)
    end

    # The Exner function Π = (p / pˢᵗ)^(Rᵐ/cᵖᵐ) lies in (0, 1] for p ≤ pˢᵗ and increases with p.
    @breeze_check function exner_function_bounded_and_increasing(q = spstn_mass_fractions(FT),
                                                                 pˢᵗ = spstn_pressures(FT; lo=9e4, hi=1.1e5),
                                                                 f = spstn_floats(FT; lo=0.01, hi=1),
                                                                 δ = spstn_floats(FT; lo=10, hi=5e4))
        θ = FT(300)
        p₁ = f * pˢᵗ
        Π₁ = exner_function(LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, p₁), thermo)
        Π₂ = exner_function(LiquidIcePotentialTemperatureState(θ, q, pˢᵗ, p₁ + δ), thermo)
        return 0 < Π₁ <= 1 && Π₂ > Π₁
    end
end

#####
##### Saturation vapor pressure
#####

using Breeze.Thermodynamics:
    TetensFormula,
    FlatauPolynomial,
    saturation_vapor_pressure,
    PlanarLiquidSurface,
    PlanarIceSurface,
    PlanarMixedPhaseSurface,
    dewpoint_temperature,
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

@testset "Saturation vapor pressure surfaces [$FT]" for FT in all_float_types()
    thermo = ThermodynamicConstants(FT)
    liquid_surface = PlanarLiquidSurface()
    ice_surface = PlanarIceSurface()
    rtol = FT === Float64 ? 1e-12 : FT(1e-5)

    @breeze_check function homogeneous_surfaces_match_condensed_phases(T = spstn_temperatures(FT))
        pˡ = saturation_vapor_pressure(T, thermo, thermo.liquid)
        pⁱ = saturation_vapor_pressure(T, thermo, thermo.ice)
        return isapprox(saturation_vapor_pressure(T, thermo, liquid_surface), pˡ; rtol) &&
               isapprox(saturation_vapor_pressure(T, thermo, ice_surface), pⁱ; rtol)
    end

    @breeze_check function mixed_surface_matches_reference(T = spstn_temperatures(FT),
                                                           λ = spstn_unit_interval(FT))
        p_surface = saturation_vapor_pressure(T, thermo, PlanarMixedPhaseSurface(λ))
        return isapprox(p_surface, reference_mixed_surface_pressure(T, thermo, λ); rtol)
    end
end

@testset "Saturation vapor pressure formulations [$FT]" for FT in all_float_types()
    rtol = spstn_rounding_rtol(FT)
    Tᶠ = FT(273.15)

    formulations = (("ClausiusClapeyron", ThermodynamicConstants(FT)),
                    ("TetensFormula", ThermodynamicConstants(FT; saturation_vapor_pressure=TetensFormula(FT))),
                    ("FlatauPolynomial", ThermodynamicConstants(FT; saturation_vapor_pressure=FlatauPolynomial(FT))))

    @testset "$name" for (name, thermo) in formulations
        # The saturation vapor pressure over a mixed-phase surface lies between the ice and liquid
        # values (a geometric blend for Clausius-Clapeyron, an arithmetic one for Tetens and Flatau).
        # Also a regression for a missing `saturation_vapor_pressure(..., ::PlanarMixedPhaseSurface)`
        # method — without it, `SaturationAdjustment` (which uses `MixedPhaseEquilibrium`) throws a
        # `MethodError`, which also breaks GPU kernel codegen (`InvalidIRError`).
        @breeze_check function mixed_phase_svp_between_ice_and_liquid(T = spstn_temperatures(FT; hi=Tᶠ),
                                                                      λ = spstn_unit_interval(FT))
            pˡ = saturation_vapor_pressure(T, thermo, PlanarLiquidSurface())
            pⁱ = saturation_vapor_pressure(T, thermo, PlanarIceSurface())
            pᵐ = saturation_vapor_pressure(T, thermo, PlanarMixedPhaseSurface(λ))
            lo, hi = minmax(pˡ, pⁱ)
            slack = rtol * hi
            return isfinite(pᵐ) && pᵐ > 0 && lo - slack <= pᵐ <= hi + slack
        end

        # Saturation vapor pressure increases strictly with temperature. The ice fits are only
        # valid below freezing, the liquid ones up to about 330 K.
        @breeze_check function svp_strictly_increasing_in_temperature(T₁ˡ = spstn_temperatures(FT; lo=200, hi=280),
                                                                      δˡ = spstn_floats(FT; lo=0.05, hi=50),
                                                                      T₁ⁱ = spstn_temperatures(FT; lo=200, hi=260),
                                                                      δⁱ = spstn_floats(FT; lo=0.05, hi=13))
            liquid_ok = saturation_vapor_pressure(T₁ˡ + δˡ, thermo, PlanarLiquidSurface()) >
                        saturation_vapor_pressure(T₁ˡ, thermo, PlanarLiquidSurface())
            ice_ok = saturation_vapor_pressure(T₁ⁱ + δⁱ, thermo, PlanarIceSurface()) >
                     saturation_vapor_pressure(T₁ⁱ, thermo, PlanarIceSurface())
            return liquid_ok && ice_ok
        end
    end
end

@testset "Dewpoint temperature inversion [$FT]" for FT in all_float_types()
    thermo = ThermodynamicConstants(FT)

    @testset "$name" for (name, surface) in (("liquid", PlanarLiquidSurface()), ("ice", PlanarIceSurface()))
        # The dewpoint inverts the saturation vapor pressure: for pᵛ = pᵛ⁺(T⁺) with T⁺ ≤ T the
        # secant solve recovers T⁺ within its tolerance (reltol 1e-4 on pᵛ, about 1.5 mK)...
        @breeze_check function dewpoint_inverts_saturation_vapor_pressure(T⁺ = spstn_temperatures(FT; lo=200, hi=300),
                                                                          Δ = spstn_floats(FT; lo=0, hi=40))
            T = T⁺ + Δ
            pᵛ = saturation_vapor_pressure(T⁺, thermo, surface)
            return abs(dewpoint_temperature(pᵛ, T, thermo, surface) - T⁺) <= FT(0.01)
        end

        # ...and a saturated or supersaturated state (pᵛ ≥ pᵛ⁺(T)) returns T itself.
        @breeze_check function saturated_dewpoint_is_temperature(T = spstn_temperatures(FT),
                                                                 f = spstn_floats(FT; lo=1, hi=3))
            pᵛ = f * saturation_vapor_pressure(T, thermo, surface)
            return dewpoint_temperature(pᵛ, T, thermo, surface) == T
        end
    end
end

@testset "Tetens formula saturation vapor pressure [$FT]" for FT in test_float_types()
    tetens = TetensFormula()
    thermo = ThermodynamicConstants(; saturation_vapor_pressure=tetens)
    rtol = FT === Float64 ? eps(FT) : FT(1e-5)

    # Test at reference temperature (273.15 K): should return reference pressure
    Tᵣ = FT(273.15)
    pᵛ⁺_ref = saturation_vapor_pressure(Tᵣ, thermo, PlanarLiquidSurface())
    @test pᵛ⁺_ref ≈ FT(610) rtol=rtol

    # Test ice surface at reference temperature
    pⁱ_ref = saturation_vapor_pressure(Tᵣ, thermo, PlanarIceSurface())
    @test pⁱ_ref ≈ FT(610) rtol=rtol

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
end

@testset "Flatau vs Clausius-Clapeyron comparison [$FT]" for FT in test_float_types()
    flatau = FlatauPolynomial(FT)
    thermo_flatau = ThermodynamicConstants(FT; saturation_vapor_pressure=flatau)
    thermo_cc = ThermodynamicConstants(FT) # Default is Clausius-Clapeyron

    # the polynomial fits track the integrated CC form closely over the atmospheric range
    for T in FT.((240, 260, 285, 300, 310))
        pˡ_flatau = saturation_vapor_pressure(T, thermo_flatau, PlanarLiquidSurface())
        pˡ_cc = saturation_vapor_pressure(T, thermo_cc, PlanarLiquidSurface())
        @test pˡ_flatau ≈ pˡ_cc rtol=FT(0.01)
    end
    for T in FT.((200, 240, 260, 273))
        pⁱ_flatau = saturation_vapor_pressure(T, thermo_flatau, PlanarIceSurface())
        pⁱ_cc = saturation_vapor_pressure(T, thermo_cc, PlanarIceSurface())
        @test pⁱ_flatau ≈ pⁱ_cc rtol=FT(0.05)
    end

    # far below the fit range the argument is clamped rather than extrapolated
    @test saturation_vapor_pressure(FT(150), thermo_flatau, PlanarLiquidSurface()) ==
          saturation_vapor_pressure(FT(193.16), thermo_flatau, PlanarLiquidSurface())
end

@testset "Flatau mixed-phase SVP compiles on device [$FT]" for FT in test_float_types()
    flatau = FlatauPolynomial(FT)
    thermo = ThermodynamicConstants(FT; saturation_vapor_pressure = flatau)
    surface = PlanarMixedPhaseSurface(FT(0.4))
    # Broadcasting over a device array compiles a kernel that calls the mixed-phase SVP. On a GPU this
    # is the exact codegen that failed (InvalidIRError from a dynamic MethodError throw) when the
    # PlanarMixedPhaseSurface method was missing — SaturationAdjustment uses it every step.
    Ts = Oceananigans.Architectures.on_architecture(default_arch, collect(FT, 250:5:290))
    p = saturation_vapor_pressure.(Ts, Ref(thermo), Ref(surface))
    @test all(isfinite, Array(p))
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
        @test atm.O₃ === Breeze.standard_ozone_profile
        @test atm.CFC₁₁ == 0.0
    end

    @testset "Custom constructor" begin
        atm = BackgroundAtmosphere(CO₂ = 400e-6, O₃ = 30e-9)
        @test atm.CO₂ ≈ 400e-6
        @test atm.O₃ ≈ 30e-9
        @test atm.N₂ ≈ 0.78084  # default preserved
    end

    @testset "standard_ozone_profile" begin
        O₃ = Breeze.standard_ozone_profile
        @test O₃(0) ≈ 3e-8 rtol=1e-3           # tropospheric background at the surface
        @test O₃(25e3) ≈ 8e-6 rtol=1e-3        # stratospheric peak
        @test O₃(50e3) < O₃(25e3)              # decays above the peak
        @test all(z -> O₃(z) > 0, 0:1e3:60e3)
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

#####
##### materialize_surface_property
#####

using Breeze.AtmosphereModels: materialize_surface_property

# Extension point: downstream packages materialize property sources against grid + solar position.
struct TestSurfacePropertySource end
Breeze.AtmosphereModels.materialize_surface_property(::TestSurfacePropertySource, grid, solar_position) =
    convert(eltype(grid), 1//2)

@testset "materialize_surface_property [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1), y=(0, 1), z=(0, 1))

    x = materialize_surface_property(0.2, grid)
    @test x isa FT
    @test x ≈ 0.2

    α = CenterField(grid)
    @test materialize_surface_property(α, grid) === α

    # The three-argument form falls back to the two-argument form...
    @test materialize_surface_property(0.2, grid, nothing) === materialize_surface_property(0.2, grid)

    # ...and dispatches to source-specific methods.
    @test materialize_surface_property(TestSurfacePropertySource(), grid, nothing) == FT(0.5)
end
