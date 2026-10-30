include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.AtmosphereModels: total_density, thermodynamic_density
using Breeze.Thermodynamics: MoistureMassFractions, MoistEntropyState, moist_entropy, temperature_from_moist_entropy,
                             temperature, with_temperature, vapor_entropy, liquid_entropy, ice_entropy,
                             vapor_liquid_gibbs_difference, ice_liquid_gibbs_difference,
                             mixture_isochoric_heat_capacity, saturation_vapor_pressure,
                             absolute_zero_latent_heat, vapor_gas_constant, dry_air_gas_constant,
                             PlanarLiquidSurface, PlanarIceSurface
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using Oceananigans
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition
using Oceananigans.TimeSteppers: update_state!
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

# Internal energy per unit mass with Breeze's latent heats: 𝓊 = cᵛᵐ T + ℒˡ₀ qᵛ + (ℒˡ₀ − ℒⁱ₀) qⁱ
function internal_energy(T, q, constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    return mixture_isochoric_heat_capacity(q, constants) * T + ℒˡ₀ * q.vapor + (ℒˡ₀ - ℒⁱ₀) * q.ice
end

function temperature_from_internal_energy(𝓊, q, constants)
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)
    return (𝓊 - ℒˡ₀ * q.vapor - (ℒˡ₀ - ℒⁱ₀) * q.ice) / mixture_isochoric_heat_capacity(q, constants)
end

@testset "Moist entropy thermodynamics" begin
    constants = ThermodynamicConstants(Float64)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    Tᵗʳ = constants.triple_point_temperature
    pᵗʳ = constants.triple_point_pressure

    @testset "Definition and inversion round trip" begin
        for (T, ρ, q) in ((300.0, 1.15, MoistureMassFractions(0.018, 0.002, 0.0)),
                          (250.0, 0.6,  MoistureMassFractions(1e-3, 2e-4, 3e-4)),
                          (210.0, 0.01, MoistureMassFractions(1e-7, 0.0, 1e-5)),
                          (280.0, 1.0,  MoistureMassFractions(0.0, 0.0, 0.0)))
            η = moist_entropy(T, ρ, q, constants)
            @test abs(temperature_from_moist_entropy(η, ρ, q, constants) - T) / T < 1e-10

            # The mixture entropy is the sum of the constituents' partial specific entropies
            qᵈ = 1 - q.vapor - q.liquid - q.ice
            ηᵈ = constants.dry_air.heat_capacity * log(T / Tᵗʳ) - Rᵈ * log(ρ * qᵈ * Rᵈ * T / pᵗʳ)
            ηᵛ = q.vapor > 0 ? vapor_entropy(T, ρ * q.vapor, constants) : 0.0
            Σqη = qᵈ * ηᵈ + q.vapor * ηᵛ + q.liquid * liquid_entropy(T, constants) + q.ice * ice_entropy(T, constants)
            @test η ≈ Σqη rtol=1e-12

            𝒰 = with_temperature(MoistEntropyState(0.0, q, ρ), T, constants)
            @test abs(temperature(𝒰, constants) - T) / T < 1e-10
        end
    end

    @testset "Gibbs functions match the saturation vapor pressures" begin
        T = 285.0
        ρᵛ = 0.008
        pᵛ⁺ˡ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
        @test vapor_liquid_gibbs_difference(T, ρᵛ, constants) ≈ Rᵛ * T * log(ρᵛ * Rᵛ * T / pᵛ⁺ˡ) rtol=1e-10

        T = 255.0
        pᵛ⁺ˡ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
        pᵛ⁺ⁱ = saturation_vapor_pressure(T, constants, PlanarIceSurface())
        @test ice_liquid_gibbs_difference(T, constants) ≈ Rᵛ * T * log(pᵛ⁺ⁱ / pᵛ⁺ˡ) rtol=1e-10
        @test ice_liquid_gibbs_difference(Tᵗʳ, constants) ≈ 0 atol=1e-8
    end

    @testset "Phase change at fixed internal energy and density" begin
        ρ = 1.0
        T₀ = 290.0
        qᵛ⁺ = saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface()) / (Rᵛ * T₀ * ρ)

        # Reversible condensation at saturation: dη/dqˡ = 0, so Δη is second order in the increment
        Δη(δ) = begin
            q₀ = MoistureMassFractions(qᵛ⁺, 1e-3, 0.0)
            q₁ = MoistureMassFractions(qᵛ⁺ - δ, 1e-3 + δ, 0.0)
            T₁ = temperature_from_internal_energy(internal_energy(T₀, q₀, constants), q₁, constants)
            moist_entropy(T₁, ρ, q₁, constants) - moist_entropy(T₀, ρ, q₀, constants)
        end
        @test abs(Δη(1e-6) / 1e-6) < 1e-1
        @test abs(Δη(1e-6) / Δη(1e-5)) < 2e-2  # quadratic in δ

        # Condensing out of supersaturated air produces entropy, at the rate −(𝒢ᵛ − 𝒢ˡ) δ / T
        q₀ = MoistureMassFractions(1.05qᵛ⁺, 0.0, 0.0)
        δ = 0.002qᵛ⁺
        q₁ = MoistureMassFractions(1.05qᵛ⁺ - δ, δ, 0.0)
        T₁ = temperature_from_internal_energy(internal_energy(T₀, q₀, constants), q₁, constants)
        Δη₊ = moist_entropy(T₁, ρ, q₁, constants) - moist_entropy(T₀, ρ, q₀, constants)
        @test Δη₊ > 0
        @test Δη₊ ≈ vapor_liquid_gibbs_difference(T₀, ρ * q₀.vapor, constants) * δ / T₀ rtol=0.15
    end
end

@testset "MoistEntropyFormulation construction" begin
    grid = RectilinearGrid(CPU(); size=(4, 4, 4), x=(0, 1e3), y=(0, 1e3), z=(0, 1e3))

    @test_throws ArgumentError AtmosphereModel(grid; formulation = :MoistEntropy)

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = z -> 300.0)
    for formulation in (:MoistEntropy, :ρη, MoistEntropyFormulation())
        model = AtmosphereModel(grid; dynamics, formulation)
        @test model.formulation isa MoistEntropyFormulation
        @test haskey(Oceananigans.prognostic_fields(model), :ρη)
    end

    equilibrium = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    @test_throws ArgumentError AtmosphereModel(grid; dynamics, formulation = :MoistEntropy, microphysics = equilibrium)
end

@testset "Microphysics alone conserves internal energy" begin
    FT = Float64
    grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1e3), y=(0, 1e3), z=(0, 1e3))
    constants = ThermodynamicConstants(FT)
    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation,
                                              precipitation_boundary_condition = ImpenetrableBoundaryCondition())
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 7e4,
                                    reference_potential_temperature = z -> 290.0)
    model = AtmosphereModel(grid; dynamics, formulation = :MoistEntropy, microphysics, thermodynamic_constants = constants)

    # A supersaturated, mixed-phase state below freezing: condensation, deposition, autoconversion,
    # accretion, riming and the Wegener–Bergeron–Findeisen process all act.
    ρ₀ = 0.9
    T₀ = 265.0
    qᵛ⁺ = saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface()) / (vapor_gas_constant(constants) * T₀ * ρ₀)
    set!(model; T = T₀, qᵛ = 1.1qᵛ⁺, qᶜˡ = 1e-3, qʳ = 5e-4, qᶜⁱ = 2e-4, qˢⁿ = 3e-4, ρ = ρ₀)

    function state(model)
        ρ = interior(total_density(model.dynamics))[1]
        T = interior(model.temperature)[1]
        μ = model.microphysical_fields
        ρqᵛ = interior(model.moisture_density)[1]
        ρqˡ = interior(μ.ρqᶜˡ)[1] + interior(μ.ρqʳ)[1]
        ρqⁱ = interior(μ.ρqᶜⁱ)[1] + interior(μ.ρqˢⁿ)[1]
        q = MoistureMassFractions(ρqᵛ / ρ, ρqˡ / ρ, ρqⁱ / ρ)
        return (; ρ𝓊 = ρ * internal_energy(T, q, constants), ρη = ρ * moist_entropy(T, ρ, q, constants), q)
    end

    s₀ = state(model)
    for _ in 1:300
        time_step!(model, 1)
    end
    s₁ = state(model)

    @test s₁.q.ice > s₀.q.ice + 1e-3                 # substantial phase change happened
    @test abs(s₁.ρ𝓊 - s₀.ρ𝓊) / s₀.ρ𝓊 < 1e-6         # internal energy conserved
    @test s₁.ρη > s₀.ρη                               # irreversible phase change produces entropy
end

@testset "Dry rest state stays at rest" begin
    T₀ = 250.0
    g = 9.80665
    θ̄(z) = T₀ * exp(g * z / (1005 * T₀))
    grid = RectilinearGrid(CPU(); size=(6, 6, 32), halo=(5, 5, 5), x=(0, 100e3), y=(0, 100e3), z=(0, 30e3))
    constants = ThermodynamicConstants(Float64)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = θ̄,
                                    base_pressure = 1e5, standard_pressure = 1e5)
    model = AtmosphereModel(grid; dynamics, formulation = :MoistEntropy, thermodynamic_constants = constants)

    reference = model.dynamics.reference_state
    Rᵈ = dry_air_gas_constant(constants)
    parent(model.dynamics.dry_density) .= parent(reference.density)
    Breeze.AtmosphereModels.compute_total_density!(model)
    T = CenterField(grid)
    parent(T) .= parent(reference.pressure) ./ (Rᵈ .* parent(reference.density))
    Breeze.AtmosphereModels.set_thermodynamic_variable!(model, Val(:T), T)
    update_state!(model)

    @test maximum(abs, interior(model.dynamics.pressure) .- interior(reference.pressure)) / 1e5 < 1e-14

    max_w = 0.0
    for _ in 1:100
        time_step!(model, 10)
        max_w = max(max_w, maximum(abs, interior(model.velocities.w)))
    end
    @test max_w < 1e-10
end

@testset "Moist bubble: ρη changes only by its sources" begin
    # A warm, moist bubble without microphysics, closure or boundary fluxes: ρη has no source,
    # so its integral is conserved to round-off while the bubble rises.
    grid = RectilinearGrid(CPU(); size=(8, 8, 16), halo=(5, 5, 5), x=(-6e3, 6e3), y=(-6e3, 6e3), z=(0, 12e3))
    constants = ThermodynamicConstants(Float64)
    θ̄(z) = 300 + 3e-3 * z
    q̄(z) = 0.015 * exp(-z / 3000)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); base_pressure = 101325,
                                    reference_potential_temperature = θ̄, reference_vapor_mass_fraction = q̄)
    model = AtmosphereModel(grid; dynamics, formulation = :MoistEntropy, thermodynamic_constants = constants)

    ρ̄ = Array(interior(model.dynamics.reference_state.density))
    zᶜ = znodes(grid, Center())
    ρᵢ(x, y, z) = ρ̄[1, 1, clamp(searchsortedfirst(zᶜ, z - 1e-6), 1, grid.Nz)]
    b(x, y, z) = exp(-(x^2 + y^2) / 2e3^2 - ((z - 1.5e3) / 1e3)^2)
    set!(model; θ = (x, y, z) -> θ̄(z) + 2b(x, y, z), qᵗ = (x, y, z) -> q̄(z) * (1 + 0.1b(x, y, z)), ρ = ρᵢ)

    ∫ρη₀ = sum(interior(thermodynamic_density(model.formulation)))
    for _ in 1:50
        time_step!(model, 1)
    end
    ∫ρη₁ = sum(interior(thermodynamic_density(model.formulation)))

    @test maximum(abs, interior(model.velocities.w)) > 0.1
    @test abs(∫ρη₁ - ∫ρη₀) / abs(∫ρη₀) < 1e-12
end
