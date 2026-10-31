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
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition, fill_halo_regions!
using Oceananigans.Operators: Δzᶜᶜᶜ
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

# The acoustic divergence damping uses the change of p′/Cᴸ = (ρχ)′ + Eᴸρ′ over a substep, scaled by
# χ̂ᴸ + Eᴸ = c²/Cᴸ, as its proxy for the mass divergence. In dry air that proxy is the same for ρθ, ρη
# and ρE. Dividing by χ̂ᴸ = ρη/ρᵈ alone, which is negative and shrinks in magnitude with height,
# flipped the sign of the proxy's (ρw)′ ∂z χ̂ᴸ part and drove a slow gravity-wave instability in
# the stratosphere of compressible ρη simulations.
@testset "Divergence damping of ρη and ρE matches ρθ in dry stratified air" begin
    T₀ = 230.0
    g = 9.80665
    θ̄(z) = T₀ * exp(g * z / (1005 * T₀))
    Nx, Nz = 16, 40
    grid = RectilinearGrid(CPU(); size=(Nx, Nz), halo=(5, 5), x=(0, 64e3), z=(0, 30e3),
                           topology=(Periodic, Flat, Bounded))
    constants = ThermodynamicConstants(Float64)
    Rᵈ = dry_air_gas_constant(constants)
    Δτ = 1.0

    function damping_response(formulation)
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = θ̄,
                                        base_pressure = 1e5, standard_pressure = 1e5)
        model = AtmosphereModel(grid; dynamics, formulation, thermodynamic_constants = constants)
        reference = model.dynamics.reference_state
        parent(model.dynamics.dry_density) .= parent(reference.density)
        Breeze.AtmosphereModels.compute_total_density!(model)
        T = CenterField(grid)
        parent(T) .= parent(reference.pressure) ./ (Rᵈ .* parent(reference.density))
        Breeze.AtmosphereModels.set_thermodynamic_variable!(model, Val(:T), T)
        update_state!(model)

        substepper = model.timestepper.substepper
        Breeze.CompressibleEquations.refresh_linearization_basic_state!(substepper, model)
        χ̂ = interior(substepper.linearization_potential_temperature, :, 1, :)

        # One substep of a vertical mass flux (ρw)′ = sin(2πx/Lx) uniform in z: ρ′ is zero in the
        # interior and (ρχ)′ = −Δτ ∂z(χ̂ (ρw)′) is its advection of the basic-state χ̂.
        x = xnodes(grid, Center())
        Δz = [Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:Nz]
        M = [sin(2π * x[i] / 64e3) * (1 < k < Nz + 1) for i in 1:Nx, k in 1:Nz+1]
        χ̂ᶠ = [(χ̂[i, max(k - 1, 1)] + χ̂[i, min(k, Nz)]) / 2 for i in 1:Nx, k in 1:Nz+1]
        interior(substepper.density_perturbation, :, 1, :) .= [-Δτ * (M[i, k+1] - M[i, k]) / Δz[k] for i in 1:Nx, k in 1:Nz]
        interior(substepper.density_potential_temperature_perturbation, :, 1, :) .=
            [-Δτ * (χ̂ᶠ[i, k+1] * M[i, k+1] - χ̂ᶠ[i, k] * M[i, k]) / Δz[k] for i in 1:Nx, k in 1:Nz]
        fill!(substepper.previous_density_potential_temperature_perturbation, 0)
        fill!(substepper.momentum_perturbation.u, 0)
        fill_halo_regions!(substepper.density_perturbation)
        fill_halo_regions!(substepper.density_potential_temperature_perturbation)
        fill_halo_regions!(substepper.previous_density_potential_temperature_perturbation)

        Breeze.CompressibleEquations.apply_divergence_damping!(substepper.damping, substepper, grid, Δτ, constants)
        return interior(substepper.momentum_perturbation.u, :, 1, 2:Nz-1)
    end

    Δρu_θ = damping_response(:LiquidIcePotentialTemperature)
    Δρu_η = damping_response(:MoistEntropy)
    Δρu_E = damping_response(:TotalEnergy)

    @test maximum(abs, Δρu_θ) > 0
    @test maximum(abs, Δρu_η - Δρu_θ) < 0.02 * maximum(abs, Δρu_θ)
    @test maximum(abs, Δρu_E - Δρu_θ) < 0.02 * maximum(abs, Δρu_θ)
end

#####
##### The acoustic loop's pressure against the equation of state
#####

# Advance S, ρᵈ and the water densities of a smooth moist column by the fluxes of one mass flux M(z)
# and compare the true rate of change of pressure with what the acoustic loop assumes. The loop on the
# adiabatic pressure variable has no reference to the entropy offset b; a loop on (S′, ρᵈ′) with frozen
# water is off by −C Σₓ Eˣ M ∂z rˣ, nearly all of the rate at b = 0.
@testset "The acoustic loop's pressure sees the water" begin
    constants = ThermodynamicConstants(Float64)
    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ = constants.liquid.heat_capacity
    pˢᵗ = 1e5

    T̄(z) = 300 - 6.5e-3 * z
    ρ̄ᵈ(z) = 1.15 * exp(-z / 9000)
    r̄ᵛ(z) = 0.018 * exp(-z / 2500)
    r̄ˡ(z) = 1e-3 * exp(-((z - 2000) / 800)^2)

    function pressure(S, ρᵈ, ρᵛ, ρˡ, b)
        ρ = ρᵈ + ρᵛ + ρˡ
        q = MoistureMassFractions(ρᵛ / ρ, ρˡ / ρ, 0.0)
        T = temperature_from_moist_entropy(S / ρ - b * (q.vapor + q.liquid), ρ, q, constants)
        return T * (ρᵈ * Rᵈ + ρᵛ * Rᵛ)
    end

    function entropy(z, b)
        ρᵈ, ρᵛ, ρˡ = ρ̄ᵈ(z), ρ̄ᵈ(z) * r̄ᵛ(z), ρ̄ᵈ(z) * r̄ˡ(z)
        ρ = ρᵈ + ρᵛ + ρˡ
        q = MoistureMassFractions(ρᵛ / ρ, ρˡ / ρ, 0.0)
        return ρ * (moist_entropy(T̄(z), ρ, q, constants) + b * (q.vapor + q.liquid))
    end

    M(z) = 0.3 * (1 + 0.5 * sin(z / 700))
    ∂(f, z) = (f(z + 1) - f(z - 1)) / 2

    function true_rate(z, b)
        δt = 1e-3
        χ̂(z′) = entropy(z′, b) / ρ̄ᵈ(z′)
        S, ρᵈ = entropy(z, b), ρ̄ᵈ(z)
        ρᵛ, ρˡ = ρᵈ * r̄ᵛ(z), ρᵈ * r̄ˡ(z)
        S⁺ = S - δt * ∂(z′ -> χ̂(z′) * M(z′), z)
        ρᵈ⁺ = ρᵈ - δt * ∂(M, z)
        ρᵛ⁺ = ρᵛ - δt * ∂(z′ -> r̄ᵛ(z′) * M(z′), z)
        ρˡ⁺ = ρˡ - δt * ∂(z′ -> r̄ˡ(z′) * M(z′), z)
        return (pressure(S⁺, ρᵈ⁺, ρᵛ⁺, ρˡ⁺, b) - pressure(S, ρᵈ, ρᵛ, ρˡ, b)) / δt
    end

    function adiabatic_loop_rate(z)
        function variable(z′)
            ρ = ρ̄ᵈ(z′) * (1 + r̄ᵛ(z′) + r̄ˡ(z′))
            qᵛ, qˡ = ρ̄ᵈ(z′) * r̄ᵛ(z′) / ρ, ρ̄ᵈ(z′) * r̄ˡ(z′) / ρ
            qᵈ = 1 - qᵛ - qˡ
            Rᵐ = qᵈ * Rᵈ + qᵛ * Rᵛ
            cᵖᵐ = qᵈ * cᵖᵈ + qᵛ * cᵖᵛ + qˡ * cˡ
            return Breeze.CompressibleEquations.adiabatic_pressure_variable(ρ * Rᵐ * T̄(z′), cᵖᵐ, cᵖᵐ - Rᵐ, pˢᵗ, Rᵈ)
        end
        Θ(z′) = first(variable(z′)) / ρ̄ᵈ(z′)
        return - last(variable(z)) * ∂(z′ -> Θ(z′) * M(z′), z)
    end

    function frozen_water_rate(z, b)
        δt = 1e-3
        χ̂(z′) = entropy(z′, b) / ρ̄ᵈ(z′)
        S, ρᵈ = entropy(z, b), ρ̄ᵈ(z)
        r_v, r_l = r̄ᵛ(z), r̄ˡ(z)
        # rˣ held fixed, as a loop on (S′, ρᵈ′) assumes
        S⁺ = S - δt * ∂(z′ -> χ̂(z′) * M(z′), z)
        ρᵈ⁺ = ρᵈ - δt * ∂(M, z)
        return (pressure(S⁺, ρᵈ⁺, ρᵈ⁺ * r_v, ρᵈ⁺ * r_l, b) - pressure(S, ρᵈ, ρᵈ * r_v, ρᵈ * r_l, b)) / δt
    end

    for z in (300.0, 1000.0, 2000.0, 4000.0)
        reference = true_rate(z, -7000.0)
        @test true_rate(z, 0.0) ≈ reference rtol=1e-6  # b is a convention for the true pressure
        @test adiabatic_loop_rate(z) ≈ reference rtol=0.01
    end

    # The frozen-water loop's error is the water term: large at b = 0, and much smaller for vapor at b = −7000
    z = 1000.0
    @test abs(frozen_water_rate(z, 0.0) - true_rate(z, 0.0)) > 0.1 * abs(true_rate(z, 0.0))
    @test abs(frozen_water_rate(z, 0.0) - true_rate(z, 0.0)) > 5 * abs(frozen_water_rate(z, -7000.0) - true_rate(z, -7000.0))
end

#####
##### Drag heating of the adaptive implicit part of sedimentation
#####

# A test-only scheme: rain with a prescribed fall speed and no phase changes.
struct FallingRain{FT}
    fall_speed :: FT
end

const AM = Breeze.AtmosphereModels

AM.prognostic_field_names(::FallingRain) = (:ρqʳ,)
AM.moisture_prognostic_name(::FallingRain) = :ρqᵛ
AM.condensate_liquid_fraction(::FallingRain, ::Val{:ρqʳ}) = 1
AM.sedimentation_velocity(::FallingRain, μ, ::Val{:ρqʳ}) = μ.wʳ
AM.compute_microphysical_tendencies!(::FallingRain, model) = nothing
@inline AM.microphysical_tendency(::FallingRain, name, ρ, ℳ, 𝒰, constants) = zero(ρ)
AM.microphysics_model_update!(::FallingRain, model) = nothing
@inline AM.maybe_adjust_thermodynamic_state(state, ::FallingRain, vapor, constants) = state
@inline AM.microphysical_state(::FallingRain, ρ, μ::NamedTuple, state, velocities) = (; qʳ=μ.ρqʳ / ρ)
@inline AM.moisture_fractions(::FallingRain, state::NamedTuple, vapor) = MoistureMassFractions(vapor, state.qʳ, oftype(vapor, 0))
AM.liquid_mass_fraction(::FallingRain, model) = model.microphysical_fields.qʳ
AM.ice_mass_fraction(::FallingRain, model) = nothing

function AM.materialize_microphysical_fields(::FallingRain, grid, bcs)
    return (; ρqʳ=CenterField(grid; boundary_conditions=bcs.ρqʳ),
              qʳ=CenterField(grid), qᵛ=CenterField(grid), wʳ=AM.sedimentation_velocity_field(grid))
end

@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, m::FallingRain, state, ρ, thermo, constants)
    @inbounds μ.qʳ[i, j, k] = state.qʳ
    @inbounds μ.qᵛ[i, j, k] = thermo.moisture_mass_fractions.vapor
    @inbounds μ.wʳ[i, j, k] = - m.fall_speed
    return nothing
end

# Over one step, the stored energy ∫ρ(𝓊 + g z) changes only by the enthalpy that leaves through the
# bottom, for explicit sedimentation and when most of the fall is the adaptive implicit remainder.
@testset "Drag heating of sedimenting rain closes the energy budget [$label]" for (label, rain_advection) in (
        ("explicit", UpwindBiased(order=1)),
        ("adaptive implicit", WENO(time_discretization=Oceananigans.Utils.AdaptiveVerticallyImplicitDiscretization(cfl=0.01))))

    grid = RectilinearGrid(CPU(); size=8, halo=3, z=(0, 2000), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants(Float64)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 290)
    model = AtmosphereModel(grid; dynamics, formulation = :MoistEntropy, microphysics = FallingRain(8.0),
                            thermodynamic_constants = constants,
                            scalar_advection = (; ρqʳ = rain_advection))
    rain(z) = ifelse((z > 600) & (z < 1600), 2e-3, 0.0)
    set!(model; ρ = model.dynamics.reference_state.density, θ = 290, qᵛ = 0.005, qʳ = rain)

    g = constants.gravitational_acceleration
    zc = collect(znodes(grid, Center()))
    Δz = 250.0
    ρ_field(model) = vec(interior(total_density(model.dynamics)))
    ρᵛ_field(model) = vec(interior(model.moisture_density))
    function stored_energy(model)
        T = vec(interior(model.temperature)); ρ = ρ_field(model)
        ρʳ = vec(interior(model.microphysical_fields.ρqʳ))
        ρᵛ = ρᵛ_field(model)
        return sum(map(zc, T, ρ, ρʳ, ρᵛ) do z, T, ρ, ρʳ, ρᵛ
            q = MoistureMassFractions(ρᵛ / ρ, ρʳ / ρ, 0.0)
            ρ * (internal_energy(T, q, constants) + g * z)
        end) * Δz
    end
    # The rain sits above the ground, so nothing leaves in this short step.
    E₀ = stored_energy(model)
    m₀ = copy(vec(interior(model.microphysical_fields.ρqʳ)))
    Δt = 0.5
    time_step!(model, Δt)
    update_state!(model)
    m₁ = vec(interior(model.microphysical_fields.ρqʳ))
    released = - g * sum(zc .* (m₁ .- m₀)) * Δz
    @test released > 0
    # Energy changes by less than a few percent of the potential energy the rain released
    @test abs(stored_energy(model) - E₀) < 0.05 * released
end
