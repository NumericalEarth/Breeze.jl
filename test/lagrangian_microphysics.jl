include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.LagrangianMicrophysics: kelvin_length, equilibrium_supersaturation_derivative,
                                     vapor_diffusivity, thermal_conductivity
using Breeze.Thermodynamics: saturation_vapor_pressure, liquid_latent_heat, dry_air_gas_constant,
                             PlanarLiquidSurface
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.TimeSteppers: step_lagrangian_particles!
using StructArrays: StructArray
using Test

#####
##### κ-Köhler droplet physics
#####
##### The reference particle is Anderson et al. (2023)'s NaCl aerosol: dry diameter 130 nm,
##### hygroscopicity κ = 1, at 290 K and 1 atm.
#####

# Dilute-limit (Petters & Kreidenweis 2007) critical point: 𝒮ᵉ ≈ A/D − κ Dᵈ³/D³ has its maximum
# at Dᶜ = √(3 κ Dᵈ³ / A) with 𝒮ᶜ = √(4 A³ / (27 κ Dᵈ³)).
dilute_critical_diameter(A, Dᵈ, κ) = sqrt(3 * κ * Dᵈ^3 / A)
dilute_critical_supersaturation(A, Dᵈ, κ) = sqrt(4 * A^3 / (27 * κ * Dᵈ^3))

@testset "κ-Köhler droplet physics [$(FT)]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    Dᵈ = FT(130e-9)
    κ = FT(1)
    T = FT(290)
    p = FT(101325)
    tolerance = FT === Float32 ? 1e-3 : 1e-8

    @testset "Equilibrium supersaturation" begin
        # A dry particle is infinitely subsaturated; a huge droplet is in equilibrium with saturated air
        @test equilibrium_supersaturation(Dᵈ, Dᵈ, κ, T, constants) ≈ -1
        @test abs(equilibrium_supersaturation(FT(1e-2), Dᵈ, κ, T, constants)) < 1e-6

        # The derivative matches a finite difference away from the singular dry limit
        D = 3Dᵈ
        δ = FT(1e-4) * D
        finite_difference = (equilibrium_supersaturation(D + δ, Dᵈ, κ, T, constants) -
                             equilibrium_supersaturation(D - δ, Dᵈ, κ, T, constants)) / (2δ)
        @test equilibrium_supersaturation_derivative(D, Dᵈ, κ, T, constants) ≈ finite_difference rtol=(FT === Float32 ? 1e-2 : 1e-6)
    end

    @testset "Critical point" begin
        A = kelvin_length(T, constants)
        Dᶜ = critical_diameter(Dᵈ, κ, T, constants)
        𝒮ᶜ = critical_supersaturation(Dᵈ, κ, T, constants)

        # Anderson's aerosol activates near 0.085 % at a diameter near 1.7 μm
        @test 1.5e-6 < Dᶜ < 2e-6
        @test 7e-4 < 𝒮ᶜ < 1e-3

        # Close to the dilute-limit formulas (the correction is of order (Dᵈ/Dᶜ)³ ≈ 4 × 10⁻⁴)
        @test Dᶜ ≈ dilute_critical_diameter(A, Dᵈ, κ) rtol=5e-3
        @test 𝒮ᶜ ≈ dilute_critical_supersaturation(A, Dᵈ, κ) rtol=5e-3

        # It is the maximum of the Köhler curve
        @test 𝒮ᶜ ≥ equilibrium_supersaturation(FT(0.9) * Dᶜ, Dᵈ, κ, T, constants)
        @test 𝒮ᶜ ≥ equilibrium_supersaturation(FT(1.1) * Dᶜ, Dᵈ, κ, T, constants)
        @test abs(equilibrium_supersaturation_derivative(Dᶜ, Dᵈ, κ, T, constants)) * Dᶜ < 1e-5

        # Equilibrium haze at 99 % relative humidity sits between the dry and critical diameters
        𝒮 = FT(-0.01)
        Dᵉ = equilibrium_diameter(𝒮, Dᵈ, κ, T, constants)
        @test Dᵈ < Dᵉ < Dᶜ
        @test equilibrium_supersaturation(Dᵉ, Dᵈ, κ, T, constants) ≈ 𝒮 atol=tolerance
    end

    @testset "Growth coefficient" begin
        dynamics = DropletDynamics(FT)
        G = D -> growth_coefficient(T, p, D, dynamics, constants)

        # Order of magnitude: a 10 μm droplet at 1 % supersaturation grows about a micron in ten seconds
        @test 3e-11 < G(FT(10e-6)) < 3e-10

        # The kinetic corrections slow small droplets and vanish for large ones
        @test G(FT(0.2e-6)) < G(FT(2e-6)) < G(FT(20e-6))
        @test G(FT(1e-3)) ≈ G(FT(1e-2)) rtol=1e-3

        # The continuum limit of the corrected coefficients
        @test vapor_diffusivity(T, p, FT(1), dynamics.accommodation, constants) ≈ FT(2.11e-5) * (T / 273)^FT(1.94) rtol=1e-4
        ρᵃ = p / (dry_air_gas_constant(constants) * T)
        @test thermal_conductivity(T, FT(1), ρᵃ, dynamics.thermal_accommodation, constants) ≈ FT(1e-3) * (FT(4.39) + FT(0.071) * T) rtol=1e-4

        # In the continuum limit, G matches the Maxwell–Mason coefficient assembled by hand
        Dᵛ = FT(2.11e-5) * (T / 273)^FT(1.94)
        kᵃ = FT(1e-3) * (FT(4.39) + FT(0.071) * T)
        R = constants.molar_gas_constant
        Mʷ = constants.vapor.molar_mass
        ρʷ = constants.liquid.density
        ℒ = liquid_latent_heat(T, constants)
        pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
        Gᵃ = ρʷ * R * T / (pᵛ⁺ * Dᵛ * Mʷ)
        Gᵇ = ℒ * ρʷ * (ℒ * Mʷ / (R * T) - 1) / (kᵃ * T)
        @test G(FT(1e-2)) ≈ 1 / (Gᵃ + Gᵇ) rtol=1e-3
    end

    @testset "Implicit growth" begin
        dynamics = DropletDynamics(FT)
        Δt = FT(0.02)

        # A haze particle equilibrates with subsaturated air
        𝒮 = FT(-0.02)
        D² = (FT(1.5) * Dᵈ)^2
        for _ in 1:500
            D² = implicit_growth_step(D², 𝒮, T, p, Dᵈ, κ, Δt, dynamics, constants)
        end
        @test sqrt(D²) ≈ equilibrium_diameter(𝒮, Dᵈ, κ, T, constants) rtol=(FT === Float32 ? 1e-3 : 1e-6)

        # Far from the dry limit the Köhler correction is negligible and D² grows linearly, D² = D₀² + 8 G 𝒮 t
        𝒮 = FT(0.01)
        D₀ = FT(20e-6)
        D² = D₀^2
        steps = 50
        for _ in 1:steps
            D² = implicit_growth_step(D², 𝒮, T, p, Dᵈ, κ, Δt, dynamics, constants)
        end
        Dₘ = (D₀ + sqrt(D²)) / 2
        expected = D₀^2 + 8 * growth_coefficient(T, p, Dₘ, dynamics, constants) * 𝒮 * steps * Δt
        @test D² ≈ expected rtol=1e-2

        # Sustained supersaturation above the critical value activates the particle within a minute;
        # below it the particle settles on the haze branch
        𝒮ᶜ = critical_supersaturation(Dᵈ, κ, T, constants)
        Dᶜ = critical_diameter(Dᵈ, κ, T, constants)
        D²_active = Dᵈ^2
        D²_haze = Dᵈ^2
        for _ in 1:3000
            D²_active = implicit_growth_step(D²_active, 2𝒮ᶜ, T, p, Dᵈ, κ, Δt, dynamics, constants)
            D²_haze = implicit_growth_step(D²_haze, 𝒮ᶜ / 2, T, p, Dᵈ, κ, Δt, dynamics, constants)
        end
        @test D²_active > Dᶜ^2
        @test D²_haze < Dᶜ^2
        @test sqrt(D²_haze) ≈ equilibrium_diameter(𝒮ᶜ / 2, Dᵈ, κ, T, constants) rtol=(FT === Float32 ? 1e-2 : 1e-4)

        # Evaporation never shrinks a particle below its dry diameter
        D² = (FT(1.2) * Dᵈ)^2
        for _ in 1:100
            D² = implicit_growth_step(D², FT(-0.5), T, p, Dᵈ, κ, Δt, dynamics, constants)
            @test D² ≥ Dᵈ^2
        end
    end

    @testset "Droplets in an AtmosphereModel" begin
        old_FT = Oceananigans.defaults.FloatType
        Oceananigans.defaults.FloatType = FT
        grid = RectilinearGrid(default_arch, FT; size=(4, 4, 4), x=(0, 400), y=(0, 400), z=(0, 400))
        N = 8
        Dᶜ = critical_diameter(Dᵈ, κ, T, constants)
        D²₀ = (FT(1.5) * Dᵈ)^2
        column(x) = on_architecture(default_arch, fill(FT(x), N))
        droplets = StructArray{Droplet{FT}}((on_architecture(default_arch, FT.(range(50, 350, length=N))),
                                             column(250), column(250),
                                             column(Dᵈ), column(κ), column(D²₀), column(Dᶜ),
                                             column(0), column(0), column(0), column(0)))
        dynamics = DropletDynamics(FT; thermodynamic_constants=constants)
        particles = LagrangianParticles(droplets; dynamics)
        model = AtmosphereModel(grid; particles)
        @test model.particles === particles

        # Uniform air at 0.5 % supersaturation and a uniform wind
        ℋ = FT(1.01)
        set!(model; T, ℋ, u=FT(1))
        Δt = FT(0.02)
        step_lagrangian_particles!(model, Δt)

        # The droplets see the supersaturation the model itself diagnoses at their cells, and
        # advance by exactly one implicit step
        ℋ_model = RelativeHumidityField(model)
        𝒮 = @allowscalar droplets.𝒮[1]
        @test 𝒮 ≈ (@allowscalar ℋ_model[1, 3, 3]) - 1 atol=(FT === Float32 ? 1e-4 : 1e-8)
        Tₙ, qᵛₙ, pₙ = @allowscalar (droplets.T[1], droplets.qᵛ[1], droplets.p[1])
        expected = implicit_growth_step(D²₀, ambient_supersaturation(Tₙ, qᵛₙ, pₙ, constants), Tₙ, pₙ, Dᵈ, κ, Δt, dynamics, constants)
        @test all(Array(droplets.D²) .≈ expected)
        @test all(Array(droplets.x) .≈ FT.(range(50, 350, length=N)) .+ Δt)
        @test activated_fraction(droplets) == 0

        # Sustained supersaturation activates every droplet
        for _ in 1:3000
            step_lagrangian_particles!(model, Δt)
        end
        @test activated_fraction(droplets) == 1
        @test all(Array(activated(droplets)))

        Oceananigans.defaults.FloatType = old_FT
    end
end
