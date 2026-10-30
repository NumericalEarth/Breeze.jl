include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.ParticleMicrophysics: droplet_water_mass, ice_particle_mass, ventilation_factor,
                                   vapor_diffusivity, thermal_conductivity
using Breeze.Thermodynamics: saturation_vapor_pressure, vapor_gas_constant, dry_air_gas_constant, liquid_latent_heat,
                             ice_latent_heat, mixture_heat_capacity, mixture_gas_constant, supersaturation,
                             reference_value, MoistureMassFractions, PlanarLiquidSurface, PlanarIceSurface
using Oceananigans
using Statistics: mean
using Test

quasirandom(n, k=1) = mod(n * sqrt(2 + k), 1)

# Vapor mass fraction with vapor pressure `saturation_ratio * pᵛ⁺(T)` over `surface` at pressure p
function vapor_mass_fraction(saturation_ratio, T, p, constants, surface)
    pᵛ = saturation_ratio * saturation_vapor_pressure(T, constants, surface)
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    return ε * pᵛ / (p - (1 - ε) * pᵛ)
end

@testset "Ice physics" begin
    for FT in test_float_types()
        constants = ThermodynamicConstants(FT)
        parameters = IcePhysicsParameters(FT)
        T, p = FT(263), FT(8e4)

        @testset "Type stability and allocations [$FT]" begin
            r, 𝒮ⁱ = FT(1e-5), FT(0.1)
            @test @inferred(depositional_growth_step(r, 𝒮ⁱ, one(FT), T, p, FT(1), constants, parameters)) isa
                  NamedTuple{(:radius, :deposited_mass), Tuple{FT, FT}}
            @test @inferred(ice_terminal_velocity(r, T, p, constants, parameters)) isa FT
            depositional_growth_step(r, 𝒮ⁱ, one(FT), T, p, FT(1), constants, parameters)
            @test (@allocated depositional_growth_step(r, 𝒮ⁱ, one(FT), T, p, FT(1), constants, parameters)) == 0
        end

        @testset "Capacitance of spheroids" begin
            @test spheroid_capacitance_ratio(1.0) == 1
            # Continuous through the sphere, and a thin disk of radius a has C = 2a / π
            @test spheroid_capacitance_ratio(1 - 1e-6) ≈ 1 rtol = 1e-5
            @test spheroid_capacitance_ratio(1 + 1e-6) ≈ 1 rtol = 1e-5
            φ = 1e-6
            @test spheroid_capacitance_ratio(φ) * φ^(1 / 3) ≈ 2 / π rtol = 1e-5
            # A sphere has the smallest capacitance of all shapes of equal volume
            @test all(spheroid_capacitance_ratio(φ) > 1 for φ in (0.05, 0.3, 0.9, 1.1, 3.0, 20.0))
            @test IcePhysicsParameters(FT; aspect_ratio=0.1).capacitance_ratio ≈ spheroid_capacitance_ratio(0.1)
        end

        @testset "Analytic r² ∝ t growth law of a sphere [$FT]" begin
            # dm/dt = 4π C 𝒮ⁱ / (Fᵛ + Fᵀ) with C = r, Fᵛ = Rᵛ T / (Dᵛ pᵛ⁺ⁱ), Fᵀ = (ℒⁱ / Rᵛ T - 1) ℒⁱ / (Kᵀ T);
            # with m = 4π ρᴵ r³ / 3, r² = r₀² + 2 𝒮ⁱ t / (ρᴵ (Fᵛ + Fᵀ)). The backward-Euler step is exact.
            plain = IcePhysicsParameters(FT; kinetic_correction=NoKineticCorrection())
            Rᵛ = vapor_gas_constant(constants)
            ℒⁱ = ice_latent_heat(T, constants)
            pᵛ⁺ⁱ = saturation_vapor_pressure(T, constants, PlanarIceSurface())
            Dᵛ = vapor_diffusivity(plain.vapor_diffusivity, T, p)
            Kᵀ = thermal_conductivity(plain.thermal_conductivity, T, p)
            Fᵛ = Rᵛ * T / (Dᵛ * pᵛ⁺ⁱ)
            Fᵀ = (ℒⁱ / (Rᵛ * T) - 1) * ℒⁱ / (Kᵀ * T)
            ρᴵ = plain.density
            r₀, t = FT(2e-5), FT(20)
            rtol = FT == Float64 ? 1e-8 : 1e-4
            for 𝒮ⁱ in (FT(0.15), FT(-0.05))
                r★ = sqrt(r₀^2 + 2 * 𝒮ⁱ * t / (ρᴵ * (Fᵛ + Fᵀ)))
                r, Δm = r₀, zero(FT)
                for _ in 1:20
                    step = depositional_growth_step(r, 𝒮ⁱ, one(FT), T, p, t / 20, constants, plain)
                    r = step.radius
                    Δm += step.deposited_mass
                end
                @test r ≈ r★ rtol = rtol
                @test Δm ≈ ice_particle_mass(r★, plain) - ice_particle_mass(r₀, plain) rtol = 10rtol
                @test depositional_growth_rate(r₀, 𝒮ⁱ, one(FT), T, p, constants, plain) ≈ 𝒮ⁱ / (ρᴵ * (Fᵛ + Fᵀ)) rtol = rtol
            end

            # A spheroid grows as a sphere of radius C = χ r
            plates = IcePhysicsParameters(FT; aspect_ratio=0.2, kinetic_correction=NoKineticCorrection())
            χ = plates.capacitance_ratio
            @test depositional_growth_rate(r₀, FT(0.1), one(FT), T, p, constants, plates) ≈
                  χ * depositional_growth_rate(r₀, FT(0.1), one(FT), T, p, constants, plain) rtol = rtol

            # The gas-kinetic correction slows the growth of small crystals most
            ratio(r) = depositional_growth_rate(r, FT(0.1), one(FT), T, p, constants, parameters) /
                       depositional_growth_rate(r, FT(0.1), one(FT), T, p, constants, plain)
            @test ratio(FT(1e-6)) < ratio(FT(1e-5)) < ratio(FT(1e-4)) < 1
        end

        @testset "Large time steps and complete sublimation [$FT]" begin
            # Growth over a large step converges to the solution of many small steps
            r, rs = FT(2e-6), FT(2e-6)
            big = depositional_growth_step(r, FT(0.1), one(FT), T, p, FT(30), constants, parameters)
            for _ in 1:3000
                rs = depositional_growth_step(rs, FT(0.1), one(FT), T, p, FT(0.01), constants, parameters).radius
            end
            @test big.radius ≈ rs rtol = 1e-2
            @test big.radius > r

            # A small crystal in strongly subsaturated air disappears, returning all of its mass
            r₀ = FT(1e-6)
            r, Δm = r₀, zero(FT)
            for _ in 1:100
                step = depositional_growth_step(r, FT(-0.3), one(FT), T, p, FT(0.1), constants, parameters)
                r = step.radius
                Δm += step.deposited_mass
            end
            @test r == 0
            @test Δm ≈ -ice_particle_mass(r₀, parameters) rtol = sqrt(eps(FT))
            # and stays inactive, even in supersaturated air
            step = depositional_growth_step(zero(FT), FT(0.3), one(FT), T, p, FT(1), constants, parameters)
            @test step.radius == 0 && step.deposited_mass == 0
        end

        @testset "Terminal velocity of ice spheres [$FT]" begin
            𝕎(r) = ice_terminal_velocity(FT(r), T, p, constants, parameters)
            # Stokes regime for small spheres, and slower than Stokes when the Reynolds number is not small
            μ = Breeze.ParticleMicrophysics.dynamic_viscosity(parameters.dynamic_viscosity, T)
            ρ = p / (dry_air_gas_constant(constants) * T)
            𝕎ˢ(r) = 2 * (parameters.density - ρ) * constants.gravitational_acceleration * r^2 / (9μ)
            @test 𝕎(5e-6) ≈ 𝕎ˢ(FT(5e-6)) rtol = 1e-3
            @test 𝕎(1e-4) < 0.8 * 𝕎ˢ(FT(1e-4))
            @test 𝕎(1e-4) > 𝕎(5e-5) > 𝕎(1e-5)
        end
    end
end

#####
##### Ice particles in an AtmosphereModel
#####

function mixed_phase_box_model(arch, FT; Nd = 200, Ni = 20, saturation_ratio = 1, rⁱ = 1e-5)
    L = FT(0.02)
    grid = RectilinearGrid(arch, FT; size=(6, 6, 6), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))
    T₀, p₀ = 263, 80000
    dynamics = BoussinesqDynamics(grid; pressure=p₀, temperature=T₀)
    n = 1:Nd
    droplets = (x = L .* quasirandom.(n, 1), y = L .* quasirandom.(n, 2), z = L .* quasirandom.(n, 3),
                r = fill(5e-6, Nd), rᵈ = fill(5e-8, Nd), κ = fill(0.6, Nd))
    m = 1:Ni
    ice = (x = L .* quasirandom.(m, 4), y = L .* quasirandom.(m, 5), z = L .* quasirandom.(m, 6), r = fill(rⁱ, Ni))
    microphysics = LagrangianMicrophysics(FT; droplets, ice)
    model = AtmosphereModel(grid; dynamics, microphysics)

    constants = model.thermodynamic_constants
    qᵛ₀ = vapor_mass_fraction(saturation_ratio, T₀, p₀, constants, PlanarLiquidSurface())
    set!(model; T=T₀, qᵗ=qᵛ₀, u=(x, y, z) -> sin(2π * x / L) / 20, v=(x, y, z) -> -cos(2π * y / L) / 20)
    return model
end

function water_masses(model)
    grid = model.grid
    V = prod(size(grid)) * minimum_xspacing(grid) * minimum_yspacing(grid) * minimum_zspacing(grid)
    vapor = mean(Array(interior(model.moisture_density))) * V
    constants = model.thermodynamic_constants
    droplets = model.microphysics.droplets.properties
    liquid = sum(droplet_water_mass.(Array(droplets.r), Array(droplets.rᵈ), Ref(constants)))
    ice = sum(ice_particle_mass.(Array(model.microphysics.ice.properties.r), Ref(model.microphysics.ice_physics)))
    return (; vapor, liquid, ice, air = reference_value(model.dynamics.reference_state.density) * V)
end

@testset "LagrangianMicrophysics with ice [$FT]" for FT in test_float_types()
    model = mixed_phase_box_model(default_arch, FT)
    @test length(model.microphysics.ice) == 20
    @test model.microphysics.ice.properties isa AbstractVector{<:IceParticle}
    @test occursin("20 ice particles", sprint(show, model.microphysics))

    # Saturated over liquid at -10 °C, the air is supersaturated over ice by about 10%: ice grows,
    # and total water is conserved
    w₀ = water_masses(model)
    rⁱ₀ = Array(model.microphysics.ice.properties.r)
    T₀ = mean(model.temperature)
    for _ in 1:10
        time_step!(model, 1e-2)
    end
    w₁ = water_masses(model)
    @test all(Array(model.microphysics.ice.properties.r) .> rⁱ₀)
    @test w₁.ice > w₀.ice
    @test (w₁.vapor + w₁.liquid + w₁.ice) ≈ (w₀.vapor + w₀.liquid + w₀.ice) rtol = 100 * eps(FT)

    # The latent heats of condensation and of sublimation warm the air at constant pressure
    constants = model.thermodynamic_constants
    q = MoistureMassFractions(mean(model.microphysical_fields.qᵛ))
    cᵖᵐ = mixture_heat_capacity(q, constants)
    ΔT = (liquid_latent_heat(T₀, constants) * (w₁.liquid - w₀.liquid) +
          ice_latent_heat(T₀, constants) * (w₁.ice - w₀.ice)) / (cᵖᵐ * w₁.air)
    @test mean(model.temperature) - T₀ ≈ ΔT rtol = FT == Float64 ? 1e-3 : 1e-2

    # The ice mass fraction of the particles is diagnosed on the grid
    Nc = prod(size(model.grid))
    @test sum(Array(interior(model.microphysical_fields.qⁱ))) * w₁.air / Nc ≈ w₁.ice rtol = sqrt(eps(FT))
    @test Breeze.AtmosphereModels.ice_mass_fraction(model.microphysics, model) === model.microphysical_fields.qⁱ

    # Ice particles fall and stay in the periodic domain
    ice = model.microphysics.ice.properties
    @test all(>(0), Array(ice.𝕎))
    @test all(z -> 0 ≤ z ≤ FT(0.02), Array(ice.z))

    # Air subsaturated over ice: small ice particles sublimate completely, and their mass returns to the vapor
    model = mixed_phase_box_model(default_arch, FT; Nd = 0, saturation_ratio = 0.7, rⁱ = 3e-7)
    w₀ = water_masses(model)
    for _ in 1:20
        time_step!(model, 1e-1)
    end
    w₁ = water_masses(model)
    @test all(iszero, Array(model.microphysics.ice.properties.r))
    @test w₁.ice == 0
    @test w₁.vapor ≈ w₀.vapor + w₀.ice rtol = 100 * eps(FT)
end

# Growth of a single ice sphere in quiescent air, compared to the growth law at the mean state,
# with and without the correction for the particle's own sink in its cell
function single_ice_growth_error(self_interaction; FT = Float64, N = 16, L = 0.008, Δt = 2e-3, spin_up = 100)
    grid = RectilinearGrid(CPU(), FT; size=(N, N, N), x=(0, L), y=(0, L), z=(0, L),
                           topology=(Periodic, Periodic, Periodic))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    T₀, p₀ = 263, 80000
    dynamics = BoussinesqDynamics(grid, constants; pressure=p₀, temperature=T₀)
    ice = (x = [L / 2 + 2.5e-4], y = [L / 2 + 2.5e-4], z = [L / 2 + 2.5e-4], r = [5e-6])
    microphysics = LagrangianMicrophysics(FT; ice, self_interaction)
    closure = ScalarDiffusivity(ν=1.6e-5, κ=(ρθ=2.2e-5, ρqᵛ=2.5e-5))
    model = AtmosphereModel(grid; dynamics, microphysics, closure, thermodynamic_constants=constants)
    set!(model; T=T₀, qᵗ=vapor_mass_fraction(1.05, T₀, p₀, constants, PlanarIceSurface()))

    for _ in 1:spin_up
        time_step!(model, Δt)
    end

    properties = model.microphysics.ice.properties
    r₀ = Array(properties.r)[1]
    time_step!(model, Δt)
    r₁ = Array(properties.r)[1]
    Φ = (r₁^2 - r₀^2) / 2Δt

    T = mean(model.temperature)
    q = MoistureMassFractions(mean(model.microphysical_fields.qᵛ))
    ρᵗ = p₀ / (mixture_gas_constant(q, constants) * T)
    𝒮ⁱ = supersaturation(T, ρᵗ, q, constants, PlanarIceSurface())
    physics = model.microphysics.ice_physics
    fᵛ = ventilation_factor(r₁, zero(FT), T, p₀, constants, physics)
    Φ★ = depositional_growth_rate(r₁, 𝒮ⁱ, fᵛ, T, p₀, constants, physics)
    return Φ / Φ★ - 1
end

@testset "Self-interaction correction of ice growth" begin
    uncorrected = single_ice_growth_error(nothing)
    corrected = single_ice_growth_error(DiffusiveSelfInteraction())
    # Without the correction the crystal sees its own sink: error ≈ -4π g C ≈ -3% for C / Δ = 0.01,
    # less the periodic images and the diffusion during the time step
    @test uncorrected < -0.015
    @test abs(corrected) < 0.1 * abs(uncorrected)
end
