include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.LagrangianMicrophysics: curvature_coefficient, droplet_water_mass, ventilation_factor,
                                     drag_correction, vapor_diffusivity, thermal_conductivity,
                                     dynamic_viscosity
using Breeze.Thermodynamics: saturation_vapor_pressure, liquid_latent_heat, vapor_gas_constant,
                             dry_air_gas_constant, PlanarLiquidSurface
using Oceananigans.Architectures: on_architecture, device
using Oceananigans.Utils: KernelAbstractions
using .KernelAbstractions: @kernel, @index
using Test

# Deterministic quasi-random numbers in [0, 1): additive recurrence with the fractional
# part of a quadratic irrational (`k` selects the irrational)
quasirandom(n, k=1) = mod(n * sqrt(2 + k), 1)

# Integrate `n` implicit steps of size Δt at fixed ambient conditions; return the final radius
# and the accumulated condensed mass.
function integrate_growth(r, rᵈ, κ, 𝒮, T, p, Δt, n, constants, parameters)
    Δm = zero(r)
    for _ in 1:n
        step = condensational_growth_step(r, rᵈ, κ, 𝒮, one(r), T, p, Δt, constants, parameters)
        r = step.radius
        Δm += step.condensed_mass
    end
    return r, Δm
end

@kernel function _droplet_physics_kernel!(radius, condensed_mass, terminal_velocity, velocity, displacement,
                                          r₀, rᵈ, 𝒮, T, p, κ, Δt, constants, parameters)
    n = @index(Global)
    @inbounds begin
        r = r₀[n]
        fᵛ = ventilation_factor(r, zero(r), T, p, constants, parameters)
        step = condensational_growth_step(r, rᵈ[n], κ, 𝒮[n], fᵛ, T, p, Δt, constants, parameters)
        radius[n] = step.radius
        condensed_mass[n] = step.condensed_mass
        𝕎 = droplet_terminal_velocity(step.radius, T, p, constants, parameters)
        terminal_velocity[n] = 𝕎
        τ = droplet_response_time(step.radius, 𝕎, T, p, constants, parameters)
        g = constants.gravitational_acceleration
        v, Δz = exponential_drag_step(zero(r), zero(r), -g, τ, Δt)
        velocity[n] = v
        displacement[n] = Δz
    end
end

@testset "Droplet physics" begin
    for FT in test_float_types()
        constants = ThermodynamicConstants(FT)
        parameters = DropletPhysicsParameters(FT)
        dilute = DropletPhysicsParameters(FT; solution_activity=DiluteSolutionActivity())
        T = FT(283)
        p = FT(9e4)
        rtol = FT == Float64 ? 1e-8 : 1e-4

        @testset "Type stability and allocations [$FT]" begin
            r, rᵈ, κ, 𝒮 = FT(1e-6), FT(5e-8), FT(0.6), FT(0.005)
            @test @inferred(condensational_growth_step(r, rᵈ, κ, 𝒮, one(FT), T, p, FT(1), constants, parameters)) isa
                  NamedTuple{(:radius, :condensed_mass), Tuple{FT, FT}}
            @test @inferred(equilibrium_supersaturation(r, rᵈ, κ, T, constants, parameters)) isa FT
            @test @inferred(critical_radius(rᵈ, κ, T, constants, parameters)) isa FT
            @test @inferred(equilibrium_radius(rᵈ, κ, FT(-0.1), T, constants, parameters)) isa FT
            @test @inferred(droplet_terminal_velocity(r, T, p, constants, parameters)) isa FT
            @test @inferred(droplet_response_time(r, FT(0.1), T, p, constants, parameters)) isa FT
            @test @inferred(ventilation_factor(r, FT(0.1), T, p, constants, parameters)) isa FT
            @test @inferred(exponential_drag_step(FT(0), FT(1), FT(-9.81), FT(1e-3), FT(1))) isa Tuple{FT, FT}
            condensational_growth_step(r, rᵈ, κ, 𝒮, one(FT), T, p, FT(1), constants, parameters)
            @test (@allocated condensational_growth_step(r, rᵈ, κ, 𝒮, one(FT), T, p, FT(1), constants, parameters)) == 0
        end

        @testset "Analytic r² ∝ t growth without curvature or solute [$FT]" begin
            plain = DropletPhysicsParameters(FT; surface_tension=0, kinetic_correction=NoKineticCorrection())
            Rᵛ = vapor_gas_constant(constants)
            ρᴸ = constants.liquid.density
            ℒˡ = liquid_latent_heat(T, constants)
            pᵛ⁺ = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
            Dᵛ = vapor_diffusivity(plain.vapor_diffusivity, T, p)
            Kᵀ = thermal_conductivity(plain.thermal_conductivity, T, p)
            G = 1 / (ρᴸ * Rᵛ * T / (Dᵛ * pᵛ⁺) + (ℒˡ / (Rᵛ * T) - 1) * ℒˡ * ρᴸ / (Kᵀ * T))

            for 𝒮 in (FT(0.01), FT(-0.01))
                r₀ = FT(10e-6)
                Δt, n = FT(0.5), 20
                r, Δm = integrate_growth(r₀, zero(FT), zero(FT), 𝒮, T, p, Δt, n, constants, plain)
                @test r^2 ≈ r₀^2 + 2 * G * 𝒮 * n * Δt rtol=10rtol
                @test Δm ≈ droplet_water_mass(r, 0, constants) - droplet_water_mass(r₀, 0, constants) rtol=sqrt(rtol)
            end
        end

        @testset "Köhler curve: critical radius and supersaturation [$FT]" begin
            κ = FT(0.6)
            A = curvature_coefficient(T, constants, dilute)
            errors = map((FT(2e-8), FT(1e-7), FT(1e-6))) do rᵈ
                rᶜ = critical_radius(rᵈ, κ, T, constants, dilute)
                𝒮ᶜ = critical_supersaturation(rᵈ, κ, T, constants, dilute)
                rᶜ★ = sqrt(3κ * rᵈ^3 / A)
                𝒮ᶜ★ = expm1(sqrt(4A^3 / (27κ * rᵈ^3)))
                # The critical radius is the maximum of the Köhler curve
                ϵ = FT == Float64 ? 1e-3 : FT(0.03) # the Köhler maximum is flat
                @test equilibrium_supersaturation(rᶜ * (1 + ϵ), rᵈ, κ, T, constants, dilute) < 𝒮ᶜ
                @test equilibrium_supersaturation(rᶜ * (1 - ϵ), rᵈ, κ, T, constants, dilute) < 𝒮ᶜ
                (abs(rᶜ / rᶜ★ - 1), abs(𝒮ᶜ / 𝒮ᶜ★ - 1))
            end
            # Analytic approximations become exact as rᶜ / rᵈ → ∞
            @test errors[1][1] < 0.05 && errors[1][2] < 0.05
            @test errors[3][1] < 1e-4 && errors[3][2] < (FT == Float64 ? 1e-4 : 1e-3)
            @test errors[1][1] > errors[2][1] > errors[3][1]

            # κ-Köhler and dilute activities agree for dilute droplets
            rᵈ = FT(1e-7)
            @test critical_supersaturation(rᵈ, κ, T, constants, parameters) ≈
                  critical_supersaturation(rᵈ, κ, T, constants, dilute) rtol=1e-3
        end

        @testset "Implicit step reaches the haze equilibrium for large Δt [$FT]" begin
            κ = FT(0.6)
            for rᵈ in (FT(1e-8), FT(5e-8), FT(2e-7)), 𝒮 in (FT(-0.3), FT(-0.05), FT(-0.005)),
                p_params in (parameters, dilute)

                rᵉ = equilibrium_radius(rᵈ, κ, 𝒮, T, constants, p_params)
                @test equilibrium_supersaturation(rᵉ, rᵈ, κ, T, constants, p_params) ≈ 𝒮 rtol=sqrt(rtol) atol=sqrt(rtol)
                for r₀ in (rᵈ, 3rᵈ, FT(2e-6))
                    r = condensational_growth_step(r₀, rᵈ, κ, 𝒮, one(FT), T, p, FT(1e5), constants, p_params).radius
                    @test r ≈ rᵉ rtol=1e-3
                end
            end

            # Just below the critical supersaturation, haze droplets do not activate spuriously
            rᵈ = FT(5e-8)
            𝒮ᶜ = critical_supersaturation(rᵈ, κ, T, constants, parameters)
            rᶜ = critical_radius(rᵈ, κ, T, constants, parameters)
            for Δt in (FT(1), FT(10), FT(1e4))
                r = condensational_growth_step(rᵈ, rᵈ, κ, FT(0.95) * 𝒮ᶜ, one(FT), T, p, Δt, constants, parameters).radius
                @test rᵈ < r < rᶜ
            end
            # ... while above it they activate
            r = condensational_growth_step(rᵈ, rᵈ, κ, FT(1.05) * 𝒮ᶜ, one(FT), T, p, FT(1e4), constants, parameters).radius
            @test r > rᶜ
        end

        @testset "First-order convergence in Δt and mass bookkeeping [$FT]" begin
            if FT == Float64
                rᵈ, κ, 𝒮 = 5e-8, 0.6, 0.004
                # An activated droplet, and a haze droplet that activates during the integration
                # (whose error is first order once Δt resolves the activation)
                for (r₀, t, steps, nʳᵉᶠ) in ((1e-6, 4.0, (8, 16, 32, 64), 2^14),
                                            (2e-7, 4.0, (1024, 2048, 4096, 8192), 2^18))
                    rₑ, _ = integrate_growth(r₀, rᵈ, κ, 𝒮, T, p, t / nʳᵉᶠ, nʳᵉᶠ, constants, parameters)
                    errors = map(steps) do n
                        r, Δm = integrate_growth(r₀, rᵈ, κ, 𝒮, T, p, t / n, n, constants, parameters)
                        @test Δm ≈ droplet_water_mass(r, rᵈ, constants) - droplet_water_mass(r₀, rᵈ, constants) rtol=1e-10
                        abs(r - rₑ)
                    end
                    ratios = errors[1:end-1] ./ errors[2:end]
                    @info "Implicit growth step from r₀ = $r₀: errors $errors, convergence ratios $ratios"
                    @test all(ρ -> 1.8 < ρ < 2.2, ratios)
                end
            end

            # In single precision the condensed mass stays accurate for tiny increments
            r₀, rᵈ, κ, 𝒮 = FT(10e-6), FT(1e-7), FT(0.6), FT(0.002)
            r, Δm = integrate_growth(r₀, rᵈ, κ, 𝒮, T, p, FT(1e-4), 1000, constants, parameters)
            r64, Δm64 = integrate_growth(Float64(r₀), Float64(rᵈ), Float64(κ), Float64(𝒮), Float64(T), Float64(p),
                                         1e-4, 1000, ThermodynamicConstants(Float64), DropletPhysicsParameters(Float64))
            @test Δm ≈ Δm64 rtol=FT == Float64 ? 1e-10 : 1e-3
        end

        @testset "Stability for large time steps [$FT]" begin
            all_finite = true
            above_dry = true
            for n in 1:2000
                rᵈ = FT(10^(-8.5 + 2.5 * quasirandom(n, 1)))   # 3 nm – 1 μm
                r₀ = rᵈ * FT(1 + 300 * quasirandom(n, 2)^3)
                κ = FT(0.05 + 1.2 * quasirandom(n, 3))
                𝒮 = FT(-0.5 + 0.52 * quasirandom(n, 4))
                Δt = FT(10^(-3 + 4 * quasirandom(n, 5)))       # 1 ms – 10 s
                Tₙ = FT(250 + 50 * quasirandom(n, 6))
                step = condensational_growth_step(r₀, rᵈ, κ, 𝒮, one(FT), Tₙ, p, Δt, constants, parameters)
                all_finite &= isfinite(step.radius) & isfinite(step.condensed_mass)
                above_dry &= step.radius ≥ rᵈ
            end
            @test all_finite
            @test above_dry
        end

        @testset "Terminal velocity [$FT]" begin
            # Gunn and Kinzer (1949), 1013 hPa and 20 °C: diameter (mm) => fall speed (m/s)
            gunn_kinzer = (0.5 => 2.06, 1.0 => 4.03, 1.4 => 5.17, 2.0 => 6.49, 3.0 => 8.06, 4.0 => 8.83, 5.0 => 9.09)
            Tˢᵗ, pˢᵗ = FT(293.15), FT(101325)
            for (d, 𝕎ᴳᴷ) in gunn_kinzer
                𝕎 = droplet_terminal_velocity(FT(d * 1e-3 / 2), Tˢᵗ, pˢᵗ, constants, parameters)
                @test 𝕎 ≈ 𝕎ᴳᴷ rtol=0.03
            end

            # Stokes regime, and the cheaper laws, for a 10 μm cloud droplet
            r = FT(10e-6)
            𝕎ᴮ = droplet_terminal_velocity(r, Tˢᵗ, pˢᵗ, constants, parameters)
            stokes = DropletPhysicsParameters(FT; terminal_velocity=StokesTerminalVelocity())
            rogers_yau = DropletPhysicsParameters(FT; terminal_velocity=RogersYauTerminalVelocity(FT))
            @test droplet_terminal_velocity(r, Tˢᵗ, pˢᵗ, constants, stokes) ≈ 𝕎ᴮ rtol=0.02
            @test droplet_terminal_velocity(r, Tˢᵗ, pˢᵗ, constants, rogers_yau) ≈ 𝕎ᴮ rtol=0.05
        end

        @testset "Exponential drag integrator [$FT]" begin
            g = constants.gravitational_acceleration
            u, v₀, τ = FT(0.3), FT(-1.2), FT(2e-3)
            for Δt in (FT(1e-5), FT(1e-3), FT(1))
                v, Δz = exponential_drag_step(v₀, u, -g, τ, Δt)
                v∞ = u - g * τ
                @test v ≈ v∞ + (v₀ - v∞) * exp(-Δt / τ) rtol=10rtol atol=10eps(FT)
                @test Δz ≈ v∞ * Δt + (v₀ - v∞) * τ * (1 - exp(-Δt / τ)) rtol=sqrt(rtol)
                # Exact for constant u: two half steps equal one full step
                v½, Δz½ = exponential_drag_step(v₀, u, -g, τ, Δt / 2)
                v₁, Δz₁ = exponential_drag_step(v½, u, -g, τ, Δt / 2)
                @test v₁ ≈ v rtol=sqrt(rtol) atol=10eps(FT)
                @test Δz½ + Δz₁ ≈ Δz rtol=sqrt(rtol)
            end

            # Stable for Δt ≫ τ, even τ → 0
            v, Δz = exponential_drag_step(FT(5), u, -g, FT(1e-12), FT(10))
            @test v ≈ u && Δz ≈ 10u

            # With the response time updated from the slip speed, a droplet released at rest in
            # still air converges to the terminal velocity of the Schiller–Naumann drag law,
            # which agrees with Beard (1976) for cloud droplets.
            Tˢᵗ, pˢᵗ = FT(293.15), FT(101325)
            for r in (FT(10e-6), FT(30e-6))
                v = zero(FT)
                for _ in 1:200
                    τᵣ = droplet_response_time(r, v, Tˢᵗ, pˢᵗ, constants, parameters)
                    v, _ = exponential_drag_step(v, zero(FT), -g, τᵣ, FT(1e-3))
                end
                τᵣ = droplet_response_time(r, v, Tˢᵗ, pˢᵗ, constants, parameters)
                @test -v ≈ g * τᵣ rtol=sqrt(rtol)
                @test -v ≈ droplet_terminal_velocity(r, Tˢᵗ, pˢᵗ, constants, parameters) rtol=0.05
            end
            @test drag_correction(zero(FT)) == 1
        end

        @testset "Droplet physics in a kernel on $(typeof(default_arch)) [$FT]" begin
            arch = default_arch
            N = 1024
            rᵈ = [FT(10^(-8 + 1.5 * quasirandom(n, 1))) for n in 1:N]
            r₀ = [rᵈ[n] * FT(1 + 100 * quasirandom(n, 2)) for n in 1:N]
            𝒮 = [FT(-0.1 + 0.11 * quasirandom(n, 4)) for n in 1:N]
            κ, Δt = FT(0.6), FT(0.5)

            on(a) = on_architecture(arch, a)
            outputs = [on(zeros(FT, N)) for _ in 1:5]
            kernel! = _droplet_physics_kernel!(device(arch), 256)
            kernel!(outputs..., on(r₀), on(rᵈ), on(𝒮), T, p, κ, Δt, constants, parameters; ndrange=N)
            KernelAbstractions.synchronize(device(arch))
            radius, condensed_mass, terminal_velocity, velocity, displacement = Array.(outputs)

            expected = [condensational_growth_step(r₀[n], rᵈ[n], κ, 𝒮[n],
                                                   ventilation_factor(r₀[n], zero(FT), T, p, constants, parameters),
                                                   T, p, Δt, constants, parameters) for n in 1:N]
            @test all(isfinite, radius) && all(isfinite, displacement)
            @test all(radius .≥ rᵈ)
            @test radius ≈ [e.radius for e in expected] rtol=sqrt(rtol)
            @test condensed_mass ≈ [e.condensed_mass for e in expected] rtol=sqrt(sqrt(rtol))
            @test all(terminal_velocity .> 0)
            @test all(velocity .< 0)
        end
    end
end
