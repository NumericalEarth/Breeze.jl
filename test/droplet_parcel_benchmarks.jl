include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.Thermodynamics: PlanarLiquidSurface, MoistureMassFractions, liquid_latent_heat, mixture_gas_constant,
                             mixture_heat_capacity, supersaturation, dry_air_gas_constant, vapor_gas_constant,
                             saturation_vapor_pressure
using Breeze.ParticleMicrophysics: condensational_growth_step, critical_radius, equilibrium_radius,
                                   droplet_water_mass
using Test

# Closed adiabatic parcel with one superdroplet rising at w(t); returns the time series of 𝒮 and r.
# The full benchmarks (and the comparison with Rogers' equations) are in validation/parcel_benchmarks.
function single_superdroplet_parcel(; T₀, p₀, ℋ₀, rᵈ, κ, N, w, t_end, Δt, parameters, r₀ = nothing)
    constants = ThermodynamicConstants(Float64)
    g = constants.gravitational_acceleration
    pᵛ = ℋ₀ * saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    qᵛ = ε * pᵛ / (p₀ - (1 - ε) * pᵛ)
    T, p = T₀, p₀
    ρ₀ = p / (mixture_gas_constant(MoistureMassFractions(qᵛ), constants) * T)
    n = N / ρ₀

    function 𝒮_of(T, p, qᵛ, qˡ)
        q = MoistureMassFractions(qᵛ, qˡ)
        ρ = p / (mixture_gas_constant(q, constants) * T)
        return supersaturation(T, ρ, q, constants, PlanarLiquidSurface())
    end

    r = isnothing(r₀) ? equilibrium_radius(rᵈ, κ, 𝒮_of(T, p, qᵛ, 0.0), T, constants, parameters) : r₀
    qˡ = n * droplet_water_mass(r, rᵈ, constants)

    times = [0.0]; 𝒮s = [𝒮_of(T, p, qᵛ, qˡ)]; radii = [r]; temperatures = [T]
    for step in 1:round(Int, t_end / Δt)
        wₜ = w((step - 1/2) * Δt)
        q = MoistureMassFractions(qᵛ, qˡ)
        p -= p / (mixture_gas_constant(q, constants) * T) * g * wₜ * Δt
        T -= g * wₜ * Δt / mixture_heat_capacity(q, constants)
        growth = condensational_growth_step(r, rᵈ, κ, 𝒮_of(T, p, qᵛ, qˡ), 1.0, T, p, Δt, constants, parameters)
        r = growth.radius
        Δqˡ = n * growth.condensed_mass
        T += liquid_latent_heat(T, constants) * Δqˡ / mixture_heat_capacity(q, constants)
        qᵛ -= Δqˡ
        qˡ += Δqˡ
        push!(times, step * Δt); push!(𝒮s, 𝒮_of(T, p, qᵛ, qˡ)); push!(radii, r); push!(temperatures, T)
    end
    return (; times, 𝒮 = 𝒮s, r = radii, T = temperatures, constants)
end

@testset "Droplet parcel benchmarks" begin
    @testset "Rogers (1975), Fig. 1" begin
        # 200 cm⁻³ droplets of 8 μm at 800 hPa, 280.15 K and saturation, rising at 10 m/s; continuum growth,
        # no curvature or solute (Rogers' assumptions)
        parameters = DropletPhysicsParameters(Float64; surface_tension = 1e-9, ventilation = NoVentilation(),
                                              kinetic_correction = NoKineticCorrection())
        sol = single_superdroplet_parcel(; T₀ = 280.15, p₀ = 80000.0, ℋ₀ = 1.0, rᵈ = 1e-12, κ = 0.6, N = 200e6,
                                         w = t -> 10.0, t_end = 20.0, Δt = 0.05, parameters, r₀ = 8e-6)
        k = argmax(sol.𝒮)
        # Rogers' figure: 𝒮max = 0.978% at 7.2 s and r = 9.68 μm at 19.9 s
        @test sol.𝒮[k] ≈ 0.0097 rtol = 0.05
        @test 6 < sol.times[k] < 8.5
        @test sol.r[end] ≈ 9.68e-6 rtol = 0.02
    end

    @testset "Arabas and Shima (2017): activation and deactivation at long time steps" begin
        # Sinusoidal updraft, w̄ = 1 m/s over 300 s, 50 cm⁻³ of κ = 0.2, 0.1 μm aerosol
        parameters = DropletPhysicsParameters(Float64; ventilation = NoVentilation())
        w(t) = π / 2 * sin(π * t / 150)
        kw = (; T₀ = 300.0, p₀ = 1e5, ℋ₀ = 0.98, rᵈ = 1e-7, κ = 0.2, N = 50e6 / 1.29, w, t_end = 300.0, parameters)
        fine = single_superdroplet_parcel(; kw..., Δt = 0.01)
        coarse = single_superdroplet_parcel(; kw..., Δt = 0.6)
        constants = fine.constants
        k = argmax(fine.r)
        @test fine.r[k] > critical_radius(1e-7, 0.2, fine.T[k], constants, parameters) # activates
        @test fine.r[end] < critical_radius(1e-7, 0.2, fine.T[end], constants, parameters) # and deactivates
        @test maximum(coarse.r) ≈ fine.r[k] rtol = 0.02
        @test coarse.r[end] < critical_radius(1e-7, 0.2, coarse.T[end], constants, parameters)
        @test maximum(coarse.𝒮) ≈ maximum(fine.𝒮) rtol = 0.05
    end
end
