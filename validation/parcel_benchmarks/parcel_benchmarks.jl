# # Parcel benchmarks for droplet activation and condensational growth
#
# Closed adiabatic parcels carrying Lagrangian (super)droplets, integrated with Breeze's moist
# thermodynamics and `Breeze.ParticleMicrophysics.condensational_growth_step`, compared with
# published parcel results:
#
#   1. Rogers (1975), Fig. 1: monodisperse 8 μm droplets at 200 cm⁻³ in a 10 m/s updraft.
#      Includes an independent integration of Rogers' own equations to explain why
#      CloudMicrophysics.jl's `parcel/Example_Liquid_only.jl` peaks later than the paper and PySDM.
#   2. Kreidenweis et al. (2003) cloud base and liquid water 1.2 km above it (PySDM's Table 3 checks).
#   3. pyrcel's documentation case: two-mode aerosol, peak supersaturation 0.62%.
#   4. Jensen and Nugent (2017), Table 4 upper rows: Sc case, modified polluted aerosol.
#   5. Arabas and Shima (2017): activation/deactivation hysteresis under a sinusoidal updraft;
#      convergence of the implicit growth step at the paper's long time steps.
#
# The setups (numbers only) follow the papers and the PySDM / pyrcel examples.
#
#   julia --project=validation/parcel_benchmarks validation/parcel_benchmarks/parcel_benchmarks.jl

using Breeze
using Breeze.Thermodynamics: PlanarLiquidSurface, MoistureMassFractions, liquid_latent_heat, mixture_gas_constant,
                             mixture_heat_capacity, supersaturation, dry_air_gas_constant, vapor_gas_constant,
                             saturation_vapor_pressure
using Breeze.ParticleMicrophysics: condensational_growth_step, equilibrium_radius, critical_radius,
                                   droplet_water_mass
using Printf
using Statistics: mean

const output_directory = joinpath(@__DIR__, "output")

#####
##### A closed adiabatic parcel with superdroplets
#####

mutable struct Parcel
    T :: Float64
    p :: Float64
    qᵛ :: Float64 # specific humidity
    qˡ :: Float64 # liquid (per kg of moist air, conserved in the closed parcel)
    z :: Float64
end

function parcel_supersaturation(parcel::Parcel, constants)
    q = MoistureMassFractions(parcel.qᵛ, parcel.qˡ)
    ρ = parcel.p / (mixture_gas_constant(q, constants) * parcel.T)
    return supersaturation(parcel.T, ρ, q, constants, PlanarLiquidSurface())
end

density(parcel::Parcel, constants) =
    parcel.p / (mixture_gas_constant(MoistureMassFractions(parcel.qᵛ, parcel.qˡ), constants) * parcel.T)

# Vapor specific humidity with vapor pressure ℋ pᵛ⁺(T) at total pressure p
function specific_humidity_from_relative_humidity(ℋ, T, p, constants)
    pᵛ = ℋ * saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
    ε = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
    return ε * pᵛ / (p - (1 - ε) * pᵛ)
end

"""
    run_parcel(; T₀, p₀, ℋ₀, rᵈ, n, κ, w, t_end, Δt, constants, parameters, r₀=nothing, output_interval)

Integrate a closed parcel rising with vertical velocity `w(t)`. `n` is the number of droplets per kilogram
of air for each superdroplet of dry radius `rᵈ`. Each step: hydrostatic adiabatic expansion
(dp = -ρ g w Δt, cᵖᵐ dT = -g w Δt), then implicit condensational growth of every superdroplet at the
updated supersaturation, then latent heating and vapor removal.
"""
function run_parcel(; T₀, p₀, ℋ₀, rᵈ, n, κ, w, t_end, Δt, constants, parameters,
                    r₀ = nothing, z₀ = 0.0, output_interval = Δt)
    qᵛ = specific_humidity_from_relative_humidity(ℋ₀, T₀, p₀, constants)
    parcel = Parcel(T₀, p₀, qᵛ, 0.0, z₀)
    𝒮₀ = parcel_supersaturation(parcel, constants)
    r = isnothing(r₀) ? [equilibrium_radius(rᵈ[i], κ[i], 𝒮₀, T₀, constants, parameters) for i in eachindex(rᵈ)] :
                        copy(r₀)
    # The initial liquid is added to the prescribed vapor, which keeps the initial relative humidity
    parcel.qˡ = sum(n[i] * droplet_water_mass(r[i], rᵈ[i], constants) for i in eachindex(r))

    g = constants.gravitational_acceleration
    Nt = round(Int, t_end / Δt)
    every = max(1, round(Int, output_interval / Δt))
    record = (t = [0.0], z = [z₀], T = [parcel.T], p = [parcel.p], 𝒮 = [parcel_supersaturation(parcel, constants)],
              qᵛ = [parcel.qᵛ], qˡ = [parcel.qˡ], r = [copy(r)])

    for step in 1:Nt
        t = (step - 1/2) * Δt
        wₜ = w(t)
        q = MoistureMassFractions(parcel.qᵛ, parcel.qˡ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        ρ = density(parcel, constants)
        parcel.p -= ρ * g * wₜ * Δt
        parcel.T -= g * wₜ * Δt / cᵖᵐ
        parcel.z += wₜ * Δt

        𝒮 = parcel_supersaturation(parcel, constants)
        Δqˡ = 0.0
        for i in eachindex(r)
            growth = condensational_growth_step(r[i], rᵈ[i], κ[i], 𝒮, 1.0, parcel.T, parcel.p, Δt,
                                                constants, parameters)
            r[i] = growth.radius
            Δqˡ += n[i] * growth.condensed_mass
        end
        q = MoistureMassFractions(parcel.qᵛ, parcel.qˡ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        parcel.qᵛ -= Δqˡ
        parcel.qˡ += Δqˡ
        parcel.T += liquid_latent_heat(parcel.T, constants) * Δqˡ / cᵖᵐ

        if step % every == 0
            push!(record.t, step * Δt); push!(record.z, parcel.z); push!(record.T, parcel.T)
            push!(record.p, parcel.p); push!(record.𝒮, parcel_supersaturation(parcel, constants))
            push!(record.qᵛ, parcel.qᵛ); push!(record.qˡ, parcel.qˡ); push!(record.r, copy(r))
        end
    end
    return record
end

droplet_parameters(; αᶜ = 1.0, αᵀ = 1.0, σ = 0.072, kinetic = true) =
    DropletPhysicsParameters(Float64; surface_tension = σ, ventilation = NoVentilation(),
                             kinetic_correction = kinetic ? KineticCorrection(Float64; condensation_coefficient = αᶜ,
                                                                             thermal_accommodation_coefficient = αᵀ) :
                                                            NoKineticCorrection())

# Deterministic logarithmic sampling of a sum of lognormal number distributions
function lognormal_superdroplets(modes, Nsd; rmin = nothing, rmax = nothing)
    rmin = isnothing(rmin) ? minimum(r * σ^-4 for (_, r, σ) in modes) : rmin
    rmax = isnothing(rmax) ? maximum(r * σ^4 for (_, r, σ) in modes) : rmax
    edges = exp.(range(log(rmin), log(rmax), length = Nsd + 1))
    cdf(x, N, r, σ) = N * (1 + erf_approx(log(x / r) / (sqrt(2) * log(σ)))) / 2
    counts = [sum(cdf(edges[i+1], m...) - cdf(edges[i], m...) for m in modes) for i in 1:Nsd]
    radii = sqrt.(edges[1:end-1] .* edges[2:end])
    return radii, counts
end

# erf by Abramowitz and Stegun 7.1.26 (absolute error < 1.5e-7, ample for sampling weights)
function erf_approx(x)
    t = 1 / (1 + 0.3275911 * abs(x))
    y = 1 - t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * exp(-x^2)
    return sign(x) * y
end

peak(record) = (i = argmax(record.𝒮); (𝒮 = record.𝒮[i], t = record.t[i], z = record.z[i]))

#####
##### 1. Rogers (1975)
#####

# Rogers' own parcel equations and constants (his eqs. 1-10 and A.1-A.3), integrated independently, to
# attribute the difference between Rogers' figure (and PySDM, which integrates these equations) and
# CloudMicrophysics.jl's parcel model. Switches reproduce CloudMicrophysics' choices one at a time.
Base.@kwdef struct RogersSystem
    constant_coefficients :: Bool = false # CloudMicrophysics: D = 2.26e-5, K = 0.024, G with (ℒ/RᵛT - 1)
    constant_number_concentration :: Bool = false # CloudMicrophysics: N per volume fixed as the parcel expands
    Δt :: Float64 = 1e-3
    euler :: Bool = false
end

function rogers_1975_equations(sys::RogersSystem; t_end = 20.0)
    R = 287.05; g = 9.80665; ℒ = 2.5e6; ε = 0.622; cᵖ = 1005.0; ρᴸ = 1000.0; Rᵛ = R / ε
    eˢ(T) = 2.75e12 * exp(-5.44e3 / T) * 0.1 # μbar → Pa
    K(T) = sys.constant_coefficients ? 0.024 : 2.42e-2 * (393 / (T + 120)) * (T / 273)^1.5
    D(p, T) = sys.constant_coefficients ? 2.26e-5 : 8.28 / 2.42 * T / (10p) * K(T) # p in dyn/cm² = 10 p[Pa]
    function G(p, T)
        Fk = sys.constant_coefficients ? ℒ / (K(T) * T) * (ℒ / (Rᵛ * T) - 1) * ρᴸ : ℒ^2 * ε * ρᴸ / (K(T) * R * T^2)
        Fd = ρᴸ * Rᵛ * T / (D(p, T) * eˢ(T))
        return 1 / (Fk + Fd)
    end
    p₀ = 80000.0; T₀ = 280.15; w = 10.0
    ρ₀ = p₀ / (R * T₀)
    ν₀ = 200e6 / ρ₀
    function rhs(y)
        p, T, Sᵢ, rᵢ = y
        ρ = p / (R * T)
        ν = sys.constant_number_concentration ? 200e6 / ρ : ν₀
        dp = -ρ * g * w
        dr = G(p, T) * (Sᵢ - 1) / rᵢ
        dχ = 4π * ρᴸ * ν * rᵢ^2 * dr
        dT = T * R / cᵖ * dp / p + ℒ / cᵖ * dχ
        Q₁ = ℒ * g * ε / (R * cᵖ * T^2) - g / (R * T)
        Q₂ = R * T / (ε * eˢ(T)) + ε * ℒ^2 / (cᵖ * T * p)
        dS = Q₁ * w - ρ * Q₂ * dχ
        return [dp, dT, dS, dr]
    end
    y = [p₀, T₀, 1.0, 8e-6]
    Nt = round(Int, t_end / sys.Δt)
    t = [0.0]; S = [0.0]; r = [8e-6]
    for n in 1:Nt
        if sys.euler
            y = y .+ sys.Δt .* rhs(y)
        else
            k₁ = rhs(y); k₂ = rhs(y .+ sys.Δt / 2 .* k₁); k₃ = rhs(y .+ sys.Δt / 2 .* k₂); k₄ = rhs(y .+ sys.Δt .* k₃)
            y = y .+ sys.Δt / 6 .* (k₁ .+ 2k₂ .+ 2k₃ .+ k₄)
        end
        push!(t, n * sys.Δt); push!(S, y[3] - 1); push!(r, y[4])
    end
    return (; t, 𝒮 = S, r)
end

function rogers_1975(io; Δt = 0.01, kinetic = false)
    constants = ThermodynamicConstants(Float64)
    parameters = droplet_parameters(; kinetic, σ = 1e-9) # Rogers neglects curvature and solute
    p₀, T₀ = 80000.0, 280.15
    qᵛ₀ = specific_humidity_from_relative_humidity(1, T₀, p₀, constants)
    ρ₀ = p₀ / (mixture_gas_constant(MoistureMassFractions(qᵛ₀), constants) * T₀)
    record = run_parcel(; T₀, p₀, ℋ₀ = 1.0, rᵈ = [1e-12], n = [200e6 / ρ₀], κ = [0.6], w = t -> 10.0,
                        t_end = 20.0, Δt, constants, parameters, r₀ = [8e-6])
    pk = peak(record)
    i = findfirst(≥(19.9 - Δt / 2), record.t)
    return (; 𝒮max = pk.𝒮, tmax = pk.t, r20 = record.r[i][1])
end

function rogers_section(io)
    println(io, "## 1. Rogers (1975), Fig. 1: N = 200 cm⁻³, r₀ = 8 μm, 800 hPa, 280.15 K, 𝒮₀ = 0, w = 10 m/s\n")
    println(io, "Reference: Rogers' figure (digitized in CloudMicrophysics.jl): 𝒮max = 0.978% at 7.16 s (0.963% at 9.79 s), r(19.9 s) = 9.68 μm.")
    println(io, "PySDM (Rogers' equations, LSODA) asserts 𝒮max = 0.97% at 7 ± 0.5 s.\n")
    println(io, "| run | 𝒮max (%) | t(𝒮max) (s) | r(19.9 s) (μm) |")
    println(io, "|---|---|---|---|")
    row(name, x) = @printf(io, "| %s | %.3f | %.2f | %.2f |\n", name, 100x.𝒮max, x.tmax, 1e6x.r20)
    function eqrow(name, sys)
        sol = rogers_1975_equations(sys)
        i = argmax(sol.𝒮)
        j = findfirst(≥(19.9 - sys.Δt / 2), sol.t)
        row(name, (𝒮max = sol.𝒮[i], tmax = sol.t[i], r20 = sol.r[j]))
    end
    eqrow("Rogers' equations and constants (RK4, 1 ms)", RogersSystem())
    eqrow("  + CloudMicrophysics' constant D = 2.26e-5, K = 0.024", RogersSystem(constant_coefficients = true))
    eqrow("  + fixed N per volume", RogersSystem(constant_coefficients = true, constant_number_concentration = true))
    eqrow("  + forward Euler, Δt = 0.5 s (CloudMicrophysics' Example_Liquid_only)",
          RogersSystem(constant_coefficients = true, constant_number_concentration = true, Δt = 0.5, euler = true))
    eqrow("Rogers' equations, forward Euler Δt = 0.5 s only", RogersSystem(Δt = 0.5, euler = true))
    row("Breeze, continuum growth (no kinetic correction), Δt = 0.01 s", rogers_1975(io))
    row("Breeze, continuum growth, Δt = 0.1 s", rogers_1975(io; Δt = 0.1))
    row("Breeze, kinetic correction αᶜ = 1 (default)", rogers_1975(io; kinetic = true))
    println(io)
end

#####
##### 2. Kreidenweis et al. (2003)
#####

function kreidenweis_2003(; Nsd = 100, Δt = 0.1, αᶜ = 1.0)
    constants = ThermodynamicConstants(Float64; gravitational_acceleration = 10) # as in PySDM's setup
    parameters = droplet_parameters(; αᶜ)
    rᵈ, counts = lognormal_superdroplets(((566e6, 0.04e-6, 2.0),), Nsd) # per kg (PySDM: ρ₀ = 1)
    κ = fill(0.61, Nsd)
    record = run_parcel(; T₀ = 285.2, p₀ = 95000.0, ℋ₀ = 0.95, rᵈ, n = counts, κ, w = t -> 0.5,
                        t_end = 2596.0, Δt, constants, parameters, output_interval = 1.0)
    base = findfirst(>(0), record.𝒮)
    ρᴸ = constants.liquid.density
    # Liquid water mixing ratio (per kg of dry air) in droplets with 0.5 μm < r < 25 μm
    rᶜ = record.r[end]
    cloud = 0.5e-6 .< rᶜ .< 25e-6
    qˡᶜ = sum(counts[i] * droplet_water_mass(rᶜ[i], rᵈ[i], constants) for i in eachindex(rᶜ) if cloud[i])
    qᵈ = 1 - record.qᵛ[end] - record.qˡ[end]
    pk = peak(record)
    return (; z_base = record.z[base], p_base = record.p[base], T_base = record.T[base], z_end = record.z[end],
            lwc = 1e3 * qˡᶜ / qᵈ, lwc_total = 1e3 * record.qˡ[end] / qᵈ, 𝒮max = pk.𝒮, z𝒮max = pk.z)
end

function kreidenweis_section(io)
    println(io, "## 2. Kreidenweis et al. (2003): 950 hPa, 285.2 K, RH 95%, w = 0.5 m/s, 566 mg⁻¹ NH₄HSO₄ (r = 0.04 μm, σ = 2, κ = 0.61)\n")
    println(io, "Reference (PySDM's Table-3 checks): cloud base z = 98 m, p = 939 hPa (±0.5%), T = 284.2 K (±0.5%); ",
                "liquid water mixing ratio 2.17 g/kg (±2%) at z = 1.3 km.\n")
    println(io, "| run | z_base (m) | p_base (hPa) | T_base (K) | z_end (m) | LWC 0.5-25 μm (g/kg) | LWC all (g/kg) | 𝒮max (%) |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for (name, kw) in (("αᶜ = 1, Δt = 0.1 s, 100 SD", (;)), ("αᶜ = 1, Δt = 1 s", (; Δt = 1.0)),
                       ("αᶜ = 0.036", (; αᶜ = 0.036)), ("αᶜ = 1, 30 SD", (; Nsd = 30)))
        x = kreidenweis_2003(; kw...)
        @printf(io, "| %s | %.0f | %.1f | %.2f | %.0f | %.3f | %.3f | %.3f |\n", name, x.z_base, x.p_base / 100,
                x.T_base, x.z_end, x.lwc, x.lwc_total, 100x.𝒮max)
    end
    println(io)
end

#####
##### 3. pyrcel documentation case
#####

function pyrcel_case(; αᶜ = 0.3, Nsd = 50, Δt = 0.05)
    constants = ThermodynamicConstants(Float64)
    parameters = droplet_parameters(; αᶜ)
    p₀, T₀ = 77500.0, 274.0
    qᵛ₀ = specific_humidity_from_relative_humidity(0.98, T₀, p₀, constants)
    ρ₀ = p₀ / (mixture_gas_constant(MoistureMassFractions(qᵛ₀), constants) * T₀)
    r₁, n₁ = lognormal_superdroplets(((850e6, 15e-9, 1.6),), Nsd)
    r₂, n₂ = lognormal_superdroplets(((10e6, 850e-9, 1.2),), Nsd)
    rᵈ = vcat(r₁, r₂); n = vcat(n₁, n₂) ./ ρ₀; κ = vcat(fill(0.54, Nsd), fill(1.2, Nsd))
    record = run_parcel(; T₀, p₀, ℋ₀ = 0.98, rᵈ, n, κ, w = t -> 1.0, t_end = 250.0, Δt, constants, parameters,
                        output_interval = 0.5)
    pk = peak(record)
    return (; 𝒮max = pk.𝒮, z𝒮max = pk.z, T_end = record.T[end], 𝒮_end = record.𝒮[end])
end

function pyrcel_section(io)
    println(io, "## 3. pyrcel docs case: 775 hPa, 274 K, RH 98%, w = 1 m/s; 850 cm⁻³ (15 nm, σ 1.6, κ 0.54) + 10 cm⁻³ (850 nm, σ 1.2, κ 1.2)\n")
    println(io, "Reference (pyrcel, αᶜ = 0.3): 𝒮max = 0.62%, T(250 m) = 272.2 K, 𝒮(250 m) = 0.139% (PySDM xfails this last one).\n")
    println(io, "| run | 𝒮max (%) | z(𝒮max) (m) | T(250 m) (K) | 𝒮(250 m) (%) |")
    println(io, "|---|---|---|---|---|")
    for (name, kw) in (("αᶜ = 0.3 (pyrcel)", (;)), ("αᶜ = 0.3, Δt = 0.5 s", (; Δt = 0.5)),
                       ("αᶜ = 1", (; αᶜ = 1.0)), ("αᶜ = 0.036", (; αᶜ = 0.036)))
        x = pyrcel_case(; kw...)
        @printf(io, "| %s | %.3f | %.1f | %.2f | %.3f |\n", name, 100x.𝒮max, x.z𝒮max, x.T_end, 100x.𝒮_end)
    end
    println(io)
end

#####
##### 4. Jensen and Nugent (2017), Table 4 upper rows
#####

function jensen_nugent_2017(; αᶜ = 0.036, αᵀ = 0.7, Nsd = 100, Δt = 0.1)
    constants = ThermodynamicConstants(Float64)
    parameters = droplet_parameters(; αᶜ, αᵀ)
    p₀, T₀, ℋ₀ = 93850.0, 284.3, 0.8561
    qᵛ₀ = specific_humidity_from_relative_humidity(ℋ₀, T₀, p₀, constants)
    pᵛ = ℋ₀ * saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
    ρᵈ₀ = (p₀ - pᵛ) / (dry_air_gas_constant(constants) * T₀)
    ρ₀ = p₀ / (mixture_gas_constant(MoistureMassFractions(qᵛ₀), constants) * T₀)
    # Modified polluted aerosol, number per unit volume at the initial state (PySDM: per volume of dry-air density)
    rᵈ, counts = lognormal_superdroplets(((48e6, 0.029e-6, 1.36), (114e6, 0.071e-6, 1.57)), Nsd)
    n = counts ./ ρᵈ₀ .* (1 - qᵛ₀) # per kg of moist air
    κ = fill(1.28, Nsd)
    w(t) = t < 1500 ? 0.4 : -0.4
    record = run_parcel(; T₀, p₀, ℋ₀, rᵈ, n, κ, w, t_end = 2250.0, Δt, constants, parameters, z₀ = 600.0,
                        output_interval = 1.0)
    rows = []
    for (label, heights, phase) in (("ascent", (50, 100, 150, 200, 250, 300), t -> t < 1500),
                                    ("descent", (250, 200, 150, 100, 50), t -> t ≥ 1500))
        for h in heights
            k = findfirst(i -> phase(record.t[i]) && round(record.z[i] - 900) == h, eachindex(record.t))
            r = record.r[k]
            rᶜ = [critical_radius(rᵈ[i], κ[i], record.T[k], constants, parameters) for i in eachindex(r)]
            active = r .> rᶜ
            N = sum(n[active])
            r̄ = sum(n[active] .* r[active]) / N
            σ = sqrt(sum(n[active] .* (r[active] .- r̄) .^ 2) / N)
            push!(rows, (label, h, 1e6r̄, 1e6σ, σ / r̄))
        end
    end
    return (; rows, 𝒮max = maximum(record.𝒮), rmax_end = 1e6 * maximum(record.r[end]))
end

const jensen_nugent_table_4 = Dict(("ascent", 50) => (5.28, 0.44, 0.083), ("ascent", 100) => (6.75, 0.38, 0.057),
                                   ("ascent", 150) => (7.75, 0.36, 0.046), ("ascent", 200) => (8.54, 0.34, 0.039),
                                   ("ascent", 250) => (9.20, 0.32, 0.035), ("ascent", 300) => (9.77, 0.31, 0.032),
                                   ("descent", 250) => (9.23, 0.33, 0.036), ("descent", 200) => (8.54, 0.36, 0.042),
                                   ("descent", 150) => (7.80, 0.39, 0.051), ("descent", 100) => (6.82, 0.45, 0.066),
                                   ("descent", 50) => (5.43, 0.57, 0.105))

function jensen_nugent_section(io)
    println(io, "## 4. Jensen and Nugent (2017), Table 4 upper rows: Sc, modified polluted, κ = 1.28, w = ±0.4 m/s, ",
                "938.5 hPa, 284.3 K, RH 85.61%, z₀ = 600 m; heights above 900 m\n")
    println(io, "Paper (PySDM checks at ±7.5%): mean activated radius r̄, spectral width σ (μm) and dispersion σ/r̄. ",
                "PySDM also checks 0.35% < 𝒮max < 0.5% and the largest drop ≈ 12 μm.\n")
    for (name, kw) in (("αᶜ = 0.036, αᵀ = 0.7 (paper)", (;)), ("αᶜ = 1, αᵀ = 1 (Breeze default)", (; αᶜ = 1.0, αᵀ = 1.0)))
        x = jensen_nugent_2017(; kw...)
        @printf(io, "**%s**: 𝒮max = %.3f%%, largest drop at the end %.2f μm\n\n", name, 100x.𝒮max, x.rmax_end)
        println(io, "| phase | z (m) | r̄ (μm) | paper | σ (μm) | paper | σ/r̄ | paper | r̄ error |")
        println(io, "|---|---|---|---|---|---|---|---|---|")
        for (label, h, r̄, σ, d) in x.rows
            ref = jensen_nugent_table_4[(label, h)]
            @printf(io, "| %s | %d | %.2f | %.2f | %.2f | %.2f | %.3f | %.3f | %+.1f%% |\n", label, h, r̄, ref[1], σ,
                    ref[2], d, ref[3], 100(r̄ / ref[1] - 1))
        end
        println(io)
    end
end

#####
##### 5. Arabas and Shima (2017)
#####

function arabas_shima_2017(; w̄, N_STP, rᵈ, Δt_fraction = 1 / 500, αᶜ = 1.0)
    constants = ThermodynamicConstants(Float64)
    parameters = droplet_parameters(; αᶜ)
    ρ_STP = 101325 / (dry_air_gas_constant(constants) * 273.15)
    z_half = 150.0
    t_end = 2z_half / w̄
    w(t) = w̄ * π / 2 * sin(π * t * w̄ / z_half)
    Δt = t_end * Δt_fraction
    record = run_parcel(; T₀ = 300.0, p₀ = 100000.0, ℋ₀ = 0.98, rᵈ = [rᵈ], n = [N_STP / ρ_STP], κ = [0.2], w, t_end,
                        Δt, constants, parameters, output_interval = Δt)
    rᶜ = critical_radius(rᵈ, 0.2, record.T[end], constants, parameters)
    r = [x[1] for x in record.r]
    k = argmax(r)
    rᶜtop = critical_radius(rᵈ, 0.2, record.T[k], constants, parameters)
    return (; 𝒮max = maximum(record.𝒮), rmax = r[k], activated = r[k] > rᶜtop, r_end = r[end], r₀ = r[1],
            deactivated = r[end] < rᶜ)
end

function arabas_shima_section(io)
    println(io, "## 5. Arabas and Shima (2017): 1000 hPa, 300 K, RH 98%, κ = 0.2, monodisperse, w = w̄ π/2 sin(π w̄ t / 150 m)\n")
    println(io, "Hysteresis: activation on the way up, deactivation (return to the haze branch) on the way down. ",
                "Reference: the same run with Δt = T/50000 (PySDM compares against a SciPy reference at 1%). ",
                "The paper's point is that the time step needed to capture (de)activation is short for fast updrafts.\n")
    println(io, "| w̄ (m/s) | N (cm⁻³ STP) | rᵈ (μm) | Δt (s) | 𝒮max (%) | r_max (μm) | activated | r_end (μm) | deactivated | r_max error vs fine |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for w̄ in (1.0, 0.002), N in (50e6, 500e6), rᵈ in (0.1e-6, 0.05e-6)
        N == 500e6 && rᵈ == 0.05e-6 && continue
        fine = arabas_shima_2017(; w̄, N_STP = N, rᵈ, Δt_fraction = 1 / 50000)
        for f in (1 / 50000, 1 / 500, 1 / 50)
            x = f == 1 / 50000 ? fine : arabas_shima_2017(; w̄, N_STP = N, rᵈ, Δt_fraction = f)
            @printf(io, "| %g | %d | %.2f | %.3g | %.4f | %.3f | %s | %.4f | %s | %+.2e |\n", w̄, N / 1e6, 1e6rᵈ,
                    2 * 150 / w̄ * f, 100x.𝒮max, 1e6x.rmax, x.activated, 1e6x.r_end, x.deactivated,
                    x.rmax / fine.rmax - 1)
        end
    end
    println(io)
end

function main()
    mkpath(output_directory)
    io = IOBuffer()
    println(io, "# Breeze parcel benchmarks for droplet activation and growth\n")
    println(io, "Breeze thermodynamics (default constants unless noted), `condensational_growth_step` (implicit), no ventilation, ",
                "operator splitting of expansion and condensation. Generated by `parcel_benchmarks.jl`.\n")
    for section in (rogers_section, kreidenweis_section, pyrcel_section, jensen_nugent_section, arabas_shima_section)
        t₀ = time_ns()
        section(io)
        @info "$(section) done in $(round((time_ns() - t₀) / 1e9, digits=1)) s"
    end
    text = String(take!(io))
    print(text)
    write(joinpath(output_directory, "REPORT.md"), text)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
