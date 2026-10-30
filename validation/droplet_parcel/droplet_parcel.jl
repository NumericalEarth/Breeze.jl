# # L0: a 0-D adiabatic parcel with Lagrangian droplets vs. the NCAR parcel model
#
# Rung L0 of the Breeze vs. NCAR parcel-DNS comparison: a closed, adiabatic parcel rising at
# U = 2 m/s for 600 s from T = 284.3 K, p = 93850 Pa and a supersaturation of -14.39%, carrying the
# 39-bin, three-mode marine aerosol of `activation.F90` (ammonium sulfate, 111.8 particles per
# "parcel"), one superdroplet per bin. Droplets grow with `Breeze.ParticleMicrophysics`'
# implicit condensational growth step; the parcel thermodynamics is integrated here.
#
# Runs:
#   (a)  constants, formulas and parcel equations matched to activation.F90
#   (a₁…a₅) cumulative switches from (a) to Breeze's defaults, to attribute the differences
#   (b)  Breeze's default thermodynamics and droplet physics
# plus time-step sensitivity of (a) and (b).
#
# Usage: julia --project=validation/droplet_parcel validation/droplet_parcel/droplet_parcel.jl [ncar_run_dir]

using Breeze
using Breeze.Thermodynamics: PlanarLiquidSurface, MoistureMassFractions, liquid_latent_heat,
                             mixture_gas_constant, mixture_heat_capacity, supersaturation,
                             dry_air_gas_constant, vapor_gas_constant, saturation_vapor_pressure
using Breeze.ParticleMicrophysics: ParticleMicrophysics, ventilation_factor, droplet_water_mass
using CairoMakie
using Printf

const ncar_directory = length(ARGS) > 0 ? ARGS[1] : "/home/greg_aeolus_earth/ncar-dns/runs/parcel0"
const output_directory = joinpath(@__DIR__, "output")
mkpath(output_directory)

#####
##### NCAR-specific formulas (activation.F90) as Breeze.ParticleMicrophysics extensions
#####

struct NCARSaturationVaporPressure end

const NCARConstants = ThermodynamicConstants{FT, C, I, NCARSaturationVaporPressure} where {FT, C, I}

Breeze.Thermodynamics.saturation_vapor_pressure(T, ::NCARConstants, ::PlanarLiquidSurface) = 2.53e11 * exp(-5.42e3 / T)

struct NCARVaporDiffusivity end
struct NCARThermalConductivity end
ParticleMicrophysics.vapor_diffusivity(::NCARVaporDiffusivity, T, p) = 1e-5 * (0.015T - 1.9)
ParticleMicrophysics.thermal_conductivity(::NCARThermalConductivity, T, p) = 1.5e-11T^3 - 4.8e-8T^2 + 1e-4T - 3.9e-4

const ℛ = 8.314462618
const ncar_Rᵈ = 287.0
const ncar_Rᵛ = 467.0
const ncar_cᵖ = 1004.0
const ncar_ℒ = 2.5e6
const ncar_g = 9.8
const ncar_σ = 7.61e-2
const U = 2.0                      # parcel ascent speed (m/s)
const parcel_mass = 1e-6           # NCAR's "1 cm³" parcel: number per 1e-6 kg of air (see README)

# NCAR thermodynamic constants: constant latent heat (cᵖᵛ = cˡ), Rᵈ = 287, Rᵛ = 467, cᵖ = 1004, g = 9.8
ncar_constants = ThermodynamicConstants(Float64;
                                        dry_air_molar_mass = ℛ / ncar_Rᵈ,
                                        vapor_molar_mass = ℛ / ncar_Rᵛ,
                                        dry_air_heat_capacity = ncar_cᵖ,
                                        vapor_heat_capacity = 4181,
                                        liquid = CondensedPhase(Float64; reference_latent_heat=ncar_ℒ,
                                                                heat_capacity=4181, density=1000),
                                        gravitational_acceleration = ncar_g,
                                        saturation_vapor_pressure = NCARSaturationVaporPressure())

breeze_constants = ThermodynamicConstants(Float64)

# activation.F90 writes sqrt(2π/(Rᵃ T)) (the dry-air gas constant) in the vapor kinetic term,
# which equals the standard sqrt(2π/(Rᵛ T)) with αᶜ → αᶜ sqrt(Rᵃ/Rᵛ).
ncar_kinetic_correction = KineticCorrection(Float64; condensation_coefficient = 0.036 * sqrt(ncar_Rᵈ / ncar_Rᵛ),
                                            thermal_accommodation_coefficient = 0.7,
                                            vapor_jump_length = 0.104e-6,
                                            thermal_jump_length = 0.216e-6)

ncar_parameters = DropletPhysicsParameters(Float64;
                                           surface_tension = ncar_σ,
                                           vapor_diffusivity = NCARVaporDiffusivity(),
                                           thermal_conductivity = NCARThermalConductivity(),
                                           kinetic_correction = ncar_kinetic_correction,
                                           solution_activity = DiluteSolutionActivity())

#####
##### Aerosol: activation.F90, IAEROSOL with disp = 30 (Xue et al. 2010 marine)
#####

function ncar_aerosol()
    nbins = 39
    rmin = 6e-9
    rᵈ = [i == 1 ? rmin : rmin * 2.0^(i / 3) for i in 1:nbins] # NB: bin 2 is 2^(2/3) rmin, as in the Fortran
    width = [i == 1 ? rᵈ[1] * (2^(1 / 3) - 1) : rᵈ[i] - rᵈ[i-1] for i in 1:nbins]
    modes = ((133.0, 0.0039e-6, 0.657), (66.6, 0.133e-6, 0.21), (3.06, 0.29e-6, 0.396))
    π₃₂ = 3.14159265
    dNdlogr = map(rᵈ) do r
        sum(modes) do (n₁, r₁, log10σ)
            logσ = log(10) * log10σ
            n₁ / (sqrt(2π₃₂) * logσ) * exp(-((log(r) - log(r₁)) / (sqrt(2) * logσ))^2)
        end
    end
    counts = dNdlogr .* width ./ rᵈ # per parcel (1e-6 kg)
    # Hygroscopicity of ammonium sulfate as written in activation.F90: B = vh (mʷ/mˢ) (ρˢ/ρʷ) rᵈ³
    κ = 2 * (18e-3 / 132.14e-3) * (1726 / 1000)
    return rᵈ, counts, κ
end

#####
##### Run configuration
#####

Base.@kwdef struct ParcelConfiguration{P}
    name :: String
    thermodynamics :: Symbol = :ncar          # :ncar (activation.F90 parcel equations) or :breeze
    parameters :: P = ncar_parameters
    ventilation :: Symbol = :ncar             # :ncar (Stokes slip, activation.F90 Re and Sc) or :breeze
    initial_moisture :: Symbol = :supersaturation # match 𝒮₀ = -14.39% (each run's own 𝒮) or :vapor (NCAR's vapor)
    Δt :: Float64 = 0.01
end

constants_of(c::ParcelConfiguration) = c.thermodynamics === :ncar ? ncar_constants : breeze_constants

# activation.F90 ventilation: Stokes fall speed kwt r², Re = 2 r 𝕎 / μ and Sc = μ / D with the
# dynamic viscosity μ = 1.78e-5 used as a kinematic one and D = 2.55e-5.
function ncar_ventilation(r)
    𝕎 = 1.233e8 * r^2
    Re = 2r * 𝕎 / 1.78e-5
    Sc = 1.78e-5 / 2.55e-5
    return ventilation_factor(BeardPruppacherVentilation(), Re, Sc)
end

function breeze_ventilation(r, T, p, constants, parameters)
    𝕎 = droplet_terminal_velocity(r, T, p, constants, parameters)
    return ventilation_factor(r, 𝕎, T, p, constants, parameters)
end

#####
##### Parcel integration
#####

mutable struct ParcelState
    T :: Float64
    p :: Float64
    ρ :: Float64
    qᵛ :: Float64      # mixing ratio (NCAR thermodynamics) or specific humidity (Breeze)
    qˡ :: Float64      # condensed water (kg per kg air)
    Δqˡ :: Float64     # condensation of the previous step (NCAR thermodynamics applies it lagged)
    𝒮 :: Float64
end

function run_parcel(config; t_end = 600.0, output_interval = 0.1)
    constants = constants_of(config)
    parameters = config.parameters
    rᵈ, counts, κ = ncar_aerosol()
    n = counts ./ parcel_mass # number per kg of air
    nbins = length(rᵈ)

    # Initial state: activation.F90's T, p, 𝒮 and vapor mixing ratio
    T₀, p₀, 𝒮₀ = 284.3, 93850.0, -14.39e-2
    pᵛ⁺ = 2.53e11 * exp(-5.42e3 / T₀)
    ε = ncar_Rᵈ / ncar_Rᵛ
    rᵛ = (1 + 𝒮₀) * ε * pᵛ⁺ / (p₀ - pᵛ⁺) # vapor mixing ratio

    if config.thermodynamics === :ncar
        ρ₀ = p₀ / (ncar_Rᵈ * (1 + 18 / 29 * rᵛ) * T₀)
        state = ParcelState(T₀, p₀, ρ₀, rᵛ, 0.0, 0.0, 𝒮₀)
    else
        if config.initial_moisture === :vapor
            qᵛ = rᵛ / (1 + rᵛ)
        else # vapor pressure (1 + 𝒮₀) pᵛ⁺(T₀) with Breeze's saturation vapor pressure
            Rᵈ = dry_air_gas_constant(constants)
            Rᵛ = vapor_gas_constant(constants)
            pᵛ = (1 + 𝒮₀) * saturation_vapor_pressure(T₀, constants, PlanarLiquidSurface())
            ρᵛ = pᵛ / (Rᵛ * T₀)
            ρᵈ = (p₀ - pᵛ) / (Rᵈ * T₀)
            qᵛ = ρᵛ / (ρᵈ + ρᵛ)
        end
        q = MoistureMassFractions(qᵛ)
        ρ₀ = p₀ / (mixture_gas_constant(q, constants) * T₀)
        𝒮 = supersaturation(T₀, ρ₀, q, constants, PlanarLiquidSurface())
        state = ParcelState(T₀, p₀, ρ₀, qᵛ, 0.0, 0.0, 𝒮)
    end

    initial_qᵛ = state.qᵛ

    # Haze droplets in equilibrium with the initial supersaturation
    r = [equilibrium_radius(rᵈ[i], κ, state.𝒮, T₀, constants, parameters) for i in 1:nbins]
    state.qˡ = sum(n[i] * droplet_water_mass(r[i], rᵈ[i], constants) for i in 1:nbins)

    Δt = config.Δt
    Nt = round(Int, t_end / Δt)
    output_every = max(1, round(Int, output_interval / Δt))
    times = Float64[0]
    𝒮_series = Float64[state.𝒮]
    T_series = Float64[state.T]
    radii = [copy(r)]
    Δm = zeros(nbins)

    for step in 1:Nt
        if config.thermodynamics === :ncar
            ncar_thermodynamics_step!(state, Δt)
        else
            breeze_ascent_step!(state, Δt, constants)
        end

        for i in 1:nbins
            fᵛ = config.ventilation === :ncar ? ncar_ventilation(r[i]) :
                                                breeze_ventilation(r[i], state.T, state.p, constants, parameters)
            growth = condensational_growth_step(r[i], rᵈ[i], κ, state.𝒮, fᵛ, state.T, state.p, Δt,
                                                constants, parameters)
            r[i] = growth.radius
            Δm[i] = growth.condensed_mass
        end
        Δqˡ = sum(n .* Δm)
        state.qˡ += Δqˡ

        if config.thermodynamics === :ncar
            state.Δqˡ = Δqˡ # applied to vapor and temperature at the start of the next step
        else
            breeze_condensation_step!(state, Δqˡ, constants)
        end

        if step % output_every == 0
            push!(times, step * Δt)
            push!(𝒮_series, state.𝒮)
            push!(T_series, state.T)
            push!(radii, copy(r))
        end
    end

    rᵛ₀ = config.thermodynamics === :ncar ? rᵛ : initial_qᵛ / (1 - initial_qᵛ)
    return (; name = config.name, Δt, times, 𝒮 = 𝒮_series, T = T_series, radii = reduce(hcat, radii),
            rᵈ, counts, κ, rᵛ₀, 𝒮₀ = 𝒮_series[1])
end

# activation.F90's parcel equations, including the one-step lag of the condensation feedback
function ncar_thermodynamics_step!(s, Δt)
    s.p = s.ρ * ncar_Rᵈ * (1 + 18 / 29 * s.qᵛ) * s.T
    T⁻ = s.T
    s.T = T⁻ - ncar_g / ncar_cᵖ * Δt * U + ncar_ℒ / ncar_cᵖ * s.Δqˡ
    pᵛ⁺ = 2.53e11 * exp(-5.42e3 / s.T)
    rᵛ⁺ = ncar_Rᵈ / ncar_Rᵛ * pᵛ⁺ / (s.p - pᵛ⁺)
    s.ρ += s.ρ * (-ncar_g * U / (ncar_Rᵈ * s.T) * Δt - (s.T - T⁻) / s.T)
    s.qᵛ -= s.Δqˡ
    s.𝒮 = s.qᵛ / rᵛ⁺ - 1
    return nothing
end

# Hydrostatic, adiabatic ascent with Breeze's moist thermodynamics: cᵖᵐ dT = dp / ρ = -g U dt
function breeze_ascent_step!(s, Δt, constants)
    g = constants.gravitational_acceleration
    q = MoistureMassFractions(s.qᵛ, s.qˡ)
    Rᵐ = mixture_gas_constant(q, constants)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    s.ρ = s.p / (Rᵐ * s.T)
    s.p -= s.ρ * g * U * Δt
    s.T -= g * U * Δt / cᵖᵐ
    s.ρ = s.p / (Rᵐ * s.T)
    s.𝒮 = supersaturation(s.T, s.ρ, q, constants, PlanarLiquidSurface())
    return nothing
end

function breeze_condensation_step!(s, Δqˡ, constants)
    q = MoistureMassFractions(s.qᵛ, s.qˡ)
    cᵖᵐ = mixture_heat_capacity(q, constants)
    s.qᵛ -= Δqˡ
    s.T += liquid_latent_heat(s.T, constants) * Δqˡ / cᵖᵐ
    return nothing
end

#####
##### NCAR reference output
#####

function read_ncar(directory)
    out = reduce(vcat, permutedims.([parse.(Float64, split(l)) for l in eachline(joinpath(directory, "new.out"))]))
    rad = reduce(vcat, permutedims.([parse.(Float64, split(l)) for l in eachline(joinpath(directory, "new.rad"))]))
    rᵈ, counts, κ = ncar_aerosol()
    # new.rad: the first row holds the dry radii (wet equilibration happens after it is written)
    @assert maximum(abs.(rad[1, 2:end] ./ rᵈ .- 1)) < 1e-6
    radii = permutedims(rad[2:end, 2:end])
    times = out[2:end, 1]
    @assert times ≈ rad[2:end, 1]
    return (; name = "NCAR activation.F90", Δt = 1e-4, times, 𝒮 = out[2:end, 3], T = out[2:end, 6], rᵛ₀ = out[1, 8], 𝒮₀ = out[1, 3],
            radii, rᵈ, counts, κ, ncar_lwc = out[2:end, 10], ρ = out[2:end, 11])
end

#####
##### Diagnostics: one code path for every run
#####

function diagnostics(run)
    n = run.counts ./ parcel_mass
    nbins, nt = size(run.radii)
    lwc = [sum(n[i] * droplet_water_mass(run.radii[i, k], run.rᵈ[i], ncar_constants) for i in 1:nbins) for k in 1:nt]
    # Activation criterion common to all runs: the critical radius of the NCAR-matched Köhler curve
    activated = falses(nbins, nt)
    for k in 1:nt, i in 1:nbins
        rᶜ = critical_radius(run.rᵈ[i], run.κ, run.T[k], ncar_constants, ncar_parameters)
        activated[i, k] = run.radii[i, k] > rᶜ
    end
    N = sum(run.counts)
    activated_fraction = [sum(run.counts[activated[:, k]]) / N for k in 1:nt]

    kmax = argmax(run.𝒮)
    final = activated[:, end]
    w = run.counts[final]
    r = run.radii[final, end]
    r̄ = sum(w .* r) / sum(w)
    σʳ = sqrt(sum(w .* (r .- r̄) .^ 2) / sum(w))
    return (; name = run.name, Δt = run.Δt, 𝒮₀ = run.𝒮₀, rᵛ₀ = run.rᵛ₀, T_end = run.T[end], 𝒮max = run.𝒮[kmax], t𝒮max = run.times[kmax], 𝒮end = run.𝒮[end],
            lwc, lwc_end = lwc[end], activated_fraction, activated_end = activated_fraction[end],
            mean_radius = r̄, spectral_width = σʳ, relative_dispersion = σʳ / r̄)
end

#####
##### Runs
#####

breeze_parameters = DropletPhysicsParameters(Float64)

# Cumulative switches from (a) toward (b)
p₂ = DropletPhysicsParameters(Float64; surface_tension = ncar_σ, vapor_diffusivity = NCARVaporDiffusivity(),
                              thermal_conductivity = NCARThermalConductivity(),
                              kinetic_correction = ncar_kinetic_correction, solution_activity = DiluteSolutionActivity())
p₃ = DropletPhysicsParameters(Float64; vapor_diffusivity = NCARVaporDiffusivity(),
                              thermal_conductivity = NCARThermalConductivity(),
                              kinetic_correction = ncar_kinetic_correction, solution_activity = DiluteSolutionActivity())
p₄ = DropletPhysicsParameters(Float64; vapor_diffusivity = NCARVaporDiffusivity(),
                              thermal_conductivity = NCARThermalConductivity(),
                              kinetic_correction = ncar_kinetic_correction)
p₅ = DropletPhysicsParameters(Float64; kinetic_correction = ncar_kinetic_correction)

configurations = [
    ParcelConfiguration(name = "(a) NCAR-matched"),
    ParcelConfiguration(name = "(a1) + Breeze thermodynamics", thermodynamics = :breeze, parameters = p₂),
    ParcelConfiguration(name = "(a2) + σ = 0.072 N/m", thermodynamics = :breeze, parameters = p₃),
    ParcelConfiguration(name = "(a3) + κ-Köhler activity", thermodynamics = :breeze, parameters = p₄),
    ParcelConfiguration(name = "(a4) + PK Dᵛ(T,p), Kᵀ(T)", thermodynamics = :breeze, parameters = p₅),
    ParcelConfiguration(name = "(a5) + αᶜ = αᵀ = 1", thermodynamics = :breeze, parameters = breeze_parameters),
    ParcelConfiguration(name = "(b) Breeze defaults", thermodynamics = :breeze, parameters = breeze_parameters,
                        ventilation = :breeze),
    ParcelConfiguration(name = "(b′) Breeze defaults, NCAR vapor", thermodynamics = :breeze,
                        parameters = breeze_parameters, ventilation = :breeze, initial_moisture = :vapor),
    ParcelConfiguration(name = "(a1′) + Breeze thermo., NCAR vapor", thermodynamics = :breeze, parameters = p₂,
                        initial_moisture = :vapor),
]

timestep_configurations = vcat(
    [ParcelConfiguration(name = "(a) NCAR-matched", Δt = Δt) for Δt in (1e-3, 0.1, 0.5, 1.0, 2.0)],
    [ParcelConfiguration(name = "(b) Breeze defaults", thermodynamics = :breeze, parameters = breeze_parameters,
                         ventilation = :breeze, Δt = Δt) for Δt in (0.1, 1.0)])

ncar = read_ncar(ncar_directory)
runs = Any[ncar]
for config in vcat(configurations, timestep_configurations)
    elapsed = @elapsed run = run_parcel(config)
    @info @sprintf("%-32s Δt = %6.3f s: %.1f s wall time", config.name, config.Δt, elapsed)
    push!(runs, run)
end
results = diagnostics.(runs)

#####
##### Summary table and CSV
#####

header = "run,dt_s,S_0_percent,qv_0_g_per_kg,T_600_K,S_max_percent,t_S_max_s,S_600_percent,LWC_600_g_per_kg,activated_fraction_600,mean_radius_600_um,spectral_width_600_um,relative_dispersion_600"
summary_lines = [header]
for d in results
    push!(summary_lines, @sprintf("\"%s\",%.4g,%.3f,%.4f,%.3f,%.5f,%.2f,%.5f,%.5f,%.5f,%.4f,%.4f,%.5f", d.name, d.Δt,
                                  100d.𝒮₀, 1e3d.rᵛ₀, d.T_end, 100d.𝒮max,
                                  d.t𝒮max, 100d.𝒮end, 1e3d.lwc_end, d.activated_end, 1e6d.mean_radius,
                                  1e6d.spectral_width, d.relative_dispersion))
end
write(joinpath(output_directory, "summary.csv"), join(summary_lines, "\n") * "\n")
foreach(println, summary_lines)

# Time series at 1 s intervals for NCAR, (a) and (b)
ia = findfirst(r -> r.name == "(a) NCAR-matched" && r.Δt == 0.01, runs)
ib = findfirst(r -> r.name == "(b) Breeze defaults" && r.Δt == 0.01, runs)
series_lines = ["time_s,S_ncar_percent,S_a_percent,S_b_percent,LWC_ncar_g_per_kg,LWC_a_g_per_kg,LWC_b_g_per_kg,activated_ncar,activated_a,activated_b"]
for k in 1:10:length(ncar.times)
    t = ncar.times[k]
    ka = findfirst(≈(t), runs[ia].times)
    kb = findfirst(≈(t), runs[ib].times)
    push!(series_lines, @sprintf("%.1f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f", t,
                                 100ncar.𝒮[k], 100runs[ia].𝒮[ka], 100runs[ib].𝒮[kb],
                                 1e3results[1].lwc[k], 1e3results[ia].lwc[ka], 1e3results[ib].lwc[kb],
                                 results[1].activated_fraction[k], results[ia].activated_fraction[ka],
                                 results[ib].activated_fraction[kb]))
end
write(joinpath(output_directory, "time_series.csv"), join(series_lines, "\n") * "\n")

# Size distribution at 600 s
dsd_lines = ["dry_radius_m,count_per_mg,r_ncar_m,r_a_m,r_b_m"]
for i in eachindex(ncar.rᵈ)
    push!(dsd_lines, @sprintf("%.6e,%.6e,%.6e,%.6e,%.6e", ncar.rᵈ[i], ncar.counts[i], ncar.radii[i, end],
                              runs[ia].radii[i, end], runs[ib].radii[i, end]))
end
write(joinpath(output_directory, "dsd_600s.csv"), join(dsd_lines, "\n") * "\n")

# Agreement of (a) with NCAR over the whole run (interpolated onto the NCAR output times)
ka_on_ncar = [findfirst(≈(t), runs[ia].times) for t in ncar.times]
@info @sprintf("(a) vs NCAR: max |ΔS| = %.2e %%, max |ΔLWC| = %.2e g/kg, max relative |Δr| at 600 s = %.2e",
               100maximum(abs.(runs[ia].𝒮[ka_on_ncar] .- ncar.𝒮)),
               1e3maximum(abs.(results[ia].lwc[ka_on_ncar] .- results[1].lwc)),
               maximum(abs.(runs[ia].radii[:, end] ./ ncar.radii[:, end] .- 1)))
@info @sprintf("NCAR LWC diagnostic (column 10) / qˡ at 600 s = %.4f (NCAR air density %.4f kg/m³)",
               ncar.ncar_lwc[end] / results[1].lwc[end], ncar.ρ[end])

#####
##### Figures
#####

fig = Figure(size = (1200, 900))
ax𝒮 = Axis(fig[1, 1], xlabel = "t (s)", ylabel = "𝒮 (%)", title = "Supersaturation")
ax𝒮z = Axis(fig[1, 2], xlabel = "t (s)", ylabel = "𝒮 (%)", title = "Supersaturation near the maximum")
axL = Axis(fig[2, 1], xlabel = "t (s)", ylabel = "qˡ (g/kg)", title = "Liquid water")
axA = Axis(fig[2, 2], xlabel = "t (s)", ylabel = "fraction", title = "Activated fraction (r > rᶜ)")
axD = Axis(fig[3, 1:2], xlabel = "wet radius at 600 s (μm)", ylabel = "number per mg of air", xscale = log10,
           yscale = log10, title = "Droplet size distribution at 600 s (one superdroplet per aerosol bin)")

styles = [(:black, :solid, 3), (:dodgerblue, :dash, 2), (:orangered, :solid, 2)]
for (j, (k, run, d)) in enumerate(zip((1, ia, ib), runs[[1, ia, ib]], results[[1, ia, ib]]))
    color, linestyle, linewidth = styles[j]
    lines!(ax𝒮, run.times, 100run.𝒮; color, linestyle, linewidth, label = run.name)
    lines!(ax𝒮z, run.times, 100run.𝒮; color, linestyle, linewidth)
    lines!(axL, run.times, 1e3d.lwc; color, linestyle, linewidth)
    lines!(axA, run.times, d.activated_fraction; color, linestyle, linewidth)
    scatter!(axD, 1e6run.radii[:, end], run.counts; color, marker = (:circle, :xcross, :utriangle)[j],
             markersize = (12, 9, 7)[j], label = run.name)
end
for (k, (run, d)) in enumerate(zip(runs, results))
    2 ≤ k ≤ 7 || continue
    k == 2 && continue
    lines!(ax𝒮z, run.times, 100run.𝒮; linewidth = 1, label = run.name)
end
xlims!(ax𝒮z, 100, 300)
ylims!(ax𝒮z, 0.8, 1.8)
ylims!(axD, 1e-5, 100)
axislegend(ax𝒮, position = :rb)
axislegend(ax𝒮z, position = :rt, labelsize = 10)
axislegend(axD, position = :lb)
save(joinpath(output_directory, "parcel_comparison.png"), fig)

# Time-step sensitivity
fig = Figure(size = (900, 400))
ax = Axis(fig[1, 1], xlabel = "Δt (s)", ylabel = "max 𝒮 (%)", xscale = log10, title = "Time-step sensitivity")
ax2 = Axis(fig[1, 2], xlabel = "Δt (s)", ylabel = "mean radius at 600 s (μm)", xscale = log10)
for (label, color) in (("(a) NCAR-matched", :dodgerblue), ("(b) Breeze defaults", :orangered))
    sel = sort([d for d in results if d.name == label], by = d -> d.Δt)
    scatterlines!(ax, [d.Δt for d in sel], [100d.𝒮max for d in sel]; color, label)
    scatterlines!(ax2, [d.Δt for d in sel], [1e6d.mean_radius for d in sel]; color)
end
hlines!(ax, [100results[1].𝒮max]; color = :black, linestyle = :dot, label = "NCAR (Δt = 1e-4 s, explicit)")
axislegend(ax, position = :rc)
save(joinpath(output_directory, "timestep_sensitivity.png"), fig)
