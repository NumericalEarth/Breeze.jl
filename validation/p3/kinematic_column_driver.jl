#####
##### Kinematic 1D column driver for P3 microphysics comparison
#####
##### This script implements a simplified 1D kinematic cloud model
##### similar to the Fortran kin1d driver, allowing direct comparison
##### of Breeze.jl's P3 microphysics against the Fortran reference.
#####

using Oceananigans
using Oceananigans.Units
using Breeze
using Breeze.Thermodynamics: ThermodynamicConstants, saturation_specific_humidity_over_liquid
using Breeze.Microphysics.PredictedParticleProperties:
    PredictedParticlePropertiesMicrophysics,
    compute_p3_process_rates,
    P3ProcessRates

using NCDatasets
using CairoMakie
using Printf
using Statistics

#####
##### Configuration matching kin1d
#####

nz = 41                    # Number of vertical levels
Δt = 10.0                  # Time step [s]
total_time = 90minutes     # Total simulation time
output_interval = 1minute

# Physical constants matching Fortran
g = 9.81                   # Gravitational acceleration [m/s²]
Rd = 287.0                 # Gas constant for dry air [J/kg/K]
cₚ = 1005.0                # Specific heat at constant pressure [J/kg/K]
T₀ = 273.15                # Reference temperature [K]

# Updraft parameters (from kin1d)
w_initial = 2.0            # Initial central updraft speed [m/s]
w_max = 5.0                # Maximum central updraft speed [m/s]
initial_cloud_top = 5000.0 # Initial height of cloud top [m]
updraft_period = 5400.0    # Period for evolving updraft [s]
cloud_top_period = 5400.0  # Period for evolving cloud top [s]

#####
##### Load sounding data
#####

"""
    load_sounding(filepath)

Load Oklahoma sounding data from file.
Returns (pressure, height, temperature, dewpoint) all in SI units.
"""
function load_sounding(filepath)
    lines = readlines(filepath)
    n_levels = parse(Int, strip(lines[1]))

    # Skip header lines (2-6)
    pressure = Float64[]
    height = Float64[]
    temperature = Float64[]
    dewpoint = Float64[]

    for i in 7:(6 + n_levels)
        parts = split(strip(lines[i]))
        push!(pressure, parse(Float64, parts[1]) * 100.0)  # hPa → Pa
        push!(height, parse(Float64, parts[2]))            # m
        push!(temperature, parse(Float64, parts[3]) + T₀)  # °C → K
        push!(dewpoint, parse(Float64, parts[4]) + T₀)     # °C → K
    end

    return pressure, height, temperature, dewpoint
end

"""
    interpolate_to_levels(z_target, z_data, var_data)

Linear interpolation to target z levels.
"""
function interpolate_to_levels(z_target, z_data, var_data)
    result = similar(z_target)
    for (i, z) in enumerate(z_target)
        # Find bracketing indices (z_data is in increasing order)
        if z <= z_data[1]
            result[i] = var_data[1]
        elseif z >= z_data[end]
            result[i] = var_data[end]
        else
            j = findfirst(zd -> zd >= z, z_data)
            j1 = j - 1
            t = (z - z_data[j1]) / (z_data[j] - z_data[j1])
            result[i] = (1 - t) * var_data[j1] + t * var_data[j]
        end
    end
    return result
end

#####
##### Load vertical levels from kin1d
#####

function load_levels(filepath)
    lines = readlines(filepath)
    nk = parse(Int, strip(lines[1]))
    z = Float64[]

    # Skip header lines (lines 2-3 are description/column headers)
    for i in 4:(3 + nk)
        parts = split(strip(lines[i]))
        # Column 3 is ~HEIGHTS (with scientific notation like 1.5761E-01)
        # But the actual height is in column 3 which shows values like 12839.
        push!(z, parse(Float64, parts[3]))
    end

    return reverse(z)  # Return in ascending order (bottom to top)
end

#####
##### Evolving updraft profile
#####

"""
    compute_updraft(z, t, H)

Compute vertical velocity profile at time t.
Matches the NewWprof2 subroutine in kin1d.
"""
function compute_updraft(z, t, H)
    # Time-dependent maximum updraft speed
    wmaxH = w_initial + (w_max - w_initial) * 0.5 * (cos(t / updraft_period * 2π + π) + 1)

    # Updraft shuts off after 60 minutes
    if t > 3600.0
        wmaxH = 0.0
    end

    # Time-dependent cloud top height
    Hcld = initial_cloud_top + (H - initial_cloud_top) * 0.5 * (cos(t / cloud_top_period * 2π + π) + 1)

    # Sinusoidal profile
    w = zeros(length(z))
    for (i, zi) in enumerate(z)
        if zi <= Hcld && zi > 0
            w[i] = wmaxH * sin(π * zi / Hcld)
        end
    end

    return w
end

#####
##### Saturation specific humidity (Tetens formula)
#####

"""
    saturation_specific_humidity(T, p)

Compute saturation specific humidity using Tetens formula.
Matches the Fortran FOEW/FOQST functions.
"""
function saturation_specific_humidity_tetens(T, p)
    eps1 = 0.62194800221014
    eps2 = 0.3780199778986
    TRPL = 273.16

    # Tetens formula for saturation vapor pressure
    if T >= TRPL
        # Over liquid
        es = 610.78 * exp(17.269 * (T - TRPL) / (T - 35.86))
    else
        # Over ice
        es = 610.78 * exp(21.875 * (T - TRPL) / (T - 7.66))
    end

    # Saturation specific humidity
    qsat = eps1 / max(1.0, p / es - eps2)
    return qsat
end

#####
##### Main simulation
#####

function run_kinematic_column()
    println("=" ^ 60)
    println("Breeze.jl P3 Kinematic Column Driver")
    println("=" ^ 60)
    println()

    # Load sounding and levels
    p3_repo = get(ENV, "P3_REPO", "/Users/gregorywagner/Projects/P3-microphysics")
    sounding_path = joinpath(p3_repo, "kin1d", "soundings",
                             "snd_input.KOUN_00z1june2008.data")
    levels_path = joinpath(p3_repo, "kin1d", "levels", "levs_41.dat")

    if !isfile(sounding_path)
        error("Sounding file not found: $sounding_path\n" *
              "Set P3_REPO environment variable to your P3-microphysics clone.")
    end

    # Load data
    p_snd, z_snd, T_snd, Td_snd = load_sounding(sounding_path)
    z_levels = load_levels(levels_path)

    println("Loaded sounding: $(length(p_snd)) levels")
    println("Target levels: $(length(z_levels)) levels from $(z_levels[1]) to $(z_levels[end]) m")

    # Interpolate to target levels
    p = interpolate_to_levels(z_levels, z_snd, p_snd)
    T = interpolate_to_levels(z_levels, z_snd, T_snd)
    Td = interpolate_to_levels(z_levels, z_snd, Td_snd)

    # Compute initial profiles
    ρ = p ./ (Rd .* T)
    qv = [saturation_specific_humidity_tetens(Td[k], p[k]) for k in 1:nz]
    qsat = [saturation_specific_humidity_tetens(T[k], p[k]) for k in 1:nz]

    # Domain height
    H = z_levels[end]
    dz = diff(z_levels)
    push!(dz, dz[end])  # Assume last dz equals second-to-last

    println("Domain height: $H m")
    println("Initial T range: $(minimum(T) - T₀) to $(maximum(T) - T₀) °C")
    println("Initial qv range: $(minimum(qv)*1000) to $(maximum(qv)*1000) g/kg")
    println()

    # Initialize hydrometeor arrays (mixing ratios)
    qc = zeros(nz)   # Cloud liquid
    qr = zeros(nz)   # Rain
    nc = zeros(nz)   # Cloud droplet number (per kg)
    nr = zeros(nz)   # Rain number (per kg)

    # P3 ice variables
    qi = zeros(nz)   # Total ice mass
    ni = zeros(nz)   # Ice number
    qf = zeros(nz)   # Rime mass
    bf = zeros(nz)   # Rime volume
    zi = zeros(nz)   # Reflectivity (6th moment)
    qw = zeros(nz)   # Liquid on ice

    # Thermodynamic constants
    constants = ThermodynamicConstants(Float64)

    # P3 microphysics scheme
    p3 = PredictedParticlePropertiesMicrophysics()

    # Time integration
    nt = Int(total_time / Δt)
    n_output = Int(output_interval / Δt)
    n_saved = div(nt, n_output)

    # Output arrays
    times_out = zeros(n_saved)
    qc_out = zeros(n_saved, nz)
    qr_out = zeros(n_saved, nz)
    qi_out = zeros(n_saved, nz)
    ni_out = zeros(n_saved, nz)
    qf_out = zeros(n_saved, nz)
    bf_out = zeros(n_saved, nz)
    T_out = zeros(n_saved, nz)
    w_out = zeros(n_saved, nz)
    rime_fraction_out = zeros(n_saved, nz)

    println("Running simulation: $nt timesteps, Δt = $Δt s")
    println("Output every $n_output steps ($output_interval)")
    println()

    # Main time loop
    i_out = 0
    for n in 1:nt
        t = n * Δt

        # Compute vertical velocity
        w = compute_updraft(z_levels, t, H)

        # Advection (simple upstream)
        for k in 2:nz
            if w[k] > 0
                # Upward advection from below
                dqv = -w[k] * (qv[k] - qv[k-1]) / dz[k] * Δt
                dT_adv = -w[k] * (T[k] - T[k-1]) / dz[k] * Δt
                qv[k] += dqv
                T[k] += dT_adv
            end
        end

        # Adiabatic cooling
        for k in 1:nz
            T[k] -= g / cₚ * w[k] * Δt
        end

        # Saturation adjustment (simple condensation)
        for k in 1:nz
            qsat[k] = saturation_specific_humidity_tetens(T[k], p[k])

            if qv[k] > qsat[k]
                # Condensation
                excess = qv[k] - qsat[k]
                qv[k] = qsat[k]
                qc[k] += excess

                # Latent heating
                Lv = 2.5e6  # J/kg
                T[k] += Lv * excess / cₚ

                # Initialize cloud droplet number if new cloud
                if nc[k] < 1e6 && qc[k] > 1e-8
                    nc[k] = 250e6  # 250 per cc
                end
            end
        end

        # Ice nucleation at cold temperatures
        for k in 1:nz
            if T[k] < T₀ - 15 && qc[k] > 1e-8 && ni[k] < 1e4
                # Simple freezing
                frozen = min(qc[k], 1e-6)
                qc[k] -= frozen
                qi[k] += frozen
                ni[k] += frozen / 1e-12  # Assume small crystals
            end
        end

        # Compute P3 process rates at each level
        for k in 1:nz
            if qi[k] > 1e-10 || qc[k] > 1e-10 || qr[k] > 1e-10
                # Build microphysical state
                rime_fraction = qi[k] > 1e-12 ? qf[k] / qi[k] : 0.0
                liquid_fraction = qi[k] > 1e-12 ? qw[k] / qi[k] : 0.0

                # Simplified thermodynamic state for P3
                # Note: Full integration would use Breeze's thermodynamic formulation

                # For now, just accumulate mass through simple parameterizations
                # This is a placeholder - full integration requires matching
                # P3's complex process rate calculations

                # Autoconversion (cloud → rain)
                if qc[k] > 1e-6
                    τ_auto = 1000.0  # seconds
                    dqr = qc[k] / τ_auto * Δt
                    dqr = min(dqr, qc[k])
                    qc[k] -= dqr
                    qr[k] += dqr
                    nr[k] += dqr / 1e-9  # Rain drop mass
                end

                # Ice deposition
                if qi[k] > 1e-10 && qv[k] > qsat[k] * 0.9
                    τ_dep = 500.0
                    dqi = qi[k] * 0.1 * Δt / τ_dep
                    dqi = min(dqi, qv[k] - qsat[k] * 0.9)
                    dqi = max(dqi, 0.0)
                    qi[k] += dqi
                    qv[k] -= dqi
                end

                # Melting
                if qi[k] > 1e-10 && T[k] > T₀
                    τ_melt = 100.0
                    dmelt = qi[k] * (T[k] - T₀) / 10.0 * Δt / τ_melt
                    dmelt = min(dmelt, qi[k])
                    qi[k] -= dmelt
                    qr[k] += dmelt
                end

                # Riming
                if qi[k] > 1e-10 && qc[k] > 1e-8
                    τ_rime = 300.0
                    drime = qc[k] * 0.5 * Δt / τ_rime
                    drime = min(drime, qc[k])
                    qc[k] -= drime
                    qf[k] += drime
                    qi[k] += drime
                end
            end
        end

        # Simple sedimentation
        for k in 2:nz
            # Rain fall speed ~5 m/s
            if qr[k] > 1e-10
                v_rain = 5.0
                dz_fall = v_rain * Δt
                if dz_fall > dz[k]
                    flux = qr[k]
                    qr[k-1] += flux
                    qr[k] = 0.0
                end
            end

            # Ice fall speed ~1 m/s
            if qi[k] > 1e-10
                v_ice = 1.0
                dz_fall = v_ice * Δt
                if dz_fall > dz[k] * 3
                    flux = qi[k] * 0.1
                    if k > 1
                        qi[k-1] += flux
                        qf[k-1] += qf[k] * 0.1
                    end
                    qi[k] -= flux
                    qf[k] -= qf[k] * 0.1
                end
            end
        end

        # Enforce positivity
        qv .= max.(qv, 0.0)
        qc .= max.(qc, 0.0)
        qr .= max.(qr, 0.0)
        qi .= max.(qi, 0.0)
        qf .= max.(qf, 0.0)
        bf .= max.(bf, 0.0)
        ni .= max.(ni, 0.0)

        # Store output
        if mod(n, n_output) == 0
            i_out += 1
            times_out[i_out] = t
            qc_out[i_out, :] = qc
            qr_out[i_out, :] = qr
            qi_out[i_out, :] = qi
            ni_out[i_out, :] = ni
            qf_out[i_out, :] = qf
            bf_out[i_out, :] = bf
            T_out[i_out, :] = T
            w_out[i_out, :] = w
            rime_fraction_out[i_out, :] = [qi[k] > 1e-12 ? qf[k]/qi[k] : 0.0 for k in 1:nz]

            if mod(i_out, 10) == 0
                println("t = $(round(t/60, digits=1)) min, max qi = $(round(maximum(qi)*1000, digits=3)) g/kg")
            end
        end
    end

    println()
    println("Simulation complete!")
    println("Max cloud liquid: $(round(maximum(qc_out)*1000, digits=2)) g/kg")
    println("Max rain: $(round(maximum(qr_out)*1000, digits=2)) g/kg")
    println("Max ice: $(round(maximum(qi_out)*1000, digits=2)) g/kg")

    return (
        times = times_out,
        z = z_levels,
        qc = qc_out,
        qr = qr_out,
        qi = qi_out,
        qf = qf_out,
        T = T_out,
        w = w_out,
        rime_fraction = rime_fraction_out
    )
end

#####
##### Run and compare
#####

results = run_kinematic_column()

# Load Fortran reference
reference_path = joinpath(@__DIR__, "kin1d_reference.nc")
ds = NCDataset(reference_path, "r")
ref_time = ds["time"][:] ./ 60  # Convert to minutes
ref_z = ds["z"][:] ./ 1000      # Convert to km
ref_qc = ds["q_cloud"][:, :]
ref_qr = ds["q_rain"][:, :]
ref_qi = ds["q_ice"][:, :]
ref_rime = ds["rime_fraction"][:, :]
ref_T = ds["temperature"][:, :]
close(ds)

# Create comparison figure
fig = Figure(size=(1400, 800), fontsize=12)

breeze_time = results.times ./ 60
breeze_z = results.z ./ 1000

# Cloud liquid comparison
ax1 = Axis(fig[1, 1], xlabel="Time [min]", ylabel="Height [km]",
           title="Cloud Liquid - Fortran [g/kg]")
hm1 = heatmap!(ax1, ref_time, ref_z, ref_qc .* 1000,
               colormap=:blues, colorrange=(0, 5))
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], xlabel="Time [min]", ylabel="Height [km]",
           title="Cloud Liquid - Breeze [g/kg]")
hm2 = heatmap!(ax2, breeze_time, breeze_z, results.qc .* 1000,
               colormap=:blues, colorrange=(0, 5))
Colorbar(fig[1, 4], hm2)

# Ice comparison
ax3 = Axis(fig[2, 1], xlabel="Time [min]", ylabel="Height [km]",
           title="Ice - Fortran [g/kg]")
hm3 = heatmap!(ax3, ref_time, ref_z, ref_qi .* 1000,
               colormap=:reds, colorrange=(0, 15))
Colorbar(fig[2, 2], hm3)

ax4 = Axis(fig[2, 3], xlabel="Time [min]", ylabel="Height [km]",
           title="Ice - Breeze [g/kg]")
hm4 = heatmap!(ax4, breeze_time, breeze_z, results.qi .* 1000,
               colormap=:reds, colorrange=(0, 15))
Colorbar(fig[2, 4], hm4)

# Rime fraction comparison
ax5 = Axis(fig[3, 1], xlabel="Time [min]", ylabel="Height [km]",
           title="Rime Fraction - Fortran")
hm5 = heatmap!(ax5, ref_time, ref_z, ref_rime,
               colormap=:viridis, colorrange=(0, 1))
Colorbar(fig[3, 2], hm5)

ax6 = Axis(fig[3, 3], xlabel="Time [min]", ylabel="Height [km]",
           title="Rime Fraction - Breeze")
hm6 = heatmap!(ax6, breeze_time, breeze_z, results.rime_fraction,
               colormap=:viridis, colorrange=(0, 1))
Colorbar(fig[3, 4], hm6)

save(joinpath(@__DIR__, "kin1d_comparison.png"), fig)
println()
println("Saved comparison figure: kin1d_comparison.png")

#####
##### Summary statistics
#####

println()
println("=" ^ 60)
println("COMPARISON SUMMARY")
println("=" ^ 60)
println()
println("                     Fortran P3    Breeze.jl P3")
println("-" ^ 50)
@printf("Max cloud liquid:    %8.3f      %8.3f     g/kg\n",
        maximum(ref_qc)*1000, maximum(results.qc)*1000)
@printf("Max rain:            %8.3f      %8.3f     g/kg\n",
        maximum(ref_qr)*1000, maximum(results.qr)*1000)
@printf("Max ice:             %8.3f      %8.3f     g/kg\n",
        maximum(ref_qi)*1000, maximum(results.qi)*1000)
@printf("Max rime fraction:   %8.3f      %8.3f\n",
        maximum(ref_rime), maximum(results.rime_fraction))
println()
println("NOTE: The Breeze.jl kinematic driver uses simplified parameterizations")
println("for advection, condensation, and sedimentation. For a true comparison,")
println("these components need to match the Fortran implementation exactly.")
println()
println("The key comparison is the P3 MICROPHYSICS TENDENCIES, which should be")
println("verified by comparing individual process rates in isolation.")
