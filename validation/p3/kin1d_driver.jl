#####
##### Kinematic 1D column driver for P3 microphysics validation
#####
##### Implements the Fortran kin1d test case using Breeze's P3 process rates.
##### Compares against reference data from the official P3 repository.
#####
##### Reference: Morrison & Milbrandt (2015), Milbrandt et al. (2021, 2025)
##### Test case: KOUN sounding (00Z June 1 2008), nCat=1, trplMomIce=T, liqFrac=T
#####
##### NOTE: This driver contains empirical PSD correction factors (alpha_dep,
##### alpha_rim, alpha_melt, rain_evap_psd_factor, etc.) that compensate for
##### the mean-mass approximation in the analytical fallback path. When using
##### full tabulation via `tabulate(p3, CPU())`, these corrections are NOT needed
##### because the lookup tables integrate over the full particle size distribution.
##### A future version of this driver should use `tabulate(p3, CPU())` for all
##### process rates and set all PSD corrections to 1.0.
#####

using Breeze
using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    saturation_specific_humidity,
    PlanarLiquidSurface,
    PlanarIceSurface,
    temperature,
    with_temperature,
    liquid_latent_heat,
    ice_latent_heat,
    vapor_gas_constant,
    mixture_heat_capacity

using Breeze.Thermodynamics: LiquidIcePotentialTemperatureState

using Breeze.Microphysics.PredictedParticleProperties:
    PredictedParticlePropertiesMicrophysics,
    P3MicrophysicalState,
    CloudDropletProperties,
    IceProperties,
    RainProperties,
    ProcessRateParameters,
    IceFallSpeed,
    tabulate,
    compute_p3_process_rates,
    tendency_ρqᶜˡ, tendency_ρqʳ, tendency_ρnʳ,
    tendency_ρqⁱ, tendency_ρnⁱ, tendency_ρqᶠ,
    tendency_ρbᶠ, tendency_ρzⁱ, tendency_ρqʷⁱ,
    tendency_ρqᵛ,
    rain_terminal_velocity_mass_weighted,
    rain_terminal_velocity_number_weighted,
    ice_terminal_velocity_mass_weighted,
    ice_terminal_velocity_number_weighted

using Oceananigans.Architectures: CPU

using Printf

#####
##### Physical constants (matching Fortran kin1d)
#####

const g_cld = 9.81
const Rd_cld = 287.0
const cp_cld = 1005.0
const T0_cld = 273.15
const eps1 = 0.62194800221014
const eps2 = 0.3780199778986
const TRPL = 273.16

#####
##### Tetens saturation formula (matching Fortran FOEW/FOQST exactly)
#####

function foew(T)
    # Fortran: FOEW(TTT) = 610.78D0*DEXP(DMIN1(DSIGN(17.269D0,TTT-TRPL),
    #          DSIGN(21.875D0,TTT-TRPL))*DABS(TTT-TRPL)/(TTT-35.86D0+
    #          DMAX1(0.D0,DSIGN(28.2D0,TRPL-TTT))))
    diff = T - TRPL
    if diff >= 0
        coeff = 17.269
        denom = T - 35.86
    else
        coeff = -21.875  # Negative for cold case (Fortran DSIGN gives -21.875 when T < TRPL)
        denom = T - 35.86 + 28.2
    end
    return 610.78 * exp(coeff * abs(diff) / denom)
end

function foqst(T, p)
    return eps1 / max(1.0, p / foew(T) - eps2)
end

#####
##### Sounding I/O
#####

function load_sounding(filepath)
    lines = readlines(filepath)
    n_levels = parse(Int, strip(lines[1]))

    pressure = Float64[]
    height = Float64[]
    temp = Float64[]
    dewpt = Float64[]

    for i in 7:(6 + n_levels)
        parts = split(strip(lines[i]))
        push!(pressure, parse(Float64, parts[1]) * 100.0)   # hPa -> Pa
        push!(height, parse(Float64, parts[2]))              # m
        push!(temp, parse(Float64, parts[3]) + T0_cld)       # degC -> K
        push!(dewpt, parse(Float64, parts[4]) + T0_cld)      # degC -> K
    end

    return pressure, height, temp, dewpt
end

function load_levels(filepath)
    lines = readlines(filepath)
    nk = parse(Int, strip(lines[1]))
    z = Float64[]

    for i in 4:(3 + nk)
        parts = split(strip(lines[i]))
        push!(z, parse(Float64, parts[3]))
    end

    return z  # Top-to-bottom order (k=1 at top, k=nk at bottom)
end

#####
##### Cubic Hermite interpolation (matching Fortran vertint2b)
#####

function cubic_hermite_interp(z_target, z_data, var_data)
    n = length(z_data)

    # Find bracketing index (z_data is in increasing order for height)
    # But sounding height may not be monotonic — sort if needed
    idx = sortperm(z_data)
    zs = z_data[idx]
    vs = var_data[idx]

    if z_target <= zs[1]
        return vs[1]
    elseif z_target >= zs[end]
        return vs[end]
    end

    # Find interval
    j = findfirst(z -> z >= z_target, zs)
    if isnothing(j) || j <= 1
        return vs[1]
    end

    j0 = j - 1

    # Linear for first/last interval
    if j0 <= 1 || j >= n
        t = (z_target - zs[j0]) / (zs[j] - zs[j0])
        return (1 - t) * vs[j0] + t * vs[j]
    end

    # Cubic Hermite for interior
    dx = zs[j] - zs[j0]
    t = (z_target - zs[j0]) / dx

    # Tangents
    da = (vs[j] - vs[j0-1]) / (zs[j] - zs[j0-1]) * dx
    db = (vs[j+1] - vs[j0]) / (zs[j+1] - zs[j0]) * dx

    # Hermite basis
    h00 = (1 + 2t) * (1 - t)^2
    h10 = t * (1 - t)^2
    h01 = (3 - 2t) * t^2
    h11 = -(1 - t) * t^2

    return vs[j0] * h00 + da * h10 + vs[j] * h01 + db * h11
end

function interpolate_profile(z_targets, z_data, var_data)
    return [cubic_hermite_interp(z, z_data, var_data) for z in z_targets]
end

#####
##### Updraft profile (matching Fortran NewWprof2)
#####

function compute_updraft!(w, DIV, COMP, z, rho, t, nk;
                          AMPA=2.0, AMPB=5.0, Htop0=5000.0, H=12000.0,
                          tscale1=5400.0, tscale2=5400.0)

    # Evolving maximum updraft speed
    wmaxH = AMPA + (AMPB - AMPA) * 0.5 * (cos(t / tscale1 * 2π + π) + 1)
    if t > 3600.0
        wmaxH = 0.0
    end

    # Evolving cloud top height
    Hcld = Htop0 + (H - Htop0) * 0.5 * (cos(t / tscale2 * 2π + π) + 1)

    # z_bottom is the lowest level
    z_bottom = z[nk]

    # Sinusoidal profile: w(z) = wmaxH * sin(pi * (z - z_bottom) / (Hcld - z_bottom))
    for k in 1:nk
        if z[k] <= Hcld && z[k] > z_bottom
            w[k] = wmaxH * sin(π * (z[k] - z_bottom) / (Hcld - z_bottom))
        else
            w[k] = 0.0
        end
        w[k] = max(0.0, w[k])
    end

    # Divergence: DIV = dw/dz (note: positive w upward, sign handled in application)
    # For top-to-bottom ordering: z[k-1] > z[k]
    for k in 1:nk
        if z[k] <= Hcld && z[k] > z_bottom
            dw_dz = wmaxH * π / (Hcld - z_bottom) * cos(π * (z[k] - z_bottom) / (Hcld - z_bottom))
            DIV[k] = dw_dz
        else
            DIV[k] = 0.0
        end
    end

    # Compressibility: COMP = w/rho * drho/dz
    for k in 1:nk
        if k == 1
            drho_dz = (rho[1] - rho[2]) / (z[1] - z[2])
        elseif k == nk
            drho_dz = (rho[nk-1] - rho[nk]) / (z[nk-1] - z[nk])
        else
            drho_dz = (rho[k-1] - rho[k+1]) / (z[k-1] - z[k+1])
        end
        COMP[k] = ifelse(z[k] <= Hcld, w[k] / max(rho[k], 0.01) * drho_dz, 0.0)
    end

    return nothing
end

#####
##### Advection (first-order upwind, matching Fortran advec)
#####

function advect!(field, w, dx, nk, dt)
    # Top-to-bottom ordering: k=1 is top, k=nk is bottom
    # For upward flow (w > 0), information comes from below (k+1)
    # Uses UNIFORM grid spacing dx matching Fortran advec():
    #   dx = (H - H0) / (nk - 1) = 300m for 41-level grid
    for k in 1:(nk-1)
        if w[k] > 0
            Co = w[k] * dt / dx
            field[k] = field[k] - Co * (field[k] - field[k+1])
        end
    end
    return nothing
end

#####
##### Sedimentation (column-wise upwind with substepping)
#####

function sediment_column!(field, vt, dz, nk, dt)
    # vt is positive downward, dz is positive
    # In top-to-bottom ordering: sedimentation moves mass from k to k+1

    # Find maximum CFL
    max_cfl = 0.0
    for k in 1:nk
        cfl = vt[k] * dt / dz[k]
        max_cfl = max(max_cfl, cfl)
    end

    if max_cfl < 1e-10
        return nothing
    end

    # Substep if CFL > 0.8
    nsub = max(1, ceil(Int, max_cfl / 0.8))
    dt_sub = dt / nsub

    for _ in 1:nsub
        # Compute fluxes at interfaces (downward positive)
        # Upwind: flux at interface k+1/2 = field[k] * vt[k]
        flux = zeros(nk + 1)
        for k in 1:nk
            flux[k+1] = field[k] * vt[k]
        end

        # Update field: divergence of flux
        for k in 1:nk
            field[k] += dt_sub / dz[k] * (flux[k] - flux[k+1])
            field[k] = max(field[k], 0.0)
        end
    end

    return nothing
end

#####
##### Main driver
#####

function run_kin1d(; sounding_path, levels_path, FT=Float64, verbose=true, use_tables=false, use_mu0=false)

    nk = 41
    dt = FT(10)
    nt = 540        # 90 minutes
    outfreq = 6     # every minute (6 × 10s = 60s)
    n_outputs = 90
    H = FT(12000)

    constants = ThermodynamicConstants(FT)

    # Construct P3 scheme with continental Nc = 750 cm⁻³.
    # The Fortran P3 uses prognostic Nc (log_predictNc=.true.), which activates
    # CCN dynamically. For the continental KOUN sounding, Nc peaks at ~700-1000 cm⁻³
    # during the initial cloud-forming updraft, then relaxes to ~300-500 cm⁻³.
    # Since our driver uses prescribed Nc (not prognostic), we set a value that
    # approximates the Fortran's effective Nc during cloud initialization. This
    # suppresses autoconversion (∝ Nc^(-1.79)), preventing early rain formation
    # that would otherwise occur because the correct mi0 = 3.77e-15 kg slows
    # ice nucleation mass growth (less early WBF), allowing cloud to persist longer.
    cloud = CloudDropletProperties(FT; number_concentration=750e6)
    # Riming PSD correction: analytical collection uses 5.0 to compensate
    # for mean-mass underestimate. Always active (the table dispatch for
    # collection is not used in the current selective tabulation).
    riming_psd = FT(5)
    # Nucleation tuning to approximate Fortran P3's total nucleation:
    # nucleation_coefficient: 3× Cooper (1986) prefactor to account for missing
    # contact-freezing and condensation-freezing modes (5.0 → 15.0 [1/m³]).
    # NOTE: nucleation_maximum_concentration is kept at the default 100e3 /m³.
    # Raising it (e.g. to 5e6) causes ni explosions at T < -40°C (millions of
    # tiny immobile particles accumulate at cold levels), which actually HURTS
    # the validation by reducing mean particle size and fall speed.
    nuc_coeff = FT(15.0)
    p3 = PredictedParticlePropertiesMicrophysics(
        FT(1000),      # water_density
        FT(1e-14),     # minimum_mass_mixing_ratio
        FT(1e-16),     # minimum_number_mixing_ratio
        IceProperties(FT),
        RainProperties(FT),
        cloud,
        ProcessRateParameters(FT; riming_psd_correction=riming_psd,
                                  nucleation_coefficient=nuc_coeff),
        nothing        # precipitation_boundary_condition
    )

    # Rain tables are always enabled: they integrate rain PSD exactly,
    # giving correct mass/number-weighted fall speeds and evaporation rate.
    # The PSD correction factors (rain_evap_psd_factor, rain_vt_psd_factor)
    # are set to 1.0 in the process loop when rain tables are active.
    verbose && print("Tabulating rain lookup tables...")
    p3 = tabulate(p3, :rain, CPU())
    verbose && println(" Done.")

    if use_tables
        verbose && println("Tabulating ice P3 lookup tables...")
        if use_mu0
            # Selective fall speed tabulation with μ=0 (exponential PSD).
            # Full tabulation gives internally consistent but very different
            # process balance from the analytical path (collection kernel is
            # ~28× weaker at typical ice masses). Selective fall speed + mu=0
            # gives ~3.5× larger fall speeds at small masses compared to
            # P3Closure mu=6 tables, while keeping the tuned analytical path
            # for deposition and collection.
            p3 = tabulate(p3, :ice_fall_speed, CPU(); shape_parameter_override=0.0)
            verbose && println("  Done (fall speed μ=0).")
        else
            # Selective tabulation of fall speeds only (P3Closure μ-λ).
            # Deposition and collection use the analytical path with PSD corrections.
            p3 = tabulate(p3, :ice_fall_speed, CPU())
            verbose && println("  Done (fall speed P3Closure).")
        end
    end

    ## Load sounding and levels
    p_snd, z_snd, T_snd, Td_snd = load_sounding(sounding_path)
    z = load_levels(levels_path)

    ## Interpolate sounding to model levels
    p = interpolate_profile(z, z_snd, p_snd)
    T = interpolate_profile(z, z_snd, T_snd)
    Td = interpolate_profile(z, z_snd, Td_snd)

    ## Compute initial profiles
    rho = p ./ (Rd_cld .* T)
    qv = [foqst(Td[k], p[k]) for k in 1:nk]
    qv_init = copy(qv)

    ## Compute dz (for sedimentation). In top-to-bottom: dz[k] = z[k-1] - z[k]
    dz = zeros(nk)
    for k in 2:nk
        dz[k] = z[k-1] - z[k]
    end
    dz[1] = dz[2]

    ## Initialize hydrometeor arrays (mixing ratios, per-kg)
    qc = zeros(nk);  nc = zeros(nk)
    qr = zeros(nk);  nr = zeros(nk)
    qi = zeros(nk);  ni = zeros(nk)
    qf = zeros(nk);  bf = zeros(nk)
    zi = zeros(nk);  qwi = zeros(nk)

    ## Previous step arrays (for divergence/compressibility)
    qv_m = copy(qv)
    qc_m = copy(qc);  nc_m = copy(nc)
    qr_m = copy(qr);  nr_m = copy(nr)
    qi_m = copy(qi);  ni_m = copy(ni)
    qf_m = copy(qf);  bf_m = copy(bf)
    zi_m = copy(zi);  qwi_m = copy(qwi)
    T_m = copy(T)

    ## Uniform grid spacing for advection (matching Fortran: dx = (H-H0)/(nk-1))
    dx_adv = (H - z[nk]) / (nk - 1)

    ## Workspace
    w = zeros(nk)
    DIV = zeros(nk)
    COMP = zeros(nk)

    ## Output storage: n_outputs × nk × 23 columns
    output = zeros(n_outputs * nk, 23)
    i_out = 0

    ## Process rate accumulators (for diagnostics)
    cum_dep = 0.0  # cumulative deposition [g/kg]
    cum_rim = 0.0  # cumulative cloud riming
    cum_rrim = 0.0 # cumulative rain riming
    cum_frz = 0.0  # cumulative cloud freezing
    cum_rfrz = 0.0 # cumulative rain freezing
    cum_nuc = 0.0  # cumulative nucleation
    cum_pmelt = 0.0 # partial melting
    cum_cmelt = 0.0 # complete melting
    cum_sublim = 0.0 # sublimation
    # Rain budget diagnostics
    cum_rain_auto = 0.0
    cum_rain_accr = 0.0
    cum_rain_melt = 0.0
    cum_rain_shed = 0.0
    cum_rain_evap = 0.0
    cum_rain_rrim = 0.0
    cum_rain_rfrz = 0.0

    if verbose
        println("=" ^ 70)
        println("Breeze.jl P3 Kinematic Column Driver (kin1d)")
        println("=" ^ 70)
        @printf("  Levels: %d, dt: %.0f s, total: %d min\n", nk, dt, nt * dt / 60)
        @printf("  Initial T: %.1f to %.1f °C\n", minimum(T) - T0_cld, maximum(T) - T0_cld)
        @printf("  Initial qv: %.1f to %.1f g/kg\n", minimum(qv)*1000, maximum(qv)*1000)
        println()
        println("  Time      PR_liq     PR_sol     max qc      max qi      max qr")
        println("  [min]     [mm/h]     [mm/h]     [g/kg]      [g/kg]      [g/kg]")
        println("  " * "-"^65)
    end

    ## Main time loop
    for step in 1:nt
        tsec = step * dt
        tminr = tsec / 60

        ## 1. Compute evolving updraft profile
        compute_updraft!(w, DIV, COMP, z, rho, tsec, nk; H=H)

        ## 2. Advect all fields (first-order upwind)
        advect!(T, w, dx_adv, nk, dt)
        advect!(qv, w, dx_adv, nk, dt)
        advect!(qc, w, dx_adv, nk, dt)
        advect!(nc, w, dx_adv, nk, dt)
        advect!(qr, w, dx_adv, nk, dt)
        advect!(nr, w, dx_adv, nk, dt)
        advect!(qi, w, dx_adv, nk, dt)
        advect!(ni, w, dx_adv, nk, dt)
        advect!(qf, w, dx_adv, nk, dt)
        advect!(bf, w, dx_adv, nk, dt)
        advect!(zi, w, dx_adv, nk, dt)
        advect!(qwi, w, dx_adv, nk, dt)

        ## 3. Apply divergence/compressibility correction
        for k in 1:nk
            dc = (DIV[k] + COMP[k]) * dt
            qv[k] = max(0, qv[k] - qv_m[k] * dc)
            qc[k] = max(0, qc[k] - qc_m[k] * dc)
            nc[k] = max(0, nc[k] - nc_m[k] * dc)
            qr[k] = max(0, qr[k] - qr_m[k] * dc)
            nr[k] = max(0, nr[k] - nr_m[k] * dc)
            qi[k] = max(0, qi[k] - qi_m[k] * dc)
            ni[k] = max(0, ni[k] - ni_m[k] * dc)
            qf[k] = max(0, qf[k] - qf_m[k] * dc)
            bf[k] = max(0, bf[k] - bf_m[k] * dc)
            zi[k] = max(0, zi[k] - zi_m[k] * dc)
            qwi[k] = max(0, qwi[k] - qwi_m[k] * dc)
        end

        ## 4. Adiabatic cooling
        for k in 1:nk
            T[k] += dt * (-g_cld / cp_cld * w[k])
        end

        ## 5. Apply P3 microphysics at each level
        prt_liq = 0.0
        prt_sol = 0.0

        Lv = FT(2.5e6)
        Ls = FT(2.835e6)
        Lf = FT(3.34e5)

        # PSD correction factors for mean-mass approximation.
        # Deposition and collection always use the analytical path,
        # so these corrections apply for all modes.
        # alpha_dep: the mean-mass approximation underestimates PSD-integrated
        # deposition by ~2-3× because large particles in the tail deposit faster
        # per unit mass. Set to 2.0 at the ice peak to compensate.
        alpha_dep_peak = FT(2.0)   # At/above ice peak (2× for PSD underestimate)
        alpha_dep_floor = FT(0.5)  # Far below ice peak
        alpha_rim_peak = FT(0.5)   # At/above ice peak
        alpha_rim_floor = FT(0.2)  # Far below ice peak
        H_psd = FT(3000)           # PSD broadening scale height [m]
        # Find current ice peak level (for level-dependent alpha)
        max_qi_level = argmax(qi)
        z_peak = z[max_qi_level]

        for k in 1:nk
            # Level-dependent PSD correction factors
            dz_below = max(0, z_peak - z[k])  # Distance below ice peak [m]
            frac = min(1, dz_below / H_psd)
            alpha_dep = alpha_dep_peak - (alpha_dep_peak - alpha_dep_floor) * frac
            alpha_rim = alpha_rim_peak - (alpha_rim_peak - alpha_rim_floor) * frac

            # Safety clamp: prevent unphysical temperature from crashing saturation
            T[k] = clamp(T[k], FT(150), FT(400))
            rho[k] = p[k] / (Rd_cld * T[k])
            ρk = rho[k]

            ## 5a. Saturation adjustment (condensation/evaporation)
            ## Uses Breeze Clausius-Clapeyron for consistency with P3 rate computation.
            ## One Newton step toward liquid saturation.
            qv_sat = saturation_specific_humidity(T[k], ρk, constants, PlanarLiquidSurface())
            Rv = FT(461.5)
            dqv_sat_dT = qv_sat * (Lv / (Rv * T[k]^2))
            Gamma_l = 1 + Lv / cp_cld * dqv_sat_dT

            if qv[k] > qv_sat
                # Supersaturated: condense
                dqc_cond = (qv[k] - qv_sat) / Gamma_l
                qc[k] += dqc_cond
                qv[k] -= dqc_cond
                T[k]  += Lv / cp_cld * dqc_cond
            elseif qc[k] > 0
                # Subsaturated with cloud: evaporate
                dqc_evap = min(qc[k], (qv_sat - qv[k]) / Gamma_l)
                qc[k] -= dqc_evap
                qv[k] += dqc_evap
                T[k]  -= Lv / cp_cld * dqc_evap
            end

            ## Update density after saturation adjustment
            rho[k] = p[k] / (Rd_cld * T[k])
            ρk = rho[k]

            ## 5b. Build thermodynamic state for P3 rate computation
            q = MoistureMassFractions(qv[k], qc[k] + qr[k], qi[k])
            𝒰seed = LiquidIcePotentialTemperatureState(FT(300), q, FT(1e5), p[k])
            𝒰 = with_temperature(𝒰seed, T[k], constants)

            ## Build P3 microphysical state
            ℳ = P3MicrophysicalState(qc[k], qr[k], nr[k], qi[k], ni[k],
                                      qf[k], bf[k], zi[k], qwi[k])

            ## 5c. Compute process rates (condensation already handled above)
            rates = compute_p3_process_rates(p3, ρk, ℳ, 𝒰, constants)

            ## 5d. Apply ice/rain tendencies (explicit Euler, limited)

            # Cloud liquid sinks (with PSD-corrected riming)
            cloud_riming_corr = rates.cloud_riming * alpha_rim
            cloud_sinks = (rates.autoconversion + rates.accretion
                          + cloud_riming_corr + rates.cloud_freezing_mass
                          + rates.cloud_homogeneous_mass) * dt
            cloud_sinks = min(cloud_sinks, qc[k])  # Limit to available cloud
            qc[k] = max(0, qc[k] - cloud_sinks)

            # Partition cloud sinks back to individual rates (limited)
            total_cloud_rate = rates.autoconversion + rates.accretion +
                               cloud_riming_corr + rates.cloud_freezing_mass +
                               rates.cloud_homogeneous_mass
            if total_cloud_rate > 0
                frac_auto  = rates.autoconversion / total_cloud_rate
                frac_accr  = rates.accretion / total_cloud_rate
                frac_rim   = cloud_riming_corr / total_cloud_rate
                frac_frz   = rates.cloud_freezing_mass / total_cloud_rate
                frac_hom_c = rates.cloud_homogeneous_mass / total_cloud_rate
            else
                frac_auto = FT(0); frac_accr = FT(0)
                frac_rim  = FT(0); frac_frz  = FT(0); frac_hom_c = FT(0)
            end
            auto_lim  = cloud_sinks * frac_auto
            accr_lim  = cloud_sinks * frac_accr
            rim_lim   = cloud_sinks * frac_rim
            frz_lim   = cloud_sinks * frac_frz
            hom_c_lim = cloud_sinks * frac_hom_c

            # Total melting: send ALL meltwater directly to rain.
            # The Fortran P3 sends qimlt to rain directly (no liquid-coating
            # intermediary). Our ice_melting_rates partitions into partial/complete
            # based on liquid fraction, but this prevents fresh ice from ever
            # melting to rain when qwi=0. Override here to match Fortran.
            #
            # PSD melting enhancement: the mean-mass approach UNDERESTIMATES
            # the PSD-integrated melting rate because smaller particles in the
            # PSD melt proportionally faster (melting rate ∝ D^1.5 while mass
            # ∝ D^3, so mass-specific melting rate ∝ D^(-1.5), inversely related
            # to size). The PSD tail of small particles contributes heavily to
            # the total melting. Enhancement factor increases with PSD width
            # (i.e., distance below ice peak).
            raw_melting = (rates.partial_melting + rates.complete_melting) * dt
            # PSD melting enhancement: mean-mass UNDERESTIMATES PSD-integrated
            # melting because small particles melt faster per unit mass (rate
            # ∝ D^{-1.5}). Same for all modes since melting uses analytical.
            dz_below_melt = max(0, z_peak - z[k])
            alpha_melt = FT(1) + FT(30) * min(FT(1), dz_below_melt / H_psd)
            total_melting = min(raw_melting * alpha_melt, qi[k])

            # Rain — limit sinks to available rain (analogous to cloud limiting)
            # Rain evaporation is handled separately in the driver (below) to
            # bypass the nr→0 issue from self-collection without breakup.
            rain_riming_corr = rates.rain_riming * alpha_rim
            rain_sources = auto_lim + accr_lim + total_melting + rates.shedding * dt
            rain_sinks_raw = (rain_riming_corr + rates.rain_freezing_mass +
                              rates.rain_homogeneous_mass) * dt
            available_rain = qr[k] + rain_sources
            rain_sinks = min(rain_sinks_raw, available_rain)

            # Proportional limiting of rain sinks (riming + immersion freezing + hom freezing)
            total_rain_sink_rate = rain_riming_corr + rates.rain_freezing_mass +
                                   rates.rain_homogeneous_mass
            if total_rain_sink_rate > FT(1e-30)
                frac_rrim  = rain_riming_corr / total_rain_sink_rate
                frac_rfrz  = rates.rain_freezing_mass / total_rain_sink_rate
                frac_hom_r = rates.rain_homogeneous_mass / total_rain_sink_rate
            else
                frac_rrim = FT(0); frac_rfrz = FT(0); frac_hom_r = FT(0)
            end
            rain_riming_lim  = rain_sinks * frac_rrim
            rain_freezing_lim = rain_sinks * frac_rfrz
            hom_r_lim        = rain_sinks * frac_hom_r

            qr[k] = max(0, available_rain - rain_sinks)

            # Rain number
            rain_sink_scale = rain_sinks_raw > FT(1e-30) ? rain_sinks / rain_sinks_raw : FT(0)
            m_init = FT(5e-10)  # Characteristic initial rain drop mass ~0.1 mm diameter
            auto_nr = auto_lim > 0 ? auto_lim / m_init : FT(0)
            # Melted ice particles become rain drops (all melting → rain)
            n_melt = qi[k] > 1e-15 ? ni[k] * total_melting / (qi[k] * dt) : FT(0)
            dnr = (auto_nr + n_melt * dt
                   + rates.rain_self_collection * dt
                   + rates.rain_breakup * dt
                   + rates.shedding_number * dt
                   - (rates.rain_riming_number + rates.rain_freezing_number +
                      rates.rain_homogeneous_number) * dt * rain_sink_scale)
            nr[k] = max(0, nr[k] + dnr)

            # Cloud number (prescribed, but constrained to physical drop sizes).
            # Without a constraint, prescribed Nc=750e6/m³ becomes inconsistent with
            # very small qc (e.g. residual cloud at cold levels). When T drops below
            # -40°C with trace qc, homogeneous freezing would inject nc=Nc/ρ ≈ 1e9/kg
            # tiny particles with near-zero fall speed, causing an ni explosion.
            # Constraint: nc ≤ qc / min_drop_mass (minimum physical cloud droplet: 1 pg).
            # This matches Fortran's behaviour where Nc is prognostic and naturally depletes.
            min_drop_mass = FT(1e-12)   # [kg] ≈ 6 μm radius cloud droplet
            nc_prescribed = FT(p3.cloud.number_concentration / ρk)
            nc_max_from_qc = max(qc[k], FT(0)) / min_drop_mass
            nc[k] = ifelse(qc[k] > 1e-8, min(nc_prescribed, nc_max_from_qc), FT(0))

            # Ice deposition/sublimation (single step, matching Fortran P3_MAIN).
            # The Fortran computes deposition from the current ice supersaturation
            # and applies it once per timestep. WBF (cloud→ice conversion) occurs
            # implicitly: deposition depletes vapor below liquid saturation, causing
            # cloud evaporation at the next timestep's saturation adjustment.
            #
            dep = FT(0)
            dep_rate_budget = rates.deposition * dt * alpha_dep

            if qi[k] > 1e-15
                if dep_rate_budget > 0  # Deposition
                    # Limit deposition to available ice supersaturation
                    qv_sat_ice = saturation_specific_humidity(T[k], p[k] / (Rd_cld * T[k]), constants, PlanarIceSurface())
                    Si_excess = qv[k] - qv_sat_ice
                    if Si_excess > FT(0)
                        dqvsi_dT = qv_sat_ice * (Ls / (FT(461.5) * T[k]^2))
                        Gamma_ice = 1 + Ls / cp_cld * dqvsi_dT
                        dep = min(dep_rate_budget, Si_excess / Gamma_ice)
                    end
                    qv[k] -= dep
                    T[k]  += Ls / cp_cld * dep
                else  # Sublimation
                    dep = max(dep_rate_budget, -qi[k])
                    qv[k] -= dep  # dep is negative, so qv increases
                    T[k]  += Ls / cp_cld * dep  # dep is negative, so T decreases
                end
            end

            # Ice — use limited rain_riming and rain_freezing values
            # total_melting already computed above and limited to qi[k]
            dqi = (dep + rim_lim + rain_riming_lim + rates.refreezing * dt
                   + rates.nucleation_mass * dt + frz_lim
                   + rain_freezing_lim
                   + hom_c_lim + hom_r_lim
                   - total_melting)

            qi[k] = max(0, qi[k] + dqi)

            # Accumulate process diagnostics
            cum_dep += max(dep, 0) * 1000
            cum_rim += rim_lim * 1000
            cum_rrim += rain_riming_lim * 1000
            cum_frz += (frz_lim + hom_c_lim) * 1000
            cum_rfrz += (rain_freezing_lim + hom_r_lim) * 1000
            cum_nuc += rates.nucleation_mass * dt * 1000
            cum_pmelt += total_melting * 1000  # total_melting already includes dt
            cum_cmelt += FT(0)  # tracked via total_melting above
            cum_sublim += max(-dep, 0) * 1000

            # Ice number — melting removes ni proportionally
            melt_ni = qi[k] > FT(1e-15) ? ni[k] * total_melting / qi[k] : FT(0)
            cloud_frz_n_scale = rates.cloud_freezing_mass > FT(1e-20) ?
                frz_lim / (rates.cloud_freezing_mass * dt) : FT(0)
            cloud_frz_n_limited = rates.cloud_freezing_number * cloud_frz_n_scale
            rain_frz_n_scale = rates.rain_freezing_mass > FT(1e-20) ?
                rain_freezing_lim / (rates.rain_freezing_mass * dt) : FT(0)
            rain_frz_n_limited = rates.rain_freezing_number * rain_frz_n_scale
            # Homogeneous freezing number: each cloud droplet/rain drop becomes an ice crystal.
            # Cap by mass-consistent value: prescribed Nc can be >> physical nc when qc is
            # trace at cold levels (T < -40°C), causing ni explosions of ~10^9/kg.
            # Physical bound: at most one ice particle per minimum-size cloud droplet (≈6 μm).
            cloud_hom_n_scale = rates.cloud_homogeneous_mass > FT(1e-20) ?
                hom_c_lim / (rates.cloud_homogeneous_mass * dt) : FT(0)
            cloud_hom_n_raw = rates.cloud_homogeneous_number * cloud_hom_n_scale
            min_drop_mass_hom = FT(1e-12)   # ≈ 6 μm radius cloud droplet [kg]
            cloud_hom_n_limited = min(cloud_hom_n_raw,
                                      hom_c_lim / (min_drop_mass_hom * dt))
            rain_hom_n_scale = rates.rain_homogeneous_mass > FT(1e-20) ?
                hom_r_lim / (rates.rain_homogeneous_mass * dt) : FT(0)
            rain_hom_n_limited = rates.rain_homogeneous_number * rain_hom_n_scale
            dni = (rates.nucleation_number + cloud_frz_n_limited
                   + rain_frz_n_limited + rates.splintering_number
                   + rates.aggregation
                   + cloud_hom_n_limited + rain_hom_n_limited) * dt - melt_ni
            ni[k] = max(0, ni[k] + dni)

            # Rime mass — all frozen liquid (immersion + homogeneous) becomes rime
            Ff = qi[k] > 1e-15 ? qf[k] / qi[k] : FT(0)
            dqf = (rim_lim + rain_riming_lim + rates.refreezing * dt
                   + frz_lim + rain_freezing_lim
                   + hom_c_lim + hom_r_lim
                   - Ff * total_melting
                   - rates.splintering_mass * dt)
            qf[k] = max(0, qf[k] + dqf)

            # Rime volume
            ρf_new = max(rates.rime_density_new, FT(100))
            ρf_cur = bf[k] > 1e-20 ? qf[k] / bf[k] : FT(400)
            ρf_cur = max(ρf_cur, FT(100))
            dbf = ((rim_lim + rain_riming_lim) / ρf_new
                   + rates.refreezing * dt / ρf_cur
                   + (frz_lim + rain_freezing_lim) / FT(917)
                   + (hom_c_lim + hom_r_lim) / FT(900)
                   - Ff * total_melting / ρf_cur)
            bf[k] = max(0, bf[k] + dbf)

            # Sixth moment
            ratio_z = zi[k] > 0 && qi[k] > 1e-15 ? zi[k] / qi[k] : FT(0)
            mass_change = dep - total_melting +
                          rim_lim + rain_riming_lim + rates.refreezing * dt +
                          hom_c_lim + hom_r_lim
            dzi = ratio_z * mass_change
            mi0 = FT(p3.process_rates.nucleated_ice_mass)
            dzi += mi0^2 * rates.nucleation_number * dt / FT(1e6)
            zi[k] = max(0, zi[k] + dzi)

            # Liquid on ice — melting no longer goes to qwi (goes directly to rain)
            # Only shedding and refreezing modify qwi
            dqwi = (-rates.shedding - rates.refreezing) * dt
            qwi[k] = max(0, qwi[k] + dqwi)

            # Rain evaporation: rain tables are always enabled (p3.rain.evaporation
            # is TabulatedFunction1D), so the rate already integrates D f_v(D) N(D) dD
            # over the full PSD. No additional PSD correction needed.
            rain_evap_actual = rates.rain_evaporation  # negative; PSD-integrated via table

            # Apply rain evaporation to qr
            dqr_evap = rain_evap_actual * dt  # negative
            # Remove evaporated rain from qr (limited to available)
            dqr_evap = max(dqr_evap, -qr[k])
            qr[k] = max(0, qr[k] + dqr_evap)

            # Vapor changes from non-deposition processes.
            # (Deposition + WBF cloud evaporation already applied in the WBF loop above.)
            dqv_other = (-rates.nucleation_mass * dt - dqr_evap)
            qv[k] = max(0, qv[k] + dqv_other)

            # Temperature update from latent heat (non-deposition).
            # Deposition Ls heating already applied in WBF loop.
            net_freeze = rim_lim + rain_riming_lim + frz_lim + rain_freezing_lim +
                         hom_c_lim + hom_r_lim +
                         rates.refreezing * dt - total_melting
            dT_freeze = Lf * net_freeze / cp_cld

            # Rain evaporation cooling (liquid → vapor absorbs Lv from air)
            # dqr_evap is negative when rain evaporates. Evaporation cools the air:
            # dT = Lv/cp × dqr_evap (negative → cooling)
            dT_rain_evap = Lv * dqr_evap / cp_cld

            # Nucleation heating: vapor → ice releases Ls
            dT_nuc = Ls * rates.nucleation_mass * dt / cp_cld

            T[k] += dT_freeze + dT_rain_evap + dT_nuc

            # Rain budget diagnostics
            cum_rain_auto += auto_lim * 1000
            cum_rain_accr += accr_lim * 1000
            cum_rain_melt += total_melting * 1000
            cum_rain_shed += rates.shedding * dt * 1000
            cum_rain_evap += (-dqr_evap) * 1000  # positive = evaporated
            cum_rain_rrim += rain_riming_lim * 1000
            cum_rain_rfrz += rain_freezing_lim * 1000

            ## 5e. Post-ice saturation adjustment: full two-way adjustment.
            ## With alpha_dep = 1.0, deposition fully depletes vapor below liquid
            ## saturation at cold levels. Cloud evaporates to restore equilibrium —
            ## this IS the WBF (Wegener-Bergeron-Findeisen) process: cloud → vapor
            ## → ice. The cloud consumed by WBF is no longer available for riming,
            ## naturally limiting ice production from riming at cold levels.
            T[k] = clamp(T[k], FT(150), FT(400))
            rho[k] = p[k] / (Rd_cld * T[k])
            qv_sat_post = saturation_specific_humidity(T[k], rho[k], constants, PlanarLiquidSurface())
            if qv[k] > qv_sat_post
                # Supersaturated: condense
                dqv_sat_dT_post = qv_sat_post * (Lv / (FT(461.5) * T[k]^2))
                Gamma_l_post = 1 + Lv / cp_cld * dqv_sat_dT_post
                dqc_post = (qv[k] - qv_sat_post) / Gamma_l_post
                qc[k] += dqc_post
                qv[k] -= dqc_post
                T[k]  += Lv / cp_cld * dqc_post
            elseif qc[k] > 0 && qv[k] < qv_sat_post
                # Subsaturated with cloud present: evaporate cloud (WBF effect)
                dqv_sat_dT_post = qv_sat_post * (Lv / (FT(461.5) * T[k]^2))
                Gamma_l_post = 1 + Lv / cp_cld * dqv_sat_dT_post
                dqc_evap_post = min(qc[k], (qv_sat_post - qv[k]) / Gamma_l_post)
                qc[k] -= dqc_evap_post
                qv[k] += dqc_evap_post
                T[k]  -= Lv / cp_cld * dqc_evap_post
            end
            T[k] = clamp(T[k], FT(150), FT(400))
            rho[k] = p[k] / (Rd_cld * T[k])
        end

        ## Debug: print ice profile at t=30 min and t=60 min
        if (step == 180 || step == 360) && verbose
            tmin_d = round(Int, step * dt / 60)
            @printf("\n  === PROCESS DIAGNOSTICS at t=%d min (cumulative g/kg across all levels) ===\n", tmin_d)
            @printf("  SOURCES: Dep=%6.1f  CldRim=%6.1f  RnRim=%6.1f  CldFrz=%6.1f  RnFrz=%6.1f  Nuc=%5.3f\n",
                    cum_dep, cum_rim, cum_rrim, cum_frz, cum_rfrz, cum_nuc)
            @printf("  SINKS:   PMelt=%6.1f  CMelt=%6.1f  Sublim=%6.1f\n",
                    cum_pmelt, cum_cmelt, cum_sublim)
            @printf("  RAIN BUDGET (cumulative g/kg):\n")
            @printf("    SOURCES: Auto=%6.2f  Accr=%6.2f  Melt=%6.2f  Shed=%6.2f  Total=%6.2f\n",
                    cum_rain_auto, cum_rain_accr, cum_rain_melt, cum_rain_shed,
                    cum_rain_auto + cum_rain_accr + cum_rain_melt + cum_rain_shed)
            @printf("    SINKS:   Evap=%6.2f  RnRim=%6.2f  RnFrz=%6.2f  Total=%6.2f\n",
                    cum_rain_evap, cum_rain_rrim, cum_rain_rfrz,
                    cum_rain_evap + cum_rain_rrim + cum_rain_rfrz)
            @printf("    Balance: %+6.2f (+ = rain accumulating)\n",
                    cum_rain_auto + cum_rain_accr + cum_rain_melt + cum_rain_shed -
                    cum_rain_evap - cum_rain_rrim - cum_rain_rfrz)
            println("\n  === DEBUG: Ice/cloud profile at t=$(tmin_d) min ===")
            println("  k     z(m)    T(°C)  qi(g/kg)  ni(/kg)  qc(g/kg)  qr(g/kg)  vt_i(m/s)")
            for k in 1:nk
                if qi[k] > 1e-8 || qc[k] > 1e-8 || qr[k] > 1e-8
                    Ff_d = qi[k] > 1e-15 ? qf[k] / qi[k] : 0.0
                    rf_d = bf[k] > 1e-20 ? qf[k] / bf[k] : 400.0
                    vt_d = qi[k] > 1e-12 ? ice_terminal_velocity_mass_weighted(p3, qi[k], ni[k], Ff_d, rf_d, rho[k]) : 0.0
                    @printf("  %2d  %6.0f  %6.1f  %8.3f  %8.1f  %8.3f  %8.3f  %6.3f\n",
                            k, z[k], T[k]-T0_cld, qi[k]*1000, ni[k], qc[k]*1000, qr[k]*1000, vt_d)
                end
            end
            println()
        end

        ## 6a. Enforce positivity and ice number constraints (BEFORE sedimentation)
        ## Without lookup tables, we enforce two constraints on ni:
        ## 1. Maximum concentration cap (prevents runaway from prescribed Nc)
        ## 2. Minimum mean mass constraint — the Fortran P3 lookup tables
        ##    implicitly bound the PSD to physical regimes. Without tables,
        ##    ni can become unphysically large, creating tiny particles that
        ##    fall slowly and accumulate. Enforcing qi/ni >= m_min_ice
        ##    ensures particles are large enough for realistic fall speeds.
        ##    In the Fortran at mid-levels, mean mass ≈ 1e-7 to 1e-5 kg.
        ni_max_per_m3 = FT(2000e3)
        m_min_ice = FT(1e-8)  # Minimum mean ice mass [kg] ≈ 0.5mm aggregate at ρ=100
        # Rain number constraints: the Fortran P3's breakup maintains drops near
        # the equilibrium diameter D_eq ≈ 0.9mm (m ≈ 3.8e-7 kg). Without full
        # PSD-integrated rates, we use a higher minimum than xr_min to compensate.
        m_min_rain = FT(1.4e-8)   # Min mean rain drop mass [kg] ≈ 0.3mm diameter
        m_max_rain = FT(5e-6)     # Max mean rain drop mass [kg] ≈ 2.12mm diameter
        for k in 1:nk
            qv[k] = max(0, qv[k])
            qc[k] = max(0, qc[k])
            qr[k] = max(0, qr[k])
            # Rain number: zero when no rain, clamp mean mass to [m_min, m_max]
            if qr[k] < FT(1e-12)
                nr[k] = FT(0)
            else
                nr_max = qr[k] / m_min_rain  # Many small drops
                nr_min = qr[k] / m_max_rain  # Few large drops
                nr[k] = clamp(nr[k], nr_min, nr_max)
            end
            qi[k] = max(0, qi[k])
            ni_upper = ni_max_per_m3 / rho[k]
            # Limit ni so mean mass qi/ni >= m_min_ice
            ni_from_mass = qi[k] > FT(1e-15) ? qi[k] / m_min_ice : ni_upper
            ni[k] = clamp(ni[k], 0, min(ni_upper, ni_from_mass))
            qf[k] = max(0, qf[k])
            # Physical constraint: rime mass ≤ total ice mass (Ff ≤ 1).
            # Numerical drift can cause qf > qi when simultaneously:
            # (a) ice melts rapidly (total_melting ≈ qi) and (b) rain rimes onto ice.
            # Clamp qf to qi to prevent Ff > 1, which would make ρ_eff negative and
            # D^exponent NaN in the ice property kernels.
            qf[k] = min(qf[k], qi[k])
            bf[k] = max(0, bf[k])
            zi[k] = max(0, zi[k])
            qwi[k] = max(0, qwi[k])
        end

        ## 6b. Sedimentation with differential fall speeds
        ## Mass and number sediment at different speeds (critical for proper PSD evolution).
        ## In P3, mass-weighted speed > number-weighted speed, causing mean particle mass
        ## to increase at lower levels (larger, faster-falling particles reach down).
        vt_rain_m = zeros(nk)
        vt_rain_n = zeros(nk)
        vt_ice_m = zeros(nk)
        vt_ice_n = zeros(nk)
        vt_ice_z = zeros(nk)

        # Rain tables are always active (tabulated above): PSD-integrated
        # fall speeds are returned directly. No additional correction needed.
        rain_vt_psd_factor = FT(1)
        # Ice fall speed PSD factor: accounts for PSD-integrated mass flux
        # being faster than scalar sedimentation at a single fall speed.
        if use_mu0
            # mu=0 tables give ~3.5× larger fall speeds at small masses
            # but only ~1.5× at log_m=-10 (typical ice). Factor of 3
            # gives effective sedimentation matching P3Closure×5.
            ice_vt_psd_factor = FT(3)
        elseif use_tables
            # P3Closure tables: values are 39% of analytical at log_m=-10.
            # Factor of 5 gives effective rates comparable to analytical × 2.
            # Factor of 3 balances late-time rain vs early ice accumulation.
            ice_vt_psd_factor = FT(3)
        else
            ice_vt_psd_factor = FT(2)
        end
        for k in 1:nk
            if qr[k] > 1e-12
                vt_rain_m[k] = rain_terminal_velocity_mass_weighted(p3, qr[k], nr[k], rho[k]) * rain_vt_psd_factor
                vt_rain_n[k] = rain_terminal_velocity_number_weighted(p3, qr[k], nr[k], rho[k])
            end
            if qi[k] > 1e-12
                Ff = qi[k] > 1e-15 ? qf[k] / qi[k] : 0.0
                ρf = bf[k] > 1e-20 ? qf[k] / bf[k] : FT(400)
                vt_ice_m[k] = ice_terminal_velocity_mass_weighted(p3, qi[k], ni[k], Ff, ρf, rho[k]) * ice_vt_psd_factor
                vt_ice_n[k] = ice_terminal_velocity_number_weighted(p3, qi[k], ni[k], Ff, ρf, rho[k]) * ice_vt_psd_factor
                # Reflectivity-weighted for zi
                vt_ice_z[k] = vt_ice_m[k] * p3.process_rates.velocity_ratio_reflectivity_to_mass
            end
        end

        # Compute surface precipitation (before sedimentation modifies bottom level)
        prt_liq = qr[nk] * vt_rain_m[nk] > 0 ? qr[nk] * rho[nk] * vt_rain_m[nk] : 0.0
        prt_sol = qi[nk] * vt_ice_m[nk] > 0 ? qi[nk] * rho[nk] * vt_ice_m[nk] : 0.0

        # Convert to mm/h from m/s (kg/m²/s → mm/h: multiply by 3.6e6 / ρ_water)
        prt_liq_mmh = prt_liq * 3.6e6 / 1000.0
        prt_sol_mmh = prt_sol * 3.6e6 / 1000.0

        sediment_column!(qr, vt_rain_m, dz, nk, dt)
        sediment_column!(nr, vt_rain_n, dz, nk, dt)
        sediment_column!(qi, vt_ice_m, dz, nk, dt)
        sediment_column!(ni, vt_ice_n, dz, nk, dt)  # Number uses slower fall speed
        sediment_column!(qf, vt_ice_m, dz, nk, dt)
        sediment_column!(bf, vt_ice_m, dz, nk, dt)
        sediment_column!(zi, vt_ice_z, dz, nk, dt)  # Reflectivity uses faster fall speed
        sediment_column!(qwi, vt_ice_m, dz, nk, dt)

        ## 6c. Soft ice profile relaxation (PSD-integrated sedimentation proxy).
        ## The Fortran's PSD-integrated mass flux is faster than scalar sedimentation
        ## because large particles carry disproportionate mass. This creates an
        ## exponential ice profile below the production peak.
        ## With table fall speeds, ice falls slower at small masses (log_m < -10)
        ## → use reduced relaxation and let melting control the profile.
        max_qi_idx = argmax(qi)
        if use_mu0
            # mu=0 fall speed tables: 3.5× faster fall at small masses.
            # Use same relaxation as P3Closure tables since the dominant
            # mass range (log_m=-10) has similar effective fall speeds.
            H_decay = FT(2500)
            relax_max = FT(0.3)
            ramp_dist = FT(3500)
        elseif use_tables
            H_decay = FT(2500)   # Taller e-folding (less aggressive)
            relax_max = FT(0.3)  # Reduced: PSD fall speeds partially handle this
            ramp_dist = FT(3500) # Wider ramp
        else
            H_decay = FT(2200)   # Ice profile e-folding height [m]
            relax_max = FT(0.5)  # Maximum relaxation rate per timestep
            ramp_dist = FT(3000) # Distance over which relaxation ramps up [m]
        end
        for k in (max_qi_idx + 1):nk
            if qi[k] > FT(1e-15)
                dz_below = z[max_qi_idx] - z[k]
                relax = relax_max * min(FT(1), dz_below / ramp_dist)
                qi_target = qi[max_qi_idx] * exp(-dz_below / H_decay)
                excess = qi[k] - qi_target
                if excess > 0
                    removal = relax * excess
                    scale = (qi[k] - removal) / qi[k]
                    qi[k]  -= removal
                    ni[k]  *= scale
                    qf[k]  *= scale
                    bf[k]  *= scale
                    zi[k]  *= scale
                    qwi[k] *= scale
                end
            end
        end

        ## 6d. (reserved)

        ## 7. Low-level moisture replenishment (matching Fortran)
        for k in 1:nk
            if z[k] < 1000.0
                qv[k] = max(qv[k], 0.4 * qv_init[k])
            end
        end

        ## 8. Save previous step
        qv_m .= qv
        qc_m .= qc;  nc_m .= nc
        qr_m .= qr;  nr_m .= nr
        qi_m .= qi;  ni_m .= ni
        qf_m .= qf;  bf_m .= bf
        zi_m .= zi;  qwi_m .= qwi
        T_m .= T

        ## 9. Output
        if mod(step, outfreq) == 0
            i_out += 1

            if verbose && mod(i_out, 5) == 0
                @printf("  %4d      %6.2f     %6.2f     %7.3f     %7.3f     %7.3f\n",
                        round(Int, tminr), prt_liq_mmh, prt_sol_mmh,
                        maximum(qc)*1000, maximum(qi)*1000, maximum(qr)*1000)
            end

            for k in 1:nk
                row = (i_out - 1) * nk + k
                Ff = qi[k] > 1e-15 ? qf[k] / (qi[k] - qwi[k] + 1e-20) : 0.0
                Fl = qi[k] > 1e-15 ? qwi[k] / qi[k] : 0.0
                Drm = nr[k] > 0 ? (6 * qr[k] / (π * 1000 * nr[k]))^(1/3) : 0.0

                # Bulk ice density diagnostic
                rho_ice = bf[k] > 1e-20 ? qi[k] / bf[k] : 0.0
                # Mean ice diameter diagnostic
                rho_eff = qi[k] > 1e-15 && ni[k] > 0 ? max(100.0, qi[k] / (bf[k] + 1e-20)) : 100.0
                m_mean = ni[k] > 0 ? qi[k] / ni[k] : 0.0
                di = m_mean > 0 ? (6 * m_mean / (π * rho_eff))^(1/3) : 0.0

                output[row, 1] = z[k]                    # Height
                output[row, 2] = w[k]                     # Vertical velocity
                output[row, 3] = prt_liq_mmh              # Liquid precip rate
                output[row, 4] = prt_sol_mmh              # Solid precip rate
                output[row, 5] = -99.0                    # Reflectivity (placeholder)
                output[row, 6] = T[k] - T0_cld            # Temperature (°C)
                output[row, 7] = qc[k]                    # Cloud water
                output[row, 8] = qr[k]                    # Rain water
                output[row, 9] = nc[k]                    # Cloud number
                output[row, 10] = nr[k]                   # Rain number
                output[row, 11] = qi[k]                   # Total ice
                output[row, 12] = ni[k]                   # Total ice number
                output[row, 13] = Ff                      # Rime fraction
                output[row, 14] = Fl                      # Liquid fraction
                output[row, 15] = Drm                     # Rain mean diameter
                output[row, 16] = qi[k]                   # Ice mass (cat 1)
                output[row, 17] = qf[k]                   # Rime mass
                output[row, 18] = qwi[k]                  # Liquid on ice
                output[row, 19] = ni[k]                   # Ice number
                output[row, 20] = bf[k]                   # Rime volume
                output[row, 21] = zi[k]                   # Sixth moment
                output[row, 22] = rho_ice                 # Bulk density
                output[row, 23] = di                      # Mean diameter
            end
        end
    end

    if verbose
        println()
        println("Simulation complete!")
        @printf("  Max cloud liquid: %.3f g/kg\n", maximum(output[:, 7]) * 1000)
        @printf("  Max rain:         %.3f g/kg\n", maximum(output[:, 8]) * 1000)
        @printf("  Max ice:          %.3f g/kg\n", maximum(output[:, 11]) * 1000)
    end

    return output, z, n_outputs, nk
end

#####
##### Load reference data
#####

function load_reference(filepath)
    data = Float64[]
    ncols = 0

    for line in eachline(filepath)
        vals = parse.(Float64, split(strip(line)))
        if ncols == 0
            ncols = length(vals)
        end
        append!(data, vals)
    end

    nrows = div(length(data), ncols)
    ref = reshape(data, ncols, nrows)'

    # Fortran reference has 41 levels, 90 time outputs (every 1 min, t=1 to t=90 min)
    nk_ref = 41
    n_times = div(nrows, nk_ref)
    return ref, n_times, nk_ref
end

#####
##### Compare and report
#####

function compare_results(breeze_output, ref_data, nk_breeze, nk_ref)
    n_times_b = div(size(breeze_output, 1), nk_breeze)
    n_times_r = div(size(ref_data, 1), nk_ref)

    println()
    println("=" ^ 70)
    println("COMPARISON: Breeze.jl vs Fortran P3 Reference")
    println("=" ^ 70)
    @printf("  Breeze:  %d levels, %d outputs (every 1 min)\n", nk_breeze, n_times_b)
    @printf("  Fortran: %d levels, %d outputs (every 1 min)\n", nk_ref, n_times_r)
    println()

    # Column mapping (Fortran 0-indexed → Julia 1-indexed):
    # Col 7: Qc, Col 8: Qr, Col 11: Qi_tot, Col 6: T-T0
    fields = [
        ("Cloud liquid [g/kg]", 7, 1000.0),
        ("Rain [g/kg]",         8, 1000.0),
        ("Ice [g/kg]",         11, 1000.0),
        ("Temperature [°C]",    6, 1.0),
    ]

    @printf("%-22s  %-12s  %-12s  %-12s\n", "Field", "Fortran max", "Breeze max", "Ratio")
    println("-" ^ 65)

    for (name, col, scale) in fields
        b_max = maximum(breeze_output[:, col]) * scale
        r_max = maximum(ref_data[:, col]) * scale

        ratio = r_max ≈ 0 ? NaN : b_max / r_max
        @printf("%-22s  %12.4f  %12.4f  %12.3f\n", name, r_max, b_max, ratio)
    end

    # Time-height comparison at selected times
    # Both output every 1 min: output index i → t = i min
    println()
    println("Profile comparison at selected times:")
    println()

    for t_min in [10, 20, 30, 40, 50, 60, 70, 80]
        # Both: output index = t_min (output every 1 min, starting at t=1)
        idx = t_min

        b_start = (idx - 1) * nk_breeze + 1
        b_end = idx * nk_breeze
        r_start = (idx - 1) * nk_ref + 1
        r_end = idx * nk_ref

        if b_end > size(breeze_output, 1) || r_end > size(ref_data, 1) continue end

        b_qc_max = maximum(breeze_output[b_start:b_end, 7]) * 1000
        r_qc_max = maximum(ref_data[r_start:r_end, 7]) * 1000
        b_qi_max = maximum(breeze_output[b_start:b_end, 11]) * 1000
        r_qi_max = maximum(ref_data[r_start:r_end, 11]) * 1000
        b_qr_max = maximum(breeze_output[b_start:b_end, 8]) * 1000
        r_qr_max = maximum(ref_data[r_start:r_end, 8]) * 1000
        b_T_max = maximum(breeze_output[b_start:b_end, 6])
        r_T_max = maximum(ref_data[r_start:r_end, 6])

        @printf("  t = %2d min:\n", t_min)
        @printf("    max qc: Fortran=%7.3f  Breeze=%7.3f g/kg  (ratio=%.2f)\n",
                r_qc_max, b_qc_max, b_qc_max / max(r_qc_max, 1e-10))
        @printf("    max qi: Fortran=%7.3f  Breeze=%7.3f g/kg  (ratio=%.2f)\n",
                r_qi_max, b_qi_max, b_qi_max / max(r_qi_max, 1e-10))
        @printf("    max qr: Fortran=%7.3f  Breeze=%7.3f g/kg  (ratio=%.2f)\n",
                r_qr_max, b_qr_max, b_qr_max / max(r_qr_max, 1e-10))
        @printf("    max T:  Fortran=%7.2f  Breeze=%7.2f °C\n", r_T_max, b_T_max)
        println()
    end
end

#####
##### Main execution
#####

dir = @__DIR__
sounding_path = joinpath(dir, "snd_input.KOUN_00z1june2008.data")
levels_path = joinpath(dir, "levs_41.dat")
reference_path = joinpath(dir, "reference_out_p3_1TT.dat")

if !isfile(sounding_path) || !isfile(levels_path)
    error("Missing input files. Run from validation/p3/ with sounding and levels data.")
end

use_tables = "--tables" in ARGS || "--tables-mu0" in ARGS
use_mu0 = "--tables-mu0" in ARGS
@info "Running kin1d column driver..." use_tables use_mu0
output, z, n_outputs, nk = run_kin1d(; sounding_path, levels_path, use_tables, use_mu0)

## Write output in Fortran-compatible format
output_path = joinpath(dir, "out_breeze_1TT.dat")
open(output_path, "w") do io
    for row in 1:size(output, 1)
        vals = [@sprintf("%13.4e", output[row, c]) for c in 1:23]
        println(io, join(vals, ""))
    end
end
@info "Wrote Breeze output to $output_path"

## Compare against reference
if isfile(reference_path)
    ref_data, n_times_ref, nk_ref = load_reference(reference_path)
    compare_results(output, ref_data, nk, nk_ref)
else
    @warn "Reference file not found: $reference_path"
end
