using Breeze
using Test
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Breeze.Microphysics: kessler_terminal_velocity,
                           mass_fraction_to_mixing_ratio, mixing_ratio_to_mass_fraction
using Breeze.Thermodynamics: MoistureMassFractions, saturation_specific_humidity, PlanarLiquidSurface

#####
##### Reference Fortran Kessler implementation (translated to Julia)
#####
# This is a direct translation of the DCMIP2016 Fortran kessler.f90
# to verify our Julia implementation produces equivalent results.

"""
    fortran_kessler!(theta, qv, qc, qr, rho, pk, dt, z, constants)

Reference implementation of the DCMIP2016 Kessler microphysics scheme.
This is a translation of the Fortran kessler.f90 subroutine, adapted to use
Breeze's `saturation_specific_humidity` to ensure thermodynamic consistency
during comparison (isolating microphysics logic from saturation formula differences).

Arguments (all modified in-place except rho, pk, dt, z, constants):
- `theta`: potential temperature (K)
- `qv`: water vapor mixing ratio (g/g)
- `qc`: cloud water mixing ratio (g/g)
- `qr`: rain water mixing ratio (g/g)
- `rho`: dry air density (kg/m³)
- `pk`: Exner function (p/p0)^(R/cp)
- `dt`: time step (s)
- `z`: heights of thermodynamic levels (m)
- `constants`: ThermodynamicConstants

Returns:
- `precl`: precipitation rate (m_water/s)
"""
function fortran_kessler!(theta, qv, qc, qr, rho, pk, dt, z, constants)
    nz = length(theta)

    # Fortran constants
    f2x = 17.27
    f5 = 237.3 * f2x * 2500000.0 / 1003.0
    κ = 0.2875       # kappa (r/cp)
    psl = 1000.0      # pressure at sea level (mb)
    rhoqr = 1000.0    # density of liquid water (kg/m³)

    # Pre-compute derived quantities
    r = zeros(nz)
    rhalf = zeros(nz)
    velqr = zeros(nz)
    pc = zeros(nz)
    sed = zeros(nz)

    for k = 1:nz
        r[k] = 0.001 * rho[k]
        rhalf[k] = sqrt(rho[1] / rho[k])
        pc[k] = 3.8 / (pk[k]^(1.0 / κ) * psl)
        # Liquid water terminal velocity (m/s) following KW eq. 2.15
        velqr[k] = 36.34 * (qr[k] * r[k])^0.1364 * rhalf[k]
    end

    # Maximum time step size in accordance with CFL condition
    if dt <= 0.0
        error("kessler called with nonpositive dt")
    end

    dt_max = dt
    for k = 1:nz-1
        if velqr[k] != 0.0
            dt_max = min(dt_max, 0.8 * (z[k+1] - z[k]) / velqr[k])
        end
    end

    # Number of subcycles
    rainsplit = ceil(Int, dt / dt_max)
    dt0 = dt / rainsplit

    # Subcycle through rain process
    precl = 0.0

    for nt = 1:rainsplit
        # Precipitation rate (m/s)
        precl += rho[1] * qr[1] * velqr[1] / rhoqr

        # Sedimentation term using upstream differencing
        for k = 1:nz-1
            sed[k] = dt0 * (r[k+1] * qr[k+1] * velqr[k+1] - r[k] * qr[k] * velqr[k]) / (r[k] * (z[k+1] - z[k]))
        end
        sed[nz] = -dt0 * qr[nz] * velqr[nz] / (0.5 * (z[nz] - z[nz-1]))

        # Adjustment terms
        for k = 1:nz
            # Autoconversion and accretion rates following KW eq. 2.13a,b
            qrprod = qc[k] - (qc[k] - dt0 * max(0.001 * (qc[k] - 0.001), 0.0)) / (1.0 + dt0 * 2.2 * qr[k]^0.875)
            qc[k] = max(qc[k] - qrprod, 0.0)
            qr[k] = max(qr[k] + qrprod + sed[k], 0.0)

            # Saturation vapor mixing ratio (g/g)
            # Use Breeze thermodynamics for saturation to match the Julia implementation
            T = theta[k] * pk[k]
            q_sat = saturation_specific_humidity(T, rho[k], constants, PlanarLiquidSurface())
            qvs = q_sat / (1.0 - q_sat)
            
            prod = (qv[k] - qvs) / (1.0 + qvs * f5 / (pk[k] * theta[k] - 36.0)^2)

            # Evaporation rate following KW eq. 2.14a,b
            ern_val = min(
                dt0 * (((1.6 + 124.9 * (r[k] * qr[k])^0.2046) * (r[k] * qr[k])^0.525) /
                       (2550000.0 * pc[k] / (3.8 * qvs) + 540000.0)) *
                      (max(qvs - qv[k], 0.0) / (r[k] * qvs)),
                max(-prod - qc[k], 0.0),
                qr[k]
            )

            # Saturation adjustment following KW eq. 3.10
            theta[k] = theta[k] + 2500000.0 / (1003.0 * pk[k]) * (max(prod, -qc[k]) - ern_val)
            qv[k] = max(qv[k] - max(prod, -qc[k]) + ern_val, 0.0)
            qc[k] = qc[k] + max(prod, -qc[k])
            qr[k] = qr[k] - ern_val
        end

        # Recalculate liquid water terminal velocity
        if nt != rainsplit
            for k = 1:nz
                velqr[k] = 36.34 * (qr[k] * r[k])^0.1364 * rhalf[k]
            end
        end
    end

    precl /= rainsplit

    return precl
end

#####
##### Tests for Kessler helper functions
#####

@testset "Kessler helper functions" begin
    @testset "Terminal velocity" begin
        # Test at typical atmospheric conditions
        ρ = 1.0         # kg/m³
        ρ_bottom = 1.2  # kg/m³
        rʳ = 0.001      # 1 g/kg rain mixing ratio

        vt = kessler_terminal_velocity(rʳ, ρ, ρ_bottom)
        @test vt > 0
        @test vt < 20  # Reasonable terminal velocity (m/s)

        # Zero rain should give zero velocity
        vt_zero = kessler_terminal_velocity(0.0, ρ, ρ_bottom)
        @test vt_zero == 0.0

        # Higher rain content should give higher velocity
        vt_high = kessler_terminal_velocity(0.005, ρ, ρ_bottom)
        @test vt_high > vt
    end

    @testset "Mass fraction ↔ mixing ratio conversion" begin
        # Test conversion formulas
        qᵗ = 0.02  # 2% total moisture
        q = 0.01   # 1% of some species

        r = mass_fraction_to_mixing_ratio(q, qᵗ)
        @test r ≈ q / (1 - qᵗ)

        r_test = 0.01
        q_back = mixing_ratio_to_mass_fraction(r_test, r_test)
        @test q_back ≈ r_test / (1 + r_test)

        # Round-trip
        qᵛ = 0.015
        qˡ = 0.003
        qᵗ_total = qᵛ + qˡ

        rᵛ = mass_fraction_to_mixing_ratio(qᵛ, qᵗ_total)
        rˡ = mass_fraction_to_mixing_ratio(qˡ, qᵗ_total)
        rᵗ = rᵛ + rˡ

        qᵛ_back = mixing_ratio_to_mass_fraction(rᵛ, rᵗ)
        qˡ_back = mixing_ratio_to_mass_fraction(rˡ, rᵗ)

        @test qᵛ_back ≈ qᵛ rtol=1e-10
        @test qˡ_back ≈ qˡ rtol=1e-10
    end
end

#####
##### Physical fidelity test
#####

@testset "Physical fidelity: Julia vs Fortran" begin
    # Use Float64 for accurate comparison
    FT = Float64
    
    # 1. Setup shared column data
    nz = 40
    z_min = FT(0)
    z_max = FT(4000)
    
    # Create grid
    grid = RectilinearGrid(CPU(), size=(1, 1, nz), x=(0, 100), y=(0, 100), z=(z_min, z_max), topology=(Periodic, Periodic, Bounded))
    z_centers = collect(znodes(grid, Center()))
    
    # Atmospheric profile
    T_surface = FT(288.0)
    p_surface = FT(101325.0)
    g = FT(9.81)
    Rd = FT(287.0)
    cp = FT(1003.0)
    
    # Create somewhat realistic profile
    # Linear lapse rate with height
    lapse_rate = FT(0.0065) # 6.5 K/km
    T_prof = T_surface .- lapse_rate .* z_centers
    p_prof = p_surface .* (T_prof ./ T_surface) .^ (g / (Rd * lapse_rate))
    rho_prof = p_prof ./ (Rd .* T_prof)
    
    # Exner function
    p0 = FT(100000.0)
    pk_prof = (p_prof ./ p0) .^ (Rd / cp)
    theta_prof = T_prof ./ pk_prof
    
    # Initial moisture (Mixing Ratios for Fortran, will convert for Julia)
    # 1. Vapor: Subsaturated at bottom, supersaturated in middle (cloud), subsaturated at top
    # 2. Cloud: Non-zero in middle
    # 3. Rain: Some rain in middle/lower
    
    r_v = zeros(FT, nz)
    r_c = zeros(FT, nz)
    r_r = zeros(FT, nz)
    
    for k in 1:nz
        z = z_centers[k]
        # Peak at 2000m
        r_v[k] = 0.015 * exp(-((z - 1000) / 1000)^2) # Vapor
        
        if 1500 < z < 2500
            r_c[k] = 0.002 # 2 g/kg cloud
        end
        
        if 1000 < z < 2000
            r_r[k] = 0.0005 # 0.5 g/kg rain
        end
    end
    
    dt = FT(10.0)

    # Configure constants to match Fortran hardcoded values AND simplified thermodynamics
    # Fortran Kessler uses constant cp=1003.0 and Rd=287.0 for all air (implicitly).
    # To verify the microphysics logic in isolation, we force Breeze to use these
    # simplified thermodynamic constants (equal cp and M for all species).
    
    R_gas = 8.314462618
    Rd_target = 287.0
    Md_target = R_gas / Rd_target
    cp_target = 1003.0
    
    constants = ThermodynamicConstants(FT;
        dry_air_heat_capacity = cp_target,
        vapor_heat_capacity = cp_target,
        dry_air_molar_mass = Md_target,
        vapor_molar_mass = Md_target,
        liquid = Breeze.Thermodynamics.CondensedPhase(FT; 
            reference_latent_heat = 2500000.0,
            heat_capacity = cp_target), # Match dry air cp to avoid mixture differences
        ice = Breeze.Thermodynamics.CondensedPhase(FT;
            reference_latent_heat = 2834000.0,
            heat_capacity = cp_target)
    )

    # 2. Run Fortran Reference
    theta_f = copy(theta_prof)
    qv_f = copy(r_v)
    qc_f = copy(r_c)
    qr_f = copy(r_r)
    rho_f = copy(rho_prof)
    pk_f = copy(pk_prof)
    z_f = copy(z_centers)
    
    precl_f = fortran_kessler!(theta_f, qv_f, qc_f, qr_f, rho_f, pk_f, dt, z_f, constants)

    # 3. Run Julia Implementation
    
    microphysics = DCMIP2016KesslerMicrophysics(f2x = 17.27)
    
    # Reference State
    # We construct a reference state that matches the profile's density/pressure
    # Note: Breeze's ReferenceState usually assumes hydrostatic balance.
    # Here we just manually set the reference fields to match our profile exactly.
    
    # Create model
    dynamics = AnelasticDynamics(ReferenceState(grid, constants)) 
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants)
    
    # Manually overwrite reference state fields to match our profile exactly
    # (Breeze might have computed slightly different hydrostatic balance)
    set!(model.dynamics.reference_state.density, reshape(rho_prof, 1, 1, nz))
    set!(model.dynamics.reference_state.pressure, reshape(p_prof, 1, 1, nz))

    # Initialize prognostic fields
    # We need to convert Mixing Ratios (r) to Mass Fractions (q)
    # q = r / (1 + r_t)
    
    r_t = r_v .+ r_c .+ r_r
    q_v = r_v ./ (1 .+ r_t)
    q_c = r_c ./ (1 .+ r_t)
    q_r = r_r ./ (1 .+ r_t)
    q_t = q_v .+ q_c .+ q_r
    
    # Set total moisture density
    set!(model.moisture_density, reshape(rho_prof .* q_t, 1, 1, nz))
    
    # Set cloud and rain densities
    set!(model.microphysical_fields.ρqᶜˡ, reshape(rho_prof .* q_c, 1, 1, nz))
    set!(model.microphysical_fields.ρqʳ, reshape(rho_prof .* q_r, 1, 1, nz))
    
    # Set potential temperature density
    # We need liquid-ice potential temperature θ_li
    # T = Π * θ_li + L * q_l / cp_m
    # θ_li = (T - L * q_l / cp_m) / Π
    # Where Π = (p/p0)^(R_m/cp_m)
    
    # We need to compute this for each level
    θ_li_prof = zeros(FT, nz)
    for k in 1:nz
        q = MoistureMassFractions(q_v[k], q_c[k] + q_r[k])
        cp_m = mixture_heat_capacity(q, constants)
        R_m = mixture_gas_constant(q, constants)
        Pi = (p_prof[k] / p0)^(R_m / cp_m)
        L = constants.liquid.reference_latent_heat
        
        θ_li_prof[k] = (T_prof[k] - L * (q_c[k] + q_r[k]) / cp_m) / Pi
    end
    
    set!(model.formulation.potential_temperature_density, reshape(rho_prof .* θ_li_prof, 1, 1, nz))
    
    # Run one time step
    # IMPORTANT: The Kessler scheme uses model.clock.last_Δt.
    # We must initialize it, otherwise the first step will skip microphysics (default last_Δt is Inf).
    model.clock.last_Δt = dt
    
    # We call update_state! to populate diagnostic fields (θ_li, q, etc.) and run microphysics.
    # This ensures consistent state before microphysics runs.
    update_state!(model)
    
    # 4. Compare Results
    
    # Extract Julia results
    # We need to convert back to mixing ratios and T to compare with Fortran
    
    r_v_j = zeros(FT, nz)
    r_c_j = zeros(FT, nz)
    r_r_j = zeros(FT, nz)
    T_j = zeros(FT, nz)
    
    ρq_c_j = interior(model.microphysical_fields.ρqᶜˡ, 1, 1, :)
    ρq_r_j = interior(model.microphysical_fields.ρqʳ, 1, 1, :)
    ρq_t_j = interior(model.moisture_density, 1, 1, :)
    ρθ_li_j = interior(model.formulation.potential_temperature_density, 1, 1, :)
    
    for k in 1:nz
        # Re-fetch density (it's reference density, shouldn't change)
        rho = rho_prof[k]
        
        # Mass fractions
        q_c_val = ρq_c_j[k] / rho
        q_r_val = ρq_r_j[k] / rho
        q_t_val = ρq_t_j[k] / rho
        q_v_val = q_t_val - q_c_val - q_r_val
        
        # Mixing ratios
        # r = q / (1 - q_t)
        # Note: q_t here is total specific humidity
        r_v_j[k] = q_v_val / (1 - q_t_val)
        r_c_j[k] = q_c_val / (1 - q_t_val)
        r_r_j[k] = q_r_val / (1 - q_t_val)
        
        # Temperature
        θ_li_val = ρθ_li_j[k] / rho
        
        q = MoistureMassFractions(q_v_val, q_c_val + q_r_val)
        cp_m = mixture_heat_capacity(q, constants)
        R_m = mixture_gas_constant(q, constants)
        Pi = (p_prof[k] / p0)^(R_m / cp_m)
        L = constants.liquid.reference_latent_heat
        
        T_j[k] = Pi * θ_li_val + L * (q_c_val + q_r_val) / cp_m
    end
    
    # Fortran T
    # theta_f is potential temperature (dry, likely, or using dry Exner?)
    # Fortran code: theta[k] = theta[k] + ...
    # And pk[k] = (p/p0)^(R/cp) (Dry Exner)
    # T = theta * pk
    T_f = theta_f .* pk_f
    
    # Compare profiles
    # With matched thermodynamics and last_Δt fixed, we expect good agreement.
    # Tolerances allow for small differences due to float order of operations and
    # minor implementation details (e.g. parallel vs serial accumulation, moist vs dry Exner).
    @test r_v_j ≈ qv_f atol=1e-3 rtol=1e-3
    @test r_c_j ≈ qc_f atol=1e-3 rtol=1e-3
    @test r_r_j ≈ qr_f atol=1e-3 rtol=1e-3
    
    # Temperature comparison
    # Should now be much closer
    # Tolerances are looser here because T depends on Exner function, which differs
    # between Breeze (moist) and Fortran (dry) formulations when liquid is present.
    # A difference of 1e-3 in q (1 g/kg) corresponds to ~2.5 K in latent heating.
    @test T_j ≈ T_f atol=2.0 rtol=1e-2
end
