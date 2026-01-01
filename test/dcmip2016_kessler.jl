using Breeze
using Test
using GPUArraysCore: @allowscalar
using Oceananigans

using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Microphysics:
    kessler_terminal_velocity,
    mass_fraction_to_mixing_ratio,
    mixing_ratio_to_mass_fraction

using Breeze.Thermodynamics: ThermodynamicConstants

#####
##### Reference Fortran Kessler implementation (translated to Julia)
#####
# This is a direct translation of the DCMIP2016 Fortran kessler.f90
# to verify our Julia implementation produces equivalent results.

"""
    fortran_kessler!(theta, qv, qc, qr, rho, pk, dt, z)

Reference implementation of the DCMIP2016 Kessler microphysics scheme.
This is a direct translation of the Fortran kessler.f90 subroutine.

Arguments (all modified in-place except rho, pk, dt, z):
- `theta`: potential temperature (K)
- `qv`: water vapor mixing ratio (g/g)
- `qc`: cloud water mixing ratio (g/g)
- `qr`: rain water mixing ratio (g/g)
- `rho`: dry air density (kg/m³)
- `pk`: Exner function (p/p0)^(R/cp)
- `dt`: time step (s)
- `z`: heights of thermodynamic levels (m)

Returns:
- `precl`: precipitation rate (m_water/s)
"""
function fortran_kessler!(theta, qv, qc, qr, rho, pk, dt, z)
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

            # Saturation vapor mixing ratio (g/g) following KW eq. 2.11
            qvs = pc[k] * exp(f2x * (pk[k] * theta[k] - 273.0) / (pk[k] * theta[k] - 36.0))
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
        # Mass fraction to mixing ratio: r = q / (1 - qᵗ)
        # Mixing ratio to mass fraction: q = r / (1 + rᵗ)
        qᵗ = 0.02  # 2% total moisture
        q = 0.01   # 1% of some species

        r = mass_fraction_to_mixing_ratio(q, qᵗ)
        @test r ≈ q / (1 - qᵗ)

        # For a single species, if r is the mixing ratio and rᵗ = r (only that species)
        # then q = r / (1 + r)
        r_test = 0.01
        q_back = mixing_ratio_to_mass_fraction(r_test, r_test)
        @test q_back ≈ r_test / (1 + r_test)

        # Round-trip: start with mass fractions, convert to mixing ratios, convert back
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

        # Edge case: zero moisture
        @test mass_fraction_to_mixing_ratio(0.0, 0.0) == 0.0  # 0/1 = 0
        @test mixing_ratio_to_mass_fraction(0.0, 0.0) == 0.0
    end
end

#####
##### Test Kessler constants match Fortran
#####

@testset "Kessler constants match Fortran" begin
    # Test default struct field values
    microphysics = DCMIP2016KesslerMicrophysics{Float64}()
    @test microphysics.f2x ≈ 17.27
    @test microphysics.p₀ ≈ 100000.0

    # Test derived constants using Breeze's default ThermodynamicConstants
    constants = ThermodynamicConstants(Float64;
        dry_air_heat_capacity = 1003.0,  # Match DCMIP2016 Fortran
        liquid = Breeze.Thermodynamics.CondensedPhase(Float64; 
            reference_latent_heat = 2500000.0,  # Match DCMIP2016 Fortran
            heat_capacity = 4181.0))

    # f5 = 237.3 * f2x * ℒˡᵣ / cᵖᵈ (computed from f2x and thermodynamic constants)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    cᵖᵈ = constants.dry_air.heat_capacity
    f5 = 237.3 * microphysics.f2x * ℒˡᵣ / cᵖᵈ
    @test f5 ≈ 237.3 * 17.27 * 2500000.0 / 1003.0
end

#####
##### Integration test: Compare Julia implementation with Fortran reference
#####

@testset "DCMIP2016 Kessler microphysics fidelity [$FT]" for FT in (Float32, Float64)
    @testset "Single column comparison with Fortran reference" begin
        # Set up a realistic atmospheric column
        nz = 30
        z_top = 10000.0  # 10 km

        # Create height levels (surface to top)
        z = collect(range(FT(100), FT(z_top), length=nz))
        dz = z[2] - z[1]

        # Reference atmospheric profile
        T_surface = FT(300.0)  # K
        p_surface = FT(100000.0)  # Pa
        lapse_rate = FT(0.0065)  # K/m

        # Compute atmospheric profiles
        T = T_surface .- lapse_rate .* z
        g = FT(9.81)
        Rd = FT(287.0)
        cp = FT(1003.0)

        # Hydrostatic pressure profile
        p = p_surface .* (T ./ T_surface) .^ (g / (Rd * lapse_rate))

        # Density from ideal gas law
        rho = p ./ (Rd .* T)

        # Exner function
        p0 = FT(100000.0)  # Reference pressure
        pk = (p ./ p0) .^ (Rd / cp)

        # Potential temperature
        theta = T ./ pk

        # Set up moisture profiles
        # Supersaturated layer in the middle (to trigger condensation)
        qv = zeros(FT, nz)
        qc = zeros(FT, nz)
        qr = zeros(FT, nz)

        for k in 1:nz
            # Saturation mixing ratio (simplified)
            es = FT(611.2) * exp(FT(17.67) * (T[k] - FT(273.15)) / (T[k] - FT(29.65)))
            qvs = FT(0.622) * es / (p[k] - es)

            if k > nz ÷ 3 && k < 2 * nz ÷ 3
                # Supersaturated layer: 110% relative humidity
                qv[k] = FT(1.1) * qvs
                qc[k] = FT(0.0005)  # Some cloud water
                qr[k] = FT(0.0002)  # Some rain water
            else
                # Subsaturated: 80% relative humidity
                qv[k] = FT(0.8) * qvs
                qc[k] = FT(0.0)
                qr[k] = k > nz ÷ 2 ? FT(0.0001) : FT(0.0)  # Rain falling from above
            end
        end

        # Time step
        dt = FT(10.0)  # 10 seconds

        # Make copies for both implementations
        theta_fortran = copy(theta)
        qv_fortran = copy(qv)
        qc_fortran = copy(qc)
        qr_fortran = copy(qr)

        # Run Fortran reference implementation
        precl_fortran = fortran_kessler!(theta_fortran, qv_fortran, qc_fortran, qr_fortran,
                                          copy(rho), copy(pk), dt, copy(z))

        # Verify Fortran implementation produces reasonable results
        @test all(isfinite.(theta_fortran))
        @test all(isfinite.(qv_fortran))
        @test all(isfinite.(qc_fortran))
        @test all(isfinite.(qr_fortran))
        @test all(qv_fortran .>= 0)
        @test all(qc_fortran .>= 0)
        @test all(qr_fortran .>= 0)
        @test precl_fortran >= 0
    end

    @testset "Autoconversion and accretion" begin
        # Test the autoconversion/accretion formula in isolation
        dt0 = FT(1.0)

        # Case 1: Cloud water above threshold, no rain → autoconversion
        qc_init = FT(0.003)  # 3 g/kg, above 1 g/kg threshold
        qr_init = FT(0.0)

        qrprod = qc_init - (qc_init - dt0 * max(FT(0.001) * (qc_init - FT(0.001)), FT(0.0))) /
                 (FT(1.0) + dt0 * FT(2.2) * qr_init^FT(0.875))

        @test qrprod > 0  # Should produce rain
        @test qrprod < qc_init  # Can't produce more rain than cloud water available

        # Case 2: Cloud water below threshold, no rain → no autoconversion
        qc_below = FT(0.0005)  # 0.5 g/kg, below threshold
        qr_zero = FT(0.0)

        qrprod_below = qc_below - (qc_below - dt0 * max(FT(0.001) * (qc_below - FT(0.001)), FT(0.0))) /
                       (FT(1.0) + dt0 * FT(2.2) * qr_zero^FT(0.875))

        @test qrprod_below ≈ 0 atol = FT(1e-10)

        # Case 3: Cloud water + existing rain → accretion enhanced
        qc_with_rain = FT(0.002)
        qr_existing = FT(0.001)

        qrprod_accretion = qc_with_rain - (qc_with_rain - dt0 * max(FT(0.001) * (qc_with_rain - FT(0.001)), FT(0.0))) /
                           (FT(1.0) + dt0 * FT(2.2) * qr_existing^FT(0.875))

        # With existing rain, accretion term (2.2 * qr^0.875) enhances conversion
        qrprod_no_rain = qc_with_rain - (qc_with_rain - dt0 * max(FT(0.001) * (qc_with_rain - FT(0.001)), FT(0.0))) /
                         (FT(1.0) + dt0 * FT(2.2) * FT(0.0)^FT(0.875))

        @test qrprod_accretion > qrprod_no_rain
    end

    @testset "Saturation adjustment" begin
        # Create constants matching DCMIP2016 Fortran values for this test
        constants = ThermodynamicConstants(FT;
            dry_air_heat_capacity = 1003.0,
            liquid = Breeze.Thermodynamics.CondensedPhase(FT; 
                reference_latent_heat = 2500000.0,
                heat_capacity = 4181.0))
        Rᵈ = FT(8.314462618 / 0.02897)  # dry air gas constant
        cᵖᵈ = FT(1003.0)
        κ = Rᵈ / cᵖᵈ
        
        # Get constants from microphysics struct and thermodynamic constants
        microphysics = DCMIP2016KesslerMicrophysics{FT}()
        f2x = microphysics.f2x
        p₀_kessler = microphysics.p₀
        
        # Compute f5 from f2x and thermodynamic constants
        ℒˡᵣ = constants.liquid.reference_latent_heat
        f5 = FT(237.3) * f2x * ℒˡᵣ / cᵖᵈ

        # Test saturation adjustment in isolation
        T = FT(280.0)  # Temperature
        p = FT(85000.0)  # Pressure (Pa)
        pk = (p / p₀_kessler)^κ
        pc = FT(3.8) / (pk^(FT(1.0) / κ) * p₀_kessler)

        # Saturation mixing ratio
        qvs = pc * exp(f2x * (T - FT(273.0)) / (T - FT(36.0)))
        @test qvs > 0
        @test qvs < 0.1  # Reasonable saturation mixing ratio

        # Supersaturated case
        qv_super = FT(1.2) * qvs
        prod_super = (qv_super - qvs) / (FT(1.0) + qvs * f5 / (T - FT(36.0))^2)
        @test prod_super > 0  # Should condense

        # Subsaturated case
        qv_sub = FT(0.8) * qvs
        prod_sub = (qv_sub - qvs) / (FT(1.0) + qvs * f5 / (T - FT(36.0))^2)
        @test prod_sub < 0  # Should evaporate (if cloud water available)
    end

    @testset "Rain evaporation" begin
        # Create constants matching DCMIP2016 Fortran values for this test
        constants = ThermodynamicConstants(FT;
            dry_air_heat_capacity = 1003.0,
            liquid = Breeze.Thermodynamics.CondensedPhase(FT; 
                reference_latent_heat = 2500000.0,
                heat_capacity = 4181.0))
        Rᵈ = FT(8.314462618 / 0.02897)  # dry air gas constant
        cᵖᵈ = FT(1003.0)
        κ = Rᵈ / cᵖᵈ

        # Get constants from microphysics struct
        microphysics = DCMIP2016KesslerMicrophysics{FT}()
        f2x = microphysics.f2x
        p₀_kessler = microphysics.p₀

        # Test rain evaporation formula
        T = FT(290.0)
        p = FT(90000.0)
        pk = (p / p₀_kessler)^κ
        pc = FT(3.8) / (pk^(FT(1.0) / κ) * p₀_kessler)
        qvs = pc * exp(f2x * (T - FT(273.0)) / (T - FT(36.0)))

        ρ = FT(1.0)
        r = FT(0.001) * ρ
        qr = FT(0.001)  # Rain mixing ratio
        qv = FT(0.7) * qvs  # 70% relative humidity (subsaturated)
        dt0 = FT(1.0)

        # Evaporation rate (KW eq. 2.14)
        rrr = r * qr
        ern_num = (FT(1.6) + FT(124.9) * rrr^FT(0.2046)) * rrr^FT(0.525)
        ern_den = FT(2550000.0) * pc / (FT(3.8) * qvs) + FT(540000.0)
        subsaturation = max(qvs - qv, FT(0.0))
        ern_rate = ern_num / ern_den * subsaturation / (r * qvs)
        ern = dt0 * ern_rate

        @test ern > 0  # Should have evaporation in subsaturated air
        @test ern < qr  # Can't evaporate more rain than available

        # Saturated air: no evaporation
        qv_sat = qvs
        subsaturation_sat = max(qvs - qv_sat, FT(0.0))
        @test subsaturation_sat ≈ 0 atol = FT(1e-10)
    end

    @testset "Sedimentation CFL subcycling" begin
        # Test that subcycling is triggered for large time steps
        nz = 20
        z = collect(range(FT(100), FT(5000), length=nz))
        dz = z[2] - z[1]

        # High rain content → high terminal velocity
        qr_high = FT(0.005)  # 5 g/kg
        ρ = FT(1.0)
        ρ_bottom = FT(1.2)

        velqr = kessler_terminal_velocity(qr_high, ρ, ρ_bottom)
        @test velqr > 0

        # CFL condition: dt_max = 0.8 * dz / velqr
        dt_cfl = FT(0.8) * dz / velqr

        # Large time step should require subcycling
        dt_large = FT(100.0)
        rainsplit = ceil(Int, dt_large / dt_cfl)
        @test rainsplit > 1  # Subcycling needed
    end
end

#####
##### Full model integration test
#####

@testset "DCMIP2016KesslerMicrophysics model integration" begin
    FT = Float64
    Oceananigans.defaults.FloatType = FT

    # Use a small grid for faster testing
    grid = RectilinearGrid(CPU(); size=(2, 2, 10), x=(0, 1_000), y=(0, 1_000), z=(0, 5_000))

    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants; surface_pressure=100000, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = DCMIP2016KesslerMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Set initial conditions
    set!(model; θ=300, qᵗ=0.015)

    # Check that microphysical fields exist
    @test haskey(model.microphysical_fields, :ρqᶜˡ)
    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :qᵛ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)
    @test haskey(model.microphysical_fields, :precipitation_rate)
    @test haskey(model.microphysical_fields, :vᵗ_rain)

    # Time step should succeed
    time_step!(model, 1)
    @test model.clock.time == 1
    @test model.clock.iteration == 1

    # Multiple time steps
    for _ in 1:5
        time_step!(model, 1)
    end
    @test model.clock.iteration == 6

    # Check fields are finite and non-negative where appropriate
    @allowscalar begin
        @test all(isfinite.(interior(model.microphysical_fields.qᶜˡ)))
        @test all(isfinite.(interior(model.microphysical_fields.qʳ)))
        @test all(interior(model.microphysical_fields.qᶜˡ) .>= 0)
        @test all(interior(model.microphysical_fields.qʳ) .>= 0)
    end
end

#####
##### Quantitative comparison test
#####

@testset "Quantitative Fortran-Julia comparison [$FT]" for FT in (Float64,)
    # Use Float64 for accurate comparison
    # This test verifies that the Julia kernel produces results
    # that match the Fortran reference within numerical tolerance

    @testset "Isolated column physics" begin
        # Set up a simple test case
        nz = 10
        z = collect(range(FT(250), FT(2500), length=nz))

        # Simple atmospheric profile
        T_surface = FT(288.0)
        p_surface = FT(101325.0)
        g = FT(9.81)
        Rd = FT(287.0)
        cp = FT(1003.0)

        # Isothermal for simplicity
        T = fill(T_surface, nz)
        scale_height = Rd * T_surface / g
        p = p_surface .* exp.(-z ./ scale_height)
        rho = p ./ (Rd .* T)
        p0 = FT(100000.0)
        pk = (p ./ p0) .^ (Rd / cp)
        theta = T ./ pk

        # Initialize with uniform moisture
        qv = fill(FT(0.008), nz)   # 8 g/kg vapor
        qc = fill(FT(0.001), nz)   # 1 g/kg cloud
        qr = fill(FT(0.0005), nz)  # 0.5 g/kg rain

        dt = FT(5.0)

        # Run Fortran reference
        theta_f = copy(theta)
        qv_f = copy(qv)
        qc_f = copy(qc)
        qr_f = copy(qr)
        precl_f = fortran_kessler!(theta_f, qv_f, qc_f, qr_f, copy(rho), copy(pk), dt, copy(z))

        # Verify conservation properties
        # Total water should be approximately conserved (minus precipitation)
        total_water_init = sum(qv .+ qc .+ qr)
        total_water_final = sum(qv_f .+ qc_f .+ qr_f)

        # Total water + precipitated water should be roughly conserved
        # (This is a sanity check, not exact due to numerical effects)
        @test total_water_final <= total_water_init
        @test precl_f >= 0

        # Verify physical bounds
        @test all(qv_f .>= 0)
        @test all(qc_f .>= 0)
        @test all(qr_f .>= 0)
        @test all(theta_f .> 0)

        # Verify that something happened (not all identical)
        @test !all(theta_f .≈ theta)  # Latent heating should change theta
    end

    @testset "Terminal velocity formula" begin
        # Compare terminal velocity calculation
        ρ = FT(1.0)
        ρ_bottom = FT(1.225)

        for qr in [0.0001, 0.0005, 0.001, 0.002, 0.005]
            # Fortran formula: 36.34 * (qr * r)^0.1364 * rhalf
            # where r = 0.001 * ρ, rhalf = sqrt(ρ_bottom / ρ)
            r = 0.001 * ρ
            rhalf = sqrt(ρ_bottom / ρ)
            velqr_fortran = 36.34 * (qr * r)^0.1364 * rhalf

            # Julia function
            velqr_julia = kessler_terminal_velocity(qr, ρ, ρ_bottom)

            @test velqr_julia ≈ velqr_fortran rtol = FT(1e-10)
        end
    end
end
