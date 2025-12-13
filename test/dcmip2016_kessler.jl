using Test
using Breeze
using Breeze.Microphysics
using Breeze.Thermodynamics
using Oceananigans
using Oceananigans.Grids
using KernelAbstractions

# Import internal kernel for testing
using Breeze.Microphysics: _kessler_microphysical_update!

# --- Fortran Logic (Translated to Julia) ---
# This is a direct translation of the KESSLER subroutine from dcmip2016_kessler_physic.f90
function kessler_fortran(theta, qv, qc, qr, rho, pk, dt, z, nz)
    # Constants
    f2x = 17.27
    f5 = 237.3 * f2x * 2500000.0 / 1003.0
    xk = 0.2875
    psl = 1000.0
    rhoqr = 1000.0

    r = zeros(nz)
    rhalf = zeros(nz)
    pc = zeros(nz)
    velqr = zeros(nz)
    sed = zeros(nz)

    # Initialization
    for k in 1:nz
        r[k] = 0.001 * rho[k]
        rhalf[k] = sqrt(rho[1] / rho[k])
        pc[k] = 3.8 / (pk[k]^(1.0/xk) * psl)
        velqr[k] = 36.34 * (qr[k] * r[k])^0.1364 * rhalf[k]
    end

    # CFL
    dt_max = dt
    for k in 1:nz-1
        if velqr[k] != 0.0
            dt_max = min(dt_max, 0.8 * (z[k+1] - z[k]) / velqr[k])
        end
    end

    rainsplit = ceil(Int, dt / dt_max)
    dt0 = dt / rainsplit
    precl = 0.0

    # Subcycling
    for nt in 1:rainsplit
        precl = precl + rho[1] * qr[1] * velqr[1] / rhoqr

        # Sedimentation
        for k in 1:nz-1
            sed[k] = dt0 * (r[k+1]*qr[k+1]*velqr[k+1] - r[k]*qr[k]*velqr[k]) / (r[k]*(z[k+1]-z[k]))
        end
        sed[nz] = -dt0 * qr[nz] * velqr[nz] / (0.5 * (z[nz] - z[nz-1]))

        # Adjustment
        for k in 1:nz
            # Autoconversion
            qrprod = qc[k] - (qc[k] - dt0 * max(0.001*(qc[k]-0.001), 0.0)) / (1.0 + dt0 * 2.2 * qr[k]^0.875)
            qc[k] = max(qc[k] - qrprod, 0.0)
            qr[k] = max(qr[k] + qrprod + sed[k], 0.0)

            # Saturation
            qvs = pc[k] * exp(f2x * (pk[k]*theta[k] - 273.0) / (pk[k]*theta[k] - 36.0))
            prod = (qv[k] - qvs) / (1.0 + qvs * f5 / (pk[k]*theta[k] - 36.0)^2)

            # Evaporation
            term1 = (1.6 + 124.9 * (r[k]*qr[k])^0.2046) * (r[k]*qr[k])^0.525
            term2 = 2550000.0 * pc[k] / (3.8 * qvs) + 540000.0
            term3 = max(qvs - qv[k], 0.0) / (r[k] * qvs)
            ern = min(dt0 * (term1 / term2) * term3, max(-prod - qc[k], 0.0), qr[k])

            # Update
            theta[k] = theta[k] + 2500000.0 / (1003.0 * pk[k]) * (max(prod, -qc[k]) - ern)
            qv[k] = max(qv[k] - max(prod, -qc[k]) + ern, 0.0)
            qc[k] = qc[k] + max(prod, -qc[k])
            qr[k] = qr[k] - ern
        end

        if nt != rainsplit
            for k in 1:nz
                velqr[k] = 36.34 * (qr[k] * r[k])^0.1364 * rhalf[k]
            end
        end
    end

    precl = precl / rainsplit
    return theta, qv, qc, qr, precl
end

@testset "DCMIP2016 Kessler Microphysics" begin
    # Setup grid
    Nz = 10
    Lz = 10000.0
    grid = RectilinearGrid(size = (1, 1, Nz), x = (0, 100), y = (0, 100), z = (0, Lz), topology = (Periodic, Periodic, Bounded))
    
    # Vertical coordinates
    z = [znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]
    
    # Atmospheric profile
    ρ = [1.2 * exp(-h/8000) for h in z]
    p = [100000.0 * exp(-h/8000) for h in z]
    p₀ = 100000.0
    
    # Initial conditions (Mixing Ratios for Fortran)
    qv_f = 0.01 * ones(Nz)
    qc_f = 0.001 * ones(Nz)
    qr_f = 0.0005 * ones(Nz)
    theta_f = 300.0 * ones(Nz)
    pk_f = (p ./ 100000.0).^0.2875 # Match Kessler xk
    
    # Initial conditions (Mass Fractions for Breeze)
    # Convert mixing ratios to mass fractions
    qt_b = zeros(Nz)
    qcl_b = zeros(Nz)
    qr_b = zeros(Nz)
    theta_li_b = zeros(Nz)
    
    # Constants for Breeze thermodynamics
    constants = ThermodynamicConstants()
    
    for k in 1:Nz
        rt = qv_f[k] + qc_f[k] + qr_f[k]
        qv_val = qv_f[k] / (1 + rt)
        qcl_val = qc_f[k] / (1 + rt)
        qr_val = qr_f[k] / (1 + rt)
        
        qt_b[k] = qv_val + qcl_val + qr_val
        qcl_b[k] = qcl_val
        qr_b[k] = qr_val
        
        # T = theta * pk
        T = theta_f[k] * pk_f[k]
        
        # Calculate θˡⁱ using Breeze thermodynamics to ensure T matches exactly
        ql = qcl_val + qr_val
        q_breeze = MoistureMassFractions(qv_val, ql)
        cpm_breeze = mixture_heat_capacity(q_breeze, constants)
        Rm_breeze = mixture_gas_constant(q_breeze, constants)
        Pi_breeze = (p[k] / p₀)^(Rm_breeze / cpm_breeze)
        L_breeze = constants.liquid.reference_latent_heat
        
        # θˡⁱ = (T - L*ql/cp) / Π
        theta_li_b[k] = (T - L_breeze * ql / cpm_breeze) / Pi_breeze
    end
    
    # Create Breeze fields
    ρ_field = CenterField(grid)
    p_field = CenterField(grid)
    θˡⁱ_field = CenterField(grid)
    ρθˡⁱ_field = CenterField(grid)
    ρqᵗ_field = CenterField(grid)
    ρqᶜˡ_field = CenterField(grid)
    ρqʳ_field = CenterField(grid)
    
    qᵛ_field = CenterField(grid)
    qᶜˡ_field = CenterField(grid)
    qʳ_field = CenterField(grid)
    precipitation_rate = Field{Center, Center, Nothing}(grid)
    vᵗ_rain = CenterField(grid)
    
    # Fill fields
    set!(ρ_field, reshape(ρ, 1, 1, Nz))
    set!(p_field, reshape(p, 1, 1, Nz))
    set!(θˡⁱ_field, reshape(theta_li_b, 1, 1, Nz))
    set!(ρθˡⁱ_field, reshape(ρ .* theta_li_b, 1, 1, Nz))
    set!(ρqᵗ_field, reshape(ρ .* qt_b, 1, 1, Nz))
    set!(ρqᶜˡ_field, reshape(ρ .* qcl_b, 1, 1, Nz))
    set!(ρqʳ_field, reshape(ρ .* qr_b, 1, 1, Nz))
    
    # Constants
    constants = ThermodynamicConstants()
    
    # Run Breeze Kernel
    dt = 10.0
    
    # We need to call the kernel manually
    # Note: The kernel expects reduced 1D arrays for ρᵣ and pᵣ if they are reference profiles,
    # but here we passed full 3D fields. The kernel signature in dcmip2016_kessler.jl uses
    # ρᵣ[k] which implies 1D access if it's a 1D array, or we need to be careful.
    # In the actual code: `ρᵣ = interior(model.formulation.reference_state.density, 1, 1, :)`
    # So we should pass 1D arrays for ρᵣ and pᵣ.
    
    ρ_1d = ρ
    p_1d = p
    
    # Launch kernel
    # We use a CPU kernel launch for testing
    workgroup = (1, 1)
    ndrange = (1, 1)
    
    kernel! = _kessler_microphysical_update!(KernelAbstractions.CPU(), workgroup)
    event = kernel!(grid, Nz, dt, ρ_1d, p_1d, p₀, constants, 
                    θˡⁱ_field, ρθˡⁱ_field,
                    ρqᵗ_field, ρqᶜˡ_field, ρqʳ_field,
                    qᵛ_field, qᶜˡ_field, qʳ_field,
                    precipitation_rate, vᵗ_rain,
                    ndrange=ndrange)
    if !isnothing(event)
        wait(event)
    end
    
    # Run Fortran Reference
    theta_out_f, qv_out_f, qc_out_f, qr_out_f, precl_f = kessler_fortran(
        copy(theta_f), copy(qv_f), copy(qc_f), copy(qr_f), 
        ρ, pk_f, dt, z, Nz
    )
    
    # Compare Results
    # We need to convert Breeze outputs back to mixing ratios and potential temperature for comparison
    
    # Extract Breeze results
    ρqᶜˡ_out = interior(ρqᶜˡ_field, 1, 1, :)
    ρqʳ_out = interior(ρqʳ_field, 1, 1, :)
    ρqᵗ_out = interior(ρqᵗ_field, 1, 1, :)
    θˡⁱ_out = interior(θˡⁱ_field, 1, 1, :)
    precip_b = interior(precipitation_rate, 1, 1, 1)[1]
    
    @test precip_b ≈ precl_f atol=1e-12
    
    for k in 1:Nz
        # Convert Breeze back to mixing ratios
        qcl_out = ρqᶜˡ_out[k] / ρ[k]
        qr_out = ρqʳ_out[k] / ρ[k]
        qt_out = ρqᵗ_out[k] / ρ[k]
        qv_out = qt_out - qcl_out - qr_out
        
        r_cl = qcl_out / (1 - qt_out)
        r_r = qr_out / (1 - qt_out)
        r_v = qv_out / (1 - qt_out)
        
        # Reconstruct Temperature and Theta using Breeze thermodynamics
        ql = qcl_out + qr_out
        q_breeze = MoistureMassFractions(qv_out, ql)
        cpm_breeze = mixture_heat_capacity(q_breeze, constants)
        Rm_breeze = mixture_gas_constant(q_breeze, constants)
        Pi_breeze = (p[k] / p₀)^(Rm_breeze / cpm_breeze)
        L_breeze = constants.liquid.reference_latent_heat
        
        # T = Π * θˡⁱ + L*ql/cp
        T_actual = Pi_breeze * θˡⁱ_out[k] + L_breeze * ql / cpm_breeze
        
        # Convert T_actual to theta using Fortran definition for comparison
        theta_rec = T_actual / pk_f[k]
        
        @test theta_rec ≈ theta_out_f[k] atol=1e-10
        @test r_v ≈ qv_out_f[k] atol=1e-10
        @test r_cl ≈ qc_out_f[k] atol=1e-10
        @test r_r ≈ qr_out_f[k] atol=1e-10
    end
end
