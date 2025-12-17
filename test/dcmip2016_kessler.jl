using Breeze
using Oceananigans
using Test

using Breeze.Microphysics: _kessler_microphysical_update!
using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity, mixture_gas_constant
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: CPU

# Fallback for standalone execution (normally set by runtests.jl)
if !@isdefined(default_arch)
    const default_arch = CPU()
end

# Reference implementation of the KESSLER subroutine from the DCMIP2016 Fortran implementation
# Reference: ClimateGlobalChange/DCMIP2016: v1.0. https://doi.org/10.5281/zenodo.1298671
function kessler_reference(theta, qv, qc, qr, rho, pk, dt, z, nz)
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

@testset "DCMIP2016KesslerMicrophysics [$(typeof(default_arch))]" begin
    Nz = 10
    Lz = 10000.0
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 100), y=(0, 100), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))

    z = [znode(1, 1, k, grid, Center(), Center(), Center()) for k in 1:Nz]
    
    # Atmospheric profile
    ρ = [1.2 * exp(-h/8000) for h in z]
    p = [100000.0 * exp(-h/8000) for h in z]
    p₀ = 100000.0

    # Initial conditions (mixing ratios for reference implementation)
    qv_ref = 0.01
    qc_ref = 0.001
    qr_ref = 0.0005
    θ_ref = 300
    κᵈ_Kessler = 0.2875
    pk = (p ./ p₀) .^ κᵈ_Kessler  # Exner function (Kessler xk)

    # Convert mixing ratios to mass fractions for Breeze
    qt_b = zeros(Nz)
    qcl_b = zeros(Nz)
    qr_b = zeros(Nz)
    θˡⁱ_b = zeros(Nz)

    constants = ThermodynamicConstants()

    for k in 1:Nz
        rt = qv_ref[k] + qc_ref[k] + qr_ref[k]
        qv_val = qv_ref[k] / (1 + rt)
        qcl_val = qc_ref[k] / (1 + rt)
        qr_val = qr_ref[k] / (1 + rt)

        qt_b[k] = qv_val + qcl_val + qr_val
        qcl_b[k] = qcl_val
        qr_b[k] = qr_val

        # T = θ × Π
        T = θ_ref[k] * pk[k]

        # Calculate θˡⁱ using Breeze thermodynamics
        ql = qcl_val + qr_val
        q = MoistureMassFractions(qv_val, ql)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Rᵐ = mixture_gas_constant(q, constants)
        Π = (p[k] / p₀)^(Rᵐ / cᵖᵐ)
        ℒˡᵣ = constants.liquid.reference_latent_heat

        θˡⁱ_b[k] = (T - ℒˡᵣ * ql / cᵖᵐ) / Π
    end
    
    # Create fields
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

    set!(θˡⁱ_field, reshape(θˡⁱ_b, 1, 1, Nz))
    set!(ρθˡⁱ_field, reshape(ρ .* θˡⁱ_b, 1, 1, Nz))
    set!(ρqᵗ_field, reshape(ρ .* qt_b, 1, 1, Nz))
    set!(ρqᶜˡ_field, reshape(ρ .* qcl_b, 1, 1, Nz))
    set!(ρqʳ_field, reshape(ρ .* qr_b, 1, 1, Nz))

    # Run Breeze kernel
    dt = 10.0
    ρ_1d = ρ
    p_1d = p
    precipitation_rate_data = interior(precipitation_rate, :, :, 1)

    launch!(default_arch, grid, :xy, _kessler_microphysical_update!,
            grid, Nz, dt, ρ_1d, p_1d, p₀, constants,
            θˡⁱ_field, ρθˡⁱ_field,
            ρqᵗ_field, ρqᶜˡ_field, ρqʳ_field,
            qᵛ_field, qᶜˡ_field, qʳ_field,
            precipitation_rate_data, vᵗ_rain)

    # Run reference implementation
    θ_out, qv_out, qc_out, qr_out, precl_ref = kessler_reference(
        copy(θ_ref), copy(qv_ref), copy(qc_ref), copy(qr_ref),
        ρ, pk, dt, z, Nz)

    # Extract Breeze results
    ρqᶜˡ_out = interior(ρqᶜˡ_field, 1, 1, :)
    ρqʳ_out = interior(ρqʳ_field, 1, 1, :)
    ρqᵗ_out = interior(ρqᵗ_field, 1, 1, :)
    θˡⁱ_out = interior(θˡⁱ_field, 1, 1, :)
    precip_b = precipitation_rate_data[1, 1]

    # Tolerances reflect thermodynamic differences between Breeze (moist Exner function,
    # mixture heat capacities) and the reference Kessler (dry approximations)

    @testset "Precipitation rate" begin
        @test precip_b >= 0
        @test precip_b ≈ precl_ref rtol=0.5
    end

    @testset "Moisture and temperature evolution" begin
        for k in 1:Nz
            qcl_k = ρqᶜˡ_out[k] / ρ[k]
            qr_k = ρqʳ_out[k] / ρ[k]
            qt_k = ρqᵗ_out[k] / ρ[k]
            qv_k = qt_k - qcl_k - qr_k

            # Convert mass fractions back to mixing ratios
            r_cl = qcl_k / (1 - qt_k)
            r_r = qr_k / (1 - qt_k)
            r_v = qv_k / (1 - qt_k)

            # Reconstruct temperature using Breeze thermodynamics
            ql = qcl_k + qr_k
            q = MoistureMassFractions(qv_k, ql)
            cᵖᵐ = mixture_heat_capacity(q, constants)
            Rᵐ = mixture_gas_constant(q, constants)
            Π = (p[k] / p₀)^(Rᵐ / cᵖᵐ)
            ℒˡᵣ = constants.liquid.reference_latent_heat

            T_k = Π * θˡⁱ_out[k] + ℒˡᵣ * ql / cᵖᵐ
            θ_k = T_k / pk[k]

            @test θ_k ≈ θ_out[k] rtol=0.01
            @test r_v ≈ qv_out[k] rtol=0.1
            @test r_cl ≈ qc_out[k] rtol=0.1
            @test r_r ≈ qr_out[k] rtol=0.1
        end
    end

    @testset "Physical constraints" begin
        for k in 1:Nz
            qcl_k = ρqᶜˡ_out[k] / ρ[k]
            qr_k = ρqʳ_out[k] / ρ[k]
            qt_k = ρqᵗ_out[k] / ρ[k]
            qv_k = qt_k - qcl_k - qr_k

            @test qcl_k >= -1e-15
            @test qr_k >= -1e-15
            @test qv_k >= -1e-15
            @test qt_k >= -1e-15
            @test qt_k ≈ qv_k + qcl_k + qr_k atol=1e-14
        end
    end
end
