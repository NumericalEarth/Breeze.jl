using Breeze
using Test
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Breeze.AtmosphereModels: microphysics_model_update!, surface_precipitation_flux
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics, kessler_terminal_velocity, saturation_adjustment_coefficient
using Breeze.Thermodynamics:
    MoistureMassFractions,
    mixture_heat_capacity,
    mixture_gas_constant,
    saturation_specific_humidity,
    PlanarLiquidSurface,
    TetensFormula

#####
##### Helper functions
#####

mass_fraction_to_mixing_ratio(q, qᵗ) = q / (1 - qᵗ)
mixing_ratio_to_mass_fraction(r, rᵗ) = r / (1 + rᵗ)

#####
##### Reference implementation
#####

"""
    dcmip2016_klemp_wilhelmson_kessler!(T, qᵛ, qᶜˡ, qʳ, ρ, p, Δt, z, constants, microphysics)

Direct translation of the DCMIP2016 Kessler microphysics with modifications
to match Breeze's thermodynamic state (liquid-ice potential temperature `θˡⁱ`).

Applies one microphysics time step to column arrays, including subcycling
for rain sedimentation CFL constraints.

References: DCMIP2016 Fortran implementation (`kessler.f90` in [DOI: 10.5281/zenodo.1298671](https://doi.org/10.5281/zenodo.1298671))

"""
function dcmip2016_klemp_wilhelmson_kessler!(T, qᵛ, qᶜˡ, qʳ, ρ, p, Δt, z, constants, microphysics;
                                             sedimentation_density = ρ,
                                             dry_air_coupled = false)
    Nz = length(T)
    FT = eltype(T)

    # Thermodynamic constants
    ℒˡᵣ = constants.liquid.reference_latent_heat
    cᵖᵈ = constants.dry_air.heat_capacity

    # Saturation adjustment parameters
    f₅ = saturation_adjustment_coefficient(microphysics.dcmip_temperature_scale, constants)
    T_offset = constants.saturation_vapor_pressure.liquid_temperature_offset

    # Autoconversion and accretion parameters
    k₁   = microphysics.autoconversion_rate
    rᶜˡ★ = microphysics.autoconversion_threshold
    k₂   = microphysics.accretion_rate
    βᵃᶜᶜ = microphysics.accretion_exponent
    Cᵨ   = microphysics.density_scale

    # Evaporation parameters
    Cᵉᵛ₁   = microphysics.evaporation_ventilation_coefficient_1
    Cᵉᵛ₂   = microphysics.evaporation_ventilation_coefficient_2
    βᵉᵛ₁   = microphysics.evaporation_ventilation_exponent_1
    βᵉᵛ₂   = microphysics.evaporation_ventilation_exponent_2
    Cᵈⁱᶠᶠ  = microphysics.diffusivity_coefficient
    Cᵗʰᵉʳᵐ = microphysics.thermal_conductivity_coefficient

    cfl = microphysics.substep_cfl
    p₀ = 100000.0

    # Initialize θˡⁱ from T
    θˡⁱ = zeros(FT, Nz)
    for k = 1:Nz
        qˡ = qᶜˡ[k] + qʳ[k]
        q = MoistureMassFractions(qᵛ[k], qˡ)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Rᵐ = mixture_gas_constant(q, constants)
        Π = (p[k] / p₀)^(Rᵐ / cᵖᵐ)
        θˡⁱ[k] = (T[k] - ℒˡᵣ * qˡ / cᵖᵐ) / Π
    end

    # Convert mass fractions to mixing ratios and compute terminal velocities
    rᵛ = zeros(FT, Nz)
    rᶜˡ = zeros(FT, Nz)
    rʳ = zeros(FT, Nz)
    𝕎ʳ = zeros(FT, Nz)

    ρ₁ = ρ[1]
    max_Δt = Δt

    for k = 1:Nz
        qᵗ = qᵛ[k] + qᶜˡ[k] + qʳ[k]
        rᵛ[k] = qᵛ[k] / (1 - qᵗ)
        rᶜˡ[k] = qᶜˡ[k] / (1 - qᵗ)
        rʳ[k] = qʳ[k] / (1 - qᵗ)
        𝕎ʳ[k] = kessler_terminal_velocity(rʳ[k], ρ[k], ρ₁, microphysics)

        if k < Nz && 𝕎ʳ[k] > 0
            Δz = z[k+1] - z[k]
            max_Δt = min(max_Δt, cfl * Δz / 𝕎ʳ[k])
        end
    end

    # Subcycling
    Ns = max(1, ceil(Int, Δt / max_Δt))
    Δtₛ = Δt / Ns
    surface_mass_flux = zero(FT)

    for s = 1:Ns
        rᵗ₁ = rᵛ[1] + rᶜˡ[1] + rʳ[1]
        qʳ₁ = rʳ[1] / (1 + rᵗ₁)
        ρqʳ₁ = ifelse(dry_air_coupled,
                      sedimentation_density[1] * rʳ[1],
                      ρ[1] * qʳ₁)
        surface_mass_flux += ρqʳ₁ * 𝕎ʳ[1]

        zᵏ = z[1]

        for k = 1:Nz
            # Recover T from θˡⁱ
            rᵗ = rᵛ[k] + rᶜˡ[k] + rʳ[k]
            qᵛ_local = rᵛ[k] / (1 + rᵗ)
            qˡ_local = (rᶜˡ[k] + rʳ[k]) / (1 + rᵗ)

            q = MoistureMassFractions(qᵛ_local, qˡ_local)
            cᵖᵐ = mixture_heat_capacity(q, constants)
            Rᵐ = mixture_gas_constant(q, constants)
            Π = (p[k] / p₀)^(Rᵐ / cᵖᵐ)
            T[k] = Π * θˡⁱ[k] + ℒˡᵣ * qˡ_local / cᵖᵐ

            # Rain sedimentation (upstream differencing)
            if k < Nz
                zᵏ⁺¹ = z[k+1]
                Δz = zᵏ⁺¹ - zᵏ
                flux_out = sedimentation_density[k+1] * rʳ[k+1] * 𝕎ʳ[k+1]
                flux_in = sedimentation_density[k] * rʳ[k] * 𝕎ʳ[k]
                Δr𝕎 = Δtₛ * (flux_out - flux_in) / (sedimentation_density[k] * Δz)
                zᵏ = zᵏ⁺¹
            else
                Δz_half = 0.5 * (z[k] - z[k-1])
                Δr𝕎 = -Δtₛ * rʳ[k] * 𝕎ʳ[k] / Δz_half
            end

            # Autoconversion and accretion (KW eq. 2.13)
            Aʳ = max(0.0, k₁ * (rᶜˡ[k] - rᶜˡ★))
            denom = 1.0 + Δtₛ * k₂ * rʳ[k]^βᵃᶜᶜ
            Δrᴾ = rᶜˡ[k] - (rᶜˡ[k] - Δtₛ * Aʳ) / denom

            rᶜˡ_new = max(0.0, rᶜˡ[k] - Δrᴾ)
            rʳ_new = max(0.0, rʳ[k] + Δrᴾ + Δr𝕎)

            # Saturation adjustment
            qᵛ⁺ = saturation_specific_humidity(T[k], ρ[k], constants, PlanarLiquidSurface())
            rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)
            Δrˢᵃᵗ = (rᵛ[k] - rᵛ⁺) / (1 + rᵛ⁺ * f₅ / (T[k] - T_offset)^2)

            # Rain evaporation (KW eq. 2.14)
            ρᵏ = ρ[k] * Cᵨ
            ρrʳ = ρᵏ * rʳ_new
            Vᵉᵛ = (Cᵉᵛ₁ + Cᵉᵛ₂ * ρrʳ^βᵉᵛ₁) * ρrʳ^βᵉᵛ₂
            Dᵗʰ = Cᵈⁱᶠᶠ / (p[k] * rᵛ⁺) + Cᵗʰᵉʳᵐ

            Δrᵛ⁺ = max(0.0, rᵛ⁺ - rᵛ[k])
            Ėʳ = Vᵉᵛ / Dᵗʰ * Δrᵛ⁺ / (ρᵏ * rᵛ⁺ + 1e-20)
            Δrᴱmax = max(0.0, -Δrˢᵃᵗ - rᶜˡ_new)
            Δrᴱ = min(min(Δtₛ * Ėʳ, Δrᴱmax), rʳ_new)

            Δrᶜ = max(Δrˢᵃᵗ, -rᶜˡ_new)

            # Update mixing ratios
            rᵛ_new = max(0.0, rᵛ[k] - Δrᶜ + Δrᴱ)
            rᶜˡ_final = rᶜˡ_new + Δrᶜ
            rʳ_final = rʳ_new - Δrᴱ

            # Update θˡⁱ via latent heating
            ΔT = (ℒˡᵣ / cᵖᵈ) * (Δrᶜ - Δrᴱ)
            T_new = T[k] + ΔT

            rᵗ_new = rᵛ_new + rᶜˡ_final + rʳ_final
            qᵛ_new = rᵛ_new / (1 + rᵗ_new)
            qˡ_new = (rᶜˡ_final + rʳ_final) / (1 + rᵗ_new)

            q_new = MoistureMassFractions(qᵛ_new, qˡ_new)
            cᵖᵐ_new = mixture_heat_capacity(q_new, constants)
            Rᵐ_new = mixture_gas_constant(q_new, constants)
            Π_new = (p[k] / p₀)^(Rᵐ_new / cᵖᵐ_new)
            θˡⁱ[k] = (T_new - ℒˡᵣ * qˡ_new / cᵖᵐ_new) / Π_new

            rᵛ[k] = rᵛ_new
            rᶜˡ[k] = rᶜˡ_final
            rʳ[k] = rʳ_final
        end

        # Recalculate terminal velocities for next subcycle
        if s < Ns
            for k = 1:Nz
                𝕎ʳ[k] = kessler_terminal_velocity(rʳ[k], ρ[k], ρ₁, microphysics)
            end
        end
    end

    # Convert back to mass fractions and recover final T
    for k = 1:Nz
        rᵗ = rᵛ[k] + rᶜˡ[k] + rʳ[k]
        qᵛ[k] = rᵛ[k] / (1 + rᵗ)
        qᶜˡ[k] = rᶜˡ[k] / (1 + rᵗ)
        qʳ[k] = rʳ[k] / (1 + rᵗ)

        q = MoistureMassFractions(qᵛ[k], qᶜˡ[k] + qʳ[k])
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Rᵐ = mixture_gas_constant(q, constants)
        Π = (p[k] / p₀)^(Rᵐ / cᵖᵐ)
        T[k] = Π * θˡⁱ[k] + ℒˡᵣ * (qᶜˡ[k] + qʳ[k]) / cᵖᵐ
    end

    return surface_mass_flux / Ns
end

#####
##### Tests for Kessler helper functions
#####

@testset "Kessler helper functions" begin
    @testset "Terminal velocity" begin
        ρ = 1.0
        ρ₁ = 1.2
        rʳ = 0.001
        microphysics = DCMIP2016KesslerMicrophysics()

        𝕎ʳ = kessler_terminal_velocity(rʳ, ρ, ρ₁, microphysics)
        @test 𝕎ʳ > 0
        @test 𝕎ʳ < 20

        𝕎ʳ_zero = kessler_terminal_velocity(0.0, ρ, ρ₁, microphysics)
        @test 𝕎ʳ_zero == 0.0

        𝕎ʳ_high = kessler_terminal_velocity(0.005, ρ, ρ₁, microphysics)
        @test 𝕎ʳ_high > 𝕎ʳ
    end

    @testset "Mass fraction ↔ mixing ratio conversion" begin
        qᵗ = 0.02
        q = 0.01

        r = mass_fraction_to_mixing_ratio(q, qᵗ)
        @test r ≈ q / (1 - qᵗ)

        r_test = 0.01
        q_back = mixing_ratio_to_mass_fraction(r_test, r_test)
        @test q_back ≈ r_test / (1 + r_test)

        # Round-trip conversion
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
    FT = Float64
    Nz = 40

    grid = RectilinearGrid(CPU(),
                           size = (1, 1, Nz),
                           x = (0, 100),
                           y = (0, 100),
                           z = (0, 4000),
                           topology = (Periodic, Periodic, Bounded))

    z_centers = collect(znodes(grid, Center()))

    # Atmospheric profile with linear lapse rate
    T_surface = FT(288.0)
    p_surface = FT(101325.0)
    g = FT(9.81)
    Rᵈ = FT(287.0)
    cᵖᵈ = FT(1003.0)
    lapse_rate = FT(0.0065)

    T_prof = T_surface .- lapse_rate .* z_centers
    p_prof = p_surface .* (T_prof ./ T_surface) .^ (g / (Rᵈ * lapse_rate))
    ρ_prof = p_prof ./ (Rᵈ .* T_prof)

    p₀ = FT(100000.0)

    # Initial moisture profiles (mixing ratios)
    rᵛ_init = zeros(FT, Nz)
    rᶜˡ_init = zeros(FT, Nz)
    rʳ_init = zeros(FT, Nz)

    for k in 1:Nz
        z = z_centers[k]
        rᵛ_init[k] = 0.015 * exp(-((z - 1000) / 1000)^2)
        if 1500 < z < 2500
            rᶜˡ_init[k] = 0.002
        end
        if 1000 < z < 2000
            rʳ_init[k] = 0.0005
        end
    end

    Δt = FT(10.0)

    # Simplified thermodynamic constants matching Fortran
    ℛ = 8.314462618
    Mᵈ = ℛ / 287.0
    cᵖ = 1003.0

    DCMIP2016_tetens_formula = TetensFormula(liquid_temperature_offset=36)

    constants = ThermodynamicConstants(FT;
        dry_air_heat_capacity = cᵖ,
        vapor_heat_capacity = cᵖ,
        dry_air_molar_mass = Mᵈ,
        vapor_molar_mass = Mᵈ,
        saturation_vapor_pressure = DCMIP2016_tetens_formula,
        liquid = Breeze.Thermodynamics.CondensedPhase(FT;
            reference_latent_heat = 2500000.0,
            heat_capacity = cᵖ),
        ice = Breeze.Thermodynamics.CondensedPhase(FT;
            reference_latent_heat = 2834000.0,
            heat_capacity = cᵖ))

    microphysics = DCMIP2016KesslerMicrophysics(FT)

    # Convert to mass fractions
    rᵗ_init = rᵛ_init .+ rᶜˡ_init .+ rʳ_init
    qᵛ_init = rᵛ_init ./ (1 .+ rᵗ_init)
    qᶜˡ_init = rᶜˡ_init ./ (1 .+ rᵗ_init)
    qʳ_init = rʳ_init ./ (1 .+ rᵗ_init)
    qᵗ_init = qᵛ_init .+ qᶜˡ_init .+ qʳ_init

    # Run reference implementation
    T_ref = copy(T_prof)
    qᵛ_ref = copy(qᵛ_init)
    qᶜˡ_ref = copy(qᶜˡ_init)
    qʳ_ref = copy(qʳ_init)

    dcmip2016_klemp_wilhelmson_kessler!(T_ref, qᵛ_ref, qᶜˡ_ref, qʳ_ref, ρ_prof, p_prof, Δt, z_centers, constants, microphysics)

    # Run Breeze implementation
    ref_state = ReferenceState(grid, constants; surface_pressure=p₀)
    dynamics = AnelasticDynamics(ref_state)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants)

    set!(model.dynamics.reference_state.density, reshape(ρ_prof, 1, 1, Nz))
    set!(model.dynamics.reference_state.pressure, reshape(p_prof, 1, 1, Nz))
    set!(model.moisture_density, reshape(ρ_prof .* qᵛ_init, 1, 1, Nz))
    set!(model.microphysical_fields.ρqᶜˡ, reshape(ρ_prof .* qᶜˡ_init, 1, 1, Nz))
    set!(model.microphysical_fields.ρqʳ, reshape(ρ_prof .* qʳ_init, 1, 1, Nz))

    # Compute initial θˡⁱ
    ℒˡᵣ = constants.liquid.reference_latent_heat
    θˡⁱ_init = zeros(FT, Nz)
    for k in 1:Nz
        q = MoistureMassFractions(qᵛ_init[k], qᶜˡ_init[k] + qʳ_init[k])
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Rᵐ = mixture_gas_constant(q, constants)
        Π = (p_prof[k] / p₀)^(Rᵐ / cᵖᵐ)
        θˡⁱ_init[k] = (T_prof[k] - ℒˡᵣ * (qᶜˡ_init[k] + qʳ_init[k]) / cᵖᵐ) / Π
    end

    set!(model.formulation.potential_temperature_density, reshape(ρ_prof .* θˡⁱ_init, 1, 1, Nz))
    model.clock.last_Δt = Δt
    # Refresh the diagnostic state from the prognostics, then apply the operator-split
    # Kessler update once, mirroring how the time-steppers call it after `update_state!`.
    update_state!(model)
    microphysics_model_update!(model.microphysics, model)

    # Extract results
    ρqᶜˡ_result = Array(interior(model.microphysical_fields.ρqᶜˡ, 1, 1, :))
    ρqʳ_result = Array(interior(model.microphysical_fields.ρqʳ, 1, 1, :))
    ρqᵛ_result = Array(interior(model.moisture_density, 1, 1, :))
    ρθˡⁱ_result = Array(interior(model.formulation.potential_temperature_density, 1, 1, :))

    qᵛ_breeze = zeros(FT, Nz)
    qᶜˡ_breeze = zeros(FT, Nz)
    qʳ_breeze = zeros(FT, Nz)
    T_breeze = zeros(FT, Nz)

    for k in 1:Nz
        ρ = ρ_prof[k]
        qᶜˡ_breeze[k] = ρqᶜˡ_result[k] / ρ
        qʳ_breeze[k] = ρqʳ_result[k] / ρ
        qᵛ_breeze[k] = ρqᵛ_result[k] / ρ

        θˡⁱ_val = ρθˡⁱ_result[k] / ρ
        q = MoistureMassFractions(qᵛ_breeze[k], qᶜˡ_breeze[k] + qʳ_breeze[k])
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Rᵐ = mixture_gas_constant(q, constants)
        Π = (p_prof[k] / p₀)^(Rᵐ / cᵖᵐ)
        T_breeze[k] = Π * θˡⁱ_val + ℒˡᵣ * (qᶜˡ_breeze[k] + qʳ_breeze[k]) / cᵖᵐ
    end

    @test T_breeze ≈ T_ref rtol=1e-12
    @test qᵛ_breeze ≈ qᵛ_ref rtol=1e-12
    @test qᶜˡ_breeze ≈ qᶜˡ_ref rtol=1e-12
    @test qʳ_breeze ≈ qʳ_ref rtol=1e-12
end

@testset "Compressible Kessler density roles" begin
    FT = Float64
    Nx = 5
    Nz = 8
    grid = RectilinearGrid(default_arch, FT;
                           size = (Nx, Nx, Nz),
                           halo = (5, 5, 5),
                           x = (0, 500),
                           y = (0, 500),
                           z = (0, 800),
                           topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants(FT; saturation_vapor_pressure = TetensFormula(FT))
    microphysics = DCMIP2016KesslerMicrophysics(FT)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                    surface_pressure = FT(1e5),
                                    standard_pressure = FT(1e5),
                                    reference_potential_temperature = z -> FT(285))
    model = AtmosphereModel(grid; dynamics, microphysics,
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)

    # A deliberately moist state keeps total and dry density measurably distinct. Cloud and rain
    # activate phase conversion and sedimentation, so using the wrong density changes the result.
    qʳ_profile(x, y, z) = FT(0.003) * exp(-z / FT(250))
    set!(model; ρ = FT(1.1), T = FT(285), qᵛ = FT(0.016),
         qᶜˡ = FT(0.004), qʳ = qʳ_profile, enforce_mass_conservation = false)
    update_state!(model)

    column(field) = vec(Array(interior(field, 1, 1, :)))
    ρ = column(model.dynamics.total_density)
    ρᵈ = column(model.dynamics.dry_density)
    p = column(model.dynamics.pressure)
    T = column(model.temperature)
    qᵛ = column(model.moisture_density) ./ ρ
    qᶜˡ = column(model.microphysical_fields.ρqᶜˡ) ./ ρ
    qʳ = column(model.microphysical_fields.ρqʳ) ./ ρ
    z = collect(znodes(grid, Center()))

    @test all(ρ .> ρᵈ)

    # Advance an independent column reference from the exact diagnosed pre-update state. The
    # Kessler physics consumes total air density; its thermodynamic prognostic remains ρᵈθˡⁱ.
    Δt = FT(20)
    T_ref = copy(T)
    qᵛ_ref = copy(qᵛ)
    qᶜˡ_ref = copy(qᶜˡ)
    qʳ_ref = copy(qʳ)
    surface_mass_flux_ref =
        dcmip2016_klemp_wilhelmson_kessler!(T_ref, qᵛ_ref, qᶜˡ_ref, qʳ_ref,
                                            ρ, p, Δt, z, constants, microphysics;
                                            sedimentation_density = ρᵈ,
                                            dry_air_coupled = true)

    qᵗ_ref = qᵛ_ref .+ qᶜˡ_ref .+ qʳ_ref
    rᵛ_ref = qᵛ_ref ./ (1 .- qᵗ_ref)
    rᶜˡ_ref = qᶜˡ_ref ./ (1 .- qᵗ_ref)
    rʳ_ref = qʳ_ref ./ (1 .- qᵗ_ref)

    ℒˡᵣ = constants.liquid.reference_latent_heat
    θˡⁱ_ref = similar(T_ref)
    for k in eachindex(T_ref)
        q = MoistureMassFractions(qᵛ_ref[k], qᶜˡ_ref[k] + qʳ_ref[k])
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Rᵐ = mixture_gas_constant(q, constants)
        Π = (p[k] / FT(1e5))^(Rᵐ / cᵖᵐ)
        θˡⁱ_ref[k] = (T_ref[k] - ℒˡᵣ * (qᶜˡ_ref[k] + qʳ_ref[k]) / cᵖᵐ) / Π
    end

    model.clock.last_Δt = Δt
    microphysics_model_update!(model.microphysics, model)

    rtol = 1e-10
    for i in 1:Nx, j in 1:Nx
        @test vec(Array(interior(model.moisture_density, i, j, :))) ≈ ρᵈ .* rᵛ_ref rtol=rtol
        @test vec(Array(interior(model.microphysical_fields.ρqᶜˡ, i, j, :))) ≈ ρᵈ .* rᶜˡ_ref rtol=rtol
        @test vec(Array(interior(model.microphysical_fields.ρqʳ, i, j, :))) ≈ ρᵈ .* rʳ_ref rtol=rtol
        @test vec(Array(interior(model.formulation.potential_temperature_density, i, j, :))) ≈
              ρᵈ .* θˡⁱ_ref rtol=rtol
    end

    # Re-diagnosing total density after writeback must recover the same q/r state, rather than
    # silently changing it because old total density was used after sedimentation.
    ρ_new = column(model.dynamics.total_density)
    @test ρ_new[1] < ρ[1] # net rain outflow makes old and final surface density distinct
    @test column(model.moisture_density) ./ ρ_new ≈ qᵛ_ref rtol=rtol
    @test column(model.microphysical_fields.ρqᶜˡ) ./ ρ_new ≈ qᶜˡ_ref rtol=rtol
    @test column(model.microphysical_fields.ρqʳ) ./ ρ_new ≈ qʳ_ref rtol=rtol

    # The public surface flux must use the compressible model's total surface density, not its
    # dry density (nor a reference-state density).
    precipitation_rate = Array(interior(model.microphysical_fields.precipitation_rate))
    precipitation_flux = Array(interior(compute!(surface_precipitation_flux(model))))
    surface_ρ = Array(interior(model.dynamics.total_density, :, :, 1))
    surface_ρᵈ = Array(interior(model.dynamics.dry_density, :, :, 1))

    @test all(precipitation_rate .> 0)
    @test precipitation_flux ≈ surface_ρ .* precipitation_rate rtol=rtol
    @test all(≈(surface_mass_flux_ref; rtol), precipitation_flux)
    @test maximum(abs.(precipitation_flux .- surface_ρᵈ .* precipitation_rate)) > 1e-8
end
