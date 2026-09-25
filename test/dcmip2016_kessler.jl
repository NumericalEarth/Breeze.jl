include(joinpath(@__DIR__, "setup.jl"))

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
    kessler_column_reference!(θ, ρᵛ, ρᶜˡ, ρʳ, ρ, ϱ, p, Δz, Δt, pˢᵗ, constants, microphysics, dry_air_coupled)

Plain-array column implementation of the DCMIP2016 Kessler step as Breeze applies it, written
with explicit formulas (no Breeze thermodynamic-state machinery) so the kernel can be checked
against it. Per sedimentation substep and cell: temperature from the prognostic `θˡⁱ`;
flux-form upwind sedimentation of the rain partial density on the finite-volume cell of
thickness `Δz`, at fixed temperature; then the local Kessler processes on dry-air mixing ratios
at fixed `θˡⁱ`, with the saturation adjustment linearized about `θˡⁱ`.

`ρ` is the density the thermodynamics sees (the reference density on the anelastic core, the
total density on the compressible core), `ϱ` the coupling density (equal to `ρ` on the anelastic
core; the dry-air density on the compressible core, `dry_air_coupled = true`). The dry-air density
carrying the mixing ratios is `ϱ` when `dry_air_coupled`, and `ϱ - ρᵗ` otherwise. Mutates the
arrays in place and returns the substep-mean surface rain mass flux.

Departures from the DCMIP2016 Fortran (`kessler.f90` in [DOI: 10.5281/zenodo.1298671](https://doi.org/10.5281/zenodo.1298671)):
cell thickness instead of level spacing in the sedimentation divergence and CFL bound (the top cell
is a full cell, not a half cell); sedimentation moves the partial density `ρqʳ` (`= ρᵈ rʳ`) rather
than `ρ rʳ` with a fixed total density; the accretion rate sees the post-sedimentation rain; and
`θˡⁱ` is carried through phase change instead of incrementing `T` by `ℒˡᵣ Δrˡ / cᵖᵈ`.
"""
function kessler_column_reference!(θ, ρᵛ, ρᶜˡ, ρʳ, ρ, ϱ, p, Δz, Δt, pˢᵗ, constants, microphysics, dry_air_coupled)
    Nz = length(θ)
    FT = eltype(θ)

    # Thermodynamic constants
    ℒ   = constants.liquid.reference_latent_heat
    cᵖᵈ = constants.dry_air.heat_capacity
    cᵖᵛ = constants.vapor.heat_capacity
    cˡ  = constants.liquid.heat_capacity
    Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
    Rᵛ  = constants.molar_gas_constant / constants.vapor.molar_mass
    a   = constants.saturation_vapor_pressure.liquid_coefficient
    δT  = constants.saturation_vapor_pressure.liquid_temperature_offset
    T_DCMIP2016 = microphysics.dcmip_temperature_scale

    # Microphysics parameters
    k₁   = microphysics.autoconversion_rate
    rᶜˡ★ = microphysics.autoconversion_threshold
    k₂   = microphysics.accretion_rate
    βᵃᶜᶜ = microphysics.accretion_exponent
    Cᵨ   = microphysics.density_scale
    Cᵉᵛ₁   = microphysics.evaporation_ventilation_coefficient_1
    Cᵉᵛ₂   = microphysics.evaporation_ventilation_coefficient_2
    βᵉᵛ₁   = microphysics.evaporation_ventilation_exponent_1
    βᵉᵛ₂   = microphysics.evaporation_ventilation_exponent_2
    Cᵈⁱᶠᶠ  = microphysics.diffusivity_coefficient
    Cᵗʰᵉʳᵐ = microphysics.thermal_conductivity_coefficient
    cfl = microphysics.substep_cfl

    dry_density(k, ρᵗ) = dry_air_coupled ? ϱ[k] : ϱ[k] - ρᵗ

    # Mixture properties from dry-air mixing ratios (rᵗ = rᵛ + rˡ)
    function mixture(rᵛ, rˡ)
        rᵗ = rᵛ + rˡ
        qᵛ = rᵛ / (1 + rᵗ)
        qˡ = rˡ / (1 + rᵗ)
        qᵈ = 1 - qᵛ - qˡ
        Rᵐ = qᵈ * Rᵈ + qᵛ * Rᵛ
        cᵖᵐ = qᵈ * cᵖᵈ + qᵛ * cᵖᵛ + qˡ * cˡ
        return qᵛ, qˡ, Rᵐ, cᵖᵐ
    end

    # T = Π θˡⁱ + ℒ qˡ / cᵖᵐ, Π = (p / pˢᵗ)^(Rᵐ / cᵖᵐ), and its inverse
    function temperature_of(θᵏ, rᵛ, rˡ, pᵏ)
        _, qˡ, Rᵐ, cᵖᵐ = mixture(rᵛ, rˡ)
        Π = (pᵏ / pˢᵗ)^(Rᵐ / cᵖᵐ)
        return Π * θᵏ + ℒ * qˡ / cᵖᵐ
    end

    function potential_temperature_of(T, rᵛ, rˡ, pᵏ)
        _, qˡ, Rᵐ, cᵖᵐ = mixture(rᵛ, rˡ)
        Π = (pᵏ / pˢᵗ)^(Rᵐ / cᵖᵐ)
        return (T - ℒ * qˡ / cᵖᵐ) / Π
    end

    # ∂T/∂rˡ at fixed θˡⁱ, p and rᵗ (vapor → liquid)
    function temperature_slope(θᵏ, rᵛ, rˡ, pᵏ)
        qᵛ, qˡ, Rᵐ, cᵖᵐ = mixture(rᵛ, rˡ)
        qᵗ = qᵛ + qˡ
        Π = (pᵏ / pˢᵗ)^(Rᵐ / cᵖᵐ)
        Δc = cˡ - cᵖᵛ
        ∂κ∂qˡ = -(Rᵛ * cᵖᵐ + Rᵐ * Δc) / cᵖᵐ^2
        ∂T∂qˡ = θᵏ * Π * log(pᵏ / pˢᵗ) * ∂κ∂qˡ + ℒ / cᵖᵐ - ℒ * qˡ * Δc / cᵖᵐ^2
        return ∂T∂qˡ * (1 - qᵗ)
    end

    # Clip negative inputs (as the kernel does)
    ρᵛ .= max.(0, ρᵛ)
    ρᶜˡ .= max.(0, ρᶜˡ)
    ρʳ .= max.(0, ρʳ)

    𝕎ʳ = zeros(FT, Nz)
    ρ₁ = ρ[1]
    function update_terminal_velocities!()
        for k in 1:Nz
            rʳ = ρʳ[k] / dry_density(k, ρᵛ[k] + ρᶜˡ[k] + ρʳ[k])
            𝕎ʳ[k] = kessler_terminal_velocity(rʳ, ρ[k], ρ₁, microphysics)
        end
    end
    update_terminal_velocities!()

    # Sedimentation CFL on every cell's thickness
    max_Δt = Δt
    for k in 1:Nz
        max_Δt = min(max_Δt, cfl * Δz[k] / 𝕎ʳ[k])
    end
    Ns = max(1, ceil(Int, Δt / max_Δt))
    Δtₛ = Δt / Ns
    surface_mass_flux = zero(FT)

    for s in 1:Ns
        surface_mass_flux += ρʳ[1] * 𝕎ʳ[1]

        for k in 1:Nz
            # Temperature of the incoming state
            ρᵈ₀ = dry_density(k, ρᵛ[k] + ρᶜˡ[k] + ρʳ[k])
            T = temperature_of(θ[k], ρᵛ[k] / ρᵈ₀, (ρᶜˡ[k] + ρʳ[k]) / ρᵈ₀, p[k])

            # Upwind flux-form sedimentation of the rain partial density (cell k+1 is still
            # at its start-of-substep value)
            Fᵗᵒᵖ = k < Nz ? ρʳ[k+1] * 𝕎ʳ[k+1] : zero(FT)
            Fᵇᵒᵗ = ρʳ[k] * 𝕎ʳ[k]
            ρʳ[k] = max(0, ρʳ[k] + Δtₛ * (Fᵗᵒᵖ - Fᵇᵒᵗ) / Δz[k])

            # Post-sedimentation mixing ratios; θˡⁱ after sedimentation at fixed T
            ρᵈ = dry_density(k, ρᵛ[k] + ρᶜˡ[k] + ρʳ[k])
            rᵛ = ρᵛ[k] / ρᵈ
            rᶜˡ = ρᶜˡ[k] / ρᵈ
            rʳ = ρʳ[k] / ρᵈ
            θ₁ = potential_temperature_of(T, rᵛ, rᶜˡ + rʳ, p[k])
            ∂T∂rˡ = temperature_slope(θ₁, rᵛ, rᶜˡ + rʳ, p[k])
            f₅ = a * T_DCMIP2016 * ∂T∂rˡ

            # Autoconversion and accretion (KW eq. 2.13)
            Aʳ = max(0, k₁ * (rᶜˡ - rᶜˡ★))
            denom = 1 + Δtₛ * k₂ * rʳ^βᵃᶜᶜ
            Δrᴾ = rᶜˡ - (rᶜˡ - Δtₛ * Aʳ) / denom
            rᶜˡ = max(0, rᶜˡ - Δrᴾ)
            rʳ = max(0, rʳ + Δrᴾ)

            # Saturation adjustment: one Newton step linearized about θˡⁱ
            qᵛ⁺ = saturation_specific_humidity(T, ρ[k], constants, PlanarLiquidSurface())
            rᵛ⁺ = qᵛ⁺ / (1 - qᵛ⁺)
            Δrˢᵃᵗ = (rᵛ - rᵛ⁺) / (1 + rᵛ⁺ * f₅ / (T - δT)^2)

            # Rain evaporation (KW eq. 2.14)
            ρᵏ = ρ[k] * Cᵨ
            ρrʳ = ρᵏ * rʳ
            Vᵉᵛ = (Cᵉᵛ₁ + Cᵉᵛ₂ * ρrʳ^βᵉᵛ₁) * ρrʳ^βᵉᵛ₂
            Dᵗʰ = Cᵈⁱᶠᶠ / (p[k] * rᵛ⁺) + Cᵗʰᵉʳᵐ
            Δrᵛ⁺ = max(0, rᵛ⁺ - rᵛ)
            Ėʳ = Vᵉᵛ / Dᵗʰ * Δrᵛ⁺ / (ρᵏ * rᵛ⁺ + FT(1e-20))
            Δrᴱmax = max(0, -Δrˢᵃᵗ - rᶜˡ)
            Δrᴱ = min(min(Δtₛ * Ėʳ, Δrᴱmax), rʳ)

            Δrᶜ = max(Δrˢᵃᵗ, -rᶜˡ)
            rᵛ = max(0, rᵛ - Δrᶜ + Δrᴱ)
            rᶜˡ = rᶜˡ + Δrᶜ
            rʳ = rʳ - Δrᴱ

            # Write back: partial densities on the (unchanged) dry-air density, θˡⁱ conserved
            # through the phase change
            ρᵛ[k] = ρᵈ * rᵛ
            ρᶜˡ[k] = ρᵈ * rᶜˡ
            ρʳ[k] = ρᵈ * rʳ
            θ[k] = θ₁
        end

        s < Ns && update_terminal_velocities!()
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

@testset "Kernel matches the column reference on a stretched grid" begin
    FT = Float64

    # Uniform 100 m cells to 2 km, then 10 % stretching to 4 km: the sedimentation geometry
    # (cell thicknesses, junction, top cell) is exercised, not only the physics.
    faces = collect(0.0:100.0:2000.0)
    Δ = 100.0
    while faces[end] < 4000 - 1e-6
        Δ *= 1.1
        push!(faces, min(faces[end] + Δ, 4000.0))
    end
    Nz = length(faces) - 1

    grid = RectilinearGrid(CPU(), FT;
                           size = (1, 1, Nz),
                           x = (0, 100),
                           y = (0, 100),
                           z = faces,
                           topology = (Periodic, Periodic, Bounded))

    z_centers = collect(znodes(grid, Center()))
    Δz = diff(faces)

    # Atmospheric profile with linear lapse rate
    T_surface = FT(288.0)
    p_surface = FT(101325.0)
    g = FT(9.81)
    Rᵈ = FT(287.0)
    lapse_rate = FT(0.0065)

    T_prof = T_surface .- lapse_rate .* z_centers
    p_prof = p_surface .* (T_prof ./ T_surface) .^ (g / (Rᵈ * lapse_rate))
    ρ_prof = p_prof ./ (Rᵈ .* T_prof)

    p₀ = FT(100000.0)

    # Initial moisture profiles (mixing ratios): a moist layer, a cloud layer above the
    # autoconversion threshold, and rain that falls through the junction and out of the top cell
    rᵛ_init = zeros(FT, Nz)
    rᶜˡ_init = zeros(FT, Nz)
    rʳ_init = zeros(FT, Nz)

    for k in 1:Nz
        z = z_centers[k]
        rᵛ_init[k] = 0.015 * exp(-((z - 1000) / 1000)^2)
        if 1500 < z < 2500
            rᶜˡ_init[k] = 0.002
        end
        if 1000 < z < 2000 || k == Nz
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
            heat_capacity = cᵖ,
            density = 1000),
        ice = Breeze.Thermodynamics.CondensedPhase(FT;
            reference_latent_heat = 2834000.0,
            heat_capacity = cᵖ,
            density = 917))

    microphysics = DCMIP2016KesslerMicrophysics(FT)

    # Convert to mass fractions
    rᵗ_init = rᵛ_init .+ rᶜˡ_init .+ rʳ_init
    qᵛ_init = rᵛ_init ./ (1 .+ rᵗ_init)
    qᶜˡ_init = rᶜˡ_init ./ (1 .+ rᵗ_init)
    qʳ_init = rʳ_init ./ (1 .+ rᵗ_init)

    # Initial θˡⁱ from T
    ℒˡᵣ = constants.liquid.reference_latent_heat
    θˡⁱ_init = zeros(FT, Nz)
    for k in 1:Nz
        q = MoistureMassFractions(qᵛ_init[k], qᶜˡ_init[k] + qʳ_init[k])
        cᵖᵐ = mixture_heat_capacity(q, constants)
        Rᵐ = mixture_gas_constant(q, constants)
        Π = (p_prof[k] / p₀)^(Rᵐ / cᵖᵐ)
        θˡⁱ_init[k] = (T_prof[k] - ℒˡᵣ * (qᶜˡ_init[k] + qʳ_init[k]) / cᵖᵐ) / Π
    end

    # Run the column reference on the partial densities (anelastic: ϱ = ρ, fixed total density)
    θ_ref = copy(θˡⁱ_init)
    ρᵛ_ref = ρ_prof .* qᵛ_init
    ρᶜˡ_ref = ρ_prof .* qᶜˡ_init
    ρʳ_ref = ρ_prof .* qʳ_init
    surface_mass_flux_ref =
        kessler_column_reference!(θ_ref, ρᵛ_ref, ρᶜˡ_ref, ρʳ_ref, ρ_prof, ρ_prof, p_prof, Δz, Δt, p₀,
                                  constants, microphysics, false)

    # Run Breeze implementation
    ref_state = ReferenceState(grid, constants; base_pressure=p₀)
    dynamics = AnelasticDynamics(ref_state)
    model = AtmosphereModel(grid; dynamics, microphysics, thermodynamic_constants=constants)

    set!(model.dynamics.reference_state.density, reshape(ρ_prof, 1, 1, Nz))
    set!(model.dynamics.reference_state.pressure, reshape(p_prof, 1, 1, Nz))
    set!(model.moisture_density, reshape(ρ_prof .* qᵛ_init, 1, 1, Nz))
    set!(model.microphysical_fields.ρqᶜˡ, reshape(ρ_prof .* qᶜˡ_init, 1, 1, Nz))
    set!(model.microphysical_fields.ρqʳ, reshape(ρ_prof .* qʳ_init, 1, 1, Nz))
    set!(model.formulation.potential_temperature_density, reshape(ρ_prof .* θˡⁱ_init, 1, 1, Nz))
    model.clock.last_Δt = Δt
    # Refresh the diagnostic state from the prognostics, then apply the operator-split
    # Kessler update once, mirroring how the time-steppers call it after `update_state!`.
    update_state!(model)
    microphysics_model_update!(model.microphysics, model)

    rtol = 1e-12
    @test vec(Array(interior(model.moisture_density))) ≈ ρᵛ_ref rtol=rtol
    @test vec(Array(interior(model.microphysical_fields.ρqᶜˡ))) ≈ ρᶜˡ_ref rtol=rtol
    @test vec(Array(interior(model.microphysical_fields.ρqʳ))) ≈ ρʳ_ref rtol=rtol
    @test vec(Array(interior(model.formulation.potential_temperature_density))) ≈ ρ_prof .* θ_ref rtol=rtol
    @test vec(Array(interior(model.microphysical_fields.precipitation_rate))) ≈ [surface_mass_flux_ref / ρ_prof[1]] rtol=rtol

    # The reference is not trivial: rain crossed the junction and left the top cell, cloud
    # converted to rain, and the invariant changed only through sedimentation
    @test ρʳ_ref[Nz] < ρ_prof[Nz] * qʳ_init[Nz]
    @test any(ρʳ_ref .> ρ_prof .* qʳ_init)
    @test surface_mass_flux_ref > 0
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
                                    base_pressure = FT(1e5),
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
    # Kessler thermodynamics consumes the total air density and the diagnosed pressure; the
    # mixing ratios are carried by the prognostic dry density, which is also the coupling
    # density of the thermodynamic prognostic ρᵈθˡⁱ.
    Δt = FT(20)
    Δz = fill(FT(100), Nz)
    pˢᵗ = FT(1e5)
    θ_ref = column(model.formulation.potential_temperature)
    ρᵛ_ref = column(model.moisture_density)
    ρᶜˡ_ref = column(model.microphysical_fields.ρqᶜˡ)
    ρʳ_ref = column(model.microphysical_fields.ρqʳ)
    surface_mass_flux_ref =
        kessler_column_reference!(θ_ref, ρᵛ_ref, ρᶜˡ_ref, ρʳ_ref, ρ, ρᵈ, p, Δz, Δt, pˢᵗ,
                                  constants, microphysics, true)

    ρᵗ_ref = ρᵛ_ref .+ ρᶜˡ_ref .+ ρʳ_ref
    qᵛ_ref = ρᵛ_ref ./ (ρᵈ .+ ρᵗ_ref)
    qᶜˡ_ref = ρᶜˡ_ref ./ (ρᵈ .+ ρᵗ_ref)
    qʳ_ref = ρʳ_ref ./ (ρᵈ .+ ρᵗ_ref)

    model.clock.last_Δt = Δt
    microphysics_model_update!(model.microphysics, model)

    rtol = 1e-10
    for i in 1:Nx, j in 1:Nx
        @test vec(Array(interior(model.moisture_density, i, j, :))) ≈ ρᵛ_ref rtol=rtol
        @test vec(Array(interior(model.microphysical_fields.ρqᶜˡ, i, j, :))) ≈ ρᶜˡ_ref rtol=rtol
        @test vec(Array(interior(model.microphysical_fields.ρqʳ, i, j, :))) ≈ ρʳ_ref rtol=rtol
        @test vec(Array(interior(model.formulation.potential_temperature_density, i, j, :))) ≈
              ρᵈ .* θ_ref rtol=rtol
    end

    # Re-diagnosing total density after writeback must recover the same q/r state, rather than
    # silently changing it because old total density was used after sedimentation.
    ρ_new = column(model.dynamics.total_density)
    @test ρ_new[1] < ρ[1] # net rain outflow makes old and final surface density distinct
    @test ρ_new ≈ ρᵈ .+ ρᵗ_ref rtol=rtol
    @test column(model.moisture_density) ./ ρ_new ≈ qᵛ_ref rtol=rtol
    @test column(model.microphysical_fields.ρqᶜˡ) ./ ρ_new ≈ qᶜˡ_ref rtol=rtol
    @test column(model.microphysical_fields.ρqʳ) ./ ρ_new ≈ qʳ_ref rtol=rtol

    # The prognostic water budget of every column closes to the surface flux
    for i in 1:Nx, j in 1:Nx
        W₀ = sum((ρ .* (qᵛ .+ qᶜˡ .+ qʳ)) .* Δz)
        W₁ = sum((vec(Array(interior(model.moisture_density, i, j, :))) .+
                  vec(Array(interior(model.microphysical_fields.ρqᶜˡ, i, j, :))) .+
                  vec(Array(interior(model.microphysical_fields.ρqʳ, i, j, :)))) .* Δz)
        @test W₁ + surface_mass_flux_ref * Δt ≈ W₀ rtol=1e-12
    end

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

@testset "Thermodynamic constants validation" begin
    FT = Float64
    grid = RectilinearGrid(CPU(), size=(1, 1, 4), extent=(1, 1, 1))
    microphysics = DCMIP2016KesslerMicrophysics(FT)

    # DCMIP2016 Kessler requires Tetens saturation vapor pressure. The default constants use
    # ClausiusClapeyron, which lacks the Tetens coefficients the scheme reads — this should be
    # rejected at construction with a clear error, not fail later inside the kernel (issue #858).
    @test_throws ArgumentError AtmosphereModel(grid; microphysics)

    # Constructing with Tetens constants succeeds.
    tetens_constants = ThermodynamicConstants(FT; saturation_vapor_pressure = TetensFormula(FT))
    model = AtmosphereModel(grid; microphysics, thermodynamic_constants=tetens_constants)
    @test model.microphysics isa DCMIP2016KesslerMicrophysics
end
