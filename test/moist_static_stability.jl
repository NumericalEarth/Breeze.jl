include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics
using Breeze.AtmosphereModels: density_potential_temperature, dynamics_pressure, standard_pressure,
                               precipitating_mass_fraction
using Breeze.AtmosphereModels.Diagnostics: microphysics_phase_equilibrium
using Breeze.Thermodynamics: density, dry_air_gas_constant, vapor_gas_constant,
                             mixture_gas_constant, saturation_specific_humidity, saturation_vapor_pressure,
                             PlanarLiquidSurface
using Breeze.TurbulenceClosures: moist_stateᶜᶜᶜ, saturatedᶜᶜᶠ, saturated_static_stabilityᶜᶜᶠ,
                                 saturation_specific_humidityᶜᶜᶜ, static_stabilityᶜᶜᶠ,
                                 nonprecipitating_waterᶜᶜᶜ
using CloudMicrophysics
using Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TurbulenceClosures: buoyancy_force, buoyancy_tracers
using Test

const MicrophysicsExtension = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)

stability_column(operation, model, arguments...) =
    vec(Array(interior(Field(KernelFunctionOperation{Center, Center, Face}(
        operation, model.grid, arguments...)))))

@testset "Moist stability thermodynamic consistency [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=8, z=(0, 8), topology=(Flat, Flat, Bounded))
    rtol = max(1e-9, 100eps(FT))

    @testset "thin anelastic cloud uses local EOS density" begin
        model = AtmosphereModel(grid; microphysics=SaturationAdjustment(FT), advection=nothing)
        set!(model; T=280, qᵗ=0.0063)
        buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)

        @allowscalar begin
            T, p, ρ, q = @inferred moist_stateᶜᶜᶜ(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
            ρᵐ = density(T, p, q, model.thermodynamic_constants)
            expected = saturation_specific_humidity(T, ρᵐ, model.thermodynamic_constants, WarmPhaseEquilibrium())
            actual = @inferred saturation_specific_humidityᶜᶜᶜ(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
            @test q.liquid > 0
            @test abs(ρᵐ / ρ - 1) > 0.01
            @test actual ≈ expected rtol=rtol
        end

        selected = stability_column(saturatedᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ)
        @test all(selected[2:8] .== 1)
        actual = stability_column(static_stabilityᶜᶜᶠ, model, MoistStaticStability(), buoyancy, tracers)
        expected = stability_column(saturated_static_stabilityᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ)
        @test actual[2:8] == expected[2:8]
    end

    @testset "equilibrium moisture is partitioned once" begin
        for equilibrium in (WarmPhaseEquilibrium(), MixedPhaseEquilibrium(FT))
            cloud_formation = SaturationAdjustment(FT; equilibrium)
            microphysics = MicrophysicsExtension.OneMomentCloudMicrophysics(FT; cloud_formation)
            model = AtmosphereModel(grid; microphysics, advection=nothing)
            set!(model; T=250, qᵗ=0.002)
            buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)
            μ = model.microphysical_fields

            @allowscalar begin
                T, p, ρ, q = @inferred moist_stateᶜᶜᶜ(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
                @test q.vapor ≈ μ.qᵛ[1, 1, 4] rtol=rtol
                @test q.vapor + q.liquid + q.ice ≈ FT(0.002) rtol=rtol
                @test q.liquid + q.ice > 0
                constants = model.thermodynamic_constants
                Rᵈ = dry_air_gas_constant(constants)
                θᵨ = mixture_gas_constant(q, constants) / Rᵈ * T *
                      (standard_pressure(model.dynamics) / p)^(Rᵈ / constants.dry_air.heat_capacity)
                @test density_potential_temperature(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ) ≈ θᵨ rtol=rtol
            end
        end
    end

    @testset "bulk phase equilibrium follows cloud formation" begin
        equilibrium = MixedPhaseEquilibrium(FT; homogeneous_ice_nucleation_temperature=260)
        cloud_formation = SaturationAdjustment(FT; equilibrium)
        microphysics = MicrophysicsExtension.OneMomentCloudMicrophysics(FT; cloud_formation)
        model = AtmosphereModel(grid; microphysics, advection=nothing)
        set!(model; T=250, qᵗ=0.002)
        buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)
        @test microphysics_phase_equilibrium(model.microphysics) === equilibrium
        @test microphysics_phase_equilibrium(InstantaneousPrecipitation(FT; equilibrium)) === equilibrium

        @allowscalar begin
            T, p, ρ, q = moist_stateᶜᶜᶜ(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
            constants = model.thermodynamic_constants
            ρᵐ = density(T, p, q, constants)
            expected = saturation_specific_humidity(T, ρᵐ, constants, equilibrium)
            warm = saturation_specific_humidity(T, ρᵐ, constants, WarmPhaseEquilibrium())
            actual = saturation_specific_humidityᶜᶜᶜ(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ)
            @test q.ice > 0
            @test actual ≈ expected rtol=rtol
            @test actual < FT(0.9) * warm
        end
    end

    @testset "rain and snow do not select saturated stability" begin
        warm_adjustment = SaturationAdjustment(FT)
        mixed_adjustment = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
        mixed_nonequilibrium = NonEquilibriumCloudFormation(nothing, ConstantRateCondensateFormation(FT(0)))
        schemes = (MicrophysicsExtension.OneMomentCloudMicrophysics(FT),
                   MicrophysicsExtension.OneMomentCloudMicrophysics(FT; cloud_formation=warm_adjustment),
                   MicrophysicsExtension.OneMomentCloudMicrophysics(FT; cloud_formation=mixed_adjustment),
                   MicrophysicsExtension.OneMomentCloudMicrophysics(FT; cloud_formation=mixed_nonequilibrium),
                   MicrophysicsExtension.TwoMomentCloudMicrophysics(FT))

        for microphysics in schemes
            model = AtmosphereModel(grid; microphysics, advection=nothing)
            set!(model; T=290, qᵗ=0.022, qʳ=0.02)
            buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)
            selected = stability_column(saturatedᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ)
            moist = stability_column(static_stabilityᶜᶜᶠ, model, MoistStaticStability(), buoyancy, tracers)
            dry = stability_column(static_stabilityᶜᶜᶠ, model, DryStaticStability(), buoyancy, tracers)
            @test all(selected[2:8] .== 0)
            @test moist[2:8] == dry[2:8]
            @allowscalar @test nonprecipitating_waterᶜᶜᶜ(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ) ≈ FT(0.002) rtol=rtol

            if haskey(model.microphysical_fields, :qˢⁿ)
                set!(model; T=290, qᵗ=0.022, qʳ=0, qˢⁿ=0.02)
                selected = stability_column(saturatedᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ)
                @test all(selected[2:8] .== 0)
                @allowscalar @test nonprecipitating_waterᶜᶜᶜ(1, 1, 4, grid, buoyancy, tracers.T, tracers.qᵛᵉ) ≈ FT(0.002) rtol=rtol
            end
        end
    end

    @testset "P3 excludes precipitation without double-counting rime" begin
        # Only the category interface is needed here; a skeleton avoids loading lookup tables.
        microphysics = P3Microphysics(FT(1e-14), FT(1e-16), nothing, nothing, nothing,
                                     nothing, nothing, nothing, nothing, nothing)
        names = (:qʳ, :qⁱ, :qʷⁱ, :qᶜˡ, :qᶠ)
        fields = NamedTuple{names}(ntuple(_ -> CenterField(grid), length(names)))
        for (name, mass) in zip(names, (0.002, 0.004, 0.001, 0.003, 0.002))
            set!(fields[name], mass)
        end
        @allowscalar @test @inferred(precipitating_mass_fraction(1, 1, 4, microphysics, fields)) ≈ FT(0.007) rtol=rtol
    end

    @testset "saturated rain contributes density loading" begin
        rain_grid = RectilinearGrid(default_arch, FT; size=8, z=(0, 800), topology=(Flat, Flat, Bounded))
        constants = ThermodynamicConstants(FT; saturation_vapor_pressure=Breeze.TetensFormula(FT))
        microphysics = DCMIP2016KesslerMicrophysics(FT)
        model = AtmosphereModel(rain_grid; microphysics, thermodynamic_constants=constants, advection=nothing)
        buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)
        μ = model.microphysical_fields
        T = FT(280)
        ϵ = dry_air_gas_constant(constants) / vapor_gas_constant(constants)
        vapor_pressure = saturation_vapor_pressure(T, constants, PlanarLiquidSurface())
        pressure = dynamics_pressure(model.dynamics)
        rˢ = ϵ * vapor_pressure / (pressure - vapor_pressure)
        rʳ = CenterField(rain_grid)
        rᶜˡ = FT(0.001)
        slope = FT(1e-6)

        profiles = map((FT(0), slope)) do gradient
            set!(rʳ, z -> gradient * z)
            qᵈ = 1 / (1 + rˢ + rᶜˡ + rʳ)
            set!(tracers.qᵛᵉ, rˢ * qᵈ)
            set!(μ.qᶜˡ, rᶜˡ * qᵈ)
            set!(μ.qʳ, rʳ * qᵈ)
            set!(tracers.T, T)
            fill_halo_regions!((tracers.qᵛᵉ, μ.qᶜˡ, μ.qʳ, tracers.T))
            (stability_column(saturated_static_stabilityᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ),
             stability_column(static_stabilityᶜᶜᶠ, model, DryStaticStability(), buoyancy, tracers))
        end

        # At fixed T, p, and vapor/cloud mixing ratios, rain adds its density loading to the buoyancy gradient ∂z b,
        # -g ∂z rʳ in the dilute-water approximation, which the saturated stability carries. (Its latent part changes
        # too: at fixed T the heat capacity of the rain changes θʷ, ĉᵖ Π θʷ = cᵖᵐ T - ℒˡ₀ qᶜˡ, and the rain dilutes the
        # mass fractions of the vapor and the cloud.)
        loading = profiles[2][2][2:8] - profiles[1][2][2:8]
        @test all(isapprox.(loading, -constants.gravitational_acceleration * slope; rtol=1e-2))
        @test all(isfinite, profiles[2][1])
    end
end

using Breeze.TurbulenceClosures: applied_flux_state, buoyancy_derivativesᶜᶜᶜ, distributed_water, distributed_condensateᶜᶜᶜ,
                                 saturated_condensate_coefficients, distributed_condensate_liquid_fraction
using Oceananigans.Operators: Δzᶜᶜᶠ

# The saturated static stability is the saturated stratification of the energetics of the applied fluxes,
# N²ᵇ + (b_qᶜ - b_qᵛ) (∂z s̄ - ∂z qᶜ), with the buoyancy gradient of the dynamics in place of the gradient N²ᵇ of the mixed
# variables: the latent part is the energetics' to round-off, and the whole differs from it by ∂z b - N²ᵇ.
@testset "Saturated static stability against the energetics of the applied fluxes [$FT, mixed phase = $mixed]" for
        FT in test_float_types(), mixed in (false, true)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size = 16, z = (0, 1600), topology = (Flat, Flat, Bounded))
    equilibrium = mixed ? MixedPhaseEquilibrium(FT) : WarmPhaseEquilibrium()
    microphysics = SaturationAdjustment(FT; equilibrium)
    model = AtmosphereModel(grid; microphysics, advection = nothing)
    # Saturated throughout: warm cloud, or a mixed-phase cloud at about 250 K
    θ₀, q₀ = mixed ? (250, 0.0016) : (285, 0.012)
    set!(model; θ = z -> θ₀ + 3e-3 * z, qᵗ = z -> q₀ * (1 - z / 8000))
    buoyancy, tracers = buoyancy_force(model), buoyancy_tracers(model)
    s = applied_flux_state(model)
    θʷ, qʷ, thermodynamics = distributed_water(s, grid)

    saturated = stability_column(saturatedᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ)
    N²⁺ = stability_column(saturated_static_stabilityᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ)
    N²ᵈ = stability_column(static_stabilityᶜᶜᶠ, model, DryStaticStability(), buoyancy, tracers)
    @test all(saturated[2:16] .== 1)

    @allowscalar for k in 2:16
        terms = map((k, k-1)) do m
            b_θ, b_qᵛ, b_qˡ, b_qⁱ, θ, q = buoyancy_derivativesᶜᶜᶜ(1, 1, m, grid, s)
            λ, _ = distributed_condensate_liquid_fraction(s.phase_equilibrium, s.T[1, 1, m])
            c_q, c_θ = saturated_condensate_coefficients(thermodynamics, 1, 1, m, grid, s, θʷ[1, 1, m])
            (; b_θ, b_qᵛ, b_qˡ, b_qⁱ, b_qᶜ = λ * b_qˡ + (1 - λ) * b_qⁱ, θ, q, c_q, c_θ,
               qʷ = qʷ[1, 1, m], θʷ = θʷ[1, 1, m], qᶜ = distributed_condensateᶜᶜᶜ(1, 1, m, grid, s))
        end
        t⁺, t⁻ = terms
        Δz = Δzᶜᶜᶠ(1, 1, k, grid)
        N²ᵇ = ((t⁺.b_θ + t⁻.b_θ) / 2 * (t⁺.θ - t⁻.θ) + (t⁺.b_qᵛ + t⁻.b_qᵛ) / 2 * (t⁺.q.vapor - t⁻.q.vapor) +
               (t⁺.b_qˡ + t⁻.b_qˡ) / 2 * (t⁺.q.liquid - t⁻.q.liquid) + (t⁺.b_qⁱ + t⁻.b_qⁱ) / 2 * (t⁺.q.ice - t⁻.q.ice)) / Δz
        bᶜᵛ = (t⁺.b_qᶜ + t⁻.b_qᶜ - (t⁺.b_qᵛ + t⁻.b_qᵛ)) / 2
        ∂z_s = ((t⁺.c_q + t⁻.c_q) / 2 * (t⁺.qʷ - t⁻.qʷ) - (t⁺.c_θ + t⁻.c_θ) / 2 * (t⁺.θʷ - t⁻.θʷ)) / Δz
        latent = bᶜᵛ * (∂z_s - (t⁺.qᶜ - t⁻.qᶜ) / Δz)

        # The latent part, to round-off of the buoyancy gradient it offsets
        @test N²⁺[k] - N²ᵈ[k] ≈ latent rtol = FT == Float32 ? 1e-3 : 1e-8
        # The whole differs from the energetics' by ∂z b - N²ᵇ, the buoyancy gradient of `DryStaticStability`,
        # g ∂z ln θᵨ, against the stratification of the mixed variables from the derivatives of the anelastic buoyancy:
        # 4 to 7 percent of the stratification in the warm column and 10 to 11 percent in the cold one
        @test abs(N²⁺[k] - (N²ᵇ + latent)) ≤ 0.15 * abs(N²ᵈ[k])
        # Latent heating offsets part of the stratification
        @test N²⁺[k] < N²ᵈ[k]
    end
end
