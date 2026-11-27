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
            stability_column(saturated_static_stabilityᶜᶜᶠ, model, buoyancy, tracers.T, tracers.qᵛᵉ)
        end

        # At fixed T, p, and vapor/cloud mixing ratios, only rain changes the parcel's
        # density loading. The dilute-water approximation used here gives -g ∂z rʳ.
        difference = profiles[2][2:8] - profiles[1][2:8]
        @test all(isapprox.(difference, -constants.gravitational_acceleration * slope; rtol=max(1e-8, 2000eps(FT))))
    end
end
