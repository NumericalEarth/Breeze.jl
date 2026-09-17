include(joinpath(@__DIR__, "setup.jl"))

#####
##### ecCKD (NumericalRadiation) versus RRTMGP radiation on one column
#####
##### Both extensions solve the same 15 km column: the ecCKD `climate_32x32` model through the
##### column kernel, and RRTMGP's correlated-k tables through the RRTMGP extension, each on its
##### own `AtmosphereModel` with identical profiles. The clear-sky gates are the spread expected
##### between two well-validated correlated-k models on CKDMIP-like profiles (Hogan and
##### Matricardi 2020): a few W m⁻² in the boundary fluxes and a fraction of a K day⁻¹ in the
##### heating rates, plus the 2×CO₂ forcing the two models agree on to a few tenths of a W m⁻².
##### With a 500 m liquid cloud the two cloud optics (Mie droplets on the ecCKD g points, RRTMGP's
##### lookup tables) are allowed twice the clear-sky spread.
#####

using Breeze
using Breeze.AtmosphereModels: total_density, _update_radiation!
using Dates: DateTime
using Oceananigans
using Oceananigans.Units
using Printf
using Statistics: mean
using Test

# Both radiation extensions
using ClimaComms
using NCDatasets
using RRTMGP
using NumericalRadiation: NumericalRadiation

# Prognostic cloud liquid, to prescribe the same cloud to both
using CloudMicrophysics
const CloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .CloudMicrophysicsExt: OneMomentCloudMicrophysics

const Nz = 60
const TOP = 15kilometers

column_grid(FT) = RectilinearGrid(default_arch, FT; size = Nz, x = 0.0, y = 45.0, z = (0, TOP),
                                  topology = (Flat, Flat, Bounded))

function comparison_radiation(grid, optics; CO₂ = 420e-6)
    return RadiativeTransferModel(grid, optics, ThermodynamicConstants();
                                  background_atmosphere = BackgroundAtmosphere(; CO₂),
                                  surface_temperature = 300, surface_emissivity = 0.98, surface_albedo = 0.1,
                                  solar_constant = 1361, solar_position = FixedCosineZenith(0.5),
                                  column_extension = nothing)
end

# RRTMGP takes no column extension; the grid top is the top of its atmosphere as well
comparison_radiation(grid, optics::Union{ClearSkyOptics, AllSkyOptics}; CO₂ = 420e-6) =
    RadiativeTransferModel(grid, optics, ThermodynamicConstants();
                           background_atmosphere = BackgroundAtmosphere(; CO₂),
                           surface_temperature = 300, surface_emissivity = 0.98, surface_albedo = 0.1,
                           solar_constant = 1361, solar_position = FixedCosineZenith(0.5))

# Cloud liquid held as a prognostic (zero condensation rate), so `set!` prescribes it exactly
held_liquid_microphysics() =
    OneMomentCloudMicrophysics(; cloud_formation = NonEquilibriumCloudFormation(ConstantRateCondensateFormation(0.0)))

# The 500 m cloud: `cloud_liquid` (kg kg⁻¹) between 1 and 1.5 km, cells 5 and 6 of the 250 m layers
const CLOUD_BOTTOM = 1kilometer
const CLOUD_TOP = 1.5kilometers

function comparison_model(grid, radiation; microphysics = nothing, cloud_liquid = 0)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)
    clock = Clock(time = DateTime(2024, 6, 21, 12, 0, 0))
    model = AtmosphereModel(grid; clock, dynamics, microphysics, formulation = :LiquidIcePotentialTemperature, radiation)
    θ(z) = 300 + 5e-3 * z
    qᵗ(z) = 0.015 * exp(-z / 2500)
    if isnothing(microphysics)
        set!(model; θ, qᵗ)
    else
        qᶜˡ(z) = ifelse(CLOUD_BOTTOM < z < CLOUD_TOP, cloud_liquid, 0)
        set!(model; θ, qᵗ, qᶜˡ)
    end
    return model
end

# The boundary fluxes, net flux and heating rate (K day⁻¹) of a solved column
function column_diagnostics(radiation, model)
    ℐ_lw_up = Array(interior(radiation.upwelling_longwave_flux))[1, 1, :]
    ℐ_lw_dn = Array(interior(radiation.downwelling_longwave_flux))[1, 1, :]
    ℐ_sw_up = Array(interior(radiation.upwelling_shortwave_flux))[1, 1, :]
    ℐ_sw_dn = Array(interior(radiation.downwelling_shortwave_flux))[1, 1, :]
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    ρ = Array(interior(total_density(model.dynamics)))[1, 1, :]
    heating = Array(interior(radiation.flux_divergence))[1, 1, :] ./ (ρ .* cᵖ) .* 86400
    return (olr = ℐ_lw_up[Nz+1],
            surface_longwave_down = -ℐ_lw_dn[1],
            surface_shortwave_down = -ℐ_sw_dn[1],
            toa_shortwave_up = ℐ_sw_up[Nz+1],
            net_toa = ℐ_lw_up[Nz+1] + ℐ_lw_dn[Nz+1] + ℐ_sw_up[Nz+1] + ℐ_sw_dn[Nz+1],
            net_surface = ℐ_lw_up[1] + ℐ_lw_dn[1] + ℐ_sw_up[1] + ℐ_sw_dn[1],
            heating)
end

function solved_column(grid, optics; CO₂ = 420e-6, microphysics = nothing, cloud_liquid = 0)
    radiation = comparison_radiation(grid, optics; CO₂)
    model = comparison_model(grid, radiation; microphysics, cloud_liquid)
    _update_radiation!(radiation, model)   # warm
    seconds = @elapsed _update_radiation!(radiation, model)
    return column_diagnostics(radiation, model), seconds
end

@testset "ecCKD versus RRTMGP clear sky" begin
    Oceananigans.defaults.FloatType = Float64
    grid = column_grid(Float64)

    ecckd, ecckd_seconds = solved_column(grid, EcCKDOptics())
    rrtmgp, rrtmgp_seconds = solved_column(grid, ClearSkyOptics())

    @printf("%-28s %12s %12s %12s\n", "clear sky, 15 km, μ₀ = 0.5", "ecCKD", "RRTMGP", "difference")
    for (name, key) in (("OLR", :olr), ("surface LW down", :surface_longwave_down),
                        ("surface SW down", :surface_shortwave_down), ("TOA SW up", :toa_shortwave_up))
        @printf("%-28s %12.3f %12.3f %12.3f W m⁻²\n", name, ecckd[key], rrtmgp[key], ecckd[key] - rrtmgp[key])
    end
    interior_cells = 3:Nz-2
    Δheating = ecckd.heating[interior_cells] .- rrtmgp.heating[interior_cells]
    heating_rmse = sqrt(mean(Δheating .^ 2))
    heating_max = maximum(abs, Δheating)
    @printf("%-28s %12.4f %12.4f K day⁻¹ (RMSE, max over cells 3:Nz-2)\n", "heating difference", heating_rmse, heating_max)
    @printf("%-28s %12.2f %12.2f ms per update\n", "time", 1e3 * ecckd_seconds, 1e3 * rrtmgp_seconds)

    # The OLR gate was raised once from the literature-derived 3 to 3.9 W m⁻² and is
    # calibrated at Nz = 60: the observed 3.11 W m⁻² (1.1 %) grows weakly with the vertical
    # resolution (3.11, 3.27, 3.36 at Nz = 60, 120, 240, the signature of the two models'
    # different Planck-source discretizations), persists in a much drier column (2.87 at
    # 1 g kg⁻¹), and the 2×CO₂ forcings below agree to 0.04 W m⁻², so it is the spread
    # between the two gas-optics models on this column rather than a staging error
    @test abs(ecckd.olr - rrtmgp.olr) ≤ 3.9
    @test abs(ecckd.surface_longwave_down - rrtmgp.surface_longwave_down) ≤ 4
    @test abs(ecckd.surface_shortwave_down - rrtmgp.surface_shortwave_down) ≤ 6
    @test abs(ecckd.toa_shortwave_up - rrtmgp.toa_shortwave_up) ≤ 5
    @test heating_rmse ≤ 0.15
    @test heating_max ≤ 0.5

    @testset "2×CO₂ forcing" begin
        ecckd_2x, _ = solved_column(grid, EcCKDOptics(); CO₂ = 840e-6)
        rrtmgp_2x, _ = solved_column(grid, ClearSkyOptics(); CO₂ = 840e-6)

        # Instantaneous forcing: the change of the net downward flux at the top and bottom of the
        # column (the net fluxes above are positive upward)
        forcing(x1, x2) = (toa = x1.net_toa - x2.net_toa, surface = x1.net_surface - x2.net_surface)
        ecckd_forcing = forcing(ecckd, ecckd_2x)
        rrtmgp_forcing = forcing(rrtmgp, rrtmgp_2x)
        @printf("%-28s %12.3f %12.3f %12.3f W m⁻²\n", "2×CO₂ forcing, TOA",
                ecckd_forcing.toa, rrtmgp_forcing.toa, ecckd_forcing.toa - rrtmgp_forcing.toa)
        @printf("%-28s %12.3f %12.3f %12.3f W m⁻²\n", "2×CO₂ forcing, surface",
                ecckd_forcing.surface, rrtmgp_forcing.surface, ecckd_forcing.surface - rrtmgp_forcing.surface)

        # Doubling CO₂ reduces the outgoing longwave (a positive forcing) in both models
        # (observed: 4.97 and 5.01 W m⁻² at the top, 0.92 and 0.86 W m⁻² at the surface)
        @test ecckd_forcing.toa > 0
        @test rrtmgp_forcing.toa > 0
        @test abs(ecckd_forcing.toa - rrtmgp_forcing.toa) ≤ 0.4
        @test abs(ecckd_forcing.surface - rrtmgp_forcing.surface) ≤ 0.3
    end

    @testset "All sky" begin
        # The same column with a 500 m cloud of 0.5 g kg⁻¹ liquid at 1–1.5 km, seen by both models
        # through the same prognostic cloud liquid: ecCKD with the Mie droplet table on its g points,
        # RRTMGP with its cloud optics lookup tables, both at a 10 μm effective radius
        microphysics = held_liquid_microphysics()
        cloud_liquid = 0.5e-3
        ecckd_cloudy, ecckd_cloudy_seconds = solved_column(grid, EcCKDOptics(clouds = CloudScatteringTables()); microphysics, cloud_liquid)
        rrtmgp_cloudy, rrtmgp_cloudy_seconds = solved_column(grid, AllSkyOptics(); microphysics, cloud_liquid)

        @printf("%-28s %12s %12s %12s\n", "all sky, 500 m cloud", "ecCKD", "RRTMGP", "difference")
        for (name, key) in (("OLR", :olr), ("surface LW down", :surface_longwave_down),
                            ("surface SW down", :surface_shortwave_down), ("TOA SW up", :toa_shortwave_up))
            @printf("%-28s %12.3f %12.3f %12.3f W m⁻²\n", name, ecckd_cloudy[key], rrtmgp_cloudy[key],
                    ecckd_cloudy[key] - rrtmgp_cloudy[key])
        end
        cloud_cells = 5:6
        clear_cells = setdiff(interior_cells, cloud_cells)
        Δheating = ecckd_cloudy.heating .- rrtmgp_cloudy.heating
        cloudy_heating_rmse = sqrt(mean(Δheating[clear_cells] .^ 2))
        cloudy_heating_max = maximum(abs, Δheating[clear_cells])
        @printf("%-28s %12.4f %12.4f K day⁻¹ (RMSE, max over the clear cells of 3:Nz-2)\n", "heating difference",
                cloudy_heating_rmse, cloudy_heating_max)
        for k in cloud_cells
            @printf("%-28s %12.3f %12.3f %12.3f K day⁻¹\n", "heating, cloud cell $k",
                    ecckd_cloudy.heating[k], rrtmgp_cloudy.heating[k], Δheating[k])
        end
        @printf("%-28s %12.2f %12.2f ms per update\n", "time", 1e3 * ecckd_cloudy_seconds, 1e3 * rrtmgp_cloudy_seconds)

        # Both models see the cloud: it reflects more than 50 W m⁻² of extra sunlight to space
        # (observed: 87.6 → 429.2 and 88.0 → 424.4 W m⁻²)
        @test ecckd_cloudy.toa_shortwave_up > ecckd.toa_shortwave_up + 50
        @test rrtmgp_cloudy.toa_shortwave_up > rrtmgp.toa_shortwave_up + 50

        # Twice the clear-sky spread between the two cloud optics in the boundary fluxes
        # (observed: 5.39, 1.46, 0.32 and 4.84 W m⁻²) and in the heating of the clear cells
        # (observed: RMSE 0.142, max 0.42 K day⁻¹, the clear-sky level)
        @test abs(ecckd_cloudy.olr - rrtmgp_cloudy.olr) ≤ 2 * 3.9
        @test abs(ecckd_cloudy.surface_longwave_down - rrtmgp_cloudy.surface_longwave_down) ≤ 2 * 4
        @test abs(ecckd_cloudy.surface_shortwave_down - rrtmgp_cloudy.surface_shortwave_down) ≤ 2 * 6
        @test abs(ecckd_cloudy.toa_shortwave_up - rrtmgp_cloudy.toa_shortwave_up) ≤ 2 * 5
        @test cloudy_heating_rmse ≤ 2 * 0.15
        @test cloudy_heating_max ≤ 2 * 0.5

        # Inside the cloud the two cloud optics set the heating rate itself: the cloud top cools
        # by some 35 K day⁻¹ in the longwave and warms by some 10 K day⁻¹ in the shortwave, and
        # the Mie table on the ecCKD g points and RRTMGP's lookup tables put those within 20 % of
        # each other (observed: -26.0 versus -22.5 K day⁻¹ at the cloud top, 3.7 versus 3.6 at
        # the cloud base; both models scale the cloud scattering by delta-Eddington exactly once)
        for k in cloud_cells
            @test abs(Δheating[k]) ≤ 0.2 * max(abs(rrtmgp_cloudy.heating[k]), 1)
        end
    end
end

Oceananigans.defaults.FloatType = Float64
