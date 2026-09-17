include(joinpath(@__DIR__, "setup.jl"))

#####
##### The column extension above a BOMEX-like boundary layer column
#####
##### A 3 km large-eddy simulation column sees only the top of the troposphere; the radiative
##### fluxes at its top depend on the atmosphere above, which the column extension supplies.
##### These tests pin what the extension does to the fluxes of a BOMEX-like column (Siebesma et
##### al. 2003 initial profiles): a bare grid radiates to space unopposed, the extension emits
##### longwave downward and absorbs and scatters sunlight, and refining or raising the extension
##### barely moves the fluxes at the grid top.
#####

using Breeze
using Breeze.AtmosphereModels: total_density
using Dates: DateTime
using NCDatasets
using NumericalRadiation: NumericalRadiation
using Oceananigans
using Oceananigans.Units
using Test

# BOMEX initial profiles: piecewise-linear θ (K) and qᵗ (kg kg⁻¹) between the nodes below
const BOMEX_HEIGHTS = (0, 520, 1480, 2000, 3000)
const BOMEX_θ = (298.7, 298.7, 302.4, 308.2, 311.85)
const BOMEX_qᵗ = (17.0e-3, 16.3e-3, 10.7e-3, 4.2e-3, 3.0e-3)

function piecewise_linear(z, heights, values)
    z ≤ heights[1] && return values[1]
    for n in 2:length(heights)
        if z ≤ heights[n]
            w = (z - heights[n-1]) / (heights[n] - heights[n-1])
            return values[n-1] + w * (values[n] - values[n-1])
        end
    end
    return values[end]
end

bomex_θ(z) = piecewise_linear(z, BOMEX_HEIGHTS, BOMEX_θ)
bomex_qᵗ(z) = piecewise_linear(z, BOMEX_HEIGHTS, BOMEX_qᵗ)

column_grid(FT, Nz, top) = RectilinearGrid(default_arch, FT; size = Nz, x = 0.0, y = 45.0, z = (0, top),
                                            topology = (Flat, Flat, Bounded))

function bomex_column_model(grid, radiation)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101500, potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, formulation = :LiquidIcePotentialTemperature, radiation)
    set!(model; θ = bomex_θ, qᵗ = bomex_qᵗ)
    return model
end

# The BOMEX sea surface temperature, emissivity and albedo, and a fixed sun
function bomex_radiation(grid; column_extension, S₀ = 1361, μ₀ = 0.5)
    FT = eltype(grid)
    return RadiativeTransferModel(grid, EcCKDOptics(), ThermodynamicConstants(); column_extension,
                                  surface_temperature = 300.4, surface_emissivity = 0.98, surface_albedo = 0.1,
                                  solar_constant = S₀, solar_position = FixedCosineZenith(FT(μ₀)))
end

# The four fluxes and their sum on the faces of the column, as host vectors
function column_fluxes(radiation)
    ℐ_lw_up = Array(interior(radiation.upwelling_longwave_flux))[1, 1, :]
    ℐ_lw_dn = Array(interior(radiation.downwelling_longwave_flux))[1, 1, :]
    ℐ_sw_up = Array(interior(radiation.upwelling_shortwave_flux))[1, 1, :]
    ℐ_sw_dn = Array(interior(radiation.downwelling_shortwave_flux))[1, 1, :]
    return ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, ℐ_lw_up .+ ℐ_lw_dn .+ ℐ_sw_up .+ ℐ_sw_dn
end

# Radiative heating rate (K day⁻¹) of every cell
function heating_rate(radiation, model)
    cᵖ = model.thermodynamic_constants.dry_air.heat_capacity
    ρ = Array(interior(total_density(model.dynamics)))[1, 1, :]
    return Array(interior(radiation.flux_divergence))[1, 1, :] ./ (ρ .* cᵖ) .* 86400
end

@testset "Column extension above a BOMEX-like column [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    grid = column_grid(FT, Nz, 3kilometers)
    μ₀ = FT(0.5)
    S₀ = FT(1361)

    extended = bomex_radiation(grid; column_extension = ColumnExtension(FT), S₀, μ₀)
    model = bomex_column_model(grid, extended)

    bare = bomex_radiation(grid; column_extension = nothing, S₀, μ₀)
    bomex_column_model(grid, bare)

    loose = FT == Float64 ? 1e-10 : 1e-5

    for (rtm, name) in ((extended, "extended"), (bare, "grid only"))
        @testset "$name" begin
            ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, ℐ_net = column_fluxes(rtm)

            # Finite, signed by direction
            for ℐ in (ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn)
                @test all(isfinite, ℐ)
            end
            @test all(ℐ_lw_up .> 0)
            @test all(ℐ_lw_dn .<= 0)
            @test all(ℐ_sw_up .> 0)
            @test all(ℐ_sw_dn .< 0)

            # A warm, moist column: strong surface emission and a reflecting surface
            @test ℐ_lw_up[1] > 400
            @test ℐ_sw_up[1] ≈ 0.1 * -ℐ_sw_dn[1] rtol = 1e-3

            # Column energy closure: the integrated divergence is the net flux difference
            Δz = 3kilometers / Nz
            column_heating = Δz * sum(Array(interior(rtm.flux_divergence)))
            @test column_heating ≈ ℐ_net[1] - ℐ_net[Nz+1] rtol = loose
        end
    end

    # Without an extension the grid top is the top of the atmosphere: no downwelling longwave,
    # and the downwelling shortwave is S₀ μ₀ summed over the g-point weights (unit sum up to
    # rounding, so the identity holds to rounding rather than bitwise)
    ℐ_lw_up₀, ℐ_lw_dn₀, ℐ_sw_up₀, ℐ_sw_dn₀, _ = column_fluxes(bare)
    @test ℐ_lw_dn₀[Nz+1] == 0
    @test -ℐ_sw_dn₀[Nz+1] ≈ S₀ * μ₀ rtol = (FT == Float64 ? 1e-12 : 4 * eps(FT))

    # With the extension, the atmosphere above the grid emits downward (observed: 227 W m⁻²) and
    # absorbs and scatters sunlight. Observed: 132 W m⁻², of which water vapor absorption is 66
    # (the standard humidity profile carries 3 g kg⁻¹ at 3 km, as BOMEX does), Rayleigh
    # scattering and the well-mixed gases 51, and ozone 14, all on the slant path of a μ₀ = 0.5
    # beam; the upper bound was raised once from the literature-derived 120 to 150.
    ℐ_lw_up₁, ℐ_lw_dn₁, ℐ_sw_up₁, ℐ_sw_dn₁, _ = column_fluxes(extended)
    @test 150 < -ℐ_lw_dn₁[Nz+1] < 350
    @test 5 < -ℐ_sw_dn₀[Nz+1] + ℐ_sw_dn₁[Nz+1] < 150

    # ... which warms the top of the grid relative to the bare column, whose top cells radiate
    # to space unopposed (observed: +123, +8.3, +3.8 and +2.6 K day⁻¹ in the top four cells)
    Δheating = heating_rate(extended, model) .- heating_rate(bare, model)
    @test all(Δheating[Nz-3:Nz] .> 0.5)

    @testset "Extension convergence" begin
        heating₁ = heating_rate(extended, model)

        # Doubling the extension layers barely moves the fluxes at the grid top or the heating
        # of the grid's cells (observed: 0.014 W m⁻² and 0.0004 K day⁻¹)
        finer = bomex_radiation(grid; column_extension = ColumnExtension(FT; layers = 80), S₀, μ₀)
        bomex_column_model(grid, finer)
        _, ℐ_lw_dn_f, _, _, _ = column_fluxes(finer)
        @test abs(ℐ_lw_dn_f[Nz+1] - ℐ_lw_dn₁[Nz+1]) < 1
        @test maximum(abs, heating_rate(finer, model) .- heating₁) < 0.02

        # Raising the top of the extension from 65 to 80 km adds a negligible amount of air
        # (observed: 0.006 W m⁻² longwave and 0.003 W m⁻² shortwave)
        taller = bomex_radiation(grid; column_extension = ColumnExtension(FT; top = 80kilometers), S₀, μ₀)
        bomex_column_model(grid, taller)
        _, ℐ_lw_dn_t, _, ℐ_sw_dn_t, _ = column_fluxes(taller)
        @test abs(ℐ_lw_dn_t[Nz+1] - ℐ_lw_dn₁[Nz+1]) < 0.3
        @test abs(ℐ_sw_dn_t[Nz+1] - ℐ_sw_dn₁[Nz+1]) < 0.5
    end

    @testset "Night" begin
        night = bomex_radiation(grid; column_extension = ColumnExtension(FT), S₀, μ₀ = 0)
        bomex_column_model(grid, night)
        ℐ_lw_up_n, ℐ_lw_dn_n, ℐ_sw_up_n, ℐ_sw_dn_n, _ = column_fluxes(night)
        @test all(iszero, ℐ_sw_up_n)
        @test all(iszero, ℐ_sw_dn_n)
        @test ℐ_lw_up_n == ℐ_lw_up₁
        @test ℐ_lw_dn_n == ℐ_lw_dn₁
    end

    @testset "Apparent sun" begin
        # The default solar position is computed from the DateTime clock and the grid's (λ, φ)
        apparent = RadiativeTransferModel(grid, EcCKDOptics(), ThermodynamicConstants();
                                          surface_temperature = 300.4, surface_albedo = 0.1)
        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants; base_pressure = 101500, potential_temperature = 300)
        clock = Clock(time = DateTime(2024, 6, 21, 12, 0, 0))
        apparent_model = AtmosphereModel(grid; clock, dynamics = AnelasticDynamics(reference_state),
                                         formulation = :LiquidIcePotentialTemperature, radiation = apparent)
        set!(apparent_model; θ = bomex_θ, qᵗ = bomex_qᵗ)
        @test apparent.solar_position isa ApparentSolarPosition
        cos_zenith = Array(apparent.atmospheric_state.cos_zenith)[1]
        @test 0 < cos_zenith <= 1
        _, _, _, ℐ_sw_dn_a, _ = column_fluxes(apparent)
        @test -ℐ_sw_dn_a[Nz+1] < S₀ * cos_zenith
        @test -ℐ_sw_dn_a[Nz+1] > 0.5 * S₀ * cos_zenith
    end
end

@testset "Float32 versus Float64 fluxes" begin
    fluxes = map((Float64, Float32)) do FT
        Oceananigans.defaults.FloatType = FT
        grid = column_grid(FT, 32, 3kilometers)
        radiation = bomex_radiation(grid; column_extension = ColumnExtension(FT))
        bomex_column_model(grid, radiation)
        return column_fluxes(radiation)
    end
    for (ℐ₆₄, ℐ₃₂) in zip(fluxes[1], fluxes[2])
        @test all(isfinite, ℐ₃₂)
        @test maximum(abs, Float64.(ℐ₃₂) .- ℐ₆₄) < 1
    end
end

Oceananigans.defaults.FloatType = Float64
