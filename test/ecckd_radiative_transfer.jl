include(joinpath(@__DIR__, "setup.jl"))

#####
##### ecCKD RadiativeTransferModel through the NumericalRadiation extension
#####

using Breeze
using Breeze.AtmosphereModels: standard_ozone_profile, top_face_temperature, top_face_pressure, bottom_face_pressure,
                               dynamics_pressure, total_density, specific_humidity, materialize_background_atmosphere
using Dates
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.Units
using Test

# Both radiation extensions load in one session: the RRTMGP optics keep working next to the ecCKD ones.
using ClimaComms
using NCDatasets
using RRTMGP
using NumericalRadiation: NumericalRadiation

const NumericalRadiationExt = Base.get_extension(Breeze, :BreezeNumericalRadiationExt)
using .NumericalRadiationExt: SpectralColumns, MaterializedColumnExtension, EcCKDLongwave, EcCKDShortwave,
                              EcCKDRadiativeTransferModel, column_atmosphere, number_of_layers

# A stand-in for an effective radius model the extension does not support yet
struct VariableRadiusParticles end

# A single-column anelastic model with a warm, moist troposphere and the given radiation
function column_model(grid, radiation; humidity_factor = 1)
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
    dynamics = AnelasticDynamics(reference_state)
    clock = Clock(time = DateTime(2024, 6, 21, 12, 0, 0))
    model = AtmosphereModel(grid; clock, dynamics, formulation = :LiquidIcePotentialTemperature, radiation)
    θ(z) = 300 + 0.01 * z / 1000
    qᵗ(z) = humidity_factor * 0.015 * exp(-z / 2500)
    set!(model; θ, qᵗ)
    return model
end

column_grid(FT, Nz, top) = RectilinearGrid(default_arch, FT; size = Nz, x = 0.0, y = 45.0, z = (0, top),
                                            topology = (Flat, Flat, Bounded))

# The four fluxes and their sum on the faces of column (i, j), as host vectors
function column_fluxes(radiation, i = 1, j = 1)
    ℐ_lw_up = Array(interior(radiation.upwelling_longwave_flux))[i, j, :]
    ℐ_lw_dn = Array(interior(radiation.downwelling_longwave_flux))[i, j, :]
    ℐ_sw_up = Array(interior(radiation.upwelling_shortwave_flux))[i, j, :]
    ℐ_sw_dn = Array(interior(radiation.downwelling_shortwave_flux))[i, j, :]
    return ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, ℐ_lw_up .+ ℐ_lw_dn .+ ℐ_sw_up .+ ℐ_sw_dn
end

@testset "ecCKD RadiativeTransferModel" begin

    @testset "Constructor argument errors" begin
        Oceananigans.defaults.FloatType = Float64
        grid = column_grid(Float64, 4, 10kilometers)
        constants = ThermodynamicConstants()

        # Albedo keyword combinations, as for the RRTMGP optics
        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          direct_surface_albedo = 0.1)

        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          diffuse_surface_albedo = 0.1)

        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300,
                                                          surface_albedo = 0.1,
                                                          direct_surface_albedo = 0.1,
                                                          diffuse_surface_albedo = 0.1)

        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300,
                                                          direct_surface_albedo = 0.1)

        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300,
                                                          diffuse_surface_albedo = 0.1)

        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300)

        # A gas the ecCKD tables do not carry
        err = try
            RadiativeTransferModel(grid, EcCKDOptics(), constants; surface_temperature = 300, surface_albedo = 0.1,
                                   background_atmosphere = BackgroundAtmosphere(CO = 1e-7))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("CO", err.msg)

        # Only constant effective radii for now
        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300, surface_albedo = 0.1,
                                                          liquid_effective_radius = VariableRadiusParticles())

        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300, surface_albedo = 0.1,
                                                          ice_effective_radius = VariableRadiusParticles())

        # All-sky optics land in a follow-up
        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(clouds = CloudScatteringTables()), constants;
                                                          surface_temperature = 300, surface_albedo = 0.1)

        # The extension needs an ozone profile it can evaluate above the grid
        materialized = materialize_background_atmosphere(BackgroundAtmosphere(), grid)
        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300, surface_albedo = 0.1,
                                                          background_atmosphere = materialized)
    end

    @testset "Construction and types [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        grid = column_grid(FT, 8, 3kilometers)
        constants = ThermodynamicConstants()

        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_albedo = 0.1,
                                           solar_position = FixedCosineZenith(0.5))

        @test radiation isa EcCKDRadiativeTransferModel
        @test radiation.atmospheric_state isa SpectralColumns
        @test radiation.atmospheric_state.extension isa MaterializedColumnExtension
        @test radiation.longwave_solver isa EcCKDLongwave
        @test radiation.shortwave_solver isa EcCKDShortwave
        @test number_of_layers(radiation.atmospheric_state) == 8 + 40
        @test length(radiation.longwave_solver.weights) == 32
        @test length(radiation.shortwave_solver.weights) == 32
        @test radiation.solar_constant isa FT
        @test radiation.shortwave_solver.solar_constant isa FT
        @test radiation.longwave_solver.mole_fractions.co2 == FT(420e-6)
        @test radiation.surface_radiation.surface_temperature.constant == FT(300)

        # Concretely typed throughout
        for T in (typeof(radiation.atmospheric_state), typeof(radiation.longwave_solver), typeof(radiation.shortwave_solver))
            @test all(isconcretetype, fieldtypes(T))
        end

        @testset "show" begin
            str = sprint(show, radiation)
            @test occursin("├── optics: EcCKDOptics with 32 longwave and 32 shortwave g-points, clear sky", str)
            @test occursin("└── column_extension: 40 layers from 3000.0 m to 65000.0 m", str)
            @test occursin("├── diffuse_surface_albedo: ConstantField(0.1)", str)
        end

        # No extension: the column stops at the grid top
        bare = RadiativeTransferModel(grid, EcCKDOptics(), constants; column_extension = nothing,
                                      surface_temperature = 300, surface_albedo = 0.1)
        @test isnothing(bare.atmospheric_state.extension)
        @test number_of_layers(bare.atmospheric_state) == 8
        @test occursin("└── column_extension: none", sprint(show, bare))

        # A grid that already reaches the extension top gets none either
        tall = RadiativeTransferModel(grid, EcCKDOptics(), constants; column_extension = ColumnExtension(FT; top = 3kilometers),
                                      surface_temperature = 300, surface_albedo = 0.1)
        @test isnothing(tall.atmospheric_state.extension)

        # A preloaded gas optics model passes straight through
        gas_model = NumericalRadiation.read_reference_ecckd_gas_optics(:climate_32x32; names = NumericalRadiationExt.ECCKD_GAS_NAMES)
        preloaded = RadiativeTransferModel(grid, EcCKDOptics(gas_model), constants; surface_temperature = 300, surface_albedo = 0.1)
        @test preloaded.longwave_solver.gas_model === gas_model
    end

    @testset "Staged columns and column extension [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = column_grid(FT, Nz, 3kilometers)
        constants = ThermodynamicConstants()

        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_emissivity = 0.98,
                                           surface_albedo = 0.1, solar_position = FixedCosineZenith(0.5))
        model = column_model(grid, radiation)

        columns = radiation.atmospheric_state
        extension = columns.extension
        Nₑ = length(extension.Δz)
        N = Nz + Nₑ
        @test number_of_layers(columns) == N

        g = constants.gravitational_acceleration
        Mᵈ = constants.dry_air.molar_mass
        Mᵛ = constants.vapor.molar_mass
        Rᵈ = constants.molar_gas_constant / Mᵈ

        p_lay = Array(columns.pressure_layers)[1, :]
        T_lay = Array(columns.temperature_layers)[1, :]
        p_int = Array(columns.pressure_interfaces)[1, :]
        T_int = Array(columns.temperature_interfaces)[1, :]
        n_dry = Array(columns.dry_air)[1, :]
        n_h2o = Array(columns.water_vapor)[1, :]
        n_o3 = Array(columns.ozone)[1, :]

        tight = FT == Float64 ? 1e-12 : 1e-5
        loose = FT == Float64 ? 1e-10 : 1e-4

        # Grid cells sit in the bottom Nz column layers, top-down
        p = Array(interior(dynamics_pressure(model.dynamics)))[1, 1, :]
        T = Array(interior(model.temperature))[1, 1, :]
        ρ = Array(interior(total_density(model.dynamics)))[1, 1, :]
        qᵛ = Array(interior(specific_humidity(model)))[1, 1, :]
        @test p_lay[N:-1:Nₑ+1] == p
        @test T_lay[N:-1:Nₑ+1] == T

        # Interfaces increase in pressure downward, and the boundary faces are extrapolated from the cells
        @test all(diff(p_int) .> 0)
        @test p_int[N+1] == bottom_face_pressure(1, 1, grid, dynamics_pressure(model.dynamics), total_density(model.dynamics), g)
        @test p_int[Nₑ+1] == top_face_pressure(1, 1, grid, dynamics_pressure(model.dynamics), total_density(model.dynamics), g)
        @test T_int[Nₑ+1] == top_face_temperature(1, 1, grid, model.temperature)
        @test T_int[1] == T_lay[1]

        # Dry mass of the grid's column: Σ n_dry Mᵈ == Σ ρ (1 - qᵗ) Δz
        Δz = 3kilometers / Nz
        @test sum(n_dry[Nₑ+1:N]) * Mᵈ ≈ sum(ρ .* (1 .- qᵛ) .* Δz) rtol = tight
        @test sum(n_h2o[Nₑ+1:N]) * Mᵛ ≈ sum(ρ .* qᵛ .* Δz) rtol = tight

        # Extension layers, bottom-up from the grid top (layer m is column layer Nₑ + 1 - m)
        Δzₑ = Array(extension.Δz)
        zₑ = Array(extension.z_layer)
        Tₑ = Array(extension.temperature_layers)
        qₑ = Array(extension.specific_humidity)
        χₑ = qₑ ./ (1 .- qₑ) .* (Mᵈ / Mᵛ)
        h = extension.blending_height
        anchor = T_int[Nₑ+1] - extension.join_temperature
        Tₘ = Tₑ .+ anchor .* exp.(-(zₑ .- extension.base) ./ h)
        Tᵥ = Tₘ .* (1 .+ (Mᵈ / Mᵛ - 1) .* qₑ)

        # Hydrostatic interface pressures with the virtual temperature, layer by layer
        p_reference = similar(p_int, Nₑ + 1)
        p_reference[Nₑ+1] = p_int[Nₑ+1]
        for m in 1:Nₑ
            p_reference[Nₑ+1-m] = p_reference[Nₑ+2-m] * exp(-g * Δzₑ[m] / (Rᵈ * Tᵥ[m]))
        end
        @test p_int[1:Nₑ+1] ≈ p_reference rtol = loose
        @test T_lay[Nₑ:-1:1] ≈ Tₘ rtol = tight

        # Moist-molar-mass gas amounts: Mᵈ n_dry + Mᵛ n_h2o == Δp / g and n_h2o / n_dry == χ
        Δp = p_int[2:Nₑ+1] .- p_int[1:Nₑ]   # top-down
        @test Mᵈ .* n_dry[1:Nₑ] .+ Mᵛ .* n_h2o[1:Nₑ] ≈ Δp ./ g rtol = tight
        @test n_h2o[Nₑ:-1:1] ./ n_dry[Nₑ:-1:1] ≈ χₑ rtol = tight
        @test n_o3[Nₑ:-1:1] ./ n_dry[Nₑ:-1:1] ≈ standard_ozone_profile.(zₑ) rtol = tight

        # The extension is clear and reaches the top
        @test all(iszero, Array(columns.liquid_water_path)[1, 1:Nₑ])
        @test all(iszero, Array(columns.ice_water_path)[1, 1:Nₑ])
        @test extension.base + sum(Δzₑ) ≈ 65e3 rtol = 100 * eps(FT)
        @test 1 < p_int[1] < 100   # ~10 Pa at 65 km

        # The anchor: the first extension layer is close to the grid's top face, and the
        # profile relaxes to the standard atmosphere far above it
        @test abs(T_lay[Nₑ] - T_int[Nₑ+1]) < 2
        @test T_lay[1] ≈ standard_atmosphere_temperature(zₑ[end]) rtol = 1e-6

        # Host round trip: the column atmosphere views the staged rows
        atmosphere = column_atmosphere(radiation, 1, 1)
        @test length(atmosphere.pressure_layers) == N
        @test length(atmosphere.pressure_interfaces) == N + 1
        @test atmosphere.gases.composite == n_dry
        @test atmosphere.gases.co2 ≈ 420e-6 .* n_dry
        @test atmosphere.surface.temperature == FT(300)
        @test atmosphere.surface.emissivity == FT(0.98)
        @test atmosphere.geometry.cos_zenith == FT(0.5)
    end

    @testset "Fluxes [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = column_grid(FT, Nz, 3kilometers)
        constants = ThermodynamicConstants()
        μ0 = FT(0.5)
        S0 = FT(1361)

        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_emissivity = 0.98,
                                           surface_albedo = 0.1, solar_constant = S0,
                                           solar_position = FixedCosineZenith(μ0))
        model = column_model(grid, radiation)

        bare = RadiativeTransferModel(grid, EcCKDOptics(), constants; column_extension = nothing,
                                      surface_temperature = 300, surface_emissivity = 0.98,
                                      surface_albedo = 0.1, solar_constant = S0,
                                      solar_position = FixedCosineZenith(μ0))
        bare_model = column_model(grid, bare)

        loose = FT == Float64 ? 1e-10 : 1e-5

        for (rtm, name) in ((radiation, "extended"), (bare, "grid only"))
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

        # Without an extension the grid top is the top of the atmosphere
        ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, _ = column_fluxes(bare)
        @test ℐ_lw_dn[Nz+1] == 0
        # S0 μ0 summed over the g-point weights (unit sum up to rounding), stored in the grid's float type
        @test -ℐ_sw_dn[Nz+1] ≈ S0 * μ0 rtol = (FT == Float64 ? 1e-12 : 4 * eps(FT))

        # With the extension, the atmosphere above the grid emits downward (203 W m⁻² observed) and
        # absorbs and scatters sunlight (131 W m⁻² observed: water vapor and ozone absorption plus
        # Rayleigh scattering of a μ0 = 0.5 beam)
        ℐ_lw_up_e, ℐ_lw_dn_e, ℐ_sw_up_e, ℐ_sw_dn_e, _ = column_fluxes(radiation)
        @test 150 < -ℐ_lw_dn_e[Nz+1] < 350
        @test 5 < -ℐ_sw_dn[Nz+1] + ℐ_sw_dn_e[Nz+1] < 150
        # ... which warms the top of the grid relative to the bare column, whose top cells radiate to
        # space unopposed (observed: +61 and +1.4 K day⁻¹ in the two top cells, +0.2 and +0.09 below)
        cᵖ = constants.dry_air.heat_capacity
        heating(rtm) = Array(interior(rtm.flux_divergence))[1, 1, :] ./
                       (Array(interior(total_density(model.dynamics)))[1, 1, :] .* cᵖ) .* 86400
        @test all(heating(radiation)[Nz-1:Nz] .- heating(bare)[Nz-1:Nz] .> 0.5)

        @testset "extension convergence" begin
            # Doubling the extension layers or raising its top barely moves the fluxes at the grid top
            finer = RadiativeTransferModel(grid, EcCKDOptics(), constants; column_extension = ColumnExtension(FT; layers = 80),
                                           surface_temperature = 300, surface_emissivity = 0.98,
                                           surface_albedo = 0.1, solar_constant = S0, solar_position = FixedCosineZenith(μ0))
            column_model(grid, finer)
            _, ℐ_lw_dn_f, _, _, _ = column_fluxes(finer)
            @test abs(ℐ_lw_dn_f[Nz+1] - ℐ_lw_dn_e[Nz+1]) < 1
            @test maximum(abs, heating(finer) .- heating(radiation)) < 0.02

            taller = RadiativeTransferModel(grid, EcCKDOptics(), constants; column_extension = ColumnExtension(FT; top = 80kilometers),
                                            surface_temperature = 300, surface_emissivity = 0.98,
                                            surface_albedo = 0.1, solar_constant = S0, solar_position = FixedCosineZenith(μ0))
            column_model(grid, taller)
            _, ℐ_lw_dn_t, _, ℐ_sw_dn_t, _ = column_fluxes(taller)
            @test abs(ℐ_lw_dn_t[Nz+1] - ℐ_lw_dn_e[Nz+1]) < 0.3
            @test abs(ℐ_sw_dn_t[Nz+1] - ℐ_sw_dn_e[Nz+1]) < 0.5
        end

        @testset "night" begin
            night = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_emissivity = 0.98,
                                           surface_albedo = 0.1, solar_constant = S0,
                                           solar_position = FixedCosineZenith(0))
            column_model(grid, night)
            ℐ_lw_up_n, ℐ_lw_dn_n, ℐ_sw_up_n, ℐ_sw_dn_n, _ = column_fluxes(night)
            @test all(iszero, ℐ_sw_up_n)
            @test all(iszero, ℐ_sw_dn_n)
            @test ℐ_lw_up_n == ℐ_lw_up_e
            @test ℐ_lw_dn_n == ℐ_lw_dn_e
        end

        @testset "apparent sun" begin
            # The default solar position is computed from the DateTime clock and the grid's (λ, φ)
            apparent = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                              surface_temperature = 300, surface_albedo = 0.1)
            column_model(grid, apparent)
            @test apparent.solar_position isa ApparentSolarPosition
            cos_zenith = Array(apparent.atmospheric_state.cos_zenith)[1]
            @test 0 < cos_zenith <= 1
            _, _, _, ℐ_sw_dn_a, _ = column_fluxes(apparent)
            @test -ℐ_sw_dn_a[Nz+1] < S0 * cos_zenith
            @test -ℐ_sw_dn_a[Nz+1] > 0.5 * S0 * cos_zenith
        end
    end

    @testset "Several columns [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 8
        grid = RectilinearGrid(default_arch, FT; size = (2, 1, Nz), x = (0, 2), y = (0, 1), z = (0, 3kilometers),
                               topology = (Periodic, Periodic, Bounded))
        constants = ThermodynamicConstants()
        reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
        dynamics = AnelasticDynamics(reference_state)

        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_albedo = 0.1,
                                           solar_position = FixedCosineZenith(0.5))
        model = AtmosphereModel(grid; dynamics, formulation = :LiquidIcePotentialTemperature, radiation)

        # Column 2 is twice as humid as column 1
        θ(x, y, z) = 300 + 0.01 * z / 1000
        qᵗ(x, y, z) = (x < 1 ? 1 : 2) * 0.015 * exp(-z / 2500)
        set!(model; θ, qᵗ)

        ℐ_lw_up₁, ℐ_lw_dn₁, ℐ_sw_up₁, ℐ_sw_dn₁, _ = column_fluxes(radiation, 1, 1)
        ℐ_lw_up₂, ℐ_lw_dn₂, ℐ_sw_up₂, ℐ_sw_dn₂, _ = column_fluxes(radiation, 2, 1)
        @test all(isfinite, ℐ_lw_dn₁) && all(isfinite, ℐ_lw_dn₂)
        @test ℐ_lw_dn₂[1] < ℐ_lw_dn₁[1]   # the humid column emits more toward the surface
        @test -ℐ_sw_dn₂[1] < -ℐ_sw_dn₁[1] # and absorbs more sunlight

        # Each column reproduces the single-column model with the same profile, bit for bit
        for (i, humidity_factor) in ((1, 1), (2, 2))
            single_grid = column_grid(FT, Nz, 3kilometers)
            single = RadiativeTransferModel(single_grid, EcCKDOptics(), constants;
                                            surface_temperature = 300, surface_albedo = 0.1,
                                            solar_position = FixedCosineZenith(0.5))
            column_model(single_grid, single; humidity_factor)
            ℐ_lw_upₛ, ℐ_lw_dnₛ, ℐ_sw_upₛ, ℐ_sw_dnₛ, _ = column_fluxes(single)
            ℐ_lw_upᵢ, ℐ_lw_dnᵢ, ℐ_sw_upᵢ, ℐ_sw_dnᵢ, _ = column_fluxes(radiation, i, 1)
            @test ℐ_lw_upᵢ == ℐ_lw_upₛ
            @test ℐ_lw_dnᵢ == ℐ_lw_dnₛ
            @test ℐ_sw_upᵢ == ℐ_sw_upₛ
            @test ℐ_sw_dnᵢ == ℐ_sw_dnₛ
        end
    end

    @testset "Scheduling" begin
        Oceananigans.defaults.FloatType = Float64
        grid = column_grid(Float64, 8, 3kilometers)
        constants = ThermodynamicConstants()
        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_albedo = 0.1,
                                           solar_position = FixedCosineZenith(0.5),
                                           schedule = IterationInterval(3))
        model = column_model(grid, radiation)
        ℐ_lw_up = radiation.upwelling_longwave_flux

        @allowscalar @test ℐ_lw_up[1, 1, 1] > 100
        interior(ℐ_lw_up) .= 0

        model.clock.iteration = 1
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies = false)
        @allowscalar @test ℐ_lw_up[1, 1, 1] == 0

        model.clock.iteration = 3
        Oceananigans.TimeSteppers.update_state!(model; compute_tendencies = false)
        @allowscalar @test ℐ_lw_up[1, 1, 1] > 100
    end
end

Oceananigans.defaults.FloatType = Float64
