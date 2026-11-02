include(joinpath(@__DIR__, "setup.jl"))

#####
##### ecCKD RadiativeTransferModel through the NumericalRadiation extension
#####

using Breeze
using Breeze.AtmosphereModels: standard_ozone_profile, top_face_pressure, bottom_face_pressure,
                               dynamics_pressure, total_density, specific_humidity, materialize_background_atmosphere,
                               _update_radiation!
using Adapt: adapt
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
using .NumericalRadiationExt: ColumnWorkspace, EcCKDLongwave, EcCKDShortwave,
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


# The four fluxes and their sum on the faces of column (i, j), as host vectors
function column_fluxes(radiation, i = 1, j = 1)
    ℐ_lw_up = Array(interior(radiation.upwelling_longwave_flux))[i, j, :]
    ℐ_lw_dn = Array(interior(radiation.downwelling_longwave_flux))[i, j, :]
    ℐ_sw_up = Array(interior(radiation.upwelling_shortwave_flux))[i, j, :]
    ℐ_sw_dn = Array(interior(radiation.downwelling_shortwave_flux))[i, j, :]
    return ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, ℐ_lw_up .+ ℐ_lw_dn .+ ℐ_sw_up .+ ℐ_sw_dn
end

previous_float_type = Oceananigans.defaults.FloatType

@testset "ecCKD RadiativeTransferModel" begin

    @testset "Constructor argument errors" begin
        Oceananigans.defaults.FloatType = Float64
        grid = radiation_column_grid(Float64, 4, 10kilometers)
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

        # The extension needs an ozone profile it can evaluate above the grid
        materialized = materialize_background_atmosphere(BackgroundAtmosphere(), grid)
        @test_throws ArgumentError RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                                          surface_temperature = 300, surface_albedo = 0.1,
                                                          background_atmosphere = materialized)
    end

    @testset "Construction and types [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        grid = radiation_column_grid(FT, 8, 3kilometers)
        constants = ThermodynamicConstants()

        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_albedo = 0.1,
                                           solar_position = FixedCosineZenith(0.5))

        @test radiation isa EcCKDRadiativeTransferModel
        @test radiation.atmospheric_state isa ColumnWorkspace
        @test radiation.atmospheric_state.extension isa OverlyingAtmosphereColumn{FT}
        @test radiation.longwave_solver isa EcCKDLongwave
        @test radiation.shortwave_solver isa EcCKDShortwave
        @test number_of_layers(radiation.atmospheric_state) == 8 + 40
        @test length(radiation.longwave_solver.weights) == 32
        @test length(radiation.shortwave_solver.weights) == 32
        @test radiation.solar_constant isa FT
        @test radiation.shortwave_solver.solar_constant isa FT
        @test radiation.longwave_solver.mole_fractions.CO₂ == FT(420e-6)
        @test radiation.surface_radiation.surface_temperature.constant == FT(300)

        # Concretely typed throughout
        for T in (typeof(radiation.atmospheric_state), typeof(radiation.longwave_solver), typeof(radiation.shortwave_solver))
            @test all(isconcretetype, fieldtypes(T))
        end

        @testset "show" begin
            str = sprint(show, radiation)
            @test occursin("├── optics: EcCKDOptics with 32 longwave and 32 shortwave g-points, clear sky", str)
            @test occursin("└── overlying_atmosphere: 40 layers from 3000.0 m to 65000.0 m", str)
            @test occursin("├── diffuse_surface_albedo: ConstantField(0.1)", str)
        end

        # No extension: the column stops at the grid top
        bare = RadiativeTransferModel(grid, EcCKDOptics(), constants; overlying_atmosphere = nothing,
                                      surface_temperature = 300, surface_albedo = 0.1)
        @test isnothing(bare.atmospheric_state.extension)
        @test number_of_layers(bare.atmospheric_state) == 8
        @test occursin("└── overlying_atmosphere: none", sprint(show, bare))

        # A grid that already reaches the extension top gets none either
        tall = RadiativeTransferModel(grid, EcCKDOptics(), constants; overlying_atmosphere = OverlyingAtmosphereColumn(FT; top = 3kilometers),
                                      surface_temperature = 300, surface_albedo = 0.1)
        @test isnothing(tall.atmospheric_state.extension)

        # A preloaded gas optics model is converted to the grid's float type
        gas_model = NumericalRadiation.read_reference_ecckd_gas_optics(:climate_32x32; names = NumericalRadiationExt.ECCKD_GAS_NAMES)
        preloaded = RadiativeTransferModel(grid, EcCKDOptics(gas_model), constants; surface_temperature = 300, surface_albedo = 0.1)
        @test eltype(preloaded.longwave_solver.gas_model) == FT
        @test Array(preloaded.longwave_solver.gas_model.longwave_absorption) == FT.(gas_model.longwave_absorption)
        @test eltype(radiation.longwave_solver.gas_model) == FT

        # A user-supplied extension is sampled in the grid's float type, whatever its own
        # (the default `OverlyingAtmosphereColumn()` is Float64)
        Oceananigans.defaults.FloatType = Float64
        default_extension = RadiativeTransferModel(grid, EcCKDOptics(), constants; overlying_atmosphere = OverlyingAtmosphereColumn(),
                                                   surface_temperature = 300, surface_albedo = 0.1)
        @test default_extension.atmospheric_state.extension isa OverlyingAtmosphereColumn{FT}
        @test eltype(default_extension.atmospheric_state.extension.Δzᵃᵃᶜ) == FT
        Oceananigans.defaults.FloatType = FT

        # Profiles may be numbers, constant with height
        isothermal = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                            overlying_atmosphere = OverlyingAtmosphereColumn(FT; temperature = 220, specific_humidity = 0),
                                            surface_temperature = 300, surface_albedo = 0.1)
        @test all(==(FT(220)), Array(isothermal.atmospheric_state.extension.T))
        @test all(iszero, Array(isothermal.atmospheric_state.extension.qᵛ))
        @test isothermal.atmospheric_state.extension.profile_temperature_at_bottom == FT(220)
    end

    @testset "Column state and overlying atmosphere [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = radiation_column_grid(FT, Nz, 3kilometers)
        constants = ThermodynamicConstants()

        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_emissivity = 0.98,
                                           surface_albedo = 0.1, solar_position = FixedCosineZenith(0.5))
        model = column_model(grid, radiation)

        workspace = radiation.atmospheric_state
        extension = workspace.extension
        Nₑ = length(extension.Δzᵃᵃᶜ)
        N = Nz + Nₑ
        @test number_of_layers(workspace) == N

        g = constants.gravitational_acceleration
        mᵈ = constants.dry_air.molar_mass
        mᵛ = constants.vapor.molar_mass
        Rᵈ = constants.molar_gas_constant / mᵈ

        # The column the kernel solves, evaluated on the host by the same accessors
        atmosphere = column_atmosphere(radiation, model, 1, 1)
        p_lay = atmosphere.pressure_layers
        T_lay = atmosphere.temperature_layers
        p_int = atmosphere.pressure_interfaces
        T_int = atmosphere.temperature_interfaces
        dry_air_moles = atmosphere.gases.composite
        water_vapor_moles = atmosphere.gases.h2o
        ozone_moles = atmosphere.gases.o3

        tight = FT == Float64 ? 1e-12 : 1e-5
        loose = FT == Float64 ? 1e-10 : 1e-4

        # Grid cells sit in the bottom Nz column layers, top-down
        p = Array(interior(dynamics_pressure(model.dynamics)))[1, 1, :]
        T = Array(interior(model.temperature))[1, 1, :]
        ρ = Array(interior(total_density(model.dynamics)))[1, 1, :]
        qᵛ = Array(interior(specific_humidity(model)))[1, 1, :]
        @test p_lay[N:-1:Nₑ+1] == p
        @test T_lay[N:-1:Nₑ+1] == T

        # Interfaces increase in pressure downward; the boundary faces' pressures are extrapolated
        # hydrostatically from the cells; the top face takes the top cell's temperature (the
        # halo mirrors it) and the bottom face continues the lowest cells' gradient
        @test all(>(0), diff(p_int))
        @allowscalar begin
            @test p_int[N+1] == bottom_face_pressure(1, 1, grid, dynamics_pressure(model.dynamics), total_density(model.dynamics), g)
            @test p_int[Nₑ+1] == top_face_pressure(1, 1, grid, dynamics_pressure(model.dynamics), total_density(model.dynamics), g)
        end
        @test T_int[Nₑ+1] == T[Nz]
        @test T_int[N+1] ≈ (3 * T[1] - T[2]) / 2 rtol = tight
        @test T_int[1] == T_lay[1]

        # Column amounts in the dry convention of the ecCKD tables: the composite amount carries
        # the layer's total mass, `Σ dry_air_moles mᵈ == Σ ρ Δz`, and the vapor is counted on top of it
        Δz = 3kilometers / Nz
        @test sum(dry_air_moles[Nₑ+1:N]) * mᵈ ≈ sum(ρ .* Δz) rtol = tight
        @test sum(water_vapor_moles[Nₑ+1:N]) * mᵛ ≈ sum(ρ .* qᵛ .* Δz) rtol = tight

        # Extension layers, bottom-up from the grid top (layer m is column layer Nₑ + 1 - m)
        Δzₑ = Array(extension.Δzᵃᵃᶜ)
        zₑ = Array(extension.zᵃᵃᶜ)
        Tₑ = Array(extension.T)
        qₑ = Array(extension.qᵛ)
        χₑ = qₑ ./ (1 .- qₑ) .* (mᵈ / mᵛ)
        h = extension.blending_height
        anchor = T_int[Nₑ+1] - extension.profile_temperature_at_bottom
        Tₘ = Tₑ .+ anchor .* exp.(-(zₑ .- extension.bottom) ./ h)
        Tᵛ = Tₘ .* (1 .+ (mᵈ / mᵛ - 1) .* qₑ)

        # Hydrostatic interface pressures with the virtual temperature, layer by layer
        p_reference = similar(p_int, Nₑ + 1)
        p_reference[Nₑ+1] = p_int[Nₑ+1]
        for m in 1:Nₑ
            p_reference[Nₑ+1-m] = p_reference[Nₑ+2-m] * exp(-g * Δzₑ[m] / (Rᵈ * Tᵛ[m]))
        end
        @test p_int[1:Nₑ+1] ≈ p_reference rtol = loose
        @test T_lay[Nₑ:-1:1] ≈ Tₘ rtol = tight

        # The extension's interfaces interpolate its anchored layers linearly in height (the face
        # above layer m is column interface Nₑ + 1 - m)
        T_faces = [(Δzₑ[m+1] * Tₘ[m] + Δzₑ[m] * Tₘ[m+1]) / (Δzₑ[m] + Δzₑ[m+1]) for m in 1:Nₑ-1]
        @test T_int[Nₑ:-1:2] ≈ T_faces rtol = tight

        # Dry-convention gas amounts: `dry_air_moles == Δp / (g mᵈ)` and `water_vapor_moles / dry_air_moles == χ`
        Δp = p_int[2:Nₑ+1] .- p_int[1:Nₑ]   # top-down
        @test dry_air_moles[1:Nₑ] ≈ Δp ./ (g * mᵈ) rtol = tight
        @test water_vapor_moles[Nₑ:-1:1] ./ dry_air_moles[Nₑ:-1:1] ≈ χₑ rtol = tight
        @test ozone_moles[Nₑ:-1:1] ./ dry_air_moles[Nₑ:-1:1] ≈ standard_ozone_profile.(zₑ) rtol = tight

        # The extension is clear and reaches the top
        state = NumericalRadiationExt.ColumnState(radiation, model, 0)
        extension_layers = @allowscalar [NumericalRadiationExt.layer_state(1, 1, k, grid, state) for k in 1:Nₑ]
        @test all(layer -> iszero(layer.liquid_water_path) && iszero(layer.ice_water_path), extension_layers)
        @test extension.bottom + sum(Δzₑ) ≈ 65e3 rtol = 100 * eps(FT)
        @test 1 < p_int[1] < 100   # ~10 Pa at 65 km

        # The anchor: the first extension layer is close to the grid's top face, and the
        # profile relaxes to the standard atmosphere far above it
        @test abs(T_lay[Nₑ] - T_int[Nₑ+1]) < 2
        @test T_lay[1] ≈ standard_atmosphere_temperature(zₑ[end]) rtol = 1e-6

        # Host round trip: the column atmosphere is on the host and carries the model's constants
        @test atmosphere.pressure_layers isa Vector{FT}
        @test length(atmosphere.pressure_layers) == N
        @test length(atmosphere.pressure_interfaces) == N + 1
        @test atmosphere.gases.co2 ≈ 420e-6 .* dry_air_moles
        @test atmosphere.surface.temperature == FT(300)
        @test atmosphere.surface.emissivity == FT(0.98)
        @test atmosphere.geometry.cos_zenith == FT(0.5)
        @test atmosphere.constants.gravity == FT(g)
        @test atmosphere.constants.dry_air_molar_mass == FT(mᵈ)
        @test atmosphere.constants.water_molar_mass == FT(mᵛ)
        @test atmosphere.constants.stefan_boltzmann == radiation.longwave_solver.gas_model.stefan_boltzmann
    end

    @testset "Identities [$(FT)]" for FT in test_float_types()
        Oceananigans.defaults.FloatType = FT
        Nz = 16
        grid = radiation_column_grid(FT, Nz, 3kilometers)
        constants = ThermodynamicConstants()
        ε = 0.98
        radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                           surface_temperature = 300, surface_emissivity = ε,
                                           surface_albedo = 0.1, solar_position = FixedCosineZenith(0.5))
        model = column_model(grid, radiation)
        ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, ℐ_net = column_fluxes(radiation)

        # The upwelling longwave at the surface is the surface's emission plus the reflected
        # downwelling flux, `ε Σ w_g B_g(Tₛ) + (1 - ε) ℐ_lw_dn`
        gas_model = adapt(Array, radiation.longwave_solver.gas_model)   # the emission is evaluated on the host
        weights = gas_model.longwave_weights
        emission = sum(weights .* NumericalRadiation.surface_longwave_emission(gas_model, FT(300)))
        @test ℐ_lw_up[1] ≈ ε * emission + (1 - ε) * -ℐ_lw_dn[1] atol = 0.5

        # Without an extension the grid top is the top of the atmosphere: no downwelling longwave
        bare = RadiativeTransferModel(grid, EcCKDOptics(), constants; overlying_atmosphere = nothing,
                                      surface_temperature = 300, surface_emissivity = ε,
                                      surface_albedo = 0.1, solar_position = FixedCosineZenith(0.5))
        column_model(grid, bare)
        @test column_fluxes(bare)[2][Nz+1] == 0

        # Column energy closure: the integrated divergence is the net flux difference
        Δz = 3kilometers / Nz
        column_heating = Δz * sum(Array(interior(radiation.flux_divergence)))
        @test column_heating ≈ ℐ_net[1] - ℐ_net[Nz+1] rtol = (FT == Float64 ? 1e-10 : 1e-5)
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
            single_grid = radiation_column_grid(FT, Nz, 3kilometers)
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

    @testset "Allocation scaling" begin
        # The update allocates only the fixed overhead of its kernel launches, not per column:
        # 1 and 64 columns allocate the same, within 1 KiB
        Oceananigans.defaults.FloatType = Float64
        constants = ThermodynamicConstants()
        allocations = map((1, 64)) do Nx
            grid = RectilinearGrid(default_arch; size = (Nx, 1, 8), x = (0, Nx), y = (0, 1), z = (0, 3kilometers),
                                   topology = (Periodic, Periodic, Bounded))
            radiation = RadiativeTransferModel(grid, EcCKDOptics(), constants;
                                               surface_temperature = 300, surface_albedo = 0.1,
                                               solar_position = FixedCosineZenith(0.5))
            reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
            model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                                    formulation = :LiquidIcePotentialTemperature, radiation)
            set!(model; θ = (x, y, z) -> 300 + 0.01 * z / 1000, qᵗ = (x, y, z) -> 0.015 * exp(-z / 2500))
            _update_radiation!(radiation, model)
            return @allocated _update_radiation!(radiation, model)
        end
        @test abs(allocations[2] - allocations[1]) < 1024
    end

    @testset "Column batches" begin
        # Three batches of 2 rows of 3 columns (the last overlapping the second, since 2 rows do
        # not divide 5) solve the same fluxes as one batch of the whole grid, bit for bit, and
        # hold only one batch of per-column storage
        Oceananigans.defaults.FloatType = Float64
        constants = ThermodynamicConstants()
        grid = RectilinearGrid(default_arch; size = (3, 5, 8), x = (0, 3), y = (0, 5), z = (0, 3kilometers),
                               topology = (Periodic, Periodic, Bounded))
        fluxes = map((3, nothing)) do column_batches
            radiation = RadiativeTransferModel(grid, EcCKDOptics(clouds = CloudScatteringTables()), constants;
                                               surface_temperature = 300, surface_albedo = 0.1,
                                               solar_position = FixedCosineZenith(0.5), column_batches)
            reference_state = ReferenceState(grid, constants; base_pressure = 101325, potential_temperature = 300)
            model = AtmosphereModel(grid; dynamics = AnelasticDynamics(reference_state),
                                    formulation = :LiquidIcePotentialTemperature, radiation,
                                    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()))
            # Every column differs, and the moist ones are cloudy
            θ(x, y, z) = 300 + 0.01 * z / 1000 + 0.1 * x + 0.2 * y
            qᵗ(x, y, z) = (1 + 0.1 * x + 0.3 * y) * 0.012 * exp(-z / 2500)
            set!(model; θ, qᵗ)
            workspace = radiation.atmospheric_state
            return (rows = workspace.rows, columns = NumericalRadiationExt.batch_columns(workspace),
                    lw_up = Array(interior(radiation.upwelling_longwave_flux)),
                    lw_dn = Array(interior(radiation.downwelling_longwave_flux)),
                    sw_up = Array(interior(radiation.upwelling_shortwave_flux)),
                    sw_dn = Array(interior(radiation.downwelling_shortwave_flux)))
        end
        batched, whole = fluxes
        @test batched.rows == 2 && batched.columns == 6
        @test whole.rows == 5 && whole.columns == 15
        @test batched.lw_up == whole.lw_up
        @test batched.lw_dn == whole.lw_dn
        @test batched.sw_up == whole.sw_up
        @test batched.sw_dn == whole.sw_dn
        @test length(unique(batched.lw_up[:, :, end])) == 15
    end

    @testset "Scheduling" begin
        Oceananigans.defaults.FloatType = Float64
        grid = radiation_column_grid(Float64, 8, 3kilometers)
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

Oceananigans.defaults.FloatType = previous_float_type
