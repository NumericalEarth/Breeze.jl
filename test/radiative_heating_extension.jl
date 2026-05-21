using Lightflux
using Breeze
using Oceananigans
using Oceananigans.Fields: CenterField, interior, set!
using Test

@testset "RadiativeHeating extension" begin
    grid = RectilinearGrid(CPU(); size = 4, z = (0, 1), topology = (Flat, Flat, Bounded))
    constants = Breeze.Thermodynamics.ThermodynamicConstants()
    gas_optics = EcCKDGasOpticsModel(
        gas_names = (:h2o, :co2),
        longwave_absorption = fill(0.01, 2, 2),
        shortwave_absorption = fill(0.005, 2, 2),
        longwave_weights = fill(0.5, 2),
        shortwave_weights = fill(0.5, 2),
        longwave_source_scale = fill(1.0, 2),
    )

    radiation = RadiativeTransferModel(grid, RadiativeHeatingOptics(), constants;
                                       gas_optics,
                                       surface_temperature = 300,
                                       surface_albedo = 0.1)
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    workspace = radiation.atmospheric_state.workspace

    @test ext !== nothing
    @test workspace.pressure_layers === workspace.atmosphere.pressure_layers
    @test workspace.temperature_layers === workspace.atmosphere.temperature_layers
    @test size(workspace.longwave_optics.optical_depth) == (2, 4)
    @test size(workspace.shortwave_optics.optical_depth) == (2, 4)
    @test length(workspace.fluxes.longwave_up) == 5

    workspace.pressure_interfaces .= range(1000, 100000; length = 5)
    workspace.pressure_layers .= (workspace.pressure_interfaces[1:end-1] .+
                                  workspace.pressure_interfaces[2:end]) ./ 2
    workspace.temperature_layers .= range(220, 300; length = 4)
    workspace.temperature_interfaces .= range(210, 300; length = 5)
    workspace.gases.h2o .= 0.01
    workspace.gases.co2 .= 400e-6

    ext.radiative_heating_column_fluxes!(workspace, radiation)
    ext.column_heating_rates!(workspace, constants)

    @test all(isfinite, workspace.fluxes.longwave_up)
    @test all(isfinite, workspace.fluxes.longwave_down)
    @test all(isfinite, workspace.fluxes.shortwave_up)
    @test all(isfinite, workspace.fluxes.shortwave_down)
    @test all(isfinite, workspace.heating_rates)

    pressure = CenterField(grid)
    temperature = CenterField(grid)
    qᵛ = CenterField(grid)
    set!(pressure, z -> 100000 - 80000z)
    set!(temperature, z -> 300 - 80z)
    set!(qᵛ, 0.01)

    model = (
        grid = grid,
        dynamics = (reference_state = (pressure = pressure,),),
        temperature = temperature,
        microphysical_fields = (qᵛ = qᵛ,),
        clock = (iteration = 0,),
    )

    Breeze.AtmosphereModels.update_radiation!(radiation, model)
    @test all(isfinite, Array(interior(radiation.upwelling_longwave_flux, 1, 1, :)))
    @test all(isfinite, Array(interior(radiation.downwelling_longwave_flux, 1, 1, :)))
    @test all(isfinite, Array(interior(radiation.downwelling_shortwave_flux, 1, 1, :)))
    @test all(isfinite, Array(interior(radiation.flux_divergence, 1, 1, :)))

    fixed_support = ext.radiative_heating_device_support("H100", gas_optics)
    @test fixed_support.status == "supported"
    @test fixed_support.source == "BreezeLightfluxExt"
end

@testset "RadiativeHeating tabulated ecCKD column amounts" begin
    grid = RectilinearGrid(CPU(); size = 2, z = (0, 1), topology = (Flat, Flat, Bounded))
    constants = Breeze.Thermodynamics.ThermodynamicConstants()
    gas_optics = EcCKDTabulatedGasOpticsModel(
        gas_names = (:composite, :h2o, :co2),
        pressure_grid = [1.0e4, 1.0e5],
        temperature_grid = [220.0, 300.0],
        gas_reference_mole_fractions = [0.0, 0.0, 0.0],
        longwave_absorption = fill(1.0e-8, 2, 3, 2, 2),
        shortwave_absorption = fill(5.0e-9, 2, 3, 2, 2),
        longwave_weights = fill(0.5, 2),
        shortwave_weights = fill(0.5, 2),
        longwave_source_scale = fill(1.0, 2),
    )
    radiation = RadiativeTransferModel(grid, RadiativeHeatingOptics(), constants;
                                       gas_optics,
                                       gas_values = (; co2 = 420.0e-6),
                                       surface_temperature = 300,
                                       surface_albedo = 0.1)
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)

    pressure = CenterField(grid)
    temperature = CenterField(grid)
    qᵛ = CenterField(grid)
    set!(pressure, z -> 100000 - 60000z)
    set!(temperature, z -> 300 - 50z)
    set!(qᵛ, 0.01)

    model = (
        grid = grid,
        dynamics = (reference_state = (pressure = pressure,),),
        temperature = temperature,
        microphysical_fields = (qᵛ = qᵛ,),
        clock = (iteration = 0,),
    )

    workspace = radiation.atmospheric_state.workspace
    ext.fill_column_state!(workspace, radiation, model, 1, 1)

    @test all(workspace.gases.composite .> 0)
    @test all(workspace.gases.h2o .> 0)
    @test all(workspace.gases.co2 .> 0)
    @test workspace.gases.co2[1] ≈ 420.0e-6 * workspace.gases.composite[1]
    @test workspace.gases.h2o[1] > 0.01 * workspace.gases.composite[1]

    Breeze.AtmosphereModels.update_radiation!(radiation, model)
    @test all(isfinite, Array(interior(radiation.upwelling_longwave_flux, 1, 1, :)))
    @test all(isfinite, Array(interior(radiation.downwelling_longwave_flux, 1, 1, :)))
    @test all(isfinite, Array(interior(radiation.downwelling_shortwave_flux, 1, 1, :)))
    @test all(isfinite, Array(interior(radiation.flux_divergence, 1, 1, :)))

    withenv("RADIATIVE_HEATING_TABULATED_ECCKD_PARITY_JSON" => nothing) do
        support = ext.radiative_heating_device_support("H100", gas_optics)
        @test support.status == "unsupported"
        @test support.source == "BreezeLightfluxExt"
        @test occursin("parity", support.reason)
        @test !any(contains("column-molar-amount"), support.missing_kernel_requirements)
        @test !any(contains("allocation-free"), support.missing_kernel_requirements)
        @test !any(contains("radiation update path"), support.missing_kernel_requirements)
        @test any(contains("CPU/GPU parity"), support.missing_kernel_requirements)
    end
end

@testset "RadiativeHeating tabulated ecCKD device column-state workspace" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 3),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    gas_optics = EcCKDTabulatedGasOpticsModel(
        gas_names = (:composite, :h2o, :o3, :co2, :ch4, :n2o, :cfc11, :cfc12),
        pressure_grid = [1.0e4, 1.0e5],
        temperature_grid = [220.0, 300.0],
        h2o_mole_fraction_grid = [1.0e-3, 1.0e-2],
        gas_reference_mole_fractions = zeros(8),
        longwave_absorption = fill(1.0e-8, 1, 8, 2, 2),
        shortwave_absorption = fill(5.0e-9, 1, 8, 2, 2),
        longwave_h2o_absorption = fill(1.0e-8, 1, 2, 2, 2),
        shortwave_h2o_absorption = fill(5.0e-9, 1, 2, 2, 2),
        shortwave_rayleigh_molar_scattering = fill(1.0e-8, 1),
        longwave_source_temperature_grid = [200.0, 300.0],
        longwave_source_table = fill(10.0, 1, 2),
        longwave_weights = [1.0],
        shortwave_weights = [1.0],
    )
    workspace = ext.TabulatedEcCKDDeviceWorkspace(CPU(), FT, grid, gas_optics)
    pressure = CenterField(grid)
    temperature = CenterField(grid)
    qᵛ = CenterField(grid)
    set!(pressure, (x, y, z) -> 100000 - 80000z)
    set!(temperature, (x, y, z) -> 300 - 60z)
    set!(qᵛ, 0.01)

    ext.tabulated_ecckd_device_column_state!(
        workspace,
        grid,
        pressure,
        temperature,
        qᵛ;
        co2 = 420.0e-6,
        o3 = 1.0e-8,
        ch4 = 1.8e-6,
        n2o = 330.0e-9,
        cfc11 = 230.0e-12,
        cfc12 = 520.0e-12,
    )

    @test all(workspace.column_amounts.composite .> 0)
    @test all(workspace.column_amounts.h2o .> 0)
    @test workspace.column_amounts.co2[1, 1, 1] ≈ 420.0e-6 * workspace.column_amounts.composite[1, 1, 1]
    @test workspace.pressure_interfaces[1, 1, 1] > workspace.pressure_interfaces[1, 1, end]
    @test all(isfinite, workspace.temperature_layers)
end

@testset "RadiativeHeating tabulated ecCKD column amount kernel" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 3),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    state = ext.TabulatedEcCKDColumnAmountState(FT, (2, 2, 3))
    pressure_interfaces = zeros(FT, 2, 2, 4)
    qᵛ = fill(0.01, 2, 2, 3)
    for i in 1:2, j in 1:2
        pressure_interfaces[i, j, :] .= (1000, 34000, 67000, 100000)
    end

    ext.tabulated_ecckd_column_amounts!(state, grid, pressure_interfaces, qᵛ;
                                        co2 = 420e-6,
                                        o3 = 1e-8,
                                        ch4 = 1.8e-6,
                                        n2o = 330e-9,
                                        cfc11 = 230e-12,
                                        cfc12 = 520e-12)

    expected_dry = (34000 - 1000) / 9.80665 / 0.0289647
    expected_h2o = 0.01 * (34000 - 1000) / 9.80665 / 0.01801528
    @test state.composite[1, 1, 1] ≈ expected_dry
    @test state.h2o[1, 1, 1] ≈ expected_h2o
    @test state.co2[1, 1, 1] ≈ 420e-6 * expected_dry
    @test state.o3[1, 1, 1] ≈ 1e-8 * expected_dry
    @test state.ch4[1, 1, 1] ≈ 1.8e-6 * expected_dry
    @test state.n2o[1, 1, 1] ≈ 330e-9 * expected_dry
    @test state.cfc11[1, 1, 1] ≈ 230e-12 * expected_dry
    @test state.cfc12[1, 1, 1] ≈ 520e-12 * expected_dry
end

@testset "RadiativeHeating tabulated ecCKD interpolation state kernel" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 3),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    amounts = ext.TabulatedEcCKDColumnAmountState(FT, (2, 2, 3))
    interp = ext.TabulatedEcCKDInterpolationState(FT, (2, 2, 3))
    pressure_layers = fill(1.0e4, 2, 2, 3)
    temperature_layers = fill(275.0, 2, 2, 3)
    amounts.composite .= 1000.0
    amounts.h2o .= 10.0

    ext.tabulated_ecckd_interpolation_state!(
        interp,
        grid,
        pressure_layers,
        temperature_layers,
        amounts,
        [1.0e3, 1.0e4, 1.0e5],
        [200.0, 250.0, 300.0],
        [1.0e-3, 1.0e-2, 1.0e-1],
    )

    @test interp.pressure_lo[1, 1, 1] == 2
    @test interp.pressure_hi[1, 1, 1] == 3
    @test interp.pressure_weight[1, 1, 1] ≈ 0
    @test interp.temperature_lo[1, 1, 1] == 2
    @test interp.temperature_hi[1, 1, 1] == 3
    @test interp.temperature_weight[1, 1, 1] ≈ 0.5
    @test interp.h2o_lo[1, 1, 1] == 2
    @test interp.h2o_hi[1, 1, 1] == 3
    @test interp.h2o_weight[1, 1, 1] ≈ 0

    temperature_layers .= 235.0
    ext.tabulated_ecckd_interpolation_state!(
        interp,
        grid,
        pressure_layers,
        temperature_layers,
        amounts,
        [1.0e3, 1.0e4, 1.0e5],
        [200.0 250.0 300.0; 210.0 260.0 310.0; 220.0 270.0 320.0],
        [1.0e-3, 1.0e-2, 1.0e-1],
    )

    @test interp.temperature_lo[1, 1, 1] == 1
    @test interp.temperature_hi[1, 1, 1] == 2
    @test interp.temperature_weight[1, 1, 1] ≈ 0.5
end

@testset "RadiativeHeating tabulated ecCKD absorption optical depth kernel" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 2),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    amounts = ext.TabulatedEcCKDColumnAmountState(FT, (2, 2, 2))
    interp = ext.TabulatedEcCKDInterpolationState(FT, (2, 2, 2))
    amounts.composite .= 1000
    amounts.h2o .= 20
    amounts.o3 .= 3
    amounts.co2 .= 4
    amounts.ch4 .= 5
    amounts.n2o .= 6
    amounts.cfc11 .= 7
    amounts.cfc12 .= 8
    interp.pressure_lo .= 1
    interp.pressure_hi .= 2
    interp.temperature_lo .= 1
    interp.temperature_hi .= 2
    interp.h2o_lo .= 1
    interp.h2o_hi .= 2
    interp.pressure_weight .= 0
    interp.temperature_weight .= 0
    interp.h2o_weight .= 0

    lw_absorption = zeros(FT, 2, 8, 2, 2)
    sw_absorption = zeros(FT, 1, 8, 2, 2)
    for gas in 1:8
        lw_absorption[1, gas, :, :] .= 0.001gas
        lw_absorption[2, gas, :, :] .= 0.002gas
        sw_absorption[1, gas, :, :] .= 0.003gas
    end
    lw_h2o = fill(0.05, 2, 2, 2, 2)
    sw_h2o = fill(0.07, 1, 2, 2, 2)
    gas_reference = zeros(FT, 8)
    gas_reference[4] = 1.0e-3
    lw_tau = zeros(FT, 2, 2, 2, 2)
    sw_tau = zeros(FT, 1, 2, 2, 2)

    ext.tabulated_ecckd_absorption_optical_depths!(
        lw_tau,
        sw_tau,
        grid,
        amounts,
        interp,
        gas_reference,
        lw_absorption,
        sw_absorption,
        lw_h2o,
        sw_h2o,
    )

    adjusted = [1000, 20, 3, 4 - 1.0e-3 * 1000, 5, 6, 7, 8]
    expected_lw1 = sum(0.001gas * adjusted[gas] for gas in 1:8) + 0.05 * 20
    expected_lw2 = sum(0.002gas * adjusted[gas] for gas in 1:8) + 0.05 * 20
    expected_sw = sum(0.003gas * adjusted[gas] for gas in 1:8) + 0.07 * 20
    @test lw_tau[1, 1, 1, 1] ≈ expected_lw1
    @test lw_tau[2, 1, 1, 1] ≈ expected_lw2
    @test sw_tau[1, 1, 1, 1] ≈ expected_sw
end

@testset "RadiativeHeating tabulated ecCKD Rayleigh optical depth kernel" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 2),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    amounts = ext.TabulatedEcCKDColumnAmountState(FT, (2, 2, 2))
    amounts.composite .= 1234.0
    rayleigh = zeros(FT, 2, 2, 2, 2)

    ext.tabulated_ecckd_rayleigh_optical_depths!(
        rayleigh,
        grid,
        amounts,
        [1.0e-8, 2.0e-8],
    )

    @test rayleigh[1, 1, 1, 1] ≈ 1.0e-8 * 1234.0
    @test rayleigh[2, 1, 1, 1] ≈ 2.0e-8 * 1234.0
end

@testset "RadiativeHeating tabulated ecCKD longwave source kernel" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 2),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    source = zeros(FT, 2, 2, 2, 2)
    temperature_layers = fill(250.0, 2, 2, 2)
    source_temperature_grid = [200.0, 300.0]
    source_table = [
        10.0 30.0
        100.0 300.0
    ]

    ext.tabulated_ecckd_longwave_sources!(
        source,
        grid,
        temperature_layers,
        source_temperature_grid,
        source_table,
    )

    @test source[1, 1, 1, 1] ≈ 20.0
    @test source[2, 1, 1, 1] ≈ 200.0

    temperature_layers .= 180.0
    ext.tabulated_ecckd_longwave_sources!(
        source,
        grid,
        temperature_layers,
        source_temperature_grid,
        source_table,
    )
    @test source[1, 1, 1, 1] ≈ 10.0
end

@testset "RadiativeHeating tabulated ecCKD integrated optical properties kernel" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 2),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    amounts = ext.TabulatedEcCKDColumnAmountState(FT, (2, 2, 2))
    interp = ext.TabulatedEcCKDInterpolationState(FT, (2, 2, 2))
    amounts.composite .= 1000
    amounts.h2o .= 20
    amounts.o3 .= 3
    amounts.co2 .= 4
    amounts.ch4 .= 5
    amounts.n2o .= 6
    amounts.cfc11 .= 7
    amounts.cfc12 .= 8
    interp.pressure_lo .= 1
    interp.pressure_hi .= 2
    interp.temperature_lo .= 1
    interp.temperature_hi .= 2
    interp.h2o_lo .= 1
    interp.h2o_hi .= 2
    interp.pressure_weight .= 0
    interp.temperature_weight .= 0
    interp.h2o_weight .= 0

    lw_absorption = zeros(FT, 2, 8, 2, 2)
    sw_absorption = zeros(FT, 1, 8, 2, 2)
    for gas in 1:8
        lw_absorption[1, gas, :, :] .= 0.001gas
        lw_absorption[2, gas, :, :] .= 0.002gas
        sw_absorption[1, gas, :, :] .= 0.003gas
    end
    lw_h2o = fill(0.05, 2, 2, 2, 2)
    sw_h2o = fill(0.07, 1, 2, 2, 2)
    gas_reference = zeros(FT, 8)
    gas_reference[4] = 1.0e-3
    source_temperature_grid = [200.0, 300.0]
    source_table = [
        10.0 30.0
        100.0 300.0
    ]
    temperature_layers = fill(250.0, 2, 2, 2)
    lw_tau = zeros(FT, 2, 2, 2, 2)
    sw_tau = zeros(FT, 1, 2, 2, 2)
    rayleigh = zeros(FT, 1, 2, 2, 2)
    source = zeros(FT, 2, 2, 2, 2)

    ext.tabulated_ecckd_optical_properties!(
        lw_tau,
        sw_tau,
        rayleigh,
        source,
        grid,
        temperature_layers,
        amounts,
        interp,
        gas_reference,
        lw_absorption,
        sw_absorption,
        lw_h2o,
        sw_h2o,
        [1.0e-8],
        source_temperature_grid,
        source_table,
    )

    adjusted = [1000, 20, 3, 4 - 1.0e-3 * 1000, 5, 6, 7, 8]
    @test lw_tau[1, 1, 1, 1] ≈ sum(0.001gas * adjusted[gas] for gas in 1:8) + 0.05 * 20
    @test lw_tau[2, 1, 1, 1] ≈ sum(0.002gas * adjusted[gas] for gas in 1:8) + 0.05 * 20
    @test sw_tau[1, 1, 1, 1] ≈ sum(0.003gas * adjusted[gas] for gas in 1:8) + 0.07 * 20
    @test rayleigh[1, 1, 1, 1] ≈ 1.0e-8 * 1000
    @test source[1, 1, 1, 1] ≈ 20.0
    @test source[2, 1, 1, 1] ≈ 200.0
end

@testset "RadiativeHeating tabulated ecCKD flux-divergence kernel" begin
    grid = RectilinearGrid(CPU(); size = (2, 2, 2),
                           x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    ext = Base.get_extension(Breeze, :BreezeLightfluxExt)
    FT = Float64
    lw_up = zeros(FT, 2, 2, 3)
    lw_down = zeros(FT, 2, 2, 3)
    sw_down = zeros(FT, 2, 2, 3)
    flux_divergence = zeros(FT, 2, 2, 2)
    lw_tau = fill(log(2), 1, 2, 2, 2)
    sw_tau = fill(log(2), 1, 2, 2, 2)
    rayleigh_tau = zeros(FT, 1, 2, 2, 2)
    lw_source = fill(10.0, 1, 2, 2, 2)

    ext.tabulated_ecckd_flux_divergence!(
        lw_up,
        lw_down,
        sw_down,
        flux_divergence,
        grid,
        lw_tau,
        sw_tau,
        rayleigh_tau,
        lw_source,
        [1.0],
        [1.0];
        solar_constant = 100.0,
        surface_temperature = 0.0,
        surface_albedo = 0.1,
        surface_emissivity = 1.0,
    )

    @test lw_up[1, 1, :] ≈ [0.0, 5.0, 7.5]
    @test lw_down[1, 1, :] ≈ [-7.5, -5.0, 0.0]
    @test sw_down[1, 1, :] ≈ [-25.0, -50.0, -100.0]
    @test flux_divergence[1, 1, :] ≈ [35.0, 85.0]
end
