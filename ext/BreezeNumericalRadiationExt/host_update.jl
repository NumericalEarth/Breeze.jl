#####
##### Host-loop radiation update
#####
##### A CPU-only reference path: every staged column is handed to NumericalRadiation's array
##### `optical_properties!` and `radiative_fluxes!` in turn. It allocates per column and is the
##### reference the column kernels are checked against.
#####

"""
$(TYPEDSIGNATURES)

The staged column `(i, j)` of `rtm` as a NumericalRadiation `ColumnAtmosphere`: views of the
column's rows, the well-mixed gases materialized as `χ .* dry_air`, the surface temperature and
emissivity at `(i, j)`, and the column's cosine of the solar zenith angle.
"""
function column_atmosphere(rtm::EcCKDRadiativeTransferModel, i, j)
    columns = rtm.atmospheric_state
    Nx = rtm.flux_divergence.grid.Nx
    c = column_index(i, j, Nx)

    row(a) = view(a, c, :)
    dry_air = row(columns.dry_air)
    χ = rtm.longwave_solver.mole_fractions

    gases = (composite = dry_air,
             h2o = row(columns.water_vapor),
             o3 = row(columns.ozone),
             co2 = χ.co2 .* dry_air,
             ch4 = χ.ch4 .* dry_air,
             n2o = χ.n2o .* dry_air,
             cfc11 = χ.cfc11 .* dry_air,
             cfc12 = χ.cfc12 .* dry_air)

    surface_radiation = rtm.surface_radiation
    surface = (temperature = surface_radiation.surface_temperature[i, j, 1],
               emissivity = surface_radiation.surface_emissivity[i, j, 1])

    geometry = (cos_zenith = columns.cos_zenith[c],)

    return ColumnAtmosphere(; pressure_layers = row(columns.pressure_layers),
                              pressure_interfaces = row(columns.pressure_interfaces),
                              temperature_layers = row(columns.temperature_layers),
                              temperature_interfaces = row(columns.temperature_interfaces),
                              gases, surface, geometry)
end

# Per-g-point surface emission `ε B_g(T_s)` in the model's Planck convention
surface_longwave_up(gas_model::EcCKDTabulatedGasOpticsModel, T_s, ε) =
    surface_longwave_emission(gas_model, T_s; emissivity = ε)

surface_longwave_up(gas_model::EcCKDGasOpticsModel{FT}, T_s, ε) where FT =
    FT[ε * longwave_source(gas_model, ig, T_s, source_table_bracket(gas_model, T_s))
       for ig in 1:length(gas_model.longwave_weights)]

"""
$(TYPEDSIGNATURES)

Solve the longwave and shortwave fluxes of staged column `(i, j)` of `rtm` on the host with
NumericalRadiation's array optics and cloudless solvers, writing into the column's flux rows.
"""
function solve_spectral_column!(rtm::EcCKDRadiativeTransferModel, i, j)
    columns = rtm.atmospheric_state
    gas_model = rtm.longwave_solver.gas_model
    GT = eltype(gas_model)
    N = number_of_layers(columns)
    ng_lw = length(gas_model.longwave_weights)
    ng_sw = length(gas_model.shortwave_weights)

    longwave = LongwaveOptics(zeros(GT, ng_lw, N), zeros(GT, ng_lw, N);
                              source_top = zeros(GT, ng_lw, N),
                              source_bottom = zeros(GT, ng_lw, N),
                              weights = zeros(GT, ng_lw))

    shortwave = ShortwaveOptics(zeros(GT, ng_sw, N); weights = zeros(GT, ng_sw))

    atmosphere = column_atmosphere(rtm, i, j)
    optical_properties!(longwave, shortwave, gas_model, atmosphere)

    T_s = atmosphere.surface.temperature
    ε = atmosphere.surface.emissivity
    μ0 = atmosphere.geometry.cos_zenith
    S0 = rtm.shortwave_solver.solar_constant
    α_dir = rtm.surface_radiation.direct_surface_albedo[i, j, 1]
    α_dif = rtm.surface_radiation.diffuse_surface_albedo[i, j, 1]

    longwave_bcs = LongwaveBoundaryConditions(surface_longwave_up = surface_longwave_up(gas_model, T_s, ε),
                                              surface_albedo = GT(1 - ε))

    shortwave_bcs = ShortwaveBoundaryConditions(toa_shortwave_down = GT(S0 * max(μ0, 0)),
                                                surface_albedo = GT(α_dif),
                                                surface_albedo_direct = GT(α_dir))

    c = column_index(i, j, rtm.flux_divergence.grid.Nx)
    fluxes = RadiativeFluxes(longwave_up = view(columns.flux_up_lw, c, :),
                             longwave_down = view(columns.flux_down_lw, c, :),
                             shortwave_up = view(columns.flux_up_sw, c, :),
                             shortwave_down = view(columns.flux_down_sw, c, :))

    radiative_fluxes!(fluxes, CloudlessLongwave(), longwave, atmosphere, longwave_bcs)
    radiative_fluxes!(fluxes, CloudlessShortwave(), shortwave, atmosphere, shortwave_bcs)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the ecCKD radiative fluxes of `rtm` from the current state of `model`: stage the grid's
columns and their extension, solve every column on the host, copy the fluxes onto the grid and
compute the flux divergence.
"""
function AtmosphereModels._update_radiation!(rtm::EcCKDRadiativeTransferModel, model)
    assert_bound_surface_temperature(rtm)
    grid = model.grid
    columns = rtm.atmospheric_state

    update_cos_zenith!(columns.cos_zenith, rtm.solar_position, grid, model.clock)

    stage_spectral_columns!(columns, model, rtm.background_atmosphere)
    extend_spectral_columns!(columns, columns.extension, model)

    Nx, Ny, Nz = size(grid)
    for j in 1:Ny, i in 1:Nx
        solve_spectral_column!(rtm, i, j)
    end

    copy_spectral_fluxes!(rtm, grid)
    compute_radiation_flux_divergence!(rtm, grid)

    return nothing
end
