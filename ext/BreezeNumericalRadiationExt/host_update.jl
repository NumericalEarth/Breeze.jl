#####
##### Host view of a staged column
#####
##### A CPU-only reference path: a staged column presented as NumericalRadiation's
##### `ColumnAtmosphere`, so that the array `optical_properties!` and `radiative_fluxes!` can be
##### run on it as the reference the column kernels are checked against.
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
