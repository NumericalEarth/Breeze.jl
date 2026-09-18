#####
##### Host copy of a staged column
#####
##### A staged column copied to the host and presented as NumericalRadiation's `ColumnAtmosphere`,
##### so that the array `optical_properties!` and `radiative_fluxes!` can be run on it as the
##### reference the column kernels are checked against. The copy is explicit: the column arrays
##### live on the grid's architecture, and the array path of NumericalRadiation is host-only.
#####

"""
$(TYPEDSIGNATURES)

NumericalRadiation's `PhysicalConstants` in float type `FT` with the values of Breeze's
`constants::ThermodynamicConstants` (gravity, the dry-air heat capacity, the molar masses and
the gas constants), the Stefan–Boltzmann constant of `gas_model` (which carries it for its gray
source), and `solar_constant`: the constants the staging kernels use, handed to
NumericalRadiation's array path so that both paths see the same values.
"""
function physical_constants(FT, constants::ThermodynamicConstants, gas_model, solar_constant)
    return PhysicalConstants(FT; gravity = constants.gravitational_acceleration,
                                 heat_capacity = constants.dry_air.heat_capacity,
                                 stefan_boltzmann = gas_model.stefan_boltzmann,
                                 solar_constant,
                                 dry_air_molar_mass = constants.dry_air.molar_mass,
                                 water_molar_mass = constants.vapor.molar_mass,
                                 dry_air_gas_constant = dry_air_gas_constant(constants),
                                 universal_gas_constant = constants.molar_gas_constant)
end

"""
$(TYPEDSIGNATURES)

The staged column `(i, j)` of `rtm` as a NumericalRadiation `ColumnAtmosphere` on the host:
copies of the column's rows, the well-mixed gases materialized as `χ .* dry_air`, the surface
temperature and emissivity at `(i, j)`, the column's cosine of the solar zenith angle, and the
thermodynamic constants of `model` (the `AtmosphereModel` the column was staged from) as
NumericalRadiation's `PhysicalConstants`.
"""
function column_atmosphere(rtm::EcCKDRadiativeTransferModel, model, i, j)
    columns = rtm.atmospheric_state
    Nx = rtm.flux_divergence.grid.Nx
    c = column_index(i, j, Nx)

    row(a) = Array(view(a, c, :))
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
    surface = @allowscalar (temperature = surface_radiation.surface_temperature[i, j, 1],
                            emissivity = surface_radiation.surface_emissivity[i, j, 1])

    geometry = (cos_zenith = @allowscalar(columns.cos_zenith[c]),)

    constants = physical_constants(eltype(columns), model.thermodynamic_constants,
                                   rtm.longwave_solver.gas_model, rtm.solar_constant)

    return ColumnAtmosphere(; pressure_layers = row(columns.pressure_layers),
                              pressure_interfaces = row(columns.pressure_interfaces),
                              temperature_layers = row(columns.temperature_layers),
                              temperature_interfaces = row(columns.temperature_interfaces),
                              gases, surface, geometry, constants)
end
