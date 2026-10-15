#####
##### Host copy of a column
#####
##### A column evaluated on the host through the same accessors the column kernel calls and
##### presented as NumericalRadiation's `ColumnAtmosphere`, so that the array `optical_properties!`
##### and `radiative_fluxes!` can be run on it as the reference the column kernel is checked
##### against. The array path of NumericalRadiation is host-only; the accessors read the model's
##### fields elementwise, which on a GPU is scalar indexing, allowed here.
#####

"""
$(TYPEDSIGNATURES)

NumericalRadiation's `PhysicalConstants` in float type `FT` with the values of Breeze's
`constants::ThermodynamicConstants` (gravity, the dry-air heat capacity, the molar masses and
the gas constants), the Stefan–Boltzmann constant of `gas_model` (which carries it for its gray
source), and `solar_constant`: the constants the column kernel uses, handed to
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

Column `(i, j)` of `model` as NumericalRadiation's `ColumnAtmosphere` on the host: the layer and
interface states that the column kernel of `rtm` computes (after a radiation update, which fills
the extension pressures), the well-mixed gases materialized as `χ .* dry_air`, the surface
temperature and emissivity at `(i, j)`, the column's cosine of the solar zenith angle, and the
thermodynamic constants of `model` as NumericalRadiation's `PhysicalConstants`.
"""
function column_atmosphere(rtm::EcCKDRadiativeTransferModel, model, i, j)
    state = ColumnState(rtm, model)
    N = number_of_layers(state)
    workspace = rtm.atmospheric_state
    c = column_index(i, j, model.grid.Nx)

    layers, temperature_interfaces, pressure_interfaces, surface, cos_zenith = @allowscalar begin
        ([layer_state(state, i, j, k) for k in 1:N],
         [interface_temperature(state, i, j, k) for k in 1:N+1],
         [interface_pressure(state, i, j, k) for k in 1:N+1],
         (temperature = rtm.surface_radiation.surface_temperature[i, j, 1],
          emissivity = rtm.surface_radiation.surface_emissivity[i, j, 1]),
         workspace.cos_zenith[c])
    end

    dry_air = [layer.dry_air for layer in layers]
    χ = rtm.longwave_solver.mole_fractions

    gases = (composite = dry_air,
             h2o = [layer.water_vapor for layer in layers],
             o3 = [layer.ozone for layer in layers],
             co2 = χ.co2 .* dry_air,
             ch4 = χ.ch4 .* dry_air,
             n2o = χ.n2o .* dry_air,
             cfc11 = χ.cfc11 .* dry_air,
             cfc12 = χ.cfc12 .* dry_air)

    constants = physical_constants(eltype(workspace), model.thermodynamic_constants,
                                   rtm.longwave_solver.gas_model, rtm.solar_constant)

    return ColumnAtmosphere(; pressure_layers = [layer.pressure for layer in layers],
                              pressure_interfaces,
                              temperature_layers = [layer.temperature for layer in layers],
                              temperature_interfaces,
                              gases, surface,
                              geometry = (cos_zenith = cos_zenith,),
                              constants)
end
