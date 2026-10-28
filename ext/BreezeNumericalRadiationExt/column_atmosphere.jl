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
interface states that the layer pass of `rtm` computes (with the extension pressures of the batch
of columns that holds row `j`, computed here), the gas amounts of [`layer_gas_amounts`](@ref), the surface
temperature and emissivity at `(i, j)`, the column's cosine of the solar zenith angle, and the
thermodynamic constants of `model` as NumericalRadiation's `PhysicalConstants`.
"""
function column_atmosphere(rtm::EcCKDRadiativeTransferModel, model, i, j)
    workspace = rtm.atmospheric_state
    rows = workspace.rows
    row_offset = min(rows * fld(j - 1, rows), size(model.grid, 2) - rows)   # the batch that solves row `j`
    state = ColumnState(rtm, model, row_offset)
    compute_extension_pressure!(state, (size(model.grid, 1), rows))
    grid = model.grid
    N = number_of_layers(state)
    c = column_index(i, j, grid.Nx)

    layers, interface_temperatures, interface_pressures, surface, cos_zenith = @allowscalar begin
        ([layer_state(i, j, k, grid, state) for k in 1:N],
         [interface_temperature(i, j, k, grid, state) for k in 1:N+1],
         [interface_pressure(i, j, k, grid, state) for k in 1:N+1],
         (temperature = rtm.surface_radiation.surface_temperature[i, j, 1],
          emissivity = rtm.surface_radiation.surface_emissivity[i, j, 1]),
         workspace.cos_zenith[c])
    end

    # The gas amounts of the column kernel, keyed by the gas names of the ecCKD tables
    amounts = [layer_gas_amounts(layer, rtm.longwave_solver.mole_fractions) for layer in layers]
    gases = map(name -> [amount[name] for amount in amounts], NamedTuple{ECCKD_GAS_NAMES}(ECCKD_GAS_NAMES))

    constants = physical_constants(eltype(workspace), model.thermodynamic_constants,
                                   rtm.longwave_solver.gas_model,
                                   @allowscalar(solar_constant_at(rtm.solar_constant, i, j)))

    return ColumnAtmosphere(; layer_pressures = [layer.pressure for layer in layers],
                              interface_pressures,
                              layer_temperatures = [layer.temperature for layer in layers],
                              interface_temperatures,
                              gases, surface,
                              geometry = (cos_zenith = cos_zenith,),
                              constants)
end
