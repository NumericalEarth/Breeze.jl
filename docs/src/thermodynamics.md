# Thermodynamics

The thermodynamic utilities in Breeze collect the physical constants and closures
the atmospheric models rely on to convert between prognostic variables and
states such as temperature, specific humidity, and buoyancy. This section walks
through the key types and how they fit together.

## `AtmosphereThermodynamics`

`AtmosphereThermodynamics` bundles the constants that describe a moist
atmosphere composed of dry air and water vapour. It is used throughout the
anelastic model to compute derived quantities such as mixture gas constants,
heat capacities, and reference density profiles.

The struct stores

- `molar_gas_constant`: The universal molar gas constant *R* (default
  8.314462618 J mol⁻¹ K⁻¹). Keeping this value explicit lets the code switch
  float precision while retaining the same physical constant.
- `gravitational_acceleration`: The local gravitational acceleration *g* used in
  hydrostatic balance and buoyancy calculations. The default is 9.81 m s⁻².
- `dry_air` and `vapor`: Each is an [`IdealGas`](@ref) storing the molar mass and
  specific heat capacity of dry air and water vapour. They determine the
  component-specific gas constants (`Rᵈ`, `Rᵛ`).
- `saturation`: A [`Saturation`](@ref) reference state for the
  Clausius–Clapeyron relation. It supplies the triple-point pressure and
  temperature used to evaluate saturation vapour pressure.
- `condensation` and `deposition`: [`PhaseTransition`](@ref) descriptors for
  liquid and ice formation. They provide latent heats and heat capacities that
  feed into saturation adjustment.

Together these fields provide the data needed for helper functions like
[`mixture_gas_constant`](@ref) and [`mixture_heat_capacity`](@ref), as well as
saturation adjustment routines. When you construct an
`AtmosphereThermodynamics` instance you can override any of the defaults to
model alternative planetary atmospheres or microphysical closures.

### Usage inside the model

The anelastic formulation stores a single `AtmosphereThermodynamics` instance on
each `AtmosphereModel`. Functions such as `thermodynamic_state` use it to derive
potential temperature, specific humidity, and the Exner function from the
prognostic energy and moisture variables. These diagnostics in turn drive
buoyancy, pressure, and microphysical tendencies. Understanding the content of
`AtmosphereThermodynamics` is therefore the first step in auditing the
thermodynamic pathways of the model.
