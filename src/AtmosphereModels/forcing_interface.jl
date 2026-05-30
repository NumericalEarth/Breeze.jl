#####
##### Interface functions for AtmosphereModel
#####
##### These functions are defined here and extended in BoundaryConditions and Forcings modules.
#####

"""
    materialize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                    dynamics, microphysics, surface_pressure, thermodynamic_constants,
                                                    microphysical_fields, specific_prognostic_moisture, temperature)

Regularize boundary conditions for an `AtmosphereModel`. This function is extended
by the `BoundaryConditions` module to provide atmosphere-specific boundary condition handling.

If `formulation` is `:LiquidIcePotentialTemperature` and `ρe` boundary conditions are provided,
they are automatically converted to `ρθ` boundary conditions by wrapping flux BCs in
`EnergyFluxBoundaryCondition`, which divides by the local mixture heat capacity.

The `dynamics` argument provides access to the reference state for boundary conditions
that require it, such as `VirtualPotentialTemperature` diagnostics.

The `microphysics` argument specifies the microphysics scheme used to compute moisture
fractions for mixture heat capacity and virtual potential temperature calculations.

The `microphysical_fields`, `specific_prognostic_moisture`, and `temperature` arguments are pre-created
fields used to construct the `VirtualPotentialTemperature` diagnostic for stability-dependent
boundary conditions.
"""
function materialize_atmosphere_model_boundary_conditions end

"""
    materialize_atmosphere_model_forcing(forcing, field, name, model_field_names, context)

Materialize a forcing for an `AtmosphereModel` field. This function is extended
by the `Forcings` module to handle atmosphere-specific forcing types like subsidence
and geostrophic forcings.

The `context` argument provides additional information needed for materialization,
such as grid, reference state, and thermodynamic constants.
"""
function materialize_atmosphere_model_forcing end

"""
$(TYPEDSIGNATURES)

Compute any fields or quantities needed by a forcing before it is applied.
This function is extended by the `Forcings` module for forcing types that
require pre-computation (e.g., `SubsidenceForcing` which computes horizontal averages).
"""
compute_forcing!(forcing) = nothing # Fallback - do nothing

"""
$(TYPEDSIGNATURES)

Return `true` if `forcing` produces a density-weighted tendency `F_{ρϕ}` directly
(i.e., already includes the multiplication by `ρ`).

Forcings that return density tendencies must be supplied under their density-weighted
key (e.g., `ρθ`, `ρu`) rather than the corresponding specific key (`θ`, `u`), because
the specific-key dispatch wraps user values in `SpecificForcing`, which would multiply
by `ρ` a second time. This trait is used by `atmosphere_model_forcing` to reject such
misuses with a clear error.

Defaults to `false`. Extended for `SubsidenceForcing` and `GeostrophicForcing` in the
`Forcings` module.
"""
is_density_tendency_forcing(::Any) = false

"""
    wrap_specific_forcing(value, density_name)

Wrap `value` so that the kernel-time density factor `ρ` is applied automatically when
the user supplies a forcing keyed by a specific (per-unit-mass) variable name like
`θ`, `u`, `qᵉ`. Implemented in the `Forcings` module: constructs a `SpecificForcing`,
recurses into tuples, and errors if `value` is itself a density-tendency forcing like
`SubsidenceForcing` (which would double-count `ρ`).

`density_name` is the corresponding density-weighted prognostic name (e.g. `:ρθ`) used
to produce a helpful error message when the wrap is rejected.
"""
function wrap_specific_forcing end
