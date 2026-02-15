#####
##### Interface functions for AtmosphereModel
#####
##### These functions are defined here and extended in BoundaryConditions and Forcings modules.
#####

"""
    materialize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                    dynamics, microphysics, surface_pressure, thermodynamic_constants,
                                                    microphysical_fields, specific_moisture, temperature)

Regularize boundary conditions for an `AtmosphereModel`. This function is extended
by the `BoundaryConditions` module to provide atmosphere-specific boundary condition handling.

If `formulation` is `:LiquidIcePotentialTemperature` and `ρe` boundary conditions are provided,
they are automatically converted to `ρθ` boundary conditions by wrapping flux BCs in
`EnergyFluxBoundaryCondition`, which divides by the local mixture heat capacity.

The `dynamics` argument provides access to the reference state for boundary conditions
that require it, such as `VirtualPotentialTemperature` diagnostics.

The `microphysics` argument specifies the microphysics scheme used to compute moisture
fractions for mixture heat capacity and virtual potential temperature calculations.

The `microphysical_fields`, `specific_moisture`, and `temperature` arguments are pre-created
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
