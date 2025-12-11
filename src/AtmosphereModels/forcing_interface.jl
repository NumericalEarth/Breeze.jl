#####
##### Interface functions for AtmosphereModel
#####
##### These functions are defined here and extended in BoundaryConditions and Forcings modules.
#####

"""
    regularize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation, thermodynamic_constants)

Regularize boundary conditions for an `AtmosphereModel`. This function is extended
by the `BoundaryConditions` module to provide atmosphere-specific boundary condition handling.
"""
function regularize_atmosphere_model_boundary_conditions end

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
    compute_forcing!(forcing)

Compute any fields or quantities needed by a forcing before it is applied.
This function is extended by the `Forcings` module for forcing types that
require pre-computation (e.g., `SubsidenceForcing` which computes horizontal averages).
"""
function compute_forcing! end

