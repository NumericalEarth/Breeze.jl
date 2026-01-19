#####
##### Interface functions for AtmosphereModel
#####
##### These functions are defined here and extended in BoundaryConditions and Forcings modules.
#####

"""
    regularize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                    dynamics, microphysics, surface_pressure, thermodynamic_constants)

Regularize boundary conditions for an `AtmosphereModel`. This function is extended
by the `BoundaryConditions` module to provide atmosphere-specific boundary condition handling.

If `formulation` is `:LiquidIcePotentialTemperature` and `ρe` boundary conditions are provided,
they are automatically converted to `ρθ` boundary conditions by wrapping flux BCs in
`EnergyFluxBoundaryCondition`, which divides by the local mixture heat capacity.

The `dynamics` argument provides access to the density field for boundary conditions
that require it, such as `EnergyFluxBoundaryCondition`.

The `microphysics` argument specifies the microphysics scheme used to compute moisture
fractions for mixture heat capacity calculations.
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
$(TYPEDSIGNATURES)

Compute any fields or quantities needed by a forcing before it is applied.
This function is extended by the `Forcings` module for forcing types that
require pre-computation (e.g., `SubsidenceForcing` which computes horizontal averages).
"""
compute_forcing!(forcing) = nothing # Fallback - do nothing
