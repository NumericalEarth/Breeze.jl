#####
##### Interface functions for AtmosphereModel
#####
##### These functions are defined here and extended in BoundaryConditions and Forcings modules.
#####

using Oceananigans.Forcings: ContinuousForcing

"""
    materialize_atmosphere_model_boundary_conditions(boundary_conditions, grid, formulation,
                                                    dynamics, microphysics, thermodynamic_constants,
                                                    closure = nothing)

Regularize boundary conditions for an `AtmosphereModel`. This function is extended
by the `BoundaryConditions` module to provide atmosphere-specific boundary condition handling.

Boundary conditions supplied under the energy key `ρE` (see
[`total_energy_density_name`](@ref)) are routed onto the prognostic thermodynamic variable of
`formulation`: for `:LiquidIcePotentialTemperature` they become `ρθ` boundary conditions,
with flux BCs wrapped in `EnergyFluxBoundaryCondition` to divide by the local mixture heat
capacity; for `:StaticEnergy` they pass through onto `ρs` unconverted.

Conditions supplied under the moisture key `ρqᵗ` (see [`total_moisture_density_name`](@ref))
are likewise routed onto the moisture density that `microphysics` evolves — `ρqᵛ` or `ρqᵉ`,
depending on the scheme — without conversion, since water entering the prognostic moisture is
water entering ``qᵗ`` under any of them.

The `dynamics` argument provides the standard pressure that boundary conditions need at
materialization time.

The `microphysics` argument specifies the microphysics scheme used to compute moisture
fractions for mixture heat capacity and virtual potential temperature calculations.

The turbulence `closure` sets the gustiness that bulk fluxes constructed without one take:
the subgrid turbulent kinetic energy under a `TKEBasedTurbulenceClosure`, a constant otherwise.

Nothing about the model state is captured here: surface fluxes read the pressure, density,
temperature and moisture they need from the model field tuple at evaluation time.
"""
function materialize_atmosphere_model_boundary_conditions end

"""
    adiabatic_scalar_bcs(bcs)

Return a copy of the prognostic-scalar `FieldBoundaryConditions` `bcs` with every surface flux
replaced by a no-flux condition, leaving all other (dynamical) boundary conditions untouched.
Applied to both the thermodynamic density (bulk sensible-heat / energy / θ flux) and the moisture
density (vapor flux) to strip surface sources from the adiabatic initialization twin, so its
symmetric forward/backward excursion stays pure, reversible dynamics (see
[`balance_adiabatically!`](@ref)). Extended by the `BoundaryConditions` module, which owns the
flux BC types.
"""
function adiabatic_scalar_bcs end

"""
    materialize_atmosphere_model_forcing(forcing, field, name, model_field_names, context)

Materialize a forcing for an `AtmosphereModel` field. This function is extended
by the `Forcings` module to handle atmosphere-specific forcing types like subsidence
and geostrophic forcings.

The `context` argument is a `NamedTuple` of what forcings may need at materialization time:
the Coriolis parameter, the coupling and total densities, the dynamics pressure, the
thermodynamic constants, and the specific fields keyed by name.
"""
function materialize_atmosphere_model_forcing end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Compute any fields or quantities needed by a forcing before it is applied.
This function is extended by the `Forcings` module for forcing types that
require pre-computation (e.g., `SubsidenceForcing` which computes horizontal averages).
"""
compute_forcing!(forcing) = nothing # Fallback - do nothing

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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

"""
    wrap_density_tendency_forcing(value, velocity_name)

Wrap `value`, a forcing of a momentum density supplied under `ρu` or `ρv`, for dynamics whose
prognostic variable is the velocity `velocity_name` itself: the wrapped forcing divides by the
continuity density at the velocity point at kernel time. This is the inverse of
[`wrap_specific_forcing`](@ref). Implemented in the `Forcings` module, which constructs a
`DensityTendencyForcing` and recurses into tuples.
"""
function wrap_density_tendency_forcing end

# A `ContinuousForcing` with neither parameters nor field dependencies
const FunctionOfCoordinatesAndTime = ContinuousForcing{<:Any, <:Any, <:Any, Nothing, <:Any, Tuple{}}

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `true` if the materialized `forcing` is one that dynamics with prognostic velocities
accept on `u` and `v`. Those dynamics reject every other forcing rather than apply one whose
behavior on a velocity has not been verified. Defaults to `false`; `true` for no forcing and
for a function of the coordinates and time without parameters or field dependencies, and
extended in the `Forcings` module.
"""
supports_prognostic_velocity(forcing) = false
supports_prognostic_velocity(::Returns) = true
supports_prognostic_velocity(::FunctionOfCoordinatesAndTime) = true
