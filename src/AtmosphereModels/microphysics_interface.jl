#####
##### Microphysics interface (default implementations)
#####

using ..Thermodynamics:
    temperature,
    MoistureMassFractions

#####
##### Definition of the microphysics interface, with methods for "Nothing" microphysics
#####

"""
$(TYPEDSIGNATURES)

Possibly apply saturation adjustment. If a `microphysics` scheme does not invoke saturation adjustment,
just return the `state` unmodified. In contrast to `adjust_thermodynamic_state`, this function
ingests the entire `microphysics` formulation and the `microphysical_fields`.
This is needed because some microphysics schemes apply saturation adjustment to a
subset of the thermodynamic state (for example, omitting precipitating species).

Grid indices `(i, j, k)` are provided to allow access to prognostic microphysical fields
at the current grid point.
"""
@inline maybe_adjust_thermodynamic_state(i, j, k, state, ::Nothing, microphysical_fields, qᵗ, thermo) = state

"""
$(TYPEDSIGNATURES)

Return `tuple()` - zero-moment scheme has no prognostic variables.
"""
prognostic_field_names(::Nothing) = tuple()

"""
$(TYPEDSIGNATURES)

Build microphysical fields associated with `microphysics` on `grid` and with
user defined `boundary_conditions`.
"""
materialize_microphysical_fields(microphysics::Nothing, grid, boundary_conditions) = NamedTuple()

"""
$(TYPEDSIGNATURES)

Update microphysical fields for `microphysics_scheme` given the thermodynamic `state` and
`thermo`dynamic parameters.
"""
@inline update_microphysical_fields!(microphysical_fields, microphysics::Nothing, i, j, k, grid, density, state, thermo) = nothing

"""
$(TYPEDSIGNATURES)

Build and return [`MoistureMassFractions`](@ref) at `(i, j, k)` for the given `grid`,
`microphysics`, `microphysical_fields`, and total moisture mass fraction `qᵗ`.

Dispatch is provided for `::Nothing` microphysics here. Specific microphysics
schemes may extend this method to provide tailored behavior.

Note: while ρ and qᵗ are scalars, the microphysical fields `μ` are `NamedTuple` of `Field`.
This may be changed in the future.
"""
@inline compute_moisture_fractions(i, j, k, grid, microphysics::Nothing, ρ, qᵗ, μ) = MoistureMassFractions(qᵗ)

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities associated with `microphysics` and `name`.

Must be either `nothing`, or a NamedTuple with three components `u, v, w`.
"""
@inline microphysical_velocities(microphysics::Nothing, name) = nothing

"""
$(TYPEDSIGNATURES)

Return the tendency of the microphysical field `name` associated with `microphysics` and `thermo`dynamic constants.

TODO: add the function signature when it is stable
"""
@inline microphysical_tendency(i, j, k, grid, microphysics::Nothing, name, args...) = zero(grid)

"""
$(TYPEDSIGNATURES)

Adjust the thermodynamic `state` according to the `scheme`.
For example, if `scheme isa SaturationAdjustment`, then this function
will adjust and return a new thermodynamic state given the specifications
of the saturation adjustment `scheme`.

If a scheme is non-adjusting, we just return `state`.
"""
@inline adjust_thermodynamic_state(state, scheme::Nothing, thermo) = state

#####
##### Precipitation rate diagnostic
#####

"""
    precipitation_rate(model, phase=:liquid)

Return a `KernelFunctionOperation` representing the precipitation rate for the given `phase`.

The precipitation rate is the rate at which moisture is removed from the atmosphere
by precipitation processes. For zero-moment schemes, this is computed from the
`remove_precipitation` function applied to cloud condensate.

Arguments:
- `model`: An `AtmosphereModel` with a microphysics scheme
- `phase`: Either `:liquid` (rain) or `:ice` (snow). Default is `:liquid`.

Returns a `Field` or `KernelFunctionOperation` that can be computed and visualized.
Specific microphysics schemes must extend this function.
"""
precipitation_rate(model, phase::Symbol=:liquid) = precipitation_rate(model, model.microphysics, Val(phase))

# Default: no precipitation for Nothing microphysics
# We implmement this as a fallback for convenience
# TODO: support reductions over ZeroField or the like, so we can swap
# non-precipitating microphysics schemes with precipitating ones
precipitation_rate(model, microphysics, phase) = CenterField(model.grid) 
