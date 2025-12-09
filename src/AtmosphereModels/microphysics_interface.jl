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
"""
@inline maybe_adjust_thermodynamic_state(state, ::Nothing, microphysical_fields, qᵗ, thermo) = state

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

Update microphysical fields for `microphysics_scheme` given the thermodynamic `state`,
`thermo`dynamic parameters, perturbation pressure `p′`, and timestep `Δt`.
"""
@inline update_microphysical_fields!(microphysical_fields, microphysics::Nothing, i, j, k, grid, density, state, p′, thermo, Δt) = nothing

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

Return the microphysical velocities associated with `microphysics`, `name`, and `microphysical_fields`.

Must be either `nothing`, or a NamedTuple with three components `u, v, w`.
The velocity components can be `nothing`, a constant, or a `Field`.
"""
@inline microphysical_velocities(microphysics::Nothing, name, microphysical_fields) = nothing

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

"""
    $(TYPEDSIGNATURES)

Update anything you want based on `microphysics`.
"""
microphysics_model_update!(microphysics::Nothing, model) = nothing
