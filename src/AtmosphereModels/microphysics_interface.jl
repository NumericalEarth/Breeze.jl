#####
##### Microphysics interface (default implementations)
#####

using ..Thermodynamics: MoistureMassFractions

#####
##### Definition of the microphysics interface, with methods for "Nothing" microphysics
#####

"""
$(TYPEDSIGNATURES)

Return the specific humidity (vapor mass fraction) field for the given `model`.

For `Nothing` microphysics (no condensate), the vapor mass fraction equals the total
specific moisture. For microphysics schemes with prognostic vapor (e.g., where `qᵛ`
is tracked explicitly), this function returns the appropriate vapor field.
"""
specific_humidity(model) = specific_humidity(model.microphysics, model)

specific_humidity(::Nothing, model) = model.specific_moisture

"""
$(TYPEDSIGNATURES)

Possibly apply saturation adjustment. If a `microphysics` scheme does not invoke saturation adjustment,
just return the `state` unmodified. In contrast to `adjust_thermodynamic_state`, this function
ingests the entire `microphysics` formulation and the `microphysical_fields`.
This is needed because some microphysics schemes apply saturation adjustment to a
subset of the thermodynamic state (for example, omitting precipitating species).

Grid indices `(i, j, k)` are provided to allow access to prognostic microphysical fields
at the current grid point. The reference density `ρᵣ` is passed to avoid recomputing it.
"""
@inline maybe_adjust_thermodynamic_state(i, j, k, state, ::Nothing, ρᵣ, microphysical_fields, qᵗ, thermo) = state

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

Return the microphysical velocities associated with `microphysics`, `microphysical_fields`, and tracer `name`.

Must be either `nothing`, or a NamedTuple with three components `u, v, w`.
The velocities are added to the bulk flow velocities for advecting the tracer.
For example, the terminal velocity of falling rain.
"""
@inline microphysical_velocities(microphysics::Nothing, microphysical_fields, name) = nothing

"""
$(TYPEDSIGNATURES)

Return the tendency of the microphysical field `name` associated with `microphysics`
and thermodynamic `constants`.
"""
@inline microphysical_tendency(i, j, k, grid, microphysics::Nothing, name, ρ, μ, 𝒰, constants) = zero(grid)

"""
$(TYPEDSIGNATURES)

Apply microphysics model update for the given `microphysics` scheme.

This function is called during `update_state!` to apply microphysics processes
that operate on the full model state (not the tendency fields).
Specific microphysics schemes should extend this function.
"""
microphysics_model_update!(microphysics::Nothing, model) = nothing

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

#####
##### Surface precipitation flux diagnostic
#####

"""
$(TYPEDSIGNATURES)

Return a 2D `Field` representing the flux of precipitating moisture at the bottom boundary.

The surface precipitation flux is `wʳ * ρqʳ` at the bottom face (k=1), representing
the rate at which rain mass leaves the domain through the bottom boundary.

Units: kg/m²/s (positive = downward flux out of domain)

Arguments:
- `model`: An [`AtmosphereModel`](@ref) with a microphysics scheme

Returns a 2D `Field` that can be computed and visualized.
Specific microphysics schemes must extend this function.
"""
surface_precipitation_flux(model) = surface_precipitation_flux(model, model.microphysics)

# Default: zero flux for Nothing microphysics
surface_precipitation_flux(model, ::Nothing) = Field{Center, Center, Nothing}(model.grid)

#####
##### Cloud effective radius interface
#####

"""
$(TYPEDEF)

Represents cloud particles with a constant effective radius.

# Fields
- `radius`: The effective radius in microns (μm).

# Example

```julia
liquid_radius = ConstantRadiusParticles(10.0)  # 10 μm droplets
ice_radius = ConstantRadiusParticles(30.0)     # 30 μm ice crystals
```
"""
struct ConstantRadiusParticles{FT}
    radius :: FT
end

"""
$(TYPEDSIGNATURES)

Return the effective radius of cloud liquid droplets in microns (μm).

This function dispatches on the `effective_radius_model` argument. The default
implementation for `ConstantRadiusParticles` returns a constant value.

Microphysics schemes can extend this function to provide diagnosed effective radii
based on cloud properties.
"""
@inline cloud_liquid_effective_radius(i, j, k, grid, effective_radius_model::ConstantRadiusParticles, args...) =
    effective_radius_model.radius

"""
$(TYPEDSIGNATURES)

Return the effective radius of cloud ice particles in microns (μm).

This function dispatches on the `effective_radius_model` argument. The default
implementation for [`ConstantRadiusParticles`](@ref) returns a constant value.

Microphysics schemes can extend this function to provide diagnosed effective radii
based on cloud properties.
"""
@inline cloud_ice_effective_radius(i, j, k, grid, effective_radius_model::ConstantRadiusParticles, args...) =
    effective_radius_model.radius
