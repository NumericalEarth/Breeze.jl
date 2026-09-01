#####
##### Microphysics interface (default implementations)
#####
#
# This file defines the interface that all microphysics implementations must provide.
# The key abstraction is the MicrophysicalState (ℳ), which enables the same tendency
# functions to work for any dynamics (grid-based LES, parcel models, etc.).
#
# Schemes plug in by extending one of two methods:
#
# 1. `microphysical_tendency(microphysics, Val(name), ρ, ℳ, 𝒰, constants)` for schemes
#    whose tendencies factor naturally per-name. The default
#    `compute_microphysical_tendencies!` builds ℳ once per cell and `+=`s the result
#    into each prognostic G field.
#
# 2. `compute_microphysical_tendencies!(microphysics, model)` for schemes whose
#    tendencies bundle many process rates feeding multiple prognostics
#    (e.g. mixed-phase non-equilibrium 1M, two-moment non-equilibrium). These
#    schemes write a fused kernel that computes the bundle once per cell.
#
# The model never calls `microphysical_tendency` directly during tendency assembly —
# `compute_microphysical_tendencies!` is the entry point that adds microphysical
# sources to `Gⁿ`.
#####

using Oceananigans.BoundaryConditions: BoundaryCondition, NormalFlow
using Oceananigans.Fields: set!, ZeroField, ZFaceField
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ, V⁻¹ᶜᶜᶜ, δzᵃᵃᶜ

using ..Thermodynamics: MoistureMassFractions

#####
##### MicrophysicalState abstraction
#####
#
# The AbstractMicrophysicalState type hierarchy enables microphysics schemes
# to work seamlessly in both grid-based LES and Lagrangian parcel models.
#
# Notation: ℳ (mathcal M) denotes a microphysical state, paralleling 𝒰 for
# thermodynamic state.
#####

"""
    AbstractMicrophysicalState{FT}

Abstract supertype for microphysical state structs.

Microphysical states encapsulate the local microphysical variables (e.g., cloud liquid,
rain, droplet number) needed to compute tendencies. This abstraction enables the same
tendency functions to work for both grid-based LES and Lagrangian parcel models.

Concrete subtypes should be immutable structs containing the relevant mixing ratios
and number concentrations for a given microphysics scheme.

For example, a warm-phase one-moment scheme might define a state with cloud liquid
and rain mixing ratios (`qᶜˡ`, `qʳ`).

See also [`microphysical_state`](@ref), [`microphysical_tendency`](@ref).
"""
abstract type AbstractMicrophysicalState{FT} end

@inline Base.eltype(::AbstractMicrophysicalState{FT}) where FT = FT

"""
    NothingMicrophysicalState{FT}

A microphysical state with no prognostic variables.

Used for `Nothing` microphysics and `SaturationAdjustment` schemes where
cloud condensate is diagnosed from the thermodynamic state rather than
being prognostic.
"""
struct NothingMicrophysicalState{FT} <: AbstractMicrophysicalState{FT} end

NothingMicrophysicalState(FT::DataType) = NothingMicrophysicalState{FT}()

"""
    WarmRainState{FT} <: AbstractMicrophysicalState{FT}

A simple microphysical state for warm-rain schemes with cloud liquid and rain.

# Fields
$(TYPEDFIELDS)
"""
struct WarmRainState{FT} <: AbstractMicrophysicalState{FT}
    "Specific cloud liquid water content [kg/kg]"
    qᶜˡ :: FT
    "Specific rain water content [kg/kg]"
    qʳ :: FT
end

#####
##### Prognostic field extraction
#####
#
# Extract prognostic microphysical variables at a grid point into a NamedTuple.
# This enables a generic grid-indexed wrapper that calls the gridless microphysical_state.

"""
$(TYPEDSIGNATURES)

Extract prognostic microphysical variables at grid point `(i, j, k)` into a NamedTuple
of scalar values.

Uses [`prognostic_field_names`](@ref) to determine which fields to extract. The result
is a NamedTuple with density-weighted values (e.g., `(ρqᶜˡ=..., ρqʳ=...)`).

This function enables a generic grid-indexed [`microphysical_state`](@ref) that extracts
prognostics and delegates to the gridless version.
"""
@inline function extract_microphysical_prognostics(i, j, k, microphysics, μ_fields)
    names = prognostic_field_names(microphysics)
    return _extract_prognostics(i, j, k, μ_fields, names)
end

# Base case: no prognostic fields
@inline _extract_prognostics(i, j, k, μ_fields, ::Tuple{}) = NamedTuple()

# Recursive case: extract first field, then rest
@inline function _extract_prognostics(i, j, k, μ_fields, names::Tuple{Symbol, Vararg})
    name = first(names)
    field = getproperty(μ_fields, name)
    val = @inbounds field[i, j, k]
    rest = _extract_prognostics(i, j, k, μ_fields, Base.tail(names))
    return merge(NamedTuple{(name,)}((val,)), rest)
end

#####
##### MicrophysicalState interface
#####

"""
    microphysical_state(microphysics, ρ, μ, 𝒰, velocities)

Build an [`AbstractMicrophysicalState`](@ref) (ℳ) from density-weighted prognostic
microphysical variables `μ`, density `ρ`, and thermodynamic state `𝒰`.

This is the **primary interface** that microphysics schemes must implement.
It converts density-weighted prognostics to the scheme-specific
`AbstractMicrophysicalState` type.

For **non-equilibrium schemes**, cloud condensate comes from `μ` (prognostic fields).
For **saturation adjustment schemes**, cloud condensate comes from `𝒰.moisture_mass_fractions`,
while precipitation (rain, snow) still comes from `μ`.

# Arguments
- `microphysics`: The microphysics scheme
- `ρ`: Local density (scalar)
- `μ`: NamedTuple of density-weighted prognostic variables (e.g., `(ρqᶜˡ=..., ρqʳ=...)`)
- `𝒰`: Thermodynamic state
- `velocities`: NamedTuple of velocity components `(; u, v, w)` [m/s].

# Returns
An `AbstractMicrophysicalState` subtype containing the local specific microphysical variables.

See also [`microphysical_tendency`](@ref), [`AbstractMicrophysicalState`](@ref).
"""
@inline microphysical_state(::Nothing, ρ, μ, 𝒰, velocities) = NothingMicrophysicalState(typeof(ρ))
@inline microphysical_state(::Nothing, ρ, ::Nothing, 𝒰, velocities) = NothingMicrophysicalState(typeof(ρ))
@inline microphysical_state(microphysics, ρ, ::Nothing, 𝒰, velocities) = NothingMicrophysicalState(typeof(ρ))
@inline microphysical_state(microphysics, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities) = NothingMicrophysicalState(typeof(ρ))
# Disambiguation for Nothing microphysics + empty NamedTuple
@inline microphysical_state(::Nothing, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities) = NothingMicrophysicalState(typeof(ρ))

"""
$(TYPEDSIGNATURES)

Restore scheme-specific constraints on density-weighted microphysical
`prognostics` after a parcel time-integration substep.

The default returns `prognostics` unchanged. Schemes with coupled prognostic
constraints may extend this hook to return a corrected value.
"""
@inline postprocess_microphysical_prognostics(microphysics, prognostics, ρ) = prognostics

"""
    grid_microphysical_state(i, j, k, grid, microphysics, μ_fields, ρ, 𝒰, velocities)

Build an [`AbstractMicrophysicalState`](@ref) (ℳ) at grid point `(i, j, k)`.

This is the **grid-indexed wrapper** that:
1. Extracts prognostic values from `μ_fields` via [`extract_microphysical_prognostics`](@ref)
2. Calls the gridless [`microphysical_state(microphysics, ρ, μ, 𝒰, velocities)`](@ref)

Microphysics schemes should implement the gridless version, not this one.

# Arguments
- `i, j, k`: Grid indices
- `grid`: The computational grid
- `microphysics`: The microphysics scheme
- `μ_fields`: NamedTuple of microphysical fields
- `ρ`: Local density (scalar)
- `𝒰`: Thermodynamic state
- `velocities`: Velocity fields ``(u, v, w)``. Velocities are interpolated to cell centers
                for use by microphysics schemes (e.g., aerosol activation uses vertical velocity).

# Returns
An `AbstractMicrophysicalState` subtype containing the local microphysical variables.

See also [`microphysical_tendency`](@ref), [`AbstractMicrophysicalState`](@ref).
"""
@inline function grid_microphysical_state(i, j, k, grid, microphysics, μ_fields, ρ, 𝒰, velocities)
    μ = extract_microphysical_prognostics(i, j, k, microphysics, μ_fields)
    u = ℑxᶜᵃᵃ(i, j, k, grid, velocities.u)
    v = ℑyᵃᶜᵃ(i, j, k, grid, velocities.v)
    w = ℑzᵃᵃᶜ(i, j, k, grid, velocities.w)
    U = (; u, v, w)
    return microphysical_state(microphysics, ρ, μ, 𝒰, U)
end

# Explicit Nothing fallback
@inline grid_microphysical_state(i, j, k, grid, microphysics::Nothing, μ_fields, ρ, 𝒰, velocities) =
    NothingMicrophysicalState(eltype(grid))

"""
    microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)

Compute the tendency for microphysical variable `name` from the microphysical
state `ℳ` and thermodynamic state `𝒰`.

This is the **state-based** tendency interface that operates on scalar states
without grid indexing. It works identically for grid-based LES and parcel models.

# Arguments
- `microphysics`: The microphysics scheme
- `name`: Variable name as `Val(:name)` (e.g., `Val(:ρqᶜˡ)`)
- `ρ`: Local density (scalar)
- `ℳ`: Microphysical state (e.g., `WarmPhaseOneMomentState`)
- `𝒰`: Thermodynamic state
- `constants`: Thermodynamic constants

# Returns
The tendency value (scalar, units depend on variable).

See also [`microphysical_state`](@ref), [`AbstractMicrophysicalState`](@ref).
"""
@inline microphysical_tendency(microphysics::Nothing, name, ρ, ℳ, 𝒰, constants) = zero(ρ)

"""
$(TYPEDSIGNATURES)

Compute the tendencies of `names` together, as a tuple in the order given.

The gridless counterpart of [`compute_microphysical_tendencies!`](@ref). The default maps
[`microphysical_tendency`](@ref) over `names`; schemes whose process rates are coupled
across species override it to evaluate their bundle once and distribute it.

See also [`microphysical_tendency`](@ref), [`compute_microphysical_tendencies!`](@ref).
"""
@inline microphysical_tendencies(microphysics, names::Tuple, ρ, ℳ, 𝒰, constants) =
    map(name -> microphysical_tendency(microphysics, Val(name), ρ, ℳ, 𝒰, constants), names)

#####
##### Fused microphysical tendency interface
#####
#
# `compute_microphysical_tendencies!` is the single entry point through which the
# atmosphere model adds microphysical sources during tendency assembly. The model calls it
# *after* the per-tracer dynamics kernels have written advection + diffusion + forcing
# into `Gⁿ`; microphysics contributions are added on top via `+=`.
#
# Auxiliary *state* — anything a scheme diagnoses and stores in a field, including
# sedimentation velocities — does not belong here. It belongs in
# [`update_microphysical_auxiliaries!`](@ref), which `update_state!` calls, so that every
# diagnostic carries the same time level as the prognostics. Diagnosing a field during
# tendency assembly instead would leave it one stage behind whenever it is output, and
# unwritten entirely after a `set!` (which calls `update_state!` with
# `compute_tendencies=false`).

"""
$(TYPEDSIGNATURES)

Add microphysics tendency contributions to the model's `Gⁿ` fields.

This is the only entry point through which `compute_tendencies!` adds microphysical
sources to the model's `Gⁿ` fields.
Concrete implementations add methods on the two-argument helper
`compute_microphysical_tendencies!(microphysics, model)`.

The default implementation launches a single fused kernel that builds the microphysical
state `ℳ` and thermodynamic state `𝒰` once per cell, then `+=`s the result of
[`microphysical_tendency`](@ref) for each prognostic name into the corresponding `G`
field. Schemes whose tendencies factor naturally per-name only need to extend
[`microphysical_tendency`](@ref).

Schemes whose tendencies bundle many process rates feeding multiple prognostics (e.g.
mixed-phase non-equilibrium 1M, where ~14 process rates feed 5 prognostic tendencies)
override this method directly to compute the bundle once per cell.
"""
compute_microphysical_tendencies!(model) =
    compute_microphysical_tendencies!(model.microphysics, model)

# No microphysics: nothing to add.
compute_microphysical_tendencies!(::Nothing, model) = nothing

# Default fused per-tracer kernel: ℳ and 𝒰 built once per cell, contributions
# accumulated into each G field via `+=`.
function compute_microphysical_tendencies!(microphysics, model)
    grid = model.grid
    arch = grid.architecture
    G = model.timestepper.Gⁿ

    moist_name = moisture_prognostic_name(microphysics)
    prog_names = prognostic_field_names(microphysics)
    all_names = (moist_name, prog_names...)
    G_tuple = map(n -> getproperty(G, n), all_names)
    name_tuple = map(Val, all_names)

    launch!(arch, grid, :xyz, _default_microphysical_tendencies_kernel!,
            G_tuple, name_tuple, grid, microphysics, model.dynamics, model.formulation,
            model.thermodynamic_constants, specific_prognostic_moisture(model),
            model.microphysical_fields, transport_velocities(model))

    return nothing
end

@kernel function _default_microphysical_tendencies_kernel!(G_tuple, name_tuple, grid,
                                                            microphysics, dynamics, formulation,
                                                            constants, specific_prognostic_moisture,
                                                            microphysical_fields, velocities)
    i, j, k = @index(Global, NTuple)

    ρ_field = total_density(dynamics)  # total ρ: mass fractions + microphysical state
    @inbounds ρ = ρ_field[i, j, k]
    @inbounds qᵛ = specific_prognostic_moisture[i, j, k]

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛ, microphysical_fields)
    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
    ℳ = grid_microphysical_state(i, j, k, grid, microphysics, microphysical_fields, ρ, 𝒰, velocities)

    _accumulate_microphysical_tendencies!(G_tuple, name_tuple, microphysics, i, j, k, ρ, ℳ, 𝒰, constants)
end

# Recursive Tuple iteration: type-stable and statically unrolled because the
# tuples carry their length and element types in their Tuple{...} type.
@inline _accumulate_microphysical_tendencies!(::Tuple{}, ::Tuple{}, microphysics, i, j, k, ρ, ℳ, 𝒰, constants) = nothing

@inline function _accumulate_microphysical_tendencies!(G_tuple::Tuple, name_tuple::Tuple,
                                                       microphysics, i, j, k, ρ, ℳ, 𝒰, constants)
    G = first(G_tuple)
    name = first(name_tuple)
    @inbounds G[i, j, k] += microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)
    return _accumulate_microphysical_tendencies!(Base.tail(G_tuple), Base.tail(name_tuple),
                                                 microphysics, i, j, k, ρ, ℳ, 𝒰, constants)
end

#####
##### Definition of the microphysics interface, with methods for "Nothing" microphysics
#####

"""
$(TYPEDSIGNATURES)

Return the prognostic moisture field name as a Symbol for the given microphysics scheme.

The physical meaning of the prognostic moisture field depends on the scheme:
- `Nothing` / non-equilibrium: `:ρqᵛ` (true vapor density)
- `SaturationAdjustment`: `:ρqᵉ` (equilibrium moisture density, diagnostically partitioned)
"""
moisture_prognostic_name(::Nothing) = :ρqᵛ

"""
$(TYPEDSIGNATURES)

Strip the leading `ρ` from a density-weighted field name to obtain
the specific (per-mass) name. For example, `:ρqᶜˡ` → `:qᶜˡ`.
"""
specific_field_name(name::Symbol) = (s = string(name); Symbol(s[nextind(s, 1):end]))

"""
$(TYPEDSIGNATURES)

Return the specific (per-mass) moisture field name by stripping the `ρ` prefix
from [`moisture_prognostic_name`](@ref).
"""
moisture_specific_name(microphysics) = specific_field_name(moisture_prognostic_name(microphysics))

"""
$(TYPEDSIGNATURES)

Return the prognostic specific moisture field for `model`.

This is ``qᵛ`` for non-equilibrium schemes or ``qᵉ`` for saturation adjustment schemes.
"""
specific_prognostic_moisture(model) = model.microphysical_fields[moisture_specific_name(model.microphysics)]

"""
$(TYPEDSIGNATURES)

Return the specific humidity (vapor mass fraction) field for the given `model`.

This always returns the actual vapor field ``qᵛ`` from the microphysical fields,
regardless of microphysics scheme.
"""
specific_humidity(model) = model.microphysical_fields.qᵛ

liquid_mass_fraction(model) = liquid_mass_fraction(model.microphysics, model)
ice_mass_fraction(model) = ice_mass_fraction(model.microphysics, model)

liquid_mass_fraction(::Nothing, model) = nothing
ice_mass_fraction(::Nothing, model) = nothing

"""
$(TYPEDSIGNATURES)

Possibly apply saturation adjustment. If a `microphysics` scheme does not invoke saturation adjustment,
just return the `state` unmodified.

This function takes the thermodynamic state, microphysics scheme, total moisture, and thermodynamic
constants. Schemes that use saturation adjustment override this to adjust the moisture partition.
Non-equilibrium schemes simply return the state unchanged.
"""
@inline maybe_adjust_thermodynamic_state(state, ::Nothing, qᵛ, constants) = state

"""
$(TYPEDSIGNATURES)

Return `tuple()` - `Nothing` microphysics has no prognostic variables.
"""
prognostic_field_names(::Nothing) = tuple()

"""
$(TYPEDSIGNATURES)

Return the names of prognostic fields that carry aerosol populations for `microphysics`.

Schemes without prognostic aerosol return an empty tuple by default. Microphysics schemes
with prognostic aerosol extend this interface so model components can retain or process
those fields without depending on scheme-specific names.
"""
@inline aerosol_field_names(microphysics) = tuple()

"""
$(TYPEDSIGNATURES)

Return the names of the prognostic microphysical fields that carry condensate *mass*
(condensate and precipitation densities), excluding number-concentration fields.

This is the subset of [`prognostic_field_names`](@ref) that, together with the moisture
density, is summed by [`total_condensate_density`](@ref) to form the total condensate mass per unit
volume. It defaults to all prognostic fields; schemes with prognostic number concentrations
(e.g. two-moment) override it to drop the `ρnˣ` fields.
"""
condensate_field_names(microphysics) = prognostic_field_names(microphysics)
condensate_field_names(::Nothing) = tuple()


"""
$(TYPEDSIGNATURES)

Build microphysical fields associated with `microphysics` on `grid` and with
user defined `boundary_conditions`.
"""
materialize_microphysical_fields(microphysics::Nothing, grid, boundary_conditions) = (; qᵛ=CenterField(grid))

"""
$(TYPEDSIGNATURES)

Return the aerosol population stored in a microphysics scheme's native units.

The units are the scheme's own: a volumetric distribution returns [m⁻³], while a
distribution specified per unit mass of air returns [kg⁻¹]. Use
[`initial_aerosol_number_density`](@ref) to obtain the value that the prognostic `ρnᵃ`
holds, whichever basis a scheme uses.

Returns `0` by default.
"""
initial_aerosol_number(microphysics) = 0

"""
$(TYPEDSIGNATURES)

Return the default aerosol number *density* ``ρ nᵃ`` [m⁻³] for a microphysics scheme,
given the air density `ρ` (a field for grid models, a number for parcels).

This is the value `set!` writes into the prognostic field `ρnᵃ` when the user supplies
neither `nᵃ` nor `ρnᵃ`. It is derived from the aerosol size distribution stored in the
microphysics scheme, so it stays consistent with the activation parameters.

Each scheme is responsible for the units of its own aerosol distribution: the density
argument is here so that a scheme whose distribution is specified *per unit mass*
[kg⁻¹], as `PredictedParticlePropertiesMicrophysics` is through
`AerosolMode.number_mixing_ratio`, can return the ``ρ``-weighted value that `ρnᵃ` holds,
while a scheme whose distribution is already a volumetric concentration [m⁻³] ignores `ρ`.
Breeze's convention throughout is that `nᵃ = ρnᵃ / ρ` is per unit mass [kg⁻¹]; a scheme
that omits this scaling diagnoses `nᵃ` with a spurious inverse-density dependence.

By default, forwards to [`initial_aerosol_number`](@ref), which returns `0`
unless a scheme overrides it.
"""
initial_aerosol_number_density(microphysics, ρ) = initial_aerosol_number(microphysics)

"""
$(TYPEDSIGNATURES)

Write the default aerosol reservoir [`initial_aerosol_number_density`](@ref) into `ρnᵃ`,
using the total air density [`total_density`](@ref) of `model`. A no-op for schemes without
prognostic aerosol.

Called at the end of `AtmosphereModel` construction, and again from every `set!` that does
not supply `nᵃ` or `ρnᵃ`, so the reservoir is weighted by whichever density is established
at the time: the reference density for anelastic dynamics, a prescribed density for the
kinematic driver, the reconciled total density for compressible dynamics. Compressible
density fields are zero at construction, so there the constructor writes zero and the first
`set!` carrying `ρ`, `ρᵈ`, or a [`HydrostaticallyBalancedDensity`](@ref) fills it in.

Because this runs on every such `set!`, a later call that re-initializes the state also
resets the reservoir to the distribution default. Pass `nᵃ` or `ρnᵃ` explicitly to carry a
depleted reservoir across a `set!`.
"""
function set_default_aerosol_number!(model)
    fields = model.microphysical_fields
    if :ρnᵃ ∈ keys(fields)
        ρ = total_density(model.dynamics)
        set!(fields.ρnᵃ, initial_aerosol_number_density(model.microphysics, ρ))
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update auxiliary microphysical fields at grid point `(i, j, k)`.

This is the **single interface function** for updating all auxiliary (non-prognostic)
microphysical fields. Microphysics schemes should extend this function.

The function receives:
- `μ`: NamedTuple of microphysical fields (mutated)
- `i, j, k`: Grid indices (after `μ` since this is a mutating function)
- `microphysics`: The microphysics scheme
- `ℳ`: The microphysical state at this point
- `ρ`: Local density
- `𝒰`: Thermodynamic state
- `constants`: Thermodynamic constants

## Why `i, j, k` is needed

Grid indices cannot be eliminated because:
1. Fields must be written at specific grid points
2. Some schemes need grid-dependent logic (e.g., `k == 1` for bottom boundary
   conditions in sedimentation schemes)

## What to implement

Schemes should write all auxiliary fields in one function. This includes:
- Specific moisture fractions (`qᶜˡ`, `qʳ`, etc.) from the microphysical state
- Derived quantities (`qˡ = qᶜˡ + qʳ`, `qⁱ = qᶜⁱ + qˢ`)
- Vapor mass fraction `qᵛ` from the thermodynamic state
- Terminal velocities for sedimentation

See [`WarmRainState`](@ref) implementation below for an example.
"""
function update_microphysical_auxiliaries! end

# Nothing microphysics: do nothing for any state
@inline function update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics::Nothing, ℳ, ρ, 𝒰, constants)
    return nothing
end

# Explicit disambiguation: Nothing microphysics + WarmRainState
@inline function update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics::Nothing, ℳ::WarmRainState, ρ, 𝒰, constants)
    return nothing
end

# Explicit disambiguation: Nothing microphysics + NothingMicrophysicalState
@inline function update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics::Nothing, ℳ::NothingMicrophysicalState, ρ, 𝒰, constants)
    return nothing
end

# Default for WarmRainState (used by DCMIP2016Kessler and non-precipitating warm-rain schemes)
@inline function update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics, ℳ::WarmRainState, ρ, 𝒰, constants)
    # Write state fields
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.qʳ[i, j, k] = ℳ.qʳ

    # Vapor from thermodynamic state
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor

    # Derived: total liquid
    @inbounds μ.qˡ[i, j, k] = ℳ.qᶜˡ + ℳ.qʳ

    return nothing
end

# Fallback for NothingMicrophysicalState
@inline function update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics, ℳ::NothingMicrophysicalState, ρ, 𝒰, constants)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update all microphysical fields at grid point `(i, j, k)`.

This orchestrating function:
1. Builds the microphysical state ℳ via [`microphysical_state`](@ref)
2. Calls [`update_microphysical_auxiliaries!`](@ref) to write auxiliary fields

Schemes should implement [`update_microphysical_auxiliaries!`](@ref), not this function.
"""
@inline function update_microphysical_fields!(μ, i, j, k, grid, microphysics::Nothing, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor
    return nothing
end

@inline function update_microphysical_fields!(μ, i, j, k, grid, microphysics, ρ, 𝒰, constants)
    # velocities are not used for auxiliary field updates, pass zeros
    zero_velocities = (; u = zero(ρ), v = zero(ρ), w = zero(ρ))
    ℳ = grid_microphysical_state(i, j, k, grid, microphysics, μ, ρ, 𝒰, zero_velocities)
    update_microphysical_auxiliaries!(μ, i, j, k, grid, microphysics, ℳ, ρ, 𝒰, constants)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Convert total specific moisture ``qᵗ`` to the scheme-dependent specific moisture ``qᵛᵉ``
by subtracting the appropriate condensate from the microphysical state ``ℳ``.

For non-equilibrium schemes, ``qᵛᵉ = qᵛ = qᵗ - qˡ`` (subtract all condensate).
For saturation adjustment schemes, ``qᵛᵉ = qᵉ = qᵗ - qʳ`` (subtract only precipitation).
For `Nothing` microphysics, ``qᵛᵉ = qᵗ`` (all moisture is vapor).

This is used by parcel models that store total moisture ``qᵗ`` as the prognostic
variable, to produce the correct input for [`moisture_fractions`](@ref).
"""
@inline specific_prognostic_moisture_from_total(::Nothing, qᵗ, ℳ) = qᵗ
@inline specific_prognostic_moisture_from_total(::Nothing, qᵗ, ::NothingMicrophysicalState) = qᵗ
@inline specific_prognostic_moisture_from_total(::Nothing, qᵗ, ::NamedTuple) = qᵗ

# Generic fallback: no condensate prognostics → all moisture is vapor/equilibrium.
@inline specific_prognostic_moisture_from_total(microphysics, qᵗ, ::NothingMicrophysicalState) = qᵗ

"""
$(TYPEDSIGNATURES)

Compute [`MoistureMassFractions`](@ref) from a microphysical state `ℳ` and
scheme-dependent specific moisture ``qᵛᵉ``.

The input ``qᵛᵉ`` is the scheme-dependent specific moisture: vapor for non-equilibrium
schemes, or equilibrium moisture (``qᵉ = qᵛ + qᶜˡ``) for saturation adjustment schemes.

This is the state-based (gridless) interface for computing moisture fractions.
Microphysics schemes should extend this method to partition moisture based on
their prognostic variables.

The default implementation for `Nothing` microphysics assumes all moisture is vapor.
"""
@inline moisture_fractions(::Nothing, ℳ, qᵛ) = MoistureMassFractions(qᵛ)
@inline moisture_fractions(microphysics, ::NothingMicrophysicalState, qᵛ) = MoistureMassFractions(qᵛ)
@inline moisture_fractions(::Nothing, ::NothingMicrophysicalState, qᵛ) = MoistureMassFractions(qᵛ)

# Disambiguation for Nothing microphysics + specific state types
@inline moisture_fractions(::Nothing, ℳ::WarmRainState, qᵛ) = MoistureMassFractions(qᵛ)
@inline moisture_fractions(::Nothing, ℳ::NamedTuple, qᵛ) = MoistureMassFractions(qᵛ)

# WarmRainState: cloud liquid + rain
# Input qᵛ is vapor; used with condensate to build moisture fractions.
@inline function moisture_fractions(microphysics, ℳ::WarmRainState, qᵛ)
    qˡ = ℳ.qᶜˡ + ℳ.qʳ
    return MoistureMassFractions(qᵛ, qˡ)
end

# Fallback for NamedTuple microphysical state (used by parcel models with prognostic microphysics).
# NamedTuple contains specific moisture fractions computed from ρ-weighted prognostics.
# Input qᵛᵉ is scheme-dependent specific moisture (vapor or equilibrium moisture).
@inline function moisture_fractions(microphysics, ℳ::NamedTuple, qᵛᵉ)
    z = zero(qᵛᵉ)
    qˡ = get(ℳ, :qᶜˡ, z) + get(ℳ, :qʳ, z)
    qⁱ = get(ℳ, :qᶜⁱ, z) + get(ℳ, :qˢ, z)
    return MoistureMassFractions(qᵛᵉ, qˡ, qⁱ)
end

"""
$(TYPEDSIGNATURES)

Grid-indexed version of [`moisture_fractions`](@ref).

This is the **generic wrapper** that:
1. Extracts prognostic values from `μ_fields` via [`extract_microphysical_prognostics`](@ref)
2. Builds the microphysical state via [`microphysical_state`](@ref) with `𝒰 = nothing`
3. Calls [`moisture_fractions`](@ref)

This works for **non-equilibrium schemes** where cloud condensate is prognostic.
Non-equilibrium schemes don't need `𝒰` to build their state (they use prognostic fields).

**Saturation adjustment schemes** should override this to read from diagnostic fields.
"""
@inline function grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛ, μ_fields)
    μ = extract_microphysical_prognostics(i, j, k, microphysics, μ_fields)
    # velocities are not used for moisture fraction computation, pass zeros
    zero_velocities = (; u = zero(ρ), v = zero(ρ), w = zero(ρ))
    ℳ = microphysical_state(microphysics, ρ, μ, nothing, zero_velocities)
    return moisture_fractions(microphysics, ℳ, qᵛ)
end

# Fallback for Nothing microphysics (no fields to index)
@inline grid_moisture_fractions(i, j, k, grid, microphysics::Nothing, ρ, qᵛ, μ) = MoistureMassFractions(qᵛ)

#####
##### Total condensate and total air density (diagnosed from dry density)
#####

"""
$(TYPEDSIGNATURES)

Total condensate density ``ρᵗ = ρqᵛᵉ + Σ ρqᶜ`` at `(i, j, k)`: the moisture density ``ρqᵛᵉ``
(vapor or equilibrium moisture) plus every condensed-species density named by
[`condensate_field_names`](@ref). Number-concentration fields (`ρnˣ`) are excluded. This sums
all phases of the condensable species (water by default), so other condensates can be added by
extending `condensate_field_names`.
"""
@inline function total_condensate_density(i, j, k, microphysics, moisture_density, microphysical_fields)
    ρqᵛᵉ = @inbounds moisture_density[i, j, k]
    ρqᶜ = sum_microphysical_densities(i, j, k, microphysical_fields, condensate_field_names(microphysics))
    return ρqᵛᵉ + ρqᶜ
end

# Compile-time recursion over the condensate field names (cf. `extract_microphysical_prognostics`).
# `false` is the additive identity and promotes to the field element type.
@inline sum_microphysical_densities(i, j, k, microphysical_fields, ::Tuple{}) = false
@inline function sum_microphysical_densities(i, j, k, microphysical_fields, names::Tuple{Symbol, Vararg})
    ρqˣ = @inbounds getproperty(microphysical_fields, first(names))[i, j, k]
    return ρqˣ + sum_microphysical_densities(i, j, k, microphysical_fields, Base.tail(names))
end

"""
$(TYPEDSIGNATURES)

Total air density ``ρ = ρᵈ + ρᵗ`` at `(i, j, k)`: the dry-air density `dry_density`
plus the [`total_condensate_density`](@ref) ``ρᵗ``. This is the diagnosed total mass density used
where total mass enters the physics — the gravitational/buoyancy term and the equation of state.
"""
@inline function total_density(i, j, k, dry_density, microphysics, moisture_density, microphysical_fields)
    ρᵈ = @inbounds dry_density[i, j, k]
    return ρᵈ + total_condensate_density(i, j, k, microphysics, moisture_density, microphysical_fields)
end

#####
##### Sedimentation interface
#####
#
# Microphysics schemes describe how their condensate falls through two functions:
#
#   sedimentation_velocity(microphysics, microphysical_fields, ::Val{name}) → field or nothing
#       the signed vertical velocity [m/s] the prognostic `name` falls with (negative = downward)
#   condensate_phase(microphysics, ::Val{name}) → Val(:liquid) or Val(:ice)
#       the thermodynamic phase of the condensate mass `name`; required for every name in
#       condensate_field_names(microphysics) that sediments
#
# The velocity moves the tracer: `microphysical_velocities` adds it to the transport velocity
# in `scalar_tendency`. The phase says which latent heat rides along with the falling mass
# when the thermodynamic variables are transported by sedimentation
# (`condensate_sedimentation_divergence`). The two are independent: P3's liquid on ice falls
# at the ice speed but carries liquid enthalpy.

"""
$(TYPEDSIGNATURES)

Return the sedimentation velocity field (vertical component, [m/s], negative = downward) for
the prognostic tracer `name`, or `nothing` (the default) if the tracer does not sediment.

Microphysics schemes extend this function for each sedimenting tracer, dispatching on
`::Val{name}`.
"""
@inline sedimentation_velocity(microphysics, microphysical_fields, ::Val) = nothing

"""
$(TYPEDSIGNATURES)

Return the thermodynamic phase, `Val(:liquid)` or `Val(:ice)`, of the condensate mass `name`,
or `nothing` (the default) for a mass that does not sediment.

Every name in [`condensate_field_names`](@ref) with a [`sedimentation_velocity`](@ref) must
declare its phase; [`materialize_sedimentation_constituents`](@ref) checks this at model
construction, since a falling mass without a phase would leave its latent heat behind. The
phase is the enthalpy the mass carries, not what it falls with: P3's liquid on ice `ρqʷⁱ`
falls at the ice speed yet is liquid, because no fusion enthalpy has been released for it and
[`moisture_fractions`](@ref) counts it in the liquid mass fraction.

Number moments and non-additive particle properties (P3's rime mass `ρqᶠ` and rime volume
`ρbᶠ`) are not condensate masses and need no method.
"""
@inline condensate_phase(microphysics, ::Val) = nothing

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities associated with `microphysics`, `microphysical_fields`, and tracer `name`.

Must be either `nothing`, or a NamedTuple with three components `u, v, w`.
The velocities are added to the bulk flow velocities for advecting the tracer.
For example, the terminal velocity of falling rain.

The generic implementation calls [`sedimentation_velocity`](@ref) and uses
the result as the vertical velocity component.
"""
@inline function microphysical_velocities(microphysics, microphysical_fields, name)
    w = sedimentation_velocity(microphysics, microphysical_fields, name)
    return sedimentation_velocity_tuple(w)
end

@inline sedimentation_velocity_tuple(::Nothing) = nothing
@inline sedimentation_velocity_tuple(w) = (; u = ZeroField(), v = ZeroField(), w)

#####
##### Sedimentation velocity fields
#####

"""
$(TYPEDSIGNATURES)

Build a `ZFaceField` suitable for storing a sedimentation velocity: `bottom = nothing`
ensures a kernel-set bottom-face value is preserved during `fill_halo_regions!`, while the
default impenetrable top holds `w = 0` so nothing falls in through the model top.
"""
function sedimentation_velocity_field(grid)
    boundary_conditions = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    return ZFaceField(grid; boundary_conditions)
end

# Bottom boundary treatment of a diagnosed fall speed. Index `k = 1` is the bottom face of
# the domain and carries the surface precipitation flux: `nothing` (the default
# precipitation boundary condition) keeps the diagnosed fall speed there, so precipitation
# leaves through an open surface, while an impenetrable boundary condition zeroes it, so
# precipitation accumulates in the lowest cell instead. Dispatch is on the
# boundary-condition *type*, so the choice folds to a constant per concrete scheme and
# stays GPU-safe.
#
# TODO: Use the lowest *active* face of each column rather than `k = 1` so the condition
# also applies over an immersed bottom.
const ImpenetrableSedimentationBC = BoundaryCondition{<:NormalFlow, Nothing}

@inline bottom_sedimentation_velocity(::Nothing, w) = w
@inline bottom_sedimentation_velocity(::ImpenetrableSedimentationBC, w) = zero(w)

"""
$(TYPEDSIGNATURES)

Store the fall-speed magnitude `𝕎` diagnosed at cell center `(i, j, k)` in the
sedimentation velocity field `w_field` at the cell's bottom face: microphysics libraries
return positive magnitudes while Breeze stores signed vertical velocities (negative =
downward), and sedimentation is always downward, so the donor cell for face `k` is cell
`k` itself. At `k = 1` the precipitation boundary condition `bc` is applied (see
`bottom_sedimentation_velocity`); the top face (`k = Nz + 1`) lies outside the `:xyz`
launch region and is held at zero by the impenetrable top boundary condition.
"""
@inline function write_sedimentation_velocity!(w_field, i, j, k, bc, 𝕎)
    w = -𝕎
    w₀ = bottom_sedimentation_velocity(bc, w)
    @inbounds w_field[i, j, k] = ifelse(k == 1, w₀, w)
    return nothing
end

#####
##### Sedimentation constituents
#####
#
# Everything the model needs to know about each sedimenting condensate mass, resolved once at
# construction: the velocity field it falls with, its specific-humidity field, its
# thermodynamic phase, and the advection scheme that transports it. The thermodynamic
# tendencies read these to transport the condensate part of ρθ / ρs with the falling mass,
# and the surface precipitation flux diagnostic sums their bottom-face fluxes, so both are
# built from the same declarations as the tracer transport itself.

"""
$(TYPEDSIGNATURES)

Materialize the sedimentation constituents for the selected dynamics. Eulerian dynamics use
the microphysics-only materializer by default.
"""
materialize_sedimentation_constituents(dynamics, microphysics, microphysical_fields, advection) =
    materialize_sedimentation_constituents(microphysics, microphysical_fields, advection)

"""
$(TYPEDSIGNATURES)

Return the tuple of `(; w, q, phase, advection)` sedimentation constituents of `microphysics`:
one NamedTuple for every name in [`condensate_field_names`](@ref) with a
[`sedimentation_velocity`](@ref) `w`, holding its specific-humidity field `q`, its
[`condensate_phase`](@ref) tag, and the `advection` scheme that transports the tracer's mass.
Condensate that does not sediment (for example, cloud condensate diagnosed by saturation
adjustment) is absent: it moves no mass and therefore no latent heat. The result is `()` when
nothing sediments, including for `Nothing` microphysics.

Throws an `ArgumentError` if a sedimenting condensate mass declares no phase.
"""
function materialize_sedimentation_constituents(microphysics, microphysical_fields, advection)
    constituents = map(condensate_field_names(microphysics)) do name
        sedimentation_constituent(microphysics, microphysical_fields, advection, Val(name))
    end
    return filter(!isnothing, constituents)
end

function sedimentation_constituent(microphysics, μ, advection, ::Val{name}) where name
    w = sedimentation_velocity(microphysics, μ, Val(name))
    isnothing(w) && return nothing
    phase = condensate_phase(microphysics, Val(name))
    isnothing(phase) &&
        throw(ArgumentError("Condensate mass $name sediments but declares no condensate_phase, " *
                            "so its latent heat could not follow the falling mass."))
    q = getproperty(μ, specific_field_name(name))
    return (; w, q, phase, advection = getproperty(advection, name))
end

#####
##### Sedimentation transport of the condensate part of the thermodynamic variables
#####
#
# The thermodynamic variables carry a condensate part — the deficit −(ℒˡᵣ qˡ + ℒⁱᵣ qⁱ) / (cᵖᵐ Π)
# of θˡⁱ, the content qˡ (cˡ T − ℒˡᵣ) + qⁱ (cⁱ T − ℒⁱᵣ) of s — and when condensate falls, that
# part must fall with it. Rain-out then leaves the latent warming from forming the rain aloft
# and pre-cools the layer that later evaporates it, the mechanism that drives cold pools.
#
# Each formulation supplies only its per-phase content per unit falling mass, (χˡ, χⁱ), at a
# cell; the discretization below is shared. The mass fluxes are the ones the tracer tendency
# actually applies to the cell (`sedimentation_mass_fluxes`, formed per cell because
# bounds-preserving WENO limits its reconstructions per cell), each carries the content of the
# cell it drains (`condensate_content_fluxes`) — the cell above the face when the condensate
# falls, the cell below when an updraft outruns its fall speed — and the flux is weighted by the
# density that carries the thermodynamic variable. Like the sedimentation of the tracers
# themselves, the flux is explicit in time.
#
# TODO: under adaptive implicit vertical advection the tracer's sedimentation is partly
# implicit; `sedimentation_mass_fluxes` estimates the implicit remainder explicitly at the
# current tracer state, which matches the implicit solve only to leading order in Δt when
# AIVA is used with fast-falling hydrometeors on thin near-surface cells.

"""
$(TYPEDSIGNATURES)

Return the divergence at cell `(i, j, k)` of the sedimentation flux of the condensate part of
a thermodynamic variable,

    ∂z [ρᵈ Σᵢ (χᵢ(Wᵢ) Fᵢ(Wᵢ) − χᵢ(wᵗ) Fᵢ(wᵗ))] ,

where, for each of the `constituents`, `Fᵢ(w)` is the vertical advective flux of its humidity
at velocity `w` through the faces of the cell, with the scheme that transports the tracer (see
[`sedimentation_mass_fluxes`](@ref)), `Wᵢ = wᵗ + wᵢ` is its total velocity and `wᵗ` the
resolved transport velocity, `ρᵈ` is the [`dynamics_density`](@ref) that carries the
thermodynamic variable (`ρθ = ρᵈ θ`, `ρs = ρᵈ s`), and `condensate_content(i, j, k, grid,
args...)` returns the formulation's content per unit falling mass of each phase, `(χˡ, χⁱ)`,
at cell `k` (`−ℒˣᵣ / (cᵖᵐ Π)` to leading order for `ρθ`, `(cˣ − cᵖᵈ) T − ℒˣᵣ` for `ρs`). The
tracer tendency advects each humidity at `Wᵢ` in place of `wᵗ`, so the bracket is the
sedimentation part of the mass flux advection actually applies to the cell, and each of its two
fluxes carries the content of its own upwind cell (see [`condensate_content_fluxes`](@ref)).
Returns zero when no constituent sediments.
"""
@inline condensate_sedimentation_divergence(i, j, k, grid, ::Tuple{}, args...) = zero(grid)

@inline function condensate_sedimentation_divergence(i, j, k, grid, constituents, wᵗ, dynamics, condensate_content, args...)
    # Content of the cells below, at, and above k; each face draws on the two cells flanking it.
    # The clamps keep the bottom face (nothing enters through an impenetrable bottom) and the
    # top face (the fall speeds vanish there) from reading unfilled halos.
    χ⁻ = condensate_content(i, j, max(k - 1, 1), grid, args...)
    χ⁰ = condensate_content(i, j, k, grid, args...)
    χ⁺ = condensate_content(i, j, min(k + 1, grid.Nz), grid, args...)
    Φ⁻, Φ⁺ = condensate_content_fluxes(i, j, k, grid, constituents, wᵗ, χ⁻, χ⁰, χ⁺)
    ρᵈ = dynamics_density(dynamics)
    ρᵈᶠ⁻ = ℑzᵃᵃᶠ(i, j, k,     grid, ρᵈ)
    ρᵈᶠ⁺ = ℑzᵃᵃᶠ(i, j, k + 1, grid, ρᵈ)
    return V⁻¹ᶜᶜᶜ(i, j, k, grid) * (ρᵈᶠ⁺ * Φ⁺ - ρᵈᶠ⁻ * Φ⁻)
end

"""
$(TYPEDSIGNATURES)

Return `Σᵢ [χᵢ(Wᵢ) Fᵢ(Wᵢ) − χᵢ(wᵗ) Fᵢ(wᵗ)]` through the lower and upper faces of cell
`(i, j, k)`: the sedimentation part of the advective flux of condensate content through each
face, per unit density and integrated over the face area, summed over the `constituents`.
`Fᵢ(w)` is the advective flux of constituent `i`'s humidity at velocity `w` from
[`sedimentation_mass_fluxes`](@ref), `Wᵢ = wᵗ + wᵢ` its total velocity, and `χᵢ(w)` the content
of its phase in the upwind cell of `w`: the cell above the face for a downward velocity, the
cell below for an upward one (`χ⁻`, `χ⁰` and `χ⁺` are the `(χˡ, χⁱ)` of the cells below, at and
above `k`).

An upwind flux at `Wᵢ` does not decompose into `scheme(wᵗ) + first-order(wᵢ)`, so the
difference of the two fluxes is the only form consistent with the mass the tracer tendency
moves. Giving each flux the content of the cell it drains keeps the heat with that mass whether
the condensate falls (both fluxes downward, both drain the cell above), rides an updraft that
outruns its fall speed (both upward, both drain the cell below), or falls against an updraft
(the flux at `Wᵢ` drains the cell above while the transport flux it replaces drained the cell
below). Constituents are binned by their thermodynamic phase, so P3's liquid on ice contributes
its ice-speed flux to the liquid content.
"""
@inline condensate_content_fluxes(i, j, k, grid, ::Tuple{}, wᵗ, χ⁻, χ⁰, χ⁺) = (zero(grid), zero(grid))

@inline function condensate_content_fluxes(i, j, k, grid, constituents::Tuple, wᵗ, χ⁻, χ⁰, χ⁺)
    (; w, q, phase, advection) = first(constituents)
    F⁻, F⁺ = sedimentation_mass_fluxes(i, j, k, grid, advection, wᵗ, w, q)
    χ⁻ˣ = phase_content(phase, χ⁻)
    χ⁰ˣ = phase_content(phase, χ⁰)
    χ⁺ˣ = phase_content(phase, χ⁺)
    Φ⁻ = condensate_content_flux(i, j, k,     wᵗ, w, F⁻, χ⁻ˣ, χ⁰ˣ)
    Φ⁺ = condensate_content_flux(i, j, k + 1, wᵗ, w, F⁺, χ⁰ˣ, χ⁺ˣ)
    rest⁻, rest⁺ = condensate_content_fluxes(i, j, k, grid, Base.tail(constituents), wᵗ, χ⁻, χ⁰, χ⁺)
    return Φ⁻ + rest⁻, Φ⁺ + rest⁺
end

# Content flux of one constituent through face k from its mass fluxes `F = (F(W), F(wᵗ))`: each
# flux carries the content of the cell it drains, the cell above the face for a downward
# velocity and the cell below for an upward one.
@inline function condensate_content_flux(i, j, k, wᵗ, w, F, χ_below, χ_above)
    Fᵂ, Fᵗ = F
    @inbounds wᵗₖ = wᵗ[i, j, k]
    @inbounds Wₖ = wᵗₖ + w[i, j, k]
    χᵂ = upwind_content(Wₖ, χ_below, χ_above)
    χᵗ = upwind_content(wᵗₖ, χ_below, χ_above)
    return χᵂ * Fᵂ - χᵗ * Fᵗ
end

@inline phase_content(::Val{:liquid}, χ) = χ[1]
@inline phase_content(::Val{:ice}, χ) = χ[2]

@inline upwind_content(w, χ_below, χ_above) = ifelse(w > 0, χ_below, χ_above)

"""
$(TYPEDSIGNATURES)

Return the vertical advective mass fluxes of humidity `q` through the lower and upper faces of
cell `(i, j, k)`, as `((F⁻(wᵗ + wˢ), F⁻(wᵗ)), (F⁺(wᵗ + wˢ), F⁺(wᵗ)))`: at each face, the flux at
the combined velocity and at the resolved transport velocity `wᵗ` alone, both with the tracer's
own `advection` scheme, integrated over the face area and per unit density [m³ s⁻¹]. Their
difference is the sedimentation part of the mass flux the tracer tendency applies to the cell;
[`condensate_content_fluxes`](@ref) weights each with the content of its own upwind cell. The
fluxes belong to the cell rather than to a face because bounds-preserving WENO limits its
reconstructions cell by cell, so the flux through a face differs between the two cells that
share it. Under adaptive implicit vertical advection each flux includes the first-order flux of
its implicit velocity remainder, evaluated at the current tracer state.

Implemented in `Breeze.Advection`, which owns the flux operators.
"""
function sedimentation_mass_fluxes end

"""
$(TYPEDSIGNATURES)

Apply the operator-split microphysics update for the given `microphysics` scheme.

This is called once per time step by the time-stepper (not from `update_state!`) to
apply microphysics processes that operate on the full model state by the full `Δt`,
rather than through the per-stage tendency interface. It runs after the time-stepper's
`update_state!` has refreshed the diagnostic state it reads. Schemes that mutate
prognostic fields here are responsible for restoring a consistent model state (halos,
diagnostics, and tendencies) before returning — e.g. by calling `update_state!`.
Defaults to a no-op; specific microphysics schemes extend this function.
"""
microphysics_model_update!(microphysics::Nothing, model) = nothing

"""
$(TYPEDSIGNATURES)

Validate that `microphysics` is compatible with the model's `thermodynamic_constants`.

Defaults to a no-op. Schemes that require a particular thermodynamic formulation (for
example a specific saturation vapor pressure formula) extend this method to throw a clear
`ArgumentError` at model construction, rather than failing later inside a kernel — where the
failure surfaces as an opaque dynamic `getproperty` / GPU compilation error.
"""
validate_microphysics(microphysics, thermodynamic_constants) = nothing

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
by precipitation processes.

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

Return a 2D `Field` representing the flux of precipitating moisture at the bottom boundary,
in kg/m²/s (positive = downward flux out of domain).

The default sums the bottom-face advective flux of every sedimentation constituent of the
model (see [`materialize_sedimentation_constituents`](@ref)), each evaluated with the same
advection scheme that transports that tracer, so the diagnostic matches the boundary flux the
tendency operator applies. For a scheme whose cloud condensate also sediments (the
non-equilibrium 1M and 2M schemes, and P3) this includes cloud droplet and cloud ice fallout,
not only rain and snow. For adaptive implicit vertical advection, this is the split-operator
flux evaluated at the current state rather than a time-integrated mass-loss diagnostic.
Schemes that move precipitation by internal means (such as `DCMIP2016KM`) extend
`surface_precipitation_flux(model, microphysics)` instead.

Building the returned `Field` assembles a kernel operation, so construct it once and
`compute!` it repeatedly rather than calling this inside a callback.

Arguments:
- `model`: An [`AtmosphereModel`](@ref) with a microphysics scheme

The generic method is implemented in `Breeze.Advection`, which owns the flux kernel and is
loaded after this module.
"""
surface_precipitation_flux(model) = surface_precipitation_flux(model, model.microphysics)

#####
##### Cloud effective radius interface
#####

"""
$(TYPEDEF)
$(TYPEDFIELDS)

Represents cloud particles with a constant effective radius in meters.
"""
struct ConstantRadiusParticles{FT}
    "Effective radius [m]"
    radius :: FT
end

"""
$(TYPEDSIGNATURES)

Return the effective radius of cloud liquid droplets in meters.

This function dispatches on the `effective_radius_model` argument. The default
implementation for `ConstantRadiusParticles` returns a constant value.

Microphysics schemes can extend this function to provide diagnosed effective radii
based on cloud properties.
"""
@inline cloud_liquid_effective_radius(i, j, k, grid, effective_radius_model::ConstantRadiusParticles, args...) =
    effective_radius_model.radius

"""
$(TYPEDSIGNATURES)

Return the effective radius of cloud ice particles in meters.

This function dispatches on the `effective_radius_model` argument. The default
implementation for [`ConstantRadiusParticles`](@ref) returns a constant value.

Microphysics schemes can extend this function to provide diagnosed effective radii
based on cloud properties.
"""
@inline cloud_ice_effective_radius(i, j, k, grid, effective_radius_model::ConstantRadiusParticles, args...) =
    effective_radius_model.radius
