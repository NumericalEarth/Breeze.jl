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
# `compute_microphysical_tendencies!` is the only entry point.
#####

using Oceananigans.Fields: set!, ZeroField, ZFaceField
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶜ

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

#####
##### Fused microphysical tendency interface
#####
#
# `compute_microphysical_tendencies!` is the single entry point through which the
# atmosphere model invokes microphysics during tendency assembly. The model calls it
# *after* the per-tracer dynamics kernels have written advection + diffusion + forcing
# into `Gⁿ`; microphysics contributions are added on top via `+=`.

"""
$(TYPEDSIGNATURES)

Add microphysics tendency contributions to the model's `Gⁿ` fields.

This is the only entry point through which `compute_tendencies!` invokes microphysics.
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

Return the total initial aerosol number concentration [m⁻³] for a microphysics scheme.

This is used by [`initialize_model_microphysical_fields!`](@ref) and parcel model
construction to set a physically meaningful default for the prognostic aerosol number
density `ρnᵃ`. The value is derived from the aerosol size distribution stored in the
microphysics scheme, so it stays consistent with the activation parameters.

Returns `0` by default; extensions override this for schemes with prognostic aerosol.
"""
initial_aerosol_number(microphysics) = 0

"""
$(TYPEDSIGNATURES)

Initialize default values for microphysical fields after materialization.

Sets `ρnᵃ` (aerosol number density) to [`initial_aerosol_number(microphysics)`](@ref)
if the field exists. All other microphysical fields remain at zero.
Users can override with `set!`.
"""
initialize_model_microphysical_fields!(fields, ::Nothing) = nothing

function initialize_model_microphysical_fields!(fields, microphysics)
    if :ρnᵃ ∈ keys(fields)
        set!(fields.ρnᵃ, initial_aerosol_number(microphysics))
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
##### Sedimentation velocity interface
#####
#
# The sedimentation_velocity interface returns a vertical velocity component [m/s].
# Falling hydrometeors have negative w because z is positive upward.
#
# Microphysics schemes implement:
#   sedimentation_velocity(microphysics, microphysical_fields, name) → field or nothing
#   moisture_phase(microphysics, name) → Val(:liquid), Val(:ice), or nothing
#
# The generic microphysical_velocities wrapper calls sedimentation_velocity and
# uses the result as the w component.

"""
$(TYPEDSIGNATURES)

Return the sedimentation velocity field (vertical component, [m/s]) for
the prognostic tracer `name`, or `nothing` if the tracer does not sediment.

Microphysics schemes should extend this function for each sedimenting tracer.
"""
@inline sedimentation_velocity(microphysics, microphysical_fields, name) = nothing
@inline sedimentation_velocity(microphysics::Nothing, microphysical_fields, name) = nothing

"""
$(TYPEDSIGNATURES)

Return the moisture phase (`Val(:liquid)` or `Val(:ice)`) associated with tracer `name`,
or `nothing` if the tracer has no defined phase.

Microphysics schemes should extend this function.
"""
@inline moisture_phase(microphysics, name) = nothing
@inline moisture_phase(microphysics::Nothing, name) = nothing

"""
    NegatedField{F}

A lightweight wrapper that negates field values on access.
"""
struct NegatedField{F}
    field :: F
end

@inline Base.getindex(nf::NegatedField, i, j, k) = -@inbounds nf.field[i, j, k]

Adapt.adapt_structure(to, nf::NegatedField) = NegatedField(adapt(to, nf.field))

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities associated with `microphysics`, `microphysical_fields`, and tracer `name`.

Must be either `nothing`, or a NamedTuple with three components `u, v, w`.
The velocities are added to the bulk flow velocities for advecting the tracer.
For example, the terminal velocity of falling rain.

The generic implementation calls [`sedimentation_velocity`](@ref) and uses
the result as the vertical velocity component.
"""
@inline microphysical_velocities(microphysics::Nothing, microphysical_fields, name) = nothing

@inline function microphysical_velocities(microphysics, microphysical_fields, name)
    w = sedimentation_velocity(microphysics, microphysical_fields, name)
    return sedimentation_velocity_tuple(w)
end

@inline sedimentation_velocity_tuple(::Nothing) = nothing
@inline sedimentation_velocity_tuple(w) = (; u = ZeroField(), v = ZeroField(), w)

#####
##### Effective sedimentation velocities
#####
#
# Precomputed effective sedimentation velocities for total liquid and total ice.
# These are mass-weighted averages over all sedimenting species within each phase:
#
#   wᴸ = Σᵢ(wᵢ * qᵢ) / Σᵢ(qᵢ)    for liquid species
#   wᴵ = Σᵢ(wᵢ * qᵢ) / Σᵢ(qᵢ)    for ice species

"""
$(TYPEDSIGNATURES)

Return `nothing` when microphysics is disabled.
"""
materialize_sedimentation_velocities(::Nothing, microphysical_fields, grid) = nothing

"""
$(TYPEDSIGNATURES)

Allocate `ZFaceField`s for effective sedimentation velocities and return a NamedTuple
with keys `ρqᴸ` and `ρqᴵ`, each containing a velocity NamedTuple `(u, v, w)`.

The `w` component of each entry is a `ZFaceField` that will be updated each
time step with the mass-weighted sedimentation velocity for the corresponding phase.
"""
function materialize_sedimentation_velocities(microphysics, microphysical_fields, grid)
    w_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wᴸ = ZFaceField(grid; boundary_conditions=w_bcs)
    wᴵ = ZFaceField(grid; boundary_conditions=w_bcs)
    return (; ρqᴸ = (; u = ZeroField(), v = ZeroField(), w = wᴸ),
              ρqᴵ = (; u = ZeroField(), v = ZeroField(), w = wᴵ))
end

"""
$(TYPEDSIGNATURES)

Build a tuple of `(sedimentation_velocity_field, humidity_field)` pairs for prognostic
mass tracers (names starting with "ρq") that match the given `phase` (`:liquid` or `:ice`).

Each pair consists of the sedimentation velocity field and the corresponding specific
humidity field (e.g., `:ρqʳ` maps to `:qʳ`).
"""
sedimentation_constituent(microphysics, μ, ::Tuple{}, phase) = ()

function sedimentation_constituent(microphysics, μ, names::Tuple{Symbol, Vararg}, phase)
    name = first(names)
    rest = sedimentation_constituent(microphysics, μ, Base.tail(names), phase)
    s = string(name)
    is_mass_tracer = length(s) >= 2 && s[1] == 'ρ' && s[nextind(s, 1)] == 'q'
    if is_mass_tracer && moisture_phase(microphysics, Val(name)) === phase
        w = sedimentation_velocity(microphysics, μ, Val(name))
        if !isnothing(w)
            specific_name = specific_field_name(name)
            q_field = getproperty(μ, specific_name)
            return ((w, q_field), rest...)
        end
    end
    return rest
end

"""
$(TYPEDSIGNATURES)

No-op when effective sedimentation velocities are disabled.
"""
update_sedimentation_velocities!(::Nothing, microphysics, microphysical_fields) = nothing

"""
$(TYPEDSIGNATURES)

Update the precomputed effective sedimentation velocity fields from the current
microphysical state. Builds liquid and ice sedimentation constituents and
launches the kernel to compute mass-weighted averages.
"""
function update_sedimentation_velocities!(sedimentation_velocities, microphysics, microphysical_fields)
    wᴸ = sedimentation_velocities.ρqᴸ.w
    wᴵ = sedimentation_velocities.ρqᴵ.w
    grid = wᴸ.grid
    arch = grid.architecture
    names = prognostic_field_names(microphysics)
    liquid_components = sedimentation_constituent(microphysics, microphysical_fields, names, Val(:liquid))
    ice_components = sedimentation_constituent(microphysics, microphysical_fields, names, Val(:ice))
    launch!(arch, grid, :xyz,
            _compute_sedimentation_velocities!,
            wᴸ, wᴵ, grid, liquid_components, ice_components)
    return nothing
end

@kernel function _compute_sedimentation_velocities!(wᴸ, wᴵ, grid, liquid_components, ice_components)
    i, j, k = @index(Global, NTuple)

    # Liquid phase
    numerator_l = weighted_sedimentation_velocity_sum(i, j, k, liquid_components)
    denominator_l = humidity_sum(i, j, k, liquid_components)
    wᴸ_value = ifelse(denominator_l > 0, numerator_l / denominator_l, zero(grid))
    @inbounds wᴸ[i, j, k] = wᴸ_value

    # Ice phase
    numerator_i = weighted_sedimentation_velocity_sum(i, j, k, ice_components)
    denominator_i = humidity_sum(i, j, k, ice_components)
    wᴵ_value = ifelse(denominator_i > 0, numerator_i / denominator_i, zero(grid))
    @inbounds wᴵ[i, j, k] = wᴵ_value
end

# Recursive sum: Σ(wᵢ * qᵢ)
@inline weighted_sedimentation_velocity_sum(i, j, k, ::Tuple{}) = 0
@inline function weighted_sedimentation_velocity_sum(i, j, k, components::Tuple)
    w_field, q_field = first(components)
    @inbounds w_val = w_field[i, j, k]
    @inbounds q_val = q_field[i, j, k]
    return w_val * max(0, q_val) + weighted_sedimentation_velocity_sum(i, j, k, Base.tail(components))
end

# Recursive sum: Σ qᵢ
@inline humidity_sum(i, j, k, ::Tuple{}) = 0
@inline function humidity_sum(i, j, k, components::Tuple)
    _, q_field = first(components)
    @inbounds q_val = q_field[i, j, k]
    return max(0, q_val) + humidity_sum(i, j, k, Base.tail(components))
end

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

Return a 2D `Field` representing the flux of precipitating moisture at the bottom boundary.

The surface precipitation flux is computed using the same advection scheme that
transports precipitating tracers during time stepping, evaluated at the bottom
face (`k = 1`). This ensures numerical consistency between the diagnosed flux and
the actual mass leaving the domain through the advection operator ``\\nabla \\cdot (\\rho \\boldsymbol{U} c)``.

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
