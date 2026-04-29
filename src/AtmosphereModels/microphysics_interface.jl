#####
##### Microphysics interface (default implementations)
#####
#
# This file defines the interface that all microphysics implementations must provide.
# The key abstraction is the MicrophysicalState (ℳ), which enables the same tendency
# functions to work for any dynamics (grid-based LES, parcel models, etc.).
#
# The workflow is:
#   ℳ = grid_microphysical_state(i, j, k, grid, microphysics, fields, ρ, 𝒰)
#   tendency = microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)
#
# The grid-indexed interface provides a default fallback that builds ℳ and dispatches
# to the state-based tendency. Schemes needing full grid access can override directly.
#####

using Oceananigans.Fields: set!
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
##### Grid-indexed tendency interface (default fallback)
#####

"""
    grid_microphysical_tendency(i, j, k, grid, microphysics, name, ρ, fields, 𝒰, constants, velocities)

Compute the tendency for microphysical variable `name` at grid point `(i, j, k)`.

This is the **grid-indexed** interface used by the tendency kernels. The default
implementation builds the microphysical state `ℳ` via [`microphysical_state`](@ref)
and dispatches to the state-based [`microphysical_tendency`](@ref).

Schemes that need full grid access (e.g., for non-local operations) can override
this method directly without using `microphysical_state`.

# Arguments
- `velocities`: NamedTuple of velocity components `(; u, v, w)` [m/s].
"""
# Default (no cache): build microphysical state and dispatch to microphysical_tendency.
@inline function grid_microphysical_tendency(i, j, k, grid, microphysics, name, ::Nothing,
                                             ρ, fields, 𝒰, constants, velocities)
    ℳ = grid_microphysical_state(i, j, k, grid, microphysics, fields, ρ, 𝒰, velocities)
    return microphysical_tendency(microphysics, name, ρ, ℳ, 𝒰, constants)
end

# Cache hit: read precomputed tendency. Val{N}+haskey compile-time => cache misses are branch-free zero.
@inline function grid_microphysical_tendency(i, j, k, grid, microphysics, ::Val{N},
                                             cache::NamedTuple,
                                             ρ, fields, 𝒰, constants, velocities) where N
    return haskey(cache, N) ? @inbounds(cache[N][i, j, k]) : zero(eltype(grid))
end

# Nothing microphysics — always zero, regardless of cache type.
@inline grid_microphysical_tendency(i, j, k, grid, ::Nothing, name, ::Nothing,
                                    ρ, μ, 𝒰, constants, velocities) = zero(eltype(grid))
@inline grid_microphysical_tendency(i, j, k, grid, ::Nothing, ::Val{N}, cache::NamedTuple,
                                    ρ, μ, 𝒰, constants, velocities) where N =
    haskey(cache, N) ? @inbounds(cache[N][i, j, k]) : zero(eltype(grid))

# Transitional forwarders: callers passing no cache argument route to the no-cache method.
# (To be removed in Task 6 once all call sites pass an explicit cache argument.)
@inline grid_microphysical_tendency(i, j, k, grid, microphysics, name, ρ, fields, 𝒰, constants, velocities) =
    grid_microphysical_tendency(i, j, k, grid, microphysics, name, nothing, ρ, fields, 𝒰, constants, velocities)
@inline grid_microphysical_tendency(i, j, k, grid, ::Nothing, name, ρ, μ, 𝒰, constants, velocities) =
    zero(eltype(grid))

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

Return `tuple()` - zero-moment scheme has no prognostic variables.
"""
prognostic_field_names(::Nothing) = tuple()


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

"""
$(TYPEDSIGNATURES)

Return the microphysical velocities associated with `microphysics`, `microphysical_fields`, and tracer `name`.

Must be either `nothing`, or a NamedTuple with three components `u, v, w`.
The velocities are added to the bulk flow velocities for advecting the tracer.
For example, the terminal velocity of falling rain.
"""
@inline microphysical_velocities(microphysics::Nothing, microphysical_fields, name) = nothing

# NOTE: The grid-indexed fallback for Nothing microphysics is defined above (line 159)
# via the generic fallback mechanism which calls the state-based method.

"""
    microphysics_model_update!(microphysics, model::AtmosphereModel)
    microphysics_model_update!(microphysics, model, Δt_eff)

Apply the operator-split microphysics state update for `microphysics` on `model`.
The `Δt_eff` argument is the effective integration window — for unscheduled
microphysics it equals `model.clock.last_Δt`; for scheduled microphysics it is
the wall-clock interval since the last firing.

Specific microphysics schemes extend the 3-argument form. The 2-argument shim
forwards `model.clock.last_Δt` so existing call sites keep working.

!!! warning
    Scheme implementations MUST extend the 3-argument form
    `microphysics_model_update!(microphysics, model, Δt_eff)`.
    Do NOT add a 2-argument overload — it would shadow this shim and break
    scheduled-microphysics call sites that pass an explicit `Δt_eff`.
"""
function microphysics_model_update! end

microphysics_model_update!(::Nothing, model, Δt_eff) = nothing

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

The surface precipitation flux is ``wʳ ρqʳ`` at the bottom face (`k = 1`), representing
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

"""
    MicrophysicsScheduleState{FT}

Mutable state held by `AtmosphereModel` when scheduled microphysics is active.
Tracks the time and iteration of the last microphysics firing so the driver
can compute `Δt_eff = clock.time - last_fire_time`.
"""
mutable struct MicrophysicsScheduleState{FT}
    last_fire_time      :: FT
    last_fire_iteration :: Int
end

MicrophysicsScheduleState(FT::DataType) = MicrophysicsScheduleState{FT}(zero(FT), -1)

"""
$(TYPEDSIGNATURES)

Allocate the cached microphysics tendency NamedTuple for `microphysics` on `grid`,
keyed by the prognostic names microphysics contributes to (the thermodynamic
prognostic name, the moisture prognostic, and `prognostic_field_names(microphysics)`).

Returns `nothing` when `schedule === nothing` (the default — no caching).
"""
materialize_microphysics_tendencies(microphysics, formulation, ::Nothing, grid) = nothing

function materialize_microphysics_tendencies(microphysics, formulation, schedule, grid)
    thermo_name   = thermodynamic_density_name(formulation)
    moisture_name = moisture_prognostic_name(microphysics)
    micro_names   = prognostic_field_names(microphysics)
    names = (thermo_name, moisture_name, micro_names...)
    fields = NamedTuple{names}(ntuple(_ -> CenterField(grid), length(names)))
    return fields
end

"""
$(TYPEDSIGNATURES)

Fill the cached microphysics tendency `cache` for `microphysics` on `model`.
Builds `𝒰` and `ℳ` once per grid point and writes the tendency for every
prognostic name in `keys(cache)` via static iteration over `Val(name)`.

`Δt_eff` is forwarded for diagnostic / forward-Euler-style schemes that use it;
the standard inline path ignores it.
"""
function compute_microphysics_tendencies!(cache, microphysics, model, Δt_eff)
    cache === nothing && return nothing
    grid = model.grid
    arch = grid.architecture
    fields = model.microphysical_fields
    velocities = model.velocities
    constants = model.thermodynamic_constants
    formulation = model.formulation
    dynamics = model.dynamics
    moisture = specific_prognostic_moisture(model)
    names = Val(keys(cache))

    launch!(arch, grid, :xyz,
            _compute_microphysics_tendencies!,
            cache, names, grid, microphysics, fields, formulation, dynamics, moisture, constants, velocities)
    return nothing
end

compute_microphysics_tendencies!(::Nothing, microphysics, model, Δt_eff) = nothing

@kernel function _compute_microphysics_tendencies!(cache, ::Val{names}, grid, microphysics,
                                                   fields, formulation, dynamics, moisture, constants, velocities) where names
    i, j, k = @index(Global, NTuple)

    ρ_field = dynamics_density(dynamics)
    @inbounds ρ = ρ_field[i, j, k]
    @inbounds qᵛᵉ = moisture[i, j, k]

    q = grid_moisture_fractions(i, j, k, grid, microphysics, ρ, qᵛᵉ, fields)
    𝒰 = diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
    ℳ = grid_microphysical_state(i, j, k, grid, microphysics, fields, ρ, 𝒰, velocities)

    ntuple(Val(length(names))) do n
        Base.@_inline_meta
        name = names[n]
        @inbounds cache[name][i, j, k] = microphysical_tendency(microphysics, Val(name), ρ, ℳ, 𝒰, constants)
        nothing
    end
end

"""
$(TYPEDSIGNATURES)

Drive microphysics for one `update_state!` cycle. When `model.microphysics_schedule`
is `nothing`, falls back to per-step behavior identical to the previous code path.

When a schedule is set, the operator-split state update and the cache refill are
both gated by `schedule(model)` (with a forced firing on the first iteration).
On firing, both receive `Δt_eff = clock.time − last_fire_time`.
"""
function update_microphysics!(model)
    return update_microphysics!(model.microphysics, model.microphysics_schedule, model)
end

# Unscheduled path: behaves as before.
function update_microphysics!(microphysics, ::Nothing, model)
    microphysics_model_update!(microphysics, model, model.clock.last_Δt)
    return nothing
end

# Scheduled path.
function update_microphysics!(microphysics, schedule, model)
    state = model.microphysics_state
    clock = model.clock
    first = clock.iteration == 0 && state.last_fire_iteration < 0

    if first || schedule(model)
        Δt_eff = first ? clock.last_Δt : (clock.time - state.last_fire_time)
        microphysics_model_update!(microphysics, model, Δt_eff)
        compute_microphysics_tendencies!(model.microphysics_tendencies, microphysics, model, Δt_eff)
        state.last_fire_time = clock.time
        state.last_fire_iteration = clock.iteration
    end
    return nothing
end
