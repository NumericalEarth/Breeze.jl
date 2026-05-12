#####
##### Solar position specifications for RadiativeTransferModel
#####
##### A `solar_position` tells the radiative transfer model how to obtain
##### cos(θ_z) — the cosine of the solar zenith angle — at each radiation
##### update. Two concrete subtypes of `AbstractSolarPosition` cover the
##### common cases:
#####
##### - `ApparentSolarPosition` — compute cos(θ_z) from the model clock and
#####   either an explicit `(longitude, latitude)` or the grid's coordinates.
##### - `FixedCosineZenith` — hold cos(θ_z) at a constant value, independent
#####   of the clock.
#####
##### New modes (e.g., diurnally averaged, prescribed time series) can be
##### added as new subtypes of `AbstractSolarPosition`.
#####

"""
$(TYPEDEF)

Abstract supertype for solar-position specifications passed to
[`RadiativeTransferModel`](@ref). Concrete subtypes determine how cos(θ_z)
is computed on each radiation update:

- [`ApparentSolarPosition`](@ref) — time-varying, computed from the model
  clock and grid (or explicit) longitude/latitude.
- [`FixedCosineZenith`](@ref) — constant cos(θ_z), clock-independent.
"""
abstract type AbstractSolarPosition end

"""
$(TYPEDEF)

Time-varying apparent solar position. The cosine of the solar zenith angle is
recomputed on each radiation update from the model clock and either the grid's
``(λ, φ)`` coordinates (when `coordinate === nothing`, the default) or an
explicit `(longitude, latitude)` tuple stored in `coordinate`.

When the model clock holds a floating-point time (in seconds), `epoch::DateTime`
provides the absolute reference against which `clock.time` is resolved. With a
`DateTime` clock, `epoch` is ignored.

# Fields
$(TYPEDFIELDS)
"""
struct ApparentSolarPosition{C, E} <: AbstractSolarPosition
    "Observer longitude/latitude. Either `nothing` (use grid coordinates) or a `(longitude, latitude)` tuple in degrees."
    coordinate :: C
    "DateTime anchor for floating-point clocks. Either `nothing` (requires a DateTime clock) or a `DateTime`."
    epoch :: E
end

"""
$(TYPEDSIGNATURES)

Construct an [`ApparentSolarPosition`](@ref) with optional `coordinate` and `epoch`.

```jldoctest
julia> using Breeze, Dates

julia> ApparentSolarPosition()
ApparentSolarPosition(coordinate=<from grid>, epoch=<from clock>)

julia> ApparentSolarPosition(coordinate = (-70.9, 42.5))
ApparentSolarPosition(coordinate=(-70.9, 42.5), epoch=<from clock>)

julia> ApparentSolarPosition(epoch = DateTime(2024, 1, 1))
ApparentSolarPosition(coordinate=<from grid>, epoch=2024-01-01T00:00:00)
```
"""
ApparentSolarPosition(; coordinate = nothing, epoch = nothing) =
    ApparentSolarPosition(coordinate, epoch)

"""
$(TYPEDEF)

Constant cosine of the solar zenith angle. The model clock has no effect on
the sun position; the shortwave path length is fixed at ``1 / \\cos(θ_z)`` and
the top-of-atmosphere downward shortwave flux is `solar_constant * cos_zenith`.

This is the appropriate choice for idealized studies (radiative-convective
equilibrium, RCE intercomparisons) where a diurnal or annual mean is desired.
Common values: ``\\cos(θ_z) = 0.5`` for diurnal mean at mid-latitudes,
``\\cos(θ_z) ≈ 0.41`` for the global annual mean.

# Fields
$(TYPEDFIELDS)

# Example

```jldoctest
julia> using Breeze

julia> FixedCosineZenith(0.5)
FixedCosineZenith(cos_zenith = 0.5)
```
"""
struct FixedCosineZenith{FT<:Number} <: AbstractSolarPosition
    "Cosine of the solar zenith angle. Should satisfy ``0 ≤ \\cos(θ_z) ≤ 1`` for the sun above the horizon."
    cos_zenith :: FT
end

#####
##### show methods
#####

# Use `prettysummary` so Float32 values display without the `f0` suffix
# (consistent with Oceananigans' show output, and stable across precision).
_show_coordinate(::Nothing) = "<from grid>"
_show_coordinate(coord::Tuple) = "(" * prettysummary(coord[1]) * ", " * prettysummary(coord[2]) * ")"

_show_epoch(::Nothing) = "<from clock>"
_show_epoch(epoch) = string(epoch)

function Base.show(io::IO, sp::ApparentSolarPosition)
    print(io, "ApparentSolarPosition(coordinate=", _show_coordinate(sp.coordinate),
              ", epoch=", _show_epoch(sp.epoch), ")")
end

Base.show(io::IO, sp::FixedCosineZenith) =
    print(io, "FixedCosineZenith(cos_zenith = ", prettysummary(sp.cos_zenith), ")")
