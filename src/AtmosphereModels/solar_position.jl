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

- [`ApparentSolarPosition`](@ref) — real-Earth time-varying, computed from
  the model clock and grid (or explicit) longitude/latitude.
- [`DiurnalSolarPosition`](@ref) — idealized diurnal cycle at a fixed
  latitude and declination, no calendar dependence.
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
struct FixedCosineZenith{FT} <: AbstractSolarPosition
    "Cosine of the solar zenith angle. Should satisfy ``0 ≤ \\cos(θ_z) ≤ 1`` for the sun above the horizon."
    cos_zenith :: FT
end

"""
$(TYPEDEF)

Idealized diurnal cycle with no annual variation and no calendar dependence.
cos(θ_z) is computed analytically on each radiation update from the model
clock (which must be numeric — seconds since the start of the run) as

```math
\\cos(θ_z) = \\sin(φ) \\sin(δ) + \\cos(φ) \\cos(δ) \\cos(ω),
\\qquad
ω = \\frac{2π}{T_d} (t - t_{\\text{noon}})
```

where ``φ`` is the (fixed) observer latitude, ``δ`` is the (fixed) solar
declination, ``T_d`` is the day length, and ``t_{\\text{noon}}`` is the
simulation time at which local noon occurs. ``ω = 0`` at noon and ``ω = ±π``
at local midnight. The result is clamped to be non-negative.

# Fields
$(TYPEDFIELDS)

# Examples

Perpetual equinox at 30°N (default: 24-hour day, noon at ``t = 0``):

```jldoctest
julia> using Breeze

julia> DiurnalSolarPosition(latitude = 30)
DiurnalSolarPosition(latitude = 30.0°, declination = 0.0°, day_length = 86400.0 s, noon_offset = 0.0 s)
```

Perpetual June solstice at 45°N:

```jldoctest
julia> using Breeze

julia> DiurnalSolarPosition(latitude = 45, declination = 23.5)
DiurnalSolarPosition(latitude = 45.0°, declination = 23.5°, day_length = 86400.0 s, noon_offset = 0.0 s)
```

Fast rotator with a 10-hour day, sun overhead, starting at sunrise:

```jldoctest
julia> using Breeze

julia> DiurnalSolarPosition(latitude = 0, day_length = 10 * 3600, noon_offset = 5 * 3600)
DiurnalSolarPosition(latitude = 0.0°, declination = 0.0°, day_length = 36000.0 s, noon_offset = 18000.0 s)
```
"""
struct DiurnalSolarPosition{FT} <: AbstractSolarPosition
    "Observer latitude (degrees)."
    latitude :: FT
    "Solar declination (degrees). Zero is perpetual equinox; ±23.5 is perpetual solstice."
    declination :: FT
    "Rotation period (seconds). Default `86400` is the Earth day."
    day_length :: FT
    "Simulation time (seconds) at which local noon occurs. Default `0`."
    noon_offset :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a [`DiurnalSolarPosition`](@ref) with sensible defaults: perpetual
equinox (`declination = 0`), 24-hour day (`day_length = 86400` s), and noon
at the start of the simulation (`noon_offset = 0`).

Integer inputs are promoted to floating point (via `float`) so that values
like `latitude = 30` behave like physical quantities rather than integer
counts; all four fields are then promoted to a common type so mixed inputs
(e.g. `latitude = 30, day_length = 86400.0`) work without surprises.
"""
function DiurnalSolarPosition(; latitude,
                                declination = 0,
                                day_length = 86400,
                                noon_offset = 0)
    floats = (float(latitude), float(declination), float(day_length), float(noon_offset))
    return DiurnalSolarPosition(promote(floats...)...)
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

function Base.show(io::IO, sp::DiurnalSolarPosition)
    print(io, "DiurnalSolarPosition(",
              "latitude = ",    prettysummary(sp.latitude),    "°, ",
              "declination = ", prettysummary(sp.declination), "°, ",
              "day_length = ",  prettysummary(sp.day_length),  " s, ",
              "noon_offset = ", prettysummary(sp.noon_offset), " s)")
end
