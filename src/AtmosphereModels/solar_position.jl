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

The positional argument `FT` controls the precision of the stored fields and
defaults to `Oceananigans.defaults.FloatType`. Pass `FT = Float32` (or set
`Oceananigans.defaults.FloatType = Float32`) to run in Float32:

```julia
DiurnalSolarPosition(Float32, latitude = 30)
```
"""
function DiurnalSolarPosition(FT::DataType = Oceananigans.defaults.FloatType;
                              latitude,
                              declination = 0,
                              day_length = 86400,
                              noon_offset = 0)
    return DiurnalSolarPosition{FT}(convert(FT, latitude),
                                    convert(FT, declination),
                                    convert(FT, day_length),
                                    convert(FT, noon_offset))
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

#####
##### cos(θ_z) evaluation: fill a per-column array from a solar-position specification
#####
##### Every radiation backend stores cos(θ_z) as one value per column, indexed by
##### `column_index(i, j, Nx)`, and refreshes it before each shortwave solve. The
##### functions below own that evaluation, so a backend only needs to hand over
##### its column array.
#####

using Dates: AbstractDateTime, Millisecond
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: λnode, φnode, Center
using Breeze.CelestialMechanics: cos_solar_zenith_angle

compute_datetime(dt::AbstractDateTime, epoch) = dt
compute_datetime(t::Number, epoch::AbstractDateTime) = epoch + Millisecond(round(Int, 1000t))
# When epoch is nothing and time is numeric, we can't compute datetime (used for fixed zenith angle)
compute_datetime(t::Number, epoch::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Fill `cos_zenith` once at construction from the solar-position specification.

[`ApparentSolarPosition`](@ref) and [`DiurnalSolarPosition`](@ref) need no pre-fill:
[`update_cos_zenith!`](@ref) populates the array on the first radiation update (iteration 0
always triggers one). [`FixedCosineZenith`](@ref) writes the user-supplied value here; after this
the array is never touched, since `update_cos_zenith!` is a no-op for that case.
"""
initialize_cos_zenith!(cos_zenith, ::ApparentSolarPosition) = nothing
initialize_cos_zenith!(cos_zenith, ::DiurnalSolarPosition) = nothing

function initialize_cos_zenith!(cos_zenith, sp::FixedCosineZenith)
    cos_zenith .= convert(eltype(cos_zenith), sp.cos_zenith)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Update the per-column cosine of the solar zenith angle in `cos_zenith`, dispatched on the
solar-position specification:

- [`ApparentSolarPosition`](@ref): recompute cos(θ_z) from the model clock and
  observer (λ, φ) — either an explicit coordinate or the grid's λ/φ per column.
- [`DiurnalSolarPosition`](@ref): analytical hour angle from the (numeric) clock.
- [`FixedCosineZenith`](@ref): no-op. The array was set once at construction
  by [`initialize_cos_zenith!`](@ref).

Values are clamped to be non-negative (sun above the horizon).
"""
update_cos_zenith!(cos_zenith, ::FixedCosineZenith, grid, clock) = nothing

function update_cos_zenith!(cos_zenith, sp::ApparentSolarPosition, grid, clock)
    datetime = compute_datetime(clock.time, sp.epoch)
    validate_datetime(sp, datetime)
    update_apparent_zenith!(cos_zenith, sp.coordinate, grid, datetime)
    return nothing
end

# Idealized diurnal cycle: pure analytical hour angle, no calendar / orbit.
# Requires a numeric clock (seconds since the start of the simulation).
function update_cos_zenith!(cos_zenith, sp::DiurnalSolarPosition, grid, clock)
    validate_diurnal_clock(sp, clock.time)
    t = clock.time
    # cos is periodic, so no `mod` is required — the math handles wrapping itself.
    ω = (2π / sp.day_length) * (t - sp.noon_offset)
    φ = deg2rad(sp.latitude)
    δ = deg2rad(sp.declination)
    cos_θz = sin(φ) * sin(δ) + cos(φ) * cos(δ) * cos(ω)
    cos_zenith .= max(cos_θz, 0)
    return nothing
end

@noinline validate_diurnal_clock(::DiurnalSolarPosition, ::Number) = nothing
@noinline function validate_diurnal_clock(::DiurnalSolarPosition, t)
    throw(ArgumentError(
        "DiurnalSolarPosition requires a numeric model clock (seconds since the start " *
        "of the simulation), but `model.clock.time` is a $(typeof(t)). For an idealized " *
        "diurnal cycle there is no calendar — construct the model with " *
        "`Clock(time = 0.0)` (or another numeric Clock) instead of a DateTime clock."))
end

# Helpful actionable error when the user uses a numeric clock without an epoch.
@noinline validate_datetime(::ApparentSolarPosition, ::AbstractDateTime) = nothing
@noinline function validate_datetime(::ApparentSolarPosition, ::Nothing)
    throw(ArgumentError(
        "Cannot compute apparent solar position: the model clock holds a numeric " *
        "time and `ApparentSolarPosition.epoch` is `nothing`. Either:\n" *
        "  • use a `DateTime` clock, e.g. `Clock(time=DateTime(2024,1,1,12,0,0))`,\n" *
        "  • supply an epoch, e.g. `ApparentSolarPosition(epoch=DateTime(2024,1,1))`, or\n" *
        "  • use `FixedCosineZenith(cos_zenith)` for an idealized fixed sun."))
end

# Explicit (λ, φ): one cos(θ_z) value broadcast to every column
function update_apparent_zenith!(cos_zenith, coordinate::Tuple, grid, datetime)
    cos_θz = cos_solar_zenith_angle(datetime, coordinate...)
    cos_zenith .= max.(cos_θz, 0)
    return nothing
end

# Per-column (λ, φ) from the grid: launch a 2D kernel
function update_apparent_zenith!(cos_zenith, ::Nothing, grid, datetime)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _update_apparent_zenith!, cos_zenith, grid, datetime)
    return nothing
end

@kernel function _update_apparent_zenith!(cos_zenith, grid, datetime)
    i, j = @index(Global, NTuple)
    λ = λnode(i, j, 1, grid, Center(), Center(), Center())
    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    cos_θz = cos_solar_zenith_angle(datetime, λ, φ)
    c = column_index(i, j, grid.Nx)
    @inbounds cos_zenith[c] = max(cos_θz, 0)  # Clamp to positive (sun above horizon)
end
