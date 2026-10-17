using Adapt: Adapt, adapt
using Oceananigans.Architectures: CPU, on_architecture
using Oceananigans: Oceananigans
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: location
using Oceananigans.Grids: znode

"""
$(TYPEDSIGNATURES)

Construct a linearly interpolated profile from increasing `heights`, increasing
`times`, and a matrix `values[height_index, time_index]`. Coordinates and values
must be finite. Values outside the tabulated interval are held at the nearest
endpoint. Call `profile(z, t)` to evaluate it; `profile(x, y, z, t)` also works
as an Oceananigans forcing. Times are seconds on the simulation clock.

The arrays are copied onto `architecture`. Updating `profile.values` in place
changes the forcing without reconstructing the model.

```jldoctest
profile = TimeHeightProfile([0.0, 100.0], [0.0, 10.0], [1.0 3.0; 2.0 4.0])
profile(50.0, 5.0)

# output
2.5
```
"""
struct TimeHeightProfile{Z, T, V}
    heights :: Z
    times :: T
    values :: V
end

function TimeHeightProfile(heights::AbstractVector, times::AbstractVector, values::AbstractMatrix;
                           architecture=CPU())
    length(heights) ≥ 2 || throw(ArgumentError("At least two heights are required."))
    length(times) ≥ 2 || throw(ArgumentError("At least two times are required."))
    size(values) == (length(heights), length(times)) ||
        throw(DimensionMismatch("Values must have size (length(heights), length(times))."))
    all(isfinite, heights) && all(diff(heights) .> 0) ||
        throw(ArgumentError("Heights must be finite and strictly increasing."))
    all(isfinite, times) && all(diff(times) .> 0) ||
        throw(ArgumentError("Times must be finite and strictly increasing."))
    all(isfinite, values) || throw(ArgumentError("Profile values must be finite."))
    z = on_architecture(architecture, collect(heights))
    t = on_architecture(architecture, collect(times))
    v = on_architecture(architecture, copy(values))
    return TimeHeightProfile{typeof(z), typeof(t), typeof(v)}(z, t, v)
end

function Adapt.adapt_structure(to, profile::TimeHeightProfile)
    z = adapt(to, profile.heights)
    t = adapt(to, profile.times)
    v = adapt(to, profile.values)
    return TimeHeightProfile{typeof(z), typeof(t), typeof(v)}(z, t, v)
end

@inline function profile_bracket(coordinates, x)
    low = 1
    high = length(coordinates)
    # A binary search needs at most ceil(log2(N - 1)) iterations. A fixed
    # trip count lets Reactant raise batched searches whose intervals shrink
    # at different rates. Keep a completed interval unchanged.
    iterations = 8sizeof(Int) - leading_zeros(length(coordinates) - 2)
    for iteration in 1:iterations
        middle = (low + high) ÷ 2
        before = @inbounds x < coordinates[middle]
        searching = high - low > 1
        high = ifelse(searching & before, middle, high)
        low = ifelse(searching & !before, middle, low)
    end
    @inbounds fraction = clamp((x - coordinates[low]) / (coordinates[high] - coordinates[low]), 0, 1)
    return low, fraction
end

@inline function (profile::TimeHeightProfile)(z, t)
    k, ζ = profile_bracket(profile.heights, z)
    n, τ = profile_bracket(profile.times, t)
    @inbounds begin
        a = (1 - ζ) * profile.values[k, n] + ζ * profile.values[k+1, n]
        b = (1 - ζ) * profile.values[k, n+1] + ζ * profile.values[k+1, n+1]
    end
    return (1 - τ) * a + τ * b
end

@inline (profile::TimeHeightProfile)(x, y, z, t) = profile(z, t)

@inline function time_height_value(i, j, k, grid, profile, time, ℓx, ℓy, ℓz)
    z = znode(i, j, k, grid, ℓx, ℓy, ℓz)
    return profile(z, time)
end

"""
$(TYPEDSIGNATURES)

Evaluate a `TimeHeightProfile` on `field` at `time`, on the field's architecture.
The profile must already be on that architecture.
"""
function Oceananigans.set!(field::Field, profile::TimeHeightProfile, time::Number)
    LX, LY, LZ = location(field)
    # Reduced horizontal dimensions are evaluated at cell centers.
    horizontal_location(ℓ) = ℓ === Nothing ? Center() : ℓ()
    operation = KernelFunctionOperation{LX, LY, LZ}(time_height_value, field.grid, profile, eltype(field)(time),
                                                    horizontal_location(LX), horizontal_location(LY), LZ())
    field .= operation
    return field
end
