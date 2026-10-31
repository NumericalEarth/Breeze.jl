"""
Round-off treatments for the stage updates of Breeze's time steppers.

In reduced precision a stage update ``u⁰ + δ`` rounds away any increment ``δ`` smaller than half
a unit in the last place (ulp) of ``u⁰``. For slow tendencies (radiation, subsidence, slow
microphysics) at small time steps the loss has the same sign at every step, so it is a bias rather
than noise. The treatments here decide how that sum is rounded:

- [`RoundToNearest`](@ref): IEEE round-to-nearest, the default. Bitwise identical to Breeze's
  original stage updates and free.
- [`StochasticRounding`](@ref): rounds the sum up or down with probabilities that make it exact
  in expectation. Needs no extra storage; the random numbers come from a counter-based hash.

A new treatment subtypes [`AbstractRoundingTreatment`](@ref) and extends [`rounded_add`](@ref).
Deterministic compensated summation, which carries the lost part of each increment to the next
step, would also need a persistent field per prognostic in the time stepper, zeroed by `set!` and
saved by `prognostic_state` for checkpointing; that is left for later.
"""
module RoundingTreatments

export RoundToNearest, StochasticRounding

using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES

"""
$(TYPEDEF)

Supertype of the round-off treatments a Breeze time stepper applies to its stage updates.
Subtypes extend [`rounded_add`](@ref) and [`rounding_key`](@ref).
"""
abstract type AbstractRoundingTreatment end

"""
$(TYPEDEF)

IEEE round-to-nearest for the stage updates of a time stepper: Breeze's default. Stage updates are
bitwise identical to those of a time stepper without a round-off treatment, at no extra cost.
In single precision, increments smaller than half an ulp of the prognostic variable are lost;
see [`StochasticRounding`](@ref).

```jldoctest
using Breeze

RoundToNearest()

# output
RoundToNearest()
```
"""
struct RoundToNearest <: AbstractRoundingTreatment end

"""
$(TYPEDEF)

Stochastic rounding of the stage updates of a time stepper; see the keyword constructor
`StochasticRounding(; seed = 0)`.
"""
struct StochasticRounding <: AbstractRoundingTreatment
    seed :: UInt32
end

"""
$(TYPEDSIGNATURES)

Stochastic rounding for the stage updates of a time stepper. Each stage update
``u⁰ + δ`` is computed together with its exact rounding error ``e`` (TwoSum), and the result is
moved to the neighboring float in the direction of ``e`` with probability ``|e| / \\text{gap}``,
where ``\\text{gap}`` is the spacing between the two floats. The expected value of every stage
update is then the exact sum, so sub-ulp increments accumulate correctly on average, at the cost
of a random walk of about ``\\text{ulp} \\sqrt{N}`` after ``N`` steps.

The random numbers come from a counter-based hash of `(seed, iteration, stage, field, i, j, k)`,
so there is no random-number state: runs are deterministic, bitwise reproducible, independent of
the architecture's thread scheduling, and restart reproducibly from a checkpoint (which stores
the iteration). `seed` is reduced modulo ``2³²``.

Stochastic rounding works in any floating-point precision. It is meant for `Float32`; in `Float64`
it is supported but rarely needed, and it adds ulp-sized noise to the solution.

```jldoctest
using Breeze

StochasticRounding(seed = 42)

# output
StochasticRounding(seed=42)
```
"""
StochasticRounding(; seed = 0) = StochasticRounding(seed % UInt32)

Base.show(io::IO, ::RoundToNearest) = print(io, "RoundToNearest()")
Base.show(io::IO, r::StochasticRounding) = print(io, "StochasticRounding(seed=", Int(r.seed), ")")

#####
##### Counter-based hash
#####

# The "lowbias32" integer hash of Chris Wellons (https://nullprogram.com/blog/2018/07/31/),
# a full-avalanche 32-bit mixer: every input bit flips each output bit with probability ≈ 1/2.
@inline function lowbias32(x::UInt32)
    x ⊻= x >> 16
    x *= 0x7feb352d
    x ⊻= x >> 15
    x *= 0x846ca68b
    return x ⊻ (x >> 16)
end

"""
$(TYPEDSIGNATURES)

Return the 32-bit key that, together with the grid indices, determines the random numbers of
one stage update of one prognostic `field` (an integer that distinguishes the fields updated in
the same stage). The key mixes the seed with `iteration` and `stage`, so every update draws fresh
numbers. [`RoundToNearest`](@ref) draws none and returns zero without reading its arguments.
"""
@inline rounding_key(::AbstractRoundingTreatment, iteration, stage, field) = UInt32(0)

@inline function rounding_key(r::StochasticRounding, iteration, stage, field)
    x = r.seed ⊻ lowbias32(iteration % UInt32)
    x = lowbias32(x) ⊻ lowbias32((stage % UInt32) * 0x85ebca6b)
    return lowbias32(x ⊻ (field % UInt32) * 0xc2b2ae35)
end

# A uniform random number in [0, 1) with 24 random bits, exactly representable in Float32.
@inline function uniform_random(i, j, k, key)
    h = lowbias32(key ⊻ (i % UInt32))
    h = lowbias32(h ⊻ (j % UInt32) * 0x85ebca6b)
    h = lowbias32(h ⊻ (k % UInt32) * 0xc2b2ae35)
    return Float32(h >> 8) * Float32(2)^-24
end

#####
##### Rounded sums
#####

"""
$(TYPEDSIGNATURES)

Return ``a + b`` rounded according to the round-off treatment. `i, j, k` and `key`
(from [`rounding_key`](@ref)) seed the random number of [`StochasticRounding`](@ref).

```jldoctest
julia> using Breeze.RoundingTreatments: rounded_add, rounding_key

julia> u⁰, δ = 300f0, 1f-5;  # δ is a third of the gap between u⁰ and nextfloat(u⁰)

julia> rounded_add(RoundToNearest(), u⁰, δ, 1, 1, 1, UInt32(0))
300.0f0

julia> rounding = StochasticRounding(seed = 1);

julia> key = rounding_key(rounding, 0, 1, 1);

julia> sums = [rounded_add(rounding, u⁰, δ, i, 1, 1, key) for i in 1:100000];

julia> sort(unique(sums))
2-element Vector{Float32}:
 300.0
 300.00003

julia> round(sum(Float64, sums .- u⁰) / length(sums), sigdigits = 2)
1.0e-5
```
"""
@inline rounded_add(::RoundToNearest, a, b, i, j, k, key) = a + b

@inline function rounded_add(r::StochasticRounding, a, b, i, j, k, key)
    # TwoSum (Knuth): s + e == a + b exactly, for any magnitudes of a and b, so it also covers
    # momentum near zero, where the increment can exceed the state. No fast-math: reassociating
    # these operations would set e = 0.
    s = a + b
    z = s - a
    e = (a - (s - z)) + (b - z)

    # Round toward the neighbor on the side of the exact sum with probability |e| / gap.
    neighbor = ifelse(e > 0, nextfloat(s), prevfloat(s))
    gap = abs(neighbor - s)
    ξ = convert(typeof(s), uniform_random(i, j, k, key))

    return ifelse(ξ * gap < abs(e), neighbor, s)
end

end # module
