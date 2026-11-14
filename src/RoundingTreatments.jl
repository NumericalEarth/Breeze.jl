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

using Breeze.Utils: TYPEDSIGNATURES_WITH_DEFAULTS
using DocStringExtensions: TYPEDEF
using Oceananigans.Architectures: CPU, child_architecture

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
`StochasticRounding(; seed = 0)`. `summation` is `nothing` until the time stepper chooses
[`WidenedSum`](@ref) or [`TwoSum`](@ref) for its architecture with
[`materialize_rounding`](@ref).
"""
struct StochasticRounding{S} <: AbstractRoundingTreatment
    seed :: UInt32
    summation :: S
end

"""
$(TYPEDEF)

Stochastic rounding of a `Float32` sum by computing it in `Float64`, adding random bits below
the 24 bits of the `Float32` significand, and truncating toward zero. Other precisions fall back
to [`TwoSum`](@ref). The default on CPUs.
"""
struct WidenedSum end

"""
$(TYPEDEF)

Stochastic rounding of a sum in its own precision: TwoSum gives the exact rounding error ``e``,
and the sum moves to the neighboring float in the direction of ``e`` with probability
``|e| / \\text{gap}``. The default on GPUs, where `Float64` arithmetic is slow or unavailable.
"""
struct TwoSum end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Stochastic rounding for the stage updates of a time stepper. Each stage update ``u⁰ + δ``
is rounded to one of the two floats that bracket the exact sum, choosing the upper one with
probability equal to the distance from the lower one divided by the gap between them. The
expected value of every stage update is then the exact sum, so sub-ulp increments accumulate
correctly on average, at the cost of a random walk of about ``\\text{ulp} \\sqrt{N}`` after
``N`` steps.

How the sum is rounded depends on the architecture: on CPUs a `Float32` sum is computed in
`Float64` and truncated after adding random bits ([`WidenedSum`](@ref)); on GPUs, and for
`Float64`, its rounding error is computed with TwoSum ([`TwoSum`](@ref)). Both give the same
statistics; they differ only in speed.

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
StochasticRounding(; seed = 0) = StochasticRounding(seed % UInt32, nothing)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Return `rounding` with the summation it uses on `arch`: [`WidenedSum`](@ref) on CPUs and
[`TwoSum`](@ref) on GPUs. Treatments without a choice of summation are returned unchanged.
"""
materialize_rounding(rounding, arch) = rounding
materialize_rounding(r::StochasticRounding{Nothing}, arch) =
    StochasticRounding(r.seed, default_summation(child_architecture(arch)))

default_summation(::CPU) = WidenedSum()
default_summation(arch) = TwoSum()

Base.show(io::IO, ::RoundToNearest) = print(io, "RoundToNearest()")
Base.show(io::IO, r::StochasticRounding{Nothing}) = print(io, "StochasticRounding(seed=", Int(r.seed), ")")
Base.show(io::IO, r::StochasticRounding) =
    print(io, "StochasticRounding(seed=", Int(r.seed), ", summation=", nameof(typeof(r.summation)), "())")

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
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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

# 32 random bits for the update at (i, j, k).
@inline function random_bits(i, j, k, key)
    h = lowbias32(key ⊻ (i % UInt32))
    h = lowbias32(h ⊻ (j % UInt32) * 0x85ebca6b)
    return lowbias32(h ⊻ (k % UInt32) * 0xc2b2ae35)
end

# A uniform random number in [0, 1) with 24 random bits, exactly representable in Float32.
@inline uniform_random(i, j, k, key) = Float32(random_bits(i, j, k, key) >> 8) * Float32(2)^-24

#####
##### Rounded sums
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

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

julia> round(sum(Float64, sums .- u⁰) / length(sums), sigdigits = 1)
1.0e-5
```
"""
@inline rounded_add(::RoundToNearest, a, b, i, j, k, key) = a + b

# Compute the sum in Float64, add 29 random bits below the 24-bit Float32 significand, and
# truncate toward zero: the sum rounds up with probability equal to its distance above the
# lower Float32 neighbor divided by the gap. The truncated Float64 is exactly representable
# in Float32, so the final conversion does not round again (except for Float32 subnormals).
@inline function rounded_add(r::StochasticRounding{WidenedSum}, a::Float32, b, i, j, k, key)
    s = Float64(a) + Float64(b)
    perturbation = UInt64(random_bits(i, j, k, key) >> 3)
    truncated = (reinterpret(UInt64, s) + perturbation) & 0xffff_ffff_e000_0000
    return Float32(reinterpret(Float64, truncated))
end

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
