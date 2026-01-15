#####
##### Ice Lambda Limiter
#####
##### Integrals used to limit the slope parameter λ of the gamma
##### size distribution to physically reasonable values.
#####

"""
    IceLambdaLimiter

Integrals for constraining λ to physical bounds.
See [`IceLambdaLimiter`](@ref) constructor for details.
"""
struct IceLambdaLimiter{S, L}
    small_q :: S
    large_q :: L
end

"""
$(TYPEDSIGNATURES)

Construct `IceLambdaLimiter` with quadrature-based integrals.

The slope parameter λ of the gamma size distribution can become
unrealistically large or small as prognostic moments evolve. This
happens at edges of mixed-phase regions or during rapid microphysical
adjustments.

**Physical interpretation:**
- Very large λ → all particles tiny (mean size → 0)
- Very small λ → all particles huge (mean size → ∞)

These integrals compute the limiting values:
- `small_q`: λ limit when q is small (prevents vanishingly tiny particles)
- `large_q`: λ limit when q is large (prevents unrealistically huge particles)

The limiter ensures the diagnosed size distribution remains physically
sensible even when the prognostic constraints become degenerate.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b.
"""
function IceLambdaLimiter()
    return IceLambdaLimiter(
        NumberMomentLambdaLimit(),
        MassMomentLambdaLimit()
    )
end

Base.summary(::IceLambdaLimiter) = "IceLambdaLimiter"
Base.show(io::IO, ::IceLambdaLimiter) = print(io, "IceLambdaLimiter(2 integrals)")
