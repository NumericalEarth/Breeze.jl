#####
##### Ice Lambda Limiter
#####
##### Integrals used to limit the slope parameter λ of the gamma
##### size distribution to physically reasonable values.
#####

"""
    IceLambdaLimiter{S, L}

Lambda limiter integrals for constraining the ice size distribution.

The slope parameter λ of the gamma distribution must be kept within
physical bounds. These integrals provide the limiting values for
small and large ice mass mixing ratios.

# Fields
- `small_q`: Lambda limit for small ice mass mixing ratios (i_qsmall)
- `large_q`: Lambda limit for large ice mass mixing ratios (i_qlarge)

# References

Morrison and Milbrandt (2015)
"""
struct IceLambdaLimiter{S, L}
    small_q :: S
    large_q :: L
end

"""
    IceLambdaLimiter()

Construct `IceLambdaLimiter` with quadrature-based integrals.
"""
function IceLambdaLimiter()
    return IceLambdaLimiter(
        SmallQLambdaLimit(),
        LargeQLambdaLimit()
    )
end

Base.summary(::IceLambdaLimiter) = "IceLambdaLimiter"
Base.show(io::IO, ::IceLambdaLimiter) = print(io, "IceLambdaLimiter(2 integrals)")

