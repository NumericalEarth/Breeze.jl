#####
##### Ice Sixth Moment
#####
##### Tendencies for the 6th moment (proportional to radar reflectivity)
##### of the ice particle size distribution.
##### Used in 3-moment ice schemes.
#####

"""
    IceSixthMoment{RI, DP, D1, M1, M2, AG, SH, SB, S1}

Sixth moment (Z, reflectivity) tendency integrals for 3-moment ice.

The 6th moment M₆ = ∫ D⁶ N'(D) dD is proportional to radar reflectivity.
Tracking M₆ as a prognostic variable allows better representation of
particle size distribution evolution.

# Fields (all integrals)

Growth processes:
- `rime`: Sixth moment tendency from riming (m6rime)
- `deposition`: Sixth moment tendency from vapor deposition (m6dep)
- `deposition1`: Sixth moment deposition with enhanced ventilation (m6dep1)

Melting processes:
- `melt1`: Sixth moment tendency from melting, term 1 (m6mlt1)
- `melt2`: Sixth moment tendency from melting, term 2 (m6mlt2)
- `shedding`: Sixth moment tendency from meltwater shedding (m6shd)

Collection processes:
- `aggregation`: Sixth moment tendency from aggregation (m6agg)

Sublimation:
- `sublimation`: Sixth moment tendency from sublimation (m6sub)
- `sublimation1`: Sixth moment sublimation with enhanced ventilation (m6sub1)

# References

Milbrandt and Morrison (2016), Milbrandt et al. (2024)
"""
struct IceSixthMoment{RI, DP, D1, M1, M2, AG, SH, SB, S1}
    # Growth
    rime :: RI
    deposition :: DP
    deposition1 :: D1
    # Melting
    melt1 :: M1
    melt2 :: M2
    shedding :: SH
    # Collection
    aggregation :: AG
    # Sublimation
    sublimation :: SB
    sublimation1 :: S1
end

"""
    IceSixthMoment()

Construct `IceSixthMoment` with quadrature-based integrals.
"""
function IceSixthMoment()
    return IceSixthMoment(
        SixthMomentRime(),
        SixthMomentDeposition(),
        SixthMomentDeposition1(),
        SixthMomentMelt1(),
        SixthMomentMelt2(),
        SixthMomentShedding(),
        SixthMomentAggregation(),
        SixthMomentSublimation(),
        SixthMomentSublimation1()
    )
end

Base.summary(::IceSixthMoment) = "IceSixthMoment"
Base.show(io::IO, ::IceSixthMoment) = print(io, "IceSixthMoment(9 integrals)")

