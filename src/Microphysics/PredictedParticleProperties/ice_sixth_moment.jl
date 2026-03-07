#####
##### Ice Sixth Moment
#####
##### Tendencies for the 6th moment (proportional to radar reflectivity)
##### of the ice particle size distribution.
##### Used in 3-moment ice schemes.
#####

"""
    IceSixthMoment

Sixth moment (reflectivity) tendency integrals for 3-moment ice.
See [`IceSixthMoment`](@ref) constructor for details.
"""
struct IceSixthMoment{RI, DP, D1, M1, M2, AG, SH, SB, S1}
    rime :: RI
    deposition :: DP
    deposition1 :: D1
    melt1 :: M1
    melt2 :: M2
    shedding :: SH
    aggregation :: AG
    sublimation :: SB
    sublimation1 :: S1
end

"""
$(TYPEDSIGNATURES)

Construct `IceSixthMoment` with quadrature-based integrals.

The sixth moment ``M_6 = ∫ D^6 N'(D) dD`` is proportional to radar
reflectivity Z. Prognosing M₆ (or equivalently Z) as a third moment
provides an independent constraint on the shape of the size distribution,
improving representation of differential fall speeds and collection.

Each microphysical process that affects ice mass also affects M₆:

**Growth:**
- `rime`: Riming adds mass preferentially to larger particles
- `deposition`, `deposition1`: Vapor deposition with/without ventilation

**Melting:**
- `melt1`, `melt2`: Two terms in the melting tendency
- `shedding`: Meltwater that leaves the ice particle

**Collection:**
- `aggregation`: Aggregation shifts mass to larger sizes, increasing Z

**Sublimation:**
- `sublimation`, `sublimation1`: Mass loss with/without ventilation

# References

[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) introduced 3-moment ice,
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024) refined the approach.
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
