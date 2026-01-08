#####
##### Ice Collection
#####
##### Collision-collection integrals for ice particles.
##### Includes aggregation (ice-ice) and rain collection (ice-rain).
#####

"""
    IceCollection

Ice collision-coalescence efficiencies and collection integrals.
See [`IceCollection`](@ref) constructor for details.
"""
struct IceCollection{FT, AG, RW}
    ice_cloud_collection_efficiency :: FT
    ice_rain_collection_efficiency :: FT
    aggregation :: AG
    rain_collection :: RW
end

"""
$(TYPEDSIGNATURES)

Construct `IceCollection` with parameters and quadrature-based integrals.

Collection processes describe ice particles sweeping up other hydrometeors
through gravitational settling. Two main processes are parameterized:

**Aggregation** (ice + ice → larger ice):
Ice particles collide and stick together to form larger aggregates.
This is the dominant growth mechanism for snow. The aggregation rate
depends on the differential fall speeds of particles of different sizes.

**Rain collection** (ice + rain → rime on ice):
When ice particles collect rain drops, the liquid freezes on contact
forming rime. This is a key riming pathway along with cloud droplet
collection (handled separately in the scheme).

# Keyword Arguments

- `ice_cloud_collection_efficiency`: Eⁱᶜ [-], default 0.1
- `ice_rain_collection_efficiency`: Eⁱʳ [-], default 1.0

# References

[Morrison and Milbrandt (2015a)](@citet Morrison2015parameterization) Sections 2d-e,
[Milbrandt and Yau (2005)](@citet MilbrandtYau2005).
"""
function IceCollection(FT::Type{<:AbstractFloat} = Float64;
                       ice_cloud_collection_efficiency = 0.1,
                       ice_rain_collection_efficiency = 1.0)
    return IceCollection(
        FT(ice_cloud_collection_efficiency),
        FT(ice_rain_collection_efficiency),
        AggregationNumber(),
        RainCollectionNumber()
    )
end

Base.summary(::IceCollection) = "IceCollection"

function Base.show(io::IO, c::IceCollection)
    print(io, summary(c), "(")
    print(io, "Eⁱᶜ=", c.ice_cloud_collection_efficiency, ", ")
    print(io, "Eⁱʳ=", c.ice_rain_collection_efficiency, ")")
end

