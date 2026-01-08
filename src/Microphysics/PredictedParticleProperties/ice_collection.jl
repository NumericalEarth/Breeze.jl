#####
##### Ice Collection
#####
##### Collision-collection integrals for ice particles.
##### Includes aggregation (ice-ice) and rain collection (ice-rain).
#####

"""
    IceCollection{FT, AG, RW}

Ice collection (collision-coalescence) properties and integrals.

Collection processes include:
- Aggregation: ice particles collecting other ice particles
- Rain collection: ice particles collecting rain drops (riming of rain)

# Fields

## Parameters
- `ice_cloud_collection_efficiency`: Collection efficiency for ice-cloud collisions [-]
- `ice_rain_collection_efficiency`: Collection efficiency for ice-rain collisions [-]

## Integrals
- `aggregation`: Number tendency from ice-ice aggregation
- `rain_collection`: Number tendency from rain collection by ice

# References

Morrison and Milbrandt (2015), Milbrandt and Yau (2005)
"""
struct IceCollection{FT, AG, RW}
    # Parameters
    ice_cloud_collection_efficiency :: FT
    ice_rain_collection_efficiency :: FT
    # Integrals
    aggregation :: AG
    rain_collection :: RW
end

"""
    IceCollection(FT=Float64)

Construct `IceCollection` with default parameters and quadrature-based integrals.
"""
function IceCollection(FT::Type{<:AbstractFloat} = Float64)
    return IceCollection(
        FT(0.1),   # ice_cloud_collection_efficiency [-]
        FT(1.0),   # ice_rain_collection_efficiency [-]
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

