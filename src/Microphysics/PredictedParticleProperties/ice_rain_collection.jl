#####
##### Ice-Rain Collection
#####
##### Collection integrals for ice particles collecting rain drops.
##### These are computed for multiple rain size bins in the P3 scheme.
#####

"""
    IceRainCollection{QR, NR, ZR}

Ice-rain collection integrals.

When ice particles collect rain drops, mass, number, and (for 3-moment)
sixth moment are transferred from rain to ice. These integrals are
computed for multiple rain size bins.

# Fields
- `mass`: Mass collection rate (rain mass → ice mass)
- `number`: Number collection rate (rain number reduction)
- `sixth_moment`: Sixth moment collection rate (3-moment ice)

# References

Morrison and Milbrandt (2015), Milbrandt and Morrison (2016)
"""
struct IceRainCollection{QR, NR, ZR}
    mass :: QR
    number :: NR
    sixth_moment :: ZR
end

"""
    IceRainCollection()

Construct `IceRainCollection` with quadrature-based integrals.
"""
function IceRainCollection()
    return IceRainCollection(
        IceRainMassCollection(),
        IceRainNumberCollection(),
        IceRainSixthMomentCollection()
    )
end

Base.summary(::IceRainCollection) = "IceRainCollection"
Base.show(io::IO, ::IceRainCollection) = print(io, "IceRainCollection(3 integrals)")

