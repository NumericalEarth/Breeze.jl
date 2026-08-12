#####
##### Ice-Rain Collection
#####
##### Collection integrals for ice particles collecting rain drops.
##### These are computed for multiple rain size bins in the P3 scheme.
#####

"""
    IceRainCollection

Ice collecting rain integrals for mass and number.
See [`IceRainCollection`](@ref) constructor for details.
"""
struct IceRainCollection{QR, NR}
    mass :: QR
    number :: NR
end

"""
$(TYPEDSIGNATURES)

Construct a placeholder `IceRainCollection` with `nothing` fields.

The actual ice-rain collection integrals are double integrals over both
the ice and rain size distributions, computed in the Fortran lookup tables.
This placeholder is overwritten when tables are loaded via `read_p3_table`.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization).
"""
function IceRainCollection()
    return IceRainCollection(nothing, nothing)
end

Base.summary(::IceRainCollection) = "IceRainCollection"
Base.show(io::IO, ::IceRainCollection) = print(io, "IceRainCollection(2 integrals)")
