#####
##### Ice-Rain Collection
#####
##### Collection integrals for ice particles collecting rain drops.
##### These are computed for multiple rain size bins in the P3 scheme.
#####

"""
    IceRainCollection

Ice collecting rain integrals for mass, number, and sixth moment.
See [`IceRainCollection`](@ref) constructor for details.
"""
struct IceRainCollection{QR, NR, ZR}
    mass :: QR
    number :: NR
    sixth_moment :: ZR
end

"""
$(TYPEDSIGNATURES)

Construct a placeholder `IceRainCollection` with `nothing` fields.

The actual ice-rain collection integrals are double integrals over both
the ice and rain size distributions, computed in the Fortran lookup tables.
This placeholder is overwritten when tables are loaded via `read_p3_table`.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) for sixth moment.
"""
function IceRainCollection()
    return IceRainCollection(nothing, nothing, nothing)
end

Base.summary(::IceRainCollection) = "IceRainCollection"
Base.show(io::IO, ::IceRainCollection) = print(io, "IceRainCollection(3 integrals)")
