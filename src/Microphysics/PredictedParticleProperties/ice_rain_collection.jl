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

Construct `IceRainCollection` with quadrature-based integrals.

When ice particles collect rain drops through gravitational sweepout,
the rain freezes on contact (riming). This transfers mass, number, and
reflectivity from rain to ice.

**Conservation:**
- Mass: ``dq_r/dt < 0``, ``dq_i/dt > 0``
- Number: Rain number decreases as drops are absorbed
- Sixth moment: Transferred to ice (3-moment scheme)

The collection rate depends on the collision kernel integrating over
both size distributions. P3 uses a simplified approach with rain
binned into discrete size categories.

# Integrals

- `mass`: Rate of rain mass transfer to ice
- `number`: Rate of rain drop removal
- `sixth_moment`: Rate of Z transfer (3-moment)

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Milbrandt et al. (2021)](@cite MilbrandtEtAl2021) for sixth moment.
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
