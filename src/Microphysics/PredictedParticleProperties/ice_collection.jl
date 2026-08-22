#####
##### Ice Collection
#####
##### Single-ice-PSD collision-collection integrals: ice-ice aggregation,
##### ice-cloud-water collection, and aerosol scavenging.
#####

"""
    IceCollection

Ice collision-coalescence integrals over the ice size distribution.
See [`IceCollection`](@ref) constructor for details.
"""
struct IceCollection{AG, CW, WA, IA}
    aggregation :: AG
    cloud_collection :: CW
    cloud_aerosol_collection :: WA
    ice_aerosol_collection :: IA
end

"""
$(TYPEDSIGNATURES)

Construct `IceCollection` with placeholder (`nothing`) integrals, following the
materialization pattern: [`read_lookup_tables`](@ref) replaces each field
with the corresponding tabulated integral.

Collection processes describe ice particles sweeping up other hydrometeors
through gravitational settling. The integrals held here are the ones that depend
on the ice size distribution alone:

**Aggregation** (ice + ice → larger ice):
Ice particles collide and stick together to form larger aggregates. This is the
dominant growth mechanism for snow, and depends on the differential fall speeds of
particles of different sizes. Consumed by [`ice_aggregation_rate`](@ref).

**Cloud collection** (ice + cloud droplets → rime on ice):
The PSD-integrated sweep-out kernel ``\\int V(D) A(D) N'(D) \\, dD`` [m³/s] per
particle, with the collision kernel set to zero for ice diameters below 100 μm.
Cloud droplets are small enough relative to ice that their own size distribution
does not enter the collision geometry, so a single ice-PSD integral suffices.
Consumed by [`cloud_riming_rate`](@ref) below freezing and by
[`cloud_warm_collection_rate`](@ref) above it.

**Aerosol scavenging** (`cloud_aerosol_collection`, `ice_aerosol_collection`):
Collection by ice particles of water-friendly and ice-friendly interstitial
aerosol, respectively.

Ice-rain collection is handled separately, by [`IceRainCollection`](@ref) and the
5D rain-ice block embedded in Lookup Table 1, because its kernel needs the
rain slope parameter ``λ_r`` in addition to the ice PSD.

Collection efficiencies are not stored here. They live in
[`ProcessRateParameters`](@ref) alongside the other rate parameters, as
`cloud_ice_collection_efficiency` (``E^{ci}``) and
`rain_ice_collection_efficiency` (``E^{ri}``).

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Sections 2d-e,
[Milbrandt and Yau (2005)](@cite MilbrandtYau2005).
"""
IceCollection() = IceCollection(nothing, nothing, nothing, nothing)

Base.summary(::IceCollection) = "IceCollection"

Base.show(io::IO, ::IceCollection) = print(io, "IceCollection(4 integrals)")
