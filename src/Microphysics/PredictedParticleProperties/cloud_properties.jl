#####
##### Cloud Droplet Properties
#####
##### Cloud droplet properties for the P3 scheme.
#####

"""
    CloudDropletProperties

Prescribed cloud droplet parameters for warm microphysics.
See [`CloudDropletProperties`](@ref) constructor for details.
"""
struct CloudDropletProperties{FT}
    number_concentration :: FT
    autoconversion_threshold :: FT
    condensation_timescale :: FT
end

"""
$(TYPEDSIGNATURES)

Construct `CloudDropletProperties` with prescribed parameters.

Cloud droplets in P3 are treated simply: their number concentration is
*prescribed* rather than predicted. This is a common simplification
appropriate for many applications where aerosol-cloud interactions
are not the focus.

**Why prescribe Nc?**

Predicting cloud droplet number requires treating aerosol activation
physics, which adds substantial complexity. For simulations focused
on ice processes or bulk precipitation, prescribed Nc is sufficient.

**Typical values:**
- Continental: Nc ~ 100-300 × 10⁶ m⁻³ (more CCN, smaller droplets)
- Marine: Nc ~ 50-100 × 10⁶ m⁻³ (fewer CCN, larger droplets)

**Autoconversion:**
Cloud droplets that grow past `autoconversion_threshold` are converted
to rain via collision-coalescence, following
[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

# Keyword Arguments

- `number_concentration`: Nc [1/m³], default 100×10⁶ (continental)
- `autoconversion_threshold`: Conversion diameter [m], default 25 μm
- `condensation_timescale`: Saturation relaxation [s], default 1.0

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).
"""
function CloudDropletProperties(FT = Oceananigans.defaults.FloatType;
                                number_concentration = 100e6,
                                autoconversion_threshold = 25e-6,
                                condensation_timescale = 1)
    return CloudDropletProperties(
        FT(number_concentration),
        FT(autoconversion_threshold),
        FT(condensation_timescale)
    )
end

Base.summary(::CloudDropletProperties) = "CloudDropletProperties"

function Base.show(io::IO, c::CloudDropletProperties)
    print(io, summary(c), "(")
    print(io, "nᶜˡ=", c.number_concentration, " m⁻³")
    print(io, ")")
end
