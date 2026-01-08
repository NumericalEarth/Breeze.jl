#####
##### Cloud Droplet Properties
#####
##### Cloud droplet properties for the P3 scheme.
#####

"""
    CloudDropletProperties{FT}

Cloud droplet properties for prescribed cloud droplet number concentration.

Cloud droplets are typically small enough that terminal velocity is negligible.
In this implementation, cloud droplet number concentration is prescribed
(not prognostic), which is appropriate for many applications and simplifies
the scheme.

Note: liquid water density is stored in `PredictedParticlePropertiesMicrophysics`
as it is shared between cloud and rain.

# Fields
$(TYPEDFIELDS)

# References

[Morrison2015parameterization](@cite), [KhairoutdinovKogan2000](@cite)
"""
struct CloudDropletProperties{FT}
    "Prescribed cloud droplet number concentration [1/m³]"
    number_concentration :: FT
    "Threshold diameter for autoconversion to rain [m]"
    autoconversion_threshold :: FT
    "Relaxation timescale for saturation adjustment [s]"
    condensation_timescale :: FT
end

"""
$(TYPEDSIGNATURES)

Construct `CloudDropletProperties` with specified parameters.

# Keyword Arguments
- `number_concentration`: Prescribed cloud droplet number concentration [1/m³], 
   default 100×10⁶ (typical for continental clouds; marine ~50×10⁶)
- `autoconversion_threshold`: Threshold diameter for autoconversion to rain [m],
   default 25×10⁻⁶ (25 μm)
- `condensation_timescale`: Relaxation timescale for saturation adjustment [s],
   default 1.0

Default parameters from [Morrison2015parameterization](@cite).
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
