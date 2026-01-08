#####
##### Cloud Properties
#####
##### Cloud droplet properties for the P3 scheme.
#####

"""
    CloudProperties{FT}

Cloud droplet properties.

Cloud droplets are typically small enough that terminal velocity is negligible.
The number concentration can be prescribed or diagnosed.

# Fields
- `density`: Cloud water density [kg/m³]
- `number_mode`: How to determine cloud droplet number: `:prescribed` or `:prognostic`
- `prescribed_number_concentration`: Fixed N_c if `number_mode == :prescribed` [1/m³]
- `autoconversion_threshold`: Threshold diameter for autoconversion to rain [m]
- `condensation_timescale`: Relaxation timescale for saturation adjustment [s]

# References

Morrison and Milbrandt (2015), Khairoutdinov and Kogan (2000)
"""
struct CloudProperties{FT}
    density :: FT
    number_mode :: Symbol
    prescribed_number_concentration :: FT
    autoconversion_threshold :: FT
    condensation_timescale :: FT
end

"""
    CloudProperties(FT=Float64; number_mode=:prescribed, prescribed_number_concentration=100e6)

Construct `CloudProperties` with default parameters.

# Keyword Arguments
- `number_mode`: `:prescribed` (default) or `:prognostic`
- `prescribed_number_concentration`: Default 100×10⁶ m⁻³ (continental)

Default parameters from Morrison and Milbrandt (2015).
"""
function CloudProperties(FT::Type{<:AbstractFloat} = Float64;
                         number_mode::Symbol = :prescribed,
                         prescribed_number_concentration = FT(100e6))
    return CloudProperties(
        FT(1000.0),      # density [kg/m³]
        number_mode,
        prescribed_number_concentration,
        FT(25e-6),       # autoconversion_threshold [m] = 25 μm
        FT(1.0)          # condensation_timescale [s]
    )
end

Base.summary(::CloudProperties) = "CloudProperties"

function Base.show(io::IO, c::CloudProperties)
    print(io, summary(c), "(")
    print(io, "mode=", c.number_mode, ", ")
    if c.number_mode == :prescribed
        print(io, "N_c=", c.prescribed_number_concentration, " m⁻³")
    end
    print(io, ")")
end

