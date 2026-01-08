#####
##### Rain Properties
#####
##### Rain particle properties and integrals for the P3 scheme.
#####

"""
    RainProperties{FT, MU, VN, VM, EV}

Rain particle properties and integrals.

Rain follows a gamma size distribution with diagnosed shape parameter μ_r.
Terminal velocity follows a power law similar to ice.

# Fields

## Parameters
- `density`: Rain water density [kg/m³]
- `maximum_mean_diameter`: Maximum mean raindrop diameter [m]
- `fall_speed_coefficient`: Coefficient a_r in V(D) = a_r D^{b_r} [m^{1-b_r}/s]
- `fall_speed_exponent`: Exponent b_r in V(D) = a_r D^{b_r} [-]

## Integrals
- `shape_parameter`: Diagnosed shape parameter μ_r
- `velocity_number`: Number-weighted fall speed
- `velocity_mass`: Mass-weighted fall speed
- `evaporation`: Evaporation rate integral

# References

Morrison and Milbrandt (2015), Seifert and Beheng (2006)
"""
struct RainProperties{FT, MU, VN, VM, EV}
    # Parameters
    density :: FT
    maximum_mean_diameter :: FT
    fall_speed_coefficient :: FT
    fall_speed_exponent :: FT
    # Integrals
    shape_parameter :: MU
    velocity_number :: VN
    velocity_mass :: VM
    evaporation :: EV
end

"""
    RainProperties(FT=Float64)

Construct `RainProperties` with default parameters and quadrature-based integrals.

Default parameters from Morrison and Milbrandt (2015).
"""
function RainProperties(FT::Type{<:AbstractFloat} = Float64)
    return RainProperties(
        FT(1000.0),  # density [kg/m³]
        FT(6e-3),    # maximum_mean_diameter [m] = 6 mm
        FT(4854.0),  # fall_speed_coefficient [m^{1-b}/s]
        FT(1.0),     # fall_speed_exponent [-]
        RainShapeParameter(),
        RainVelocityNumber(),
        RainVelocityMass(),
        RainEvaporation()
    )
end

Base.summary(::RainProperties) = "RainProperties"

function Base.show(io::IO, r::RainProperties)
    print(io, summary(r), "(")
    print(io, "ρ=", r.density, ", ")
    print(io, "D_max=", r.maximum_mean_diameter, ", ")
    print(io, "a=", r.fall_speed_coefficient, ", ")
    print(io, "b=", r.fall_speed_exponent, ")")
end

