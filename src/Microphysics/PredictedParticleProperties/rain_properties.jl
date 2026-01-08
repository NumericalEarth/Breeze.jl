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

Note: liquid water density is stored in `PredictedParticlePropertiesMicrophysics`
as it is shared between cloud and rain.

# Fields

## Parameters
- `maximum_mean_diameter`: Maximum mean raindrop diameter [m]
- `fall_speed_coefficient`: Coefficient aᵥ in V(D) = aᵥ D^{bᵥ} [m^{1-bᵥ}/s]
- `fall_speed_exponent`: Exponent bᵥ in V(D) = aᵥ D^{bᵥ} [-]

## Integrals
- `shape_parameter`: Diagnosed shape parameter μʳ
- `velocity_number`: Number-weighted fall speed
- `velocity_mass`: Mass-weighted fall speed
- `evaporation`: Evaporation rate integral

# References

[Morrison2015parameterization](@cite), Seifert and Beheng (2006)
"""
struct RainProperties{FT, MU, VN, VM, EV}
    # Parameters
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
$(TYPEDSIGNATURES)

Construct `RainProperties` with specified parameters and quadrature-based integrals.

# Keyword Arguments
- `maximum_mean_diameter`: Maximum mean raindrop diameter [m], default 6×10⁻³ (6 mm)
- `fall_speed_coefficient`: Coefficient aᵥ in V(D) = aᵥ D^{bᵥ} [m^{1-bᵥ}/s], default 4854
- `fall_speed_exponent`: Exponent bᵥ in V(D) = aᵥ D^{bᵥ} [-], default 1.0

Default parameters from [Morrison2015parameterization](@cite).
"""
function RainProperties(FT::Type{<:AbstractFloat} = Float64;
                        maximum_mean_diameter = 6e-3,
                        fall_speed_coefficient = 4854,
                        fall_speed_exponent = 1)
    return RainProperties(
        FT(maximum_mean_diameter),
        FT(fall_speed_coefficient),
        FT(fall_speed_exponent),
        RainShapeParameter(),
        RainVelocityNumber(),
        RainVelocityMass(),
        RainEvaporation()
    )
end

Base.summary(::RainProperties) = "RainProperties"

function Base.show(io::IO, r::RainProperties)
    print(io, summary(r), "(")
    print(io, "Dmax=", r.maximum_mean_diameter, ", ")
    print(io, "aᵥ=", r.fall_speed_coefficient, ", ")
    print(io, "bᵥ=", r.fall_speed_exponent, ")")
end

