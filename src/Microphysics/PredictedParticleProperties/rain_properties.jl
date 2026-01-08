#####
##### Rain Properties
#####
##### Rain particle properties and integrals for the P3 scheme.
#####

"""
    RainProperties

Rain particle size distribution and fall speed parameters.
See [`RainProperties`](@ref) constructor for details.
"""
struct RainProperties{FT, MU, VN, VM, EV}
    maximum_mean_diameter :: FT
    fall_speed_coefficient :: FT
    fall_speed_exponent :: FT
    shape_parameter :: MU
    velocity_number :: VN
    velocity_mass :: VM
    evaporation :: EV
end

"""
$(TYPEDSIGNATURES)

Construct `RainProperties` with parameters and quadrature-based integrals.

Rain in P3 follows a gamma size distribution similar to ice:

```math
N'(D) = N₀ D^{μ_r} e^{-λ_r D}
```

The shape parameter ``μ_r`` is diagnosed from the rain mass and number
concentrations following [Milbrandt and Yau (2005)](@citet MilbrandtYau2005).

**Terminal velocity:**

```math
V(D) = a_V D^{b_V}
```

Default coefficients give fall speeds in m/s for D in meters.

**Integrals:**

- `shape_parameter`: Diagnosed μ_r from q_r, N_r
- `velocity_number`, `velocity_mass`: Weighted fall speeds  
- `evaporation`: Rate integral for rain evaporation

# Keyword Arguments

- `maximum_mean_diameter`: Upper Dm limit [m], default 6×10⁻³ (6 mm)
- `fall_speed_coefficient`: aᵥ [m^{1-b}/s], default 4854
- `fall_speed_exponent`: bᵥ [-], default 1.0

# References

[Morrison and Milbrandt (2015a)](@citet Morrison2015parameterization),
[Milbrandt and Yau (2005)](@citet MilbrandtYau2005),
[Seifert and Beheng (2006)](@citet SeifertBeheng2006).
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

