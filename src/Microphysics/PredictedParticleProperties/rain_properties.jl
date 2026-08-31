#####
##### Rain Properties
#####
##### Rain particle properties and integrals for the P3 scheme.
#####

# Rain particle size distribution and fall-speed parameters; see the `RainProperties`
# constructor.
struct RainProperties{FT, VN, VM, EV}
    maximum_mean_diameter :: FT
    fall_speed_coefficient :: FT
    fall_speed_exponent :: FT
    velocity_number :: VN
    velocity_mass :: VM
    evaporation :: EV
end

"""
$(TYPEDSIGNATURES)

Construct `RainProperties` with parameters and quadrature-based integrals.

Rain in P3 follows an exponential size distribution, the ``μ^r = 0`` special
case of the gamma distribution used for ice:

```math
N'(D) = Nʳ₀ e^{-λ^r D}
```

There is no rain shape parameter, prognostic or diagnosed: `rain_slope_parameter`
inverts the mass integral directly as ``λ^r = (π ρ^w n^r / q^r)^{1/3}``, and
`rain_quadrature.jl` integrates against the same exponential kernel.

**Terminal velocity:**

```math
V(D) = a_V D^{b_V}
```

Default coefficients give fall speeds in m/s for D in meters.

**Integrals:**

- `velocity_number`, `velocity_mass`: Weighted fall speeds
- `evaporation`: Rate integral for rain evaporation

# Keyword Arguments

- `maximum_mean_diameter`: Upper Dm limit [m], default 2×10⁻³ (2 mm)
- `fall_speed_coefficient`: aᵥ [m^{1-b}/s], default 841.99667
- `fall_speed_exponent`: bᵥ [-], default 0.8

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Milbrandt and Yau (2005)](@cite MilbrandtYau2005),
[Seifert and Beheng (2006)](@cite SeifertBeheng2006).
"""
function RainProperties(FT::DataType = Oceananigans.defaults.FloatType;
                        maximum_mean_diameter = 2e-3,
                        fall_speed_coefficient = 841.99667,
                        fall_speed_exponent = 0.8)
    return RainProperties(
        FT(maximum_mean_diameter),
        FT(fall_speed_coefficient),
        FT(fall_speed_exponent),
        nothing, nothing, nothing,
    )
end

Base.summary(::RainProperties) = "RainProperties"

function Base.show(io::IO, r::RainProperties)
    print(io, summary(r), "(")
    print(io, "Dmax=", r.maximum_mean_diameter, ", ")
    print(io, "aᵥ=", r.fall_speed_coefficient, ", ")
    print(io, "bᵥ=", r.fall_speed_exponent, ")")
end
