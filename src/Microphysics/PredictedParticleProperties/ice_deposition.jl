#####
##### Ice Deposition
#####
##### Vapor deposition/sublimation integrals for ice particles.
##### Includes ventilation factors that account for enhanced vapor
##### transport due to particle motion through air.
#####

"""
    IceDeposition

Vapor deposition/sublimation parameters and ventilation integrals.
See [`IceDeposition`](@ref) constructor for details.
"""
struct IceDeposition{FT, V, V1, SC, SR, LC, LR}
    thermal_conductivity :: FT
    vapor_diffusivity :: FT
    ventilation :: V
    ventilation_enhanced :: V1
    small_ice_ventilation_constant :: SC
    small_ice_ventilation_reynolds :: SR
    large_ice_ventilation_constant :: LC
    large_ice_ventilation_reynolds :: LR
end

"""
$(TYPEDSIGNATURES)

Construct `IceDeposition` with parameters and quadrature-based integrals.

Ice growth/decay by vapor deposition/sublimation follows the diffusion equation
with ventilation enhancement. The ventilation factor ``fᵛᵉ`` accounts for
enhanced vapor transport due to particle motion through air:

```math
fᵛᵉ = a + b \\cdot Sc^{1/3} Re^{1/2}
```

where ``Sc`` is the Schmidt number and ``Re`` is the Reynolds number.
[Hall and Pruppacher (1976)](@cite HallPruppacher1976) showed that falling
particles have significantly enhanced vapor exchange compared to stationary
particles.

**Basic ventilation integrals:**
- `ventilation`: Integrated over full size spectrum
- `ventilation_enhanced`: For larger particles (D > 100 μm)

**Size-regime ventilation** (for melting with liquid fraction):
- `small_ice_ventilation_*`: D ≤ Dcrit, meltwater → rain
- `large_ice_ventilation_*`: D > Dcrit, meltwater → liquid on ice

# Keyword Arguments

- `thermal_conductivity`: κ [W/(m·K)], default 0.024 (~273K)
- `vapor_diffusivity`: Dᵥ [m²/s], default 2.2×10⁻⁵ (~273K)

# References

[Hall and Pruppacher (1976)](@cite HallPruppacher1976),
[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 34.
"""
function IceDeposition(FT::Type{<:AbstractFloat} = Float64;
                       thermal_conductivity = 0.024,
                       vapor_diffusivity = 2.2e-5)
    return IceDeposition(
        FT(thermal_conductivity),
        FT(vapor_diffusivity),
        Ventilation(),
        VentilationEnhanced(),
        SmallIceVentilationConstant(),
        SmallIceVentilationReynolds(),
        LargeIceVentilationConstant(),
        LargeIceVentilationReynolds()
    )
end

Base.summary(::IceDeposition) = "IceDeposition"

function Base.show(io::IO, d::IceDeposition)
    print(io, summary(d), "(")
    print(io, "κ=", d.thermal_conductivity, ", ")
    print(io, "Dᵥ=", d.vapor_diffusivity, ")")
end
