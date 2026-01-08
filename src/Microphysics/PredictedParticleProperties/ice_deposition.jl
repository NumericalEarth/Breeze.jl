#####
##### Ice Deposition
#####
##### Vapor deposition/sublimation integrals for ice particles.
##### Includes ventilation factors that account for enhanced vapor
##### transport due to particle motion through air.
#####

"""
    IceDeposition{FT, V, V1, SC, SR, LC, LR}

Ice vapor deposition/sublimation properties and integrals.

The deposition rate depends on the vapor diffusion equation with ventilation
enhancement. The ventilation factor accounts for enhanced vapor transport
due to particle motion through air, following Hall and Pruppacher (1976).

# Fields

## Parameters
- `thermal_conductivity`: Thermal conductivity of air [W/(m·K)]
- `vapor_diffusivity`: Diffusivity of water vapor in air [m²/s]

## Integrals

Basic ventilation:
- `ventilation`: Basic ventilation factor (vdep in Fortran)
- `ventilation_enhanced`: Enhanced ventilation for particles > 100 μm (vdep1)

Size-regime-specific ventilation for melting/liquid accumulation:
- `small_ice_ventilation_constant`: D ≤ D_crit, constant term → rain (vdepm1)
- `small_ice_ventilation_reynolds`: D ≤ D_crit, Re^0.5 term → rain (vdepm2)
- `large_ice_ventilation_constant`: D > D_crit, constant term → liquid on ice (vdepm3)
- `large_ice_ventilation_reynolds`: D > D_crit, Re^0.5 term → liquid on ice (vdepm4)

# References

Hall and Pruppacher (1976), Morrison and Milbrandt (2015)
"""
struct IceDeposition{FT, V, V1, SC, SR, LC, LR}
    # Parameters
    thermal_conductivity :: FT
    vapor_diffusivity :: FT
    # Basic ventilation integrals
    ventilation :: V
    ventilation_enhanced :: V1
    # Size-regime ventilation integrals
    small_ice_ventilation_constant :: SC
    small_ice_ventilation_reynolds :: SR
    large_ice_ventilation_constant :: LC
    large_ice_ventilation_reynolds :: LR
end

"""
    IceDeposition(FT=Float64)

Construct `IceDeposition` with default parameters and quadrature-based integrals.
"""
function IceDeposition(FT::Type{<:AbstractFloat} = Float64)
    return IceDeposition(
        FT(0.024),   # thermal_conductivity [W/(m·K)] at ~273K
        FT(2.2e-5),  # vapor_diffusivity [m²/s] at ~273K
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
    print(io, "D_v=", d.vapor_diffusivity, ")")
end

