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
struct IceDeposition{V, V1, SC, SR, LC, LR}
    ventilation :: V
    ventilation_enhanced :: V1
    small_ice_ventilation_constant :: SC
    small_ice_ventilation_reynolds :: SR
    large_ice_ventilation_constant :: LC
    large_ice_ventilation_reynolds :: LR
end

"""
$(TYPEDSIGNATURES)

Construct `IceDeposition` with quadrature-based ventilation integrals.

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

Thermal conductivity ``κ`` and vapor diffusivity ``Dᵥ`` are computed at runtime
from temperature and pressure via `air_transport_properties(T, P)` — they are
not stored on `IceDeposition`.

**Basic ventilation integrals:**
- `ventilation`: Integrated over full size spectrum
- `ventilation_enhanced`: For larger particles (D > 100 μm)

**Size-regime ventilation** (for melting with liquid fraction):
- `small_ice_ventilation_*`: D ≤ Dcrit, meltwater → rain
- `large_ice_ventilation_*`: D > Dcrit, meltwater → liquid on ice

# References

[Hall and Pruppacher (1976)](@cite HallPruppacher1976),
[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 34.
"""
function IceDeposition(::Type{<:AbstractFloat} = Float64)
    return IceDeposition(
        Ventilation(),
        VentilationEnhanced(),
        SmallIceVentilationConstant(),
        SmallIceVentilationReynolds(),
        LargeIceVentilationConstant(),
        LargeIceVentilationReynolds()
    )
end

Base.summary(::IceDeposition) = "IceDeposition"

Base.show(io::IO, d::IceDeposition) = print(io, summary(d), "()")
