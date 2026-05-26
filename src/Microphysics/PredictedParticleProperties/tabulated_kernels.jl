#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

using Oceananigans: Oceananigans

using Breeze.Thermodynamics: temperature,
                             adjustment_saturation_specific_humidity,
                             saturation_specific_humidity,
                             saturation_vapor_pressure,
                             PlanarLiquidSurface,
                             PlanarIceSurface,
                             density,
                             liquid_latent_heat,
                             ice_latent_heat,
                             mixture_heat_capacity,
                             vapor_gas_constant,
                             MoistureMassFractions,
                             ThermodynamicConstants
using DocStringExtensions: TYPEDSIGNATURES

#####
##### Ventilation Sc correction (H4)
#####
##### The ventilation-enhanced table stores 0.44 × ∫ C(D)√(V×D) N'(D) dD
##### with dimensions [m² s^(-1/2)]. At runtime, multiplying by
##### Sc^(1/3) × √ρ_fac / √ν restores the correct dimensions [m].
##### This helper centralizes the
##### correction so that all call sites (deposition, Z-tendency) stay in sync.
#####

"""
$(TYPEDSIGNATURES)

Schmidt number correction factor for ventilation-enhanced table values.

The P3 lookup table stores the ventilation-enhanced integral without the
`Sc^{1/3} √rhofaci / √ν` factor (matching the Fortran convention). This function
computes the correction that must be applied at runtime:

```math
f_{Sc} = \\frac{Sc^{1/3} \\sqrt{\\rho_{fac}}}{\\sqrt{\\nu}}
```

See `quadrature.jl` for the table storage convention.
"""
@inline function ventilation_sc_correction(nu, D_v, ρ_correction = one(typeof(nu)))
    FT = typeof(nu)
    Sc = nu / max(D_v, FT(1e-30))
    return cbrt(Sc) * sqrt(ρ_correction) / sqrt(nu)
end

#####
##### PSD-integrated process rate helpers (tabulated)
#####

"""
$(TYPEDSIGNATURES)

Compute per-particle ventilation integral C(D) × f_v(D) for deposition
using PSD-integrated lookup tables.
"""
@inline function deposition_ventilation(vent::P3Table5D,
                                          vent_e::P3Table5D,
                                          m_mean, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
    FT = typeof(m_mean)
    return deposition_ventilation(vent, vent_e, m_mean, Fᶠ, zero(FT), ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
end

@inline function deposition_ventilation(vent::P3Table5D,
                                          vent_e::P3Table5D,
                                          m_mean, Fᶠ, Fˡ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    # vent stores the constant ventilation term (0.65 × ∫ C(D) N'(D) dD)
    # vent_e stores the enhanced term (0.44 × ∫ C(D)√(V×D) N'(D) dD)  [m² s^(-1/2)]
    # Runtime correction via ventilation_sc_correction:
    # Sc^(1/3) × √ρ_fac / √ν [s^(1/2) m^(-1)]
    # Dimensional check: table [m² s^(-1/2)] × correction [s^(1/2)/m] = [m]
    return vent(log_m, Fᶠ, Fˡ, ρᶠ, μ) + ventilation_sc_correction(nu, D_v, ρ_correction) * vent_e(log_m, Fᶠ, Fˡ, ρᶠ, μ)
end

"""
$(TYPEDSIGNATURES)

Compute per-particle ventilation integral C(D) × f_v(D) for melting
using PSD-integrated lookup tables, blending ice (0.65, 0.44) and rain
(0.78, 0.28) ventilation coefficients weighted by liquid fraction Fl.
"""
@inline function melting_ventilation(vent::P3Table5D,
                                       vent_e::P3Table5D,
                                       m_mean, Fl, Fᶠ, ρᶠ, prp, nu, D_v, ρ_correction, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    return vent(log_m, Fᶠ, Fl, ρᶠ, μ) + ventilation_sc_correction(nu, D_v, ρ_correction) * vent_e(log_m, Fᶠ, Fl, ρᶠ, μ)
end

"""
$(TYPEDSIGNATURES)

Compute per-particle collection kernel ⟨A × V⟩ for riming.
Returns PSD-integrated ∫ V(D) A(D) N'(D) dD (per particle) from lookup table.
"""
@inline function collection_kernel_per_particle(coll::P3Table5D,
                                                  m_mean, Fᶠ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    return collection_kernel_per_particle(coll, m_mean, Fᶠ, zero(FT), ρᶠ, prp, p3, μ)
end

@inline function collection_kernel_per_particle(coll::P3Table5D,
                                                  m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    return coll(log_m, Fᶠ, Fˡ, ρᶠ, μ)
end

"""
$(TYPEDSIGNATURES)

Compute aggregation kernel for self-collection using PSD-integrated
kernel from lookup table.
"""
@inline function aggregation_kernel(coll::P3Table5D,
                                      m_mean, Fᶠ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    return aggregation_kernel(coll, m_mean, Fᶠ, zero(FT), ρᶠ, prp, p3, μ)
end

@inline function aggregation_kernel(coll::P3Table5D,
                                      m_mean, Fᶠ, Fˡ, ρᶠ, prp, p3, μ)
    FT = typeof(m_mean)
    log_m = log10(max(m_mean, p3.minimum_mass_mixing_ratio))
    # Table stores the half-integral (Fortran convention):
    # (1/2) ∫∫ (√A₁+√A₂)² |V₁-V₂| N₁ N₂ dD₁ dD₂
    # No E_agg — collection efficiency is applied by the caller.
    return coll(log_m, Fᶠ, Fˡ, ρᶠ, μ)
end
