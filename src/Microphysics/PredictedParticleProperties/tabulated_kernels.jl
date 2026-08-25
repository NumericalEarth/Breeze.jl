#####
##### P3 Process Rates
#####
##### Microphysical process rate calculations for the P3 scheme.
##### All rate functions take the P3 scheme as first positional argument
##### to access parameters. No keyword arguments (GPU compatibility).
#####
##### Notation follows docs/src/appendix/notation.md
#####

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
`Sc^{1/3} √ρ_correction / √ν` factor. This function computes the correction that
must be applied at runtime:

```math
f_{Sc} = \\frac{Sc^{1/3} \\sqrt{\\rho_{fac}}}{\\sqrt{\\nu}}
```

See `rain_quadrature.jl` for the table storage convention.
"""
@inline function ventilation_sc_correction(ν, Dᵛ, ρ_correction, floors)
    FT = typeof(ν)
    Sc = ν / max(Dᵛ, FT(floors.divisor))
    return cbrt(Sc) * sqrt(ρ_correction) / sqrt(ν)
end

#####
##### PSD-integrated process rate helpers (tabulated)
#####

"""
$(TYPEDSIGNATURES)

Compute per-particle ventilation integral C(D) × f_v(D) for deposition
using PSD-integrated lookup tables.
"""
@inline function deposition_ventilation(vent::P3Table4D,
                                          vent_e::P3Table4D,
                                          m_mean, Fᶠ, Fˡ, ρᶠ, parameters, ν, Dᵛ, ρ_correction)
    FT = typeof(m_mean)
    floors = parameters.floors
    # m_mean = qⁱ/nⁱ is a per-particle mass [kg]; floor it only with a tiny log-guard,
    # NOT the bulk mass-mixing-ratio threshold `minimum_mass_mixing_ratio` (kg/kg).
    # The table clamps the coordinate to its mass axis (min ≈ 1.56e-15 kg) rather
    # than extrapolating below it.
    log_m = log10(max(m_mean, FT(floors.mass_scale)))
    # vent stores the constant ventilation term (0.65 × ∫ C(D) N'(D) dD)
    # vent_e stores the enhanced term (0.44 × ∫ C(D)√(V×D) N'(D) dD)  [m² s^(-1/2)]
    # Runtime correction via ventilation_sc_correction:
    # Sc^(1/3) × √ρ_fac / √ν [s^(1/2) m^(-1)]
    # Dimensional check: table [m² s^(-1/2)] × correction [s^(1/2)/m] = [m]
    # Both tables share Table-1 axes, so the coordinate is bracketed once.
    prep = prepare_interpolation(vent, log_m, Fᶠ, Fˡ, ρᶠ)
    return evaluate_at(vent, prep) +
           ventilation_sc_correction(ν, Dᵛ, ρ_correction, floors) * evaluate_at(vent_e, prep)
end

"""
$(TYPEDSIGNATURES)

Compute the per-particle cloud-water collection kernel ⟨A × V⟩ for riming.
Returns the PSD-integrated ∫ V(D) A(D) N'(D) dD (per particle) from the
`IceCollection.cloud_collection` table.
"""
@inline collection_kernel_per_particle(coll::P3Table4D, m_mean, Fᶠ, Fˡ, ρᶠ, floors) =
    tabulated_ice_integral(coll, m_mean, Fᶠ, Fˡ, ρᶠ, floors)

"""
$(TYPEDSIGNATURES)

Compute aggregation kernel for self-collection using PSD-integrated
kernel from lookup table.

The table stores the half-integral,
`(1/2) ∫∫ (√A₁+√A₂)² |V₁-V₂| N₁ N₂ dD₁ dD₂`. No `E_agg` — the collection
efficiency is applied by the caller.
"""
@inline aggregation_kernel(coll::P3Table4D, m_mean, Fᶠ, Fˡ, ρᶠ, floors) =
    tabulated_ice_integral(coll, m_mean, Fᶠ, Fˡ, ρᶠ, floors)

# Evaluate a 4D ice table at a per-particle mass. The log guard is the
# per-particle one (see `deposition_ventilation`), not the bulk qmin — the tables
# are indexed by log₁₀ of mass per particle, and they clamp to their own mass axis.
@inline tabulated_ice_integral(table::P3Table4D, m_mean, Fᶠ, Fˡ, ρᶠ, floors) =
    table(log10(max(m_mean, typeof(m_mean)(floors.mass_scale))), Fᶠ, Fˡ, ρᶠ)
