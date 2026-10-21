#####
##### P3 Tabulated Kernels
#####
##### What sits between the lookup tables and the process rates: the Schmidt-number
##### correction for the ventilation-enhanced table, and `P3IceLookups`, the Table-1
##### coordinate and integrals every ice-side rate reads.
#####
##### Notation follows docs/src/appendix/notation.md
#####

#####
##### Ventilation Sc correction
#####
##### The ventilation-enhanced table stores 0.44 × ∫ C(D)√(𝕎×D) N'(D) dD
##### with dimensions [m² s^(-1/2)]. At runtime, multiplying by
##### Sc^(1/3) × √ρ_fac / √ν restores the correct dimensions [m].
##### This helper centralizes the correction so that all call sites (the vapor
##### relaxation coefficient, the wet-growth capacity, and melting) stay in sync.
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

# The ρ-independent part `Sc^{1/3} / √ν` of the correction. It depends on the cell's
# thermodynamic state only, so `compute_p3_process_rates` evaluates it once per cell and
# carries it in the `transport` tuple of the phase rates, where `ventilation_sc_correction`
# picks it up instead of re-deriving it at each of the ventilated rates.
@inline function schmidt_viscosity_factor(ν, Dᵛ, floors)
    FT = typeof(ν)
    return cbrt(ν / max(Dᵛ, FT(floors.divisor))) / sqrt(ν)
end

const TransportWithSchmidtFactor = NamedTuple{(:Dᵛ, :Kᵃ, :ν, :schmidt_factor)}

@inline ventilation_sc_correction(transport::TransportWithSchmidtFactor, ρ_correction, floors) =
    transport.schmidt_factor * sqrt(ρ_correction)

@inline ventilation_sc_correction(transport, ρ_correction, floors) =
    ventilation_sc_correction(transport.ν, transport.Dᵛ, ρ_correction, floors)

#####
##### Shared Table-1 coordinate
#####
##### Every Table-1 integral is indexed by (log₁₀ m̄, Fᶠ, Fˡ, ρᶠ), so a cell brackets that
##### coordinate once and evaluates each table it needs at the bracket.
#####

# `m_mean` [kg] is a per-particle mass: the log guard is `floors.mass_scale`, NOT the bulk
# `minimum_mass_mixing_ratio`, and the table clamps to its own mass axis (min ≈ 1.56e-15 kg).
@inline function ice_table_log_mass(m_mean, floors)
    FT = typeof(m_mean)
    return log10(max(m_mean, FT(floors.mass_scale)))
end

@inline ice_table_bracket(table, m_mean, Fᶠ, Fˡ, ρᶠ, floors) =
    ice_table_bracket_at_log_mass(table, ice_table_log_mass(m_mean, floors), Fᶠ, Fˡ, ρᶠ)

@inline ice_table_bracket_at_log_mass(table::P3Table4D, log_m, Fᶠ, Fˡ, ρᶠ) =
    prepare_interpolation(table, log_m, Fᶠ, Fˡ, ρᶠ)

# Un-materialized scheme: no table to bracket against.
@inline ice_table_bracket_at_log_mass(::Nothing, log_m, Fᶠ, Fˡ, ρᶠ) = nothing

"""
    P3IceLookups{FT, P}

Per-cell Table-1 quantities shared by every ice-side process rate: the bounded
population's mean particle mass, liquid fraction, bracketed coordinate, fall-speed
air-density correction, and two deposition ventilation integrals. Built once per cell by
[`p3_ice_lookups`](@ref).
"""
struct P3IceLookups{FT, P}
    "Mean total ice particle mass [kg]"
    m_mean :: FT
    "Liquid fraction qʷⁱ / (qⁱ + qʷⁱ) of the population [-]"
    Fˡ :: FT
    "Bracketed (log₁₀ m̄, Fᶠ, Fˡ, ρᶠ) on the Table-1 axes"
    prep :: P
    "Fall-speed air-density correction (ρ₀/ρ)^α at the ice reference density [-]"
    ρ_correction :: FT
    "Constant ventilation term 0.65 ∫ C(D) N'(D) dD [m]"
    ventilation :: FT
    "Enhanced ventilation term 0.44 ∫ C(D) √(𝕎 D) N'(D) dD [m² s^(-1/2)], before the Sc correction"
    enhanced_ventilation :: FT
    "log₁₀ of the floored mean particle mass, the Table-1 and Table-2 mass coordinate"
    log_m_mean :: FT
end

"""
$(TYPEDSIGNATURES)

Build the [`P3IceLookups`](@ref) of the ice population `(qⁱ, qʷⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ)`.
`nⁱ` is the bounded number the rates see, not the pre-limiter number.
"""
@inline function p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, Fˡ, ρᶠ, ρ)
    parameters = p3.process_rates
    floors = parameters.floors
    deposition = p3.ice.deposition
    m_mean = mean_total_ice_mass(qⁱ, qʷⁱ, nⁱ, floors)
    FT = typeof(m_mean)
    log_m = ice_table_log_mass(m_mean, floors)
    prep = ice_table_bracket_at_log_mass(deposition.ventilation, log_m, Fᶠ, Fˡ, ρᶠ)
    # The ice reference density (≈0.83 kg/m³ at 600 hPa, 253.15 K; see `IceFallSpeed`),
    # not the rain reference ≈1.275 kg/m³ of `ProcessRate`.
    ρ_correction = ice_air_density_correction(p3, fall_speed_density_power(parameters, ρ))
    return P3IceLookups{FT, typeof(prep)}(m_mean, Fˡ, prep, ρ_correction,
                                          evaluate_at(deposition.ventilation, prep),
                                          evaluate_at(deposition.enhanced_ventilation, prep),
                                          log_m)
end

# Standalone entry point: diagnose the liquid fraction, then bracket.
@inline p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, ρᶠ, ρ) =
    p3_ice_lookups(p3, qⁱ, qʷⁱ, nⁱ, Fᶠ, liquid_fraction_on_ice(qⁱ, qʷⁱ, p3.process_rates.floors),
                   ρᶠ, ρ)

# C(D) fᵛ(D) from its two tabulated terms. The Sc correction is the caller's: the vapor
# relaxation coefficient uses the thermodynamic air density, wet growth the dynamics density.
@inline ventilation_from_terms(ventilation, enhanced_ventilation, ν, Dᵛ, ρ_correction, floors) =
    ventilation + ventilation_sc_correction(ν, Dᵛ, ρ_correction, floors) * enhanced_ventilation

@inline ventilation_from_terms(ventilation, enhanced_ventilation, transport, ρ_correction, floors) =
    ventilation + ventilation_sc_correction(transport, ρ_correction, floors) * enhanced_ventilation

"""
$(TYPEDSIGNATURES)

Compute per-particle ventilation integral C(D) × f_v(D) for deposition
using PSD-integrated lookup tables.
"""
@inline function deposition_ventilation(vent::P3Table4D,
                                        vent_e::P3Table4D,
                                        m_mean, Fᶠ, Fˡ, ρᶠ, parameters, ν, Dᵛ, ρ_correction)
    floors = parameters.floors
    # Both tables share Table-1 axes, so the coordinate is bracketed once.
    prep = ice_table_bracket(vent, m_mean, Fᶠ, Fˡ, ρᶠ, floors)
    return ventilation_from_terms(evaluate_at(vent, prep), evaluate_at(vent_e, prep),
                                  ν, Dᵛ, ρ_correction, floors)
end
