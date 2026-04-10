#####
##### Ice Size Distribution
#####
##### The P3 scheme uses a generalized gamma distribution for ice particles.
#####

"""
    IceSizeDistributionState

State container for ice size distribution integration.
See [`IceSizeDistributionState`](@ref) constructor for details.
"""
struct IceSizeDistributionState{FT}
    intercept :: FT
    shape :: FT
    slope :: FT
    rime_fraction :: FT
    liquid_fraction :: FT
    rime_density :: FT
    # Mass-diameter power law parameters (α, β) from m = α D^β
    mass_coefficient :: FT
    mass_exponent :: FT
    ice_density :: FT
    # Reference air density for fall speed correction
    reference_air_density :: FT
    air_density :: FT
end

"""
$(TYPEDSIGNATURES)

Construct an `IceSizeDistributionState` for quadrature evaluation.

The ice particle size distribution follows a generalized gamma form:

```math
N'(D) = N_0 D^μ e^{-λD}
```

The gamma distribution is parameterized by three quantities:

- **N₀** (intercept): Sets the total number of particles
- **μ** (shape): Controls the relative abundance of small vs. large particles
- **λ** (slope): Sets the characteristic inverse diameter

For P3, these are determined from prognostic moments using the
[`distribution_parameters`](@ref) function.

**Rime and liquid properties** affect the mass-diameter relationship:

- `rime_fraction`: Fraction of mass that is rime (0 = pristine, 1 = graupel)
- `rime_density`: Density of the accreted rime layer
- `liquid_fraction`: Liquid water coating from partial melting

# Required Keyword Arguments

- `intercept`: N₀ [m^{-(4+μ)}]
- `shape`: μ [-]
- `slope`: λ [1/m]

# Optional Keyword Arguments

- `rime_fraction`: Fᶠ [-], default 0 (unrimed)
- `liquid_fraction`: Fˡ [-], default 0 (no meltwater)
- `rime_density`: ρᶠ [kg/m³], default 400
- `mass_coefficient`: α in m = α D^β [kg/m^β], default 0.0121
- `mass_exponent`: β in m = α D^β [-], default 1.9
- `ice_density`: Pure ice density [kg/m³], default 917
- `reference_air_density`: ρ₀ for fall speed correction [kg/m³], default ≈0.825 (P3 reference)
- `air_density`: Local air density [kg/m³], default ≈0.825 (P3 reference)

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b.
"""
function IceSizeDistributionState(FT::Type{<:AbstractFloat} = Float64;
                                   intercept,
                                   shape,
                                   slope,
                                   rime_fraction = zero(FT),
                                   liquid_fraction = zero(FT),
                                   rime_density = FT(400),
                                   mass_coefficient = FT(0.0121),
                                   mass_exponent = FT(1.9),
                                   ice_density = FT(900),
                                   reference_air_density = FT(60000 / (dry_air_gas_constant(ThermodynamicConstants()) * 253.15)),
                                   air_density = FT(60000 / (dry_air_gas_constant(ThermodynamicConstants()) * 253.15)))
    return IceSizeDistributionState(
        FT(intercept),
        FT(shape),
        FT(slope),
        FT(rime_fraction),
        FT(liquid_fraction),
        FT(rime_density),
        FT(mass_coefficient),
        FT(mass_exponent),
        FT(ice_density),
        FT(reference_air_density),
        FT(air_density)
    )
end

"""
    size_distribution(D, state::IceSizeDistributionState)

Evaluate the ice size distribution ``N'(D)`` at diameter D.

Returns the number density of particles per unit diameter interval:

```math
N'(D) = N_0 D^μ e^{-λD}
```

The total number concentration is ``N = ∫_0^∞ N'(D) dD``.
"""
@inline function size_distribution(D, state::IceSizeDistributionState)
    N₀ = state.intercept
    μ = state.shape
    λ = state.slope
    return N₀ * D^μ * exp(-λ * D)
end

"""
    dry_size_distribution(D, state::IceSizeDistributionState)

Evaluate the **dry ice** size distribution ``N_d'(D)`` at diameter D.

When liquid fraction ``Fˡ > 0`` (partial melting), the wet (total) PSD
includes liquid water in the mass constraint. For melting and deposition
integrals, the Fortran P3 v5.5.0 uses the dry PSD from
[Cholette et al. (2019)](@cite Cholette2019parameterization) which represents
only the ice portion.

The dry PSD is derived analytically by adjusting ``λ`` and ``N_0`` so that
the mass moment matches the dry ice content ``q_{dry} = q_{total}(1 - Fˡ)``,
using the effective mass-diameter exponent ``β`` from the state:

```math
λ_d = λ (1-Fˡ)^{-1/β}, \\quad N_{0,d} = N_0 (λ_d/λ)^{μ+1}
```

When ``Fˡ = 0``, ``λ_d = λ`` and the dry PSD equals the wet PSD.
"""
@inline function dry_size_distribution(D, state::IceSizeDistributionState)
    N₀ = state.intercept
    μ = state.shape
    λ = state.slope
    Fˡ = state.liquid_fraction
    β = state.mass_exponent

    # Compute λ ratio: (1-Fˡ)^(-1/β). When Fˡ = 0, ratio = 1 (dry = wet).
    one_minus_Fl = max(1 - Fˡ, eps(typeof(Fˡ)))
    λ_ratio = one_minus_Fl^(-1 / β)
    λ_dry = λ * λ_ratio
    # Adjust N₀ to conserve total number: N = N₀ Γ(μ+1)/λ^(μ+1)
    N₀_dry = N₀ * λ_ratio^(μ + 1)

    return N₀_dry * D^μ * exp(-λ_dry * D)
end
