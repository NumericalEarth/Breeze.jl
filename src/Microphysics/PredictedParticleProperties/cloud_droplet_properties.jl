#####
##### Cloud Droplet Properties
#####
##### Cloud droplet properties for the P3 scheme.
#####

"""
    CloudDropletProperties

Prescribed cloud droplet parameters for warm microphysics.
See [`CloudDropletProperties`](@ref) constructor for details.
"""
struct CloudDropletProperties{FT}
    number_concentration :: FT
    condensation_timescale :: FT
    # Cloud gamma PSD shape parameter μ_c ∈ [2, 15].
    # Diagnosed from Nc via the Liu-Daum (2000) relation in the constructor
    # (matching Fortran P3 get_cloud_dsd2). Affects immersion freezing PSD correction.
    shape_parameter :: FT
    # PSD correction for cloud immersion freezing: C(μ_c) = Γ(μ+7)Γ(μ+1)/Γ(μ+4)²
    # Precomputed at construction time from `shape_parameter` for GPU compatibility.
    freezing_psd_correction :: FT
end

"""
$(TYPEDSIGNATURES)

Diagnose the cloud droplet gamma PSD shape parameter μ_c from number concentration.

Implements the Liu-Daum (2000)-type relation used in Fortran P3 `get_cloud_dsd2`
(lines 10545–10548):

```math
\\chi = 0.0005714 \\, N_c^{\\rm cm} + 0.2714, \\qquad
\\mu_c = \\frac{1}{\\chi^2} - 1, \\qquad \\mu_c \\in [2, 15]
```

where ``N_c^{\\rm cm} = N_c \\times 10^{-6}`` is the number concentration in cm⁻³.

In the Fortran, `nc` is a specific quantity [kg⁻¹] and is multiplied by ρ to
obtain the absolute number density before applying this formula. In Julia, `Nc`
is already the absolute density [m⁻³], so no ρ is required.

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: liu_daum_shape_parameter
round(liu_daum_shape_parameter(100e6), digits=1)  # continental default

# output
8.3
```
"""
function liu_daum_shape_parameter(Nc)
    FT = typeof(float(Nc))
    Nc_cm3 = Nc * FT(1e-6)                 # m⁻³ → cm⁻³ (Fortran nc × 10⁻⁶ × ρ equivalent)
    χ = FT(0.0005714) * Nc_cm3 + FT(0.2714)   # Liu-Daum intermediate parameter
    μ_c = FT(1) / χ^2 - FT(1)
    return clamp(μ_c, FT(2), FT(15))
end

"""
$(TYPEDSIGNATURES)

Construct `CloudDropletProperties` with prescribed parameters.

Cloud droplets in P3 are treated simply: their number concentration is
*prescribed* rather than predicted. This is a common simplification
appropriate for many applications where aerosol-cloud interactions
are not the focus.

**Why prescribe Nc?**

Predicting cloud droplet number requires treating aerosol activation
physics, which adds substantial complexity. For simulations focused
on ice processes or bulk precipitation, prescribed Nc is sufficient.

**Fortran parity note:** The Fortran P3 driver carries and advects prognostic
`Nc` and `ssat` (supersaturation). The prescribed-Nc simplification means:
(1) the homogeneous freezing rate includes a mass-number consistency cap to
prevent ni explosions with trace cloud at T < −40°C, and (2) autoconversion
sensitivity to Nc is controlled by the prescribed value rather than dynamically.

**Cloud DSD shape parameter (C4 fix):** The Fortran P3 diagnoses `μ_c ∈ [2, 15]`
from Nc each timestep via a Liu–Daum (2000)-type relation (`get_cloud_dsd2`).
Since Nc is prescribed in Julia (constant), μ_c is also constant and is diagnosed
from Nc at construction time via [`liu_daum_shape_parameter`](@ref), giving the
same result as Fortran at no runtime cost. Pass `shape_parameter` explicitly to
override the diagnosis (e.g., for sensitivity studies).

The `freezing_psd_correction = Γ(μ_c+7)Γ(μ_c+1)/Γ(μ_c+4)²` is pre-computed
at construction time and used in `immersion_freezing_cloud_rate`.

**Typical values:**
- Continental: Nc ~ 100-300 × 10⁶ m⁻³ → μ_c ~ 4–8
- Marine: Nc ~ 50-100 × 10⁶ m⁻³ → μ_c ~ 8–10

**Autoconversion:**
Cloud droplets are converted to rain via collision-coalescence following
[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

# Keyword Arguments

- `number_concentration`: Nc [1/m³], default 200×10⁶ (Fortran nccnst_2)
- `condensation_timescale`: Saturation relaxation [s], default 1.0
- `shape_parameter`: μ_c for cloud gamma PSD [-], default `nothing` (diagnosed
  from Nc via Liu-Daum relation). Pass an explicit value to override.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

# Examples

```jldoctest
using Oceananigans, Breeze
using Breeze.Microphysics.PredictedParticleProperties: CloudDropletProperties
cloud = CloudDropletProperties()
round(cloud.shape_parameter, digits=1)  # μ_c diagnosed from Nc = 200×10⁶ m⁻³

# output
5.7
```
"""
function CloudDropletProperties(FT = Oceananigans.defaults.FloatType;
                                number_concentration = 200e6,
                                condensation_timescale = 1,
                                shape_parameter = nothing)
    # Diagnose μ_c from Nc via the Liu-Daum (2000) relation by default.
    # Since Nc is prescribed (not predicted), μ_c is also constant — it is
    # safe to evaluate the empirical relation once at construction time.
    μ_c = isnothing(shape_parameter) ? liu_daum_shape_parameter(number_concentration) : shape_parameter
    # Pre-compute PSD correction at construction time for GPU compatibility.
    # C(μ_c) = Γ(μ_c+7)Γ(μ_c+1)/Γ(μ_c+4)² accounts for the broader-than-mean
    # volume distribution of a gamma PSD in the immersion freezing rate.
    freezing_psd_correction = psd_correction_spherical_volume(FT(μ_c))
    return CloudDropletProperties(
        FT(number_concentration),
        FT(condensation_timescale),
        FT(μ_c),
        FT(freezing_psd_correction)
    )
end

"""
$(TYPEDSIGNATURES)

Return the cloud number concentration [1/m³] adjusted for cloud lambda bounds,
matching Fortran `get_cloud_dsd2` (lines 10557-10575 of `microphy_p3.f90`).

When the cloud mass is too small (or too large) to support the prescribed `Nᶜ` at
the given `μ_c`, the lambda parameter hits its bounds. Fortran recomputes `nc` from
the clamped lambda to maintain mass-DSD consistency. This function reproduces that
adjustment so that downstream rates (autoconversion, immersion freezing) see a
physically consistent cloud number.
"""
@inline function bounded_cloud_number(Nᶜ, μ_c, qᶜˡ, ρ)
    FT = typeof(qᶜˡ)
    ρ_water = FT(1000)
    qᶜˡ_abs = max(qᶜˡ * ρ, FT(1e-20))  # absolute cloud content [kg/m³]

    # Compute unclamped lambda from mass and number
    λ_c_uncapped = cbrt(
        FT(π) * ρ_water * Nᶜ * (μ_c + 3) * (μ_c + 2) * (μ_c + 1) /
        (FT(6) * qᶜˡ_abs)
    )

    # Fortran bounds: λ_min = (μ_c+1)×2.5e4, λ_max = (μ_c+1)×1e6
    λ_min = (μ_c + 1) * FT(2.5e4)
    λ_max = (μ_c + 1) * FT(1e6)
    λ_c = clamp(λ_c_uncapped, λ_min, λ_max)

    # If lambda was clamped, recompute N from the clamped lambda to maintain
    # mass consistency: N = qᶜˡ_abs × λ^(μ+1) × 6 / (π ρ_w Γ(μ+4)/Γ(μ+1))
    # Since Γ(μ+4)/Γ(μ+1) = (μ+3)(μ+2)(μ+1), the result simplifies to:
    Nᶜ_bounded = qᶜˡ_abs * FT(6) * λ_c^3 /
                 (FT(π) * ρ_water * (μ_c + 3) * (μ_c + 2) * (μ_c + 1))

    # Only adjust when clamping was needed; use per-volume [1/m³] convention
    needs_adjustment = (λ_c_uncapped < λ_min) | (λ_c_uncapped > λ_max)
    return ifelse(needs_adjustment, Nᶜ_bounded, Nᶜ)
end

Base.summary(::CloudDropletProperties) = "CloudDropletProperties"

function Base.show(io::IO, c::CloudDropletProperties)
    print(io, summary(c), "(")
    print(io, "nᶜˡ=", c.number_concentration, " m⁻³, ")
    print(io, "μᶜ=", round(c.shape_parameter, digits=2), ")")
end

"""
$(TYPEDSIGNATURES)

Diagnose the cloud PSD state from prognostic cloud liquid and cloud number.

This mirrors the Fortran `get_cloud_dsd2` logic used by P3: convert the
prognostic specific cloud number `nᶜˡ` [kg⁻¹] to an absolute concentration,
diagnose `μ_c` via Liu-Daum, apply the lambda bounds, and return the adjusted
cloud number together with the PSD correction used by immersion freezing.
"""
@inline function diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
    FT = typeof(qᶜˡ + nᶜˡ + ρ)
    qᶜˡ_eff = max(0, qᶜˡ)
    nᶜˡ_eff = max(1e-16, nᶜˡ)
    Nᶜ = nᶜˡ_eff * ρ

    μ_c = liu_daum_shape_parameter(Nᶜ)
    Nᶜ_bounded = bounded_cloud_number(Nᶜ, μ_c, qᶜˡ_eff, ρ)
    nᶜˡ_bounded = ifelse(iszero(ρ), zero(FT), Nᶜ_bounded / ρ)

    λ_c_uncapped = cbrt(
        FT(π) * FT(1000) * Nᶜ_bounded * (μ_c + 3) * (μ_c + 2) * (μ_c + 1) /
        (FT(6) * max(qᶜˡ_eff * ρ, FT(1e-20)))
    )
    λ_min = (μ_c + 1) * FT(2.5e4)
    λ_max = (μ_c + 1) * FT(1e6)
    λ_c = clamp(λ_c_uncapped, λ_min, λ_max)

    return (; Nᶜ = Nᶜ_bounded,
              nᶜˡ = nᶜˡ_bounded,
              μ_c,
              λ_c,
              freezing_psd_correction = psd_correction_spherical_volume(μ_c))
end
