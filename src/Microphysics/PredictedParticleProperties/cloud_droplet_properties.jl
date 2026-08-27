#####
##### Cloud Droplet Properties
#####
##### Cloud droplet properties for the P3 scheme.
#####

# Prescribed cloud droplet parameters for warm microphysics; see the
# `CloudDropletProperties` constructor.
struct CloudDropletProperties{FT}
    number_concentration :: FT
    condensation_timescale :: FT
    # Cloud gamma PSD shape parameter μᶜˡ ∈ [2, 15].
    # Diagnosed from Nᶜˡ via the Liu-Daum (2000) relation in the constructor.
    # Affects the immersion freezing PSD correction.
    shape_parameter :: FT
    # PSD correction for cloud immersion freezing: C(μᶜˡ) = Γ(μᶜˡ+7)Γ(μᶜˡ+1)/Γ(μᶜˡ+4)²
    # Precomputed at construction time from `shape_parameter` for GPU compatibility.
    freezing_psd_correction :: FT
end

"""
$(TYPEDSIGNATURES)

Diagnose the cloud droplet gamma PSD shape parameter μᶜˡ from number concentration.

Implements the Liu-Daum (2000)-type relation:

```math
\\chi = 0.0005714 \\, N^{cl}_{\\rm cm} + 0.2714, \\qquad
\\mu^{cl} = \\frac{1}{\\chi^2} - 1, \\qquad \\mu^{cl} \\in [2, 15]
```

where ``N^{cl}_{\\rm cm} = N^{cl} \\times 10^{-6}`` is the number concentration in cm⁻³.

The relation is written for the absolute number density, so a specific droplet
number [kg⁻¹] would first have to be multiplied by ρ. `Nᶜˡ` here is already the
absolute density [m⁻³], so no ρ is required.

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: liu_daum_shape_parameter
round(liu_daum_shape_parameter(100e6), digits=1)  # continental default

# output
8.3
```
"""
@inline function liu_daum_shape_parameter(Nᶜˡ)
    FT = typeof(float(Nᶜˡ))
    Nᶜˡ_cm³ = Nᶜˡ * FT(1e-6)              # m⁻³ → cm⁻³
    # χ is the relative dispersion of the droplet spectrum, and the two coefficients are
    # the Liu-Daum regression of χ on droplet concentration, fit to aircraft measurements
    # of warm cloud droplet spectra: at fixed water content, more droplets means a
    # narrower spectrum, hence a larger μᶜˡ. They belong to that fit rather than being
    # tunable model parameters, and the bounds keep μᶜˡ inside the range it was measured
    # over.
    χ = FT(0.0005714) * Nᶜˡ_cm³ + FT(0.2714)
    μᶜˡ = FT(1) / χ^2 - FT(1)
    return clamp(μᶜˡ, FT(2), FT(15))
end

"""
$(TYPEDSIGNATURES)

Construct `CloudDropletProperties` with prescribed parameters.

Cloud droplets in P3 are treated simply: their number concentration is
*prescribed* rather than predicted. This is a common simplification
appropriate for many applications where aerosol-cloud interactions
are not the focus.

**Why prescribe Nᶜˡ?**

Predicting cloud droplet number Nᶜˡ requires treating aerosol activation
physics, which adds substantial complexity. For simulations focused
on ice processes or bulk precipitation, prescribed Nᶜˡ is sufficient.

The prescribed-Nᶜˡ simplification means: (1) homogeneous freezing below −40°C
transfers the *prescribed* Nᶜˡ rather than a locally depleted droplet count, and
(2) autoconversion sensitivity to Nᶜˡ is controlled by the prescribed value rather
than dynamically. Pass `aerosol = AerosolActivation(AerosolMode())` to predict Nᶜˡ
instead.

There is no separate mass-number consistency cap on homogeneous freezing:
`homogeneous_freezing_cloud_rate` transfers all of Nᶜˡ at
`T < homogeneous_freezing_temperature`. `compute_p3_process_rates` diagnoses the
rate from the *post-process* residual cloud and rescales mass and number together
by a single `sink_limiting_factor`. This limits the frozen mass to the residual
cloud while preserving its diagnosed mass-number ratio; it does not impose a
minimum frozen-particle mass or independently limit the transferred number.

**Cloud DSD shape parameter (C4 fix):** `μᶜˡ ∈ [2, 15]` is diagnosed from Nᶜˡ via a
Liu–Daum (2000)-type relation. Since Nᶜˡ is prescribed here, μᶜˡ is constant too, so
it is diagnosed once at construction time via [`liu_daum_shape_parameter`](@ref)
rather than every timestep. Pass `shape_parameter` explicitly to override the
diagnosis (e.g., for sensitivity studies).

The `freezing_psd_correction = Γ(μᶜˡ+7)Γ(μᶜˡ+1)/Γ(μᶜˡ+4)²` is pre-computed
at construction time and used in `immersion_freezing_cloud_rate`.

**Typical values:**
- Continental: Nᶜˡ ~ 100-300 × 10⁶ m⁻³ → μᶜˡ ~ 4–8
- Marine: Nᶜˡ ~ 50-100 × 10⁶ m⁻³ → μᶜˡ ~ 8–10

**Autoconversion:**
Cloud droplets are converted to rain via collision-coalescence following
[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

# Keyword Arguments

- `number_concentration`: Nᶜˡ [1/m³], default 200×10⁶
- `condensation_timescale`: Saturation relaxation [s], default 1.0
- `shape_parameter`: μᶜˡ for cloud gamma PSD [-], default `nothing` (diagnosed
  from Nᶜˡ via Liu-Daum relation). Pass an explicit value to override.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

# Examples

```jldoctest
using Oceananigans, Breeze
using Breeze.Microphysics.PredictedParticleProperties: CloudDropletProperties
cloud = CloudDropletProperties()
round(cloud.shape_parameter, digits=1)  # μᶜˡ diagnosed from Nᶜˡ = 200×10⁶ m⁻³

# output
5.7
```
"""
function CloudDropletProperties(FT = Oceananigans.defaults.FloatType;
                                number_concentration = 200e6,
                                condensation_timescale = 1,
                                shape_parameter = nothing)
    # Diagnose μᶜˡ from Nᶜˡ via the Liu-Daum (2000) relation by default.
    # Since Nᶜˡ is prescribed (not predicted), μᶜˡ is also constant — it is
    # safe to evaluate the empirical relation once at construction time.
    μᶜˡ = isnothing(shape_parameter) ? liu_daum_shape_parameter(number_concentration) : shape_parameter
    # Pre-compute PSD correction at construction time for GPU compatibility.
    # C(μᶜˡ) = Γ(μᶜˡ+7)Γ(μᶜˡ+1)/Γ(μᶜˡ+4)² accounts for the broader-than-mean
    # volume distribution of a gamma PSD in the immersion freezing rate.
    freezing_psd_correction = psd_correction_spherical_volume(FT(μᶜˡ))
    return CloudDropletProperties(
        FT(number_concentration),
        FT(condensation_timescale),
        FT(μᶜˡ),
        FT(freezing_psd_correction)
    )
end

"""
$(TYPEDSIGNATURES)

Slope parameter λᶜˡ [1/m] of the cloud gamma PSD carrying absolute mass
`qᶜˡ_abs` [kg/m³] at number `Nᶜˡ` [1/m³] and shape `μᶜˡ`, before the
mean-diameter bounds of [`cloud_slope_bounds`](@ref) are applied.
"""
@inline function unbounded_cloud_slope_parameter(Nᶜˡ, μᶜˡ, qᶜˡ_abs, ρᴸ)
    FT = typeof(qᶜˡ_abs)
    return cbrt(FT(π) * ρᴸ * Nᶜˡ * (μᶜˡ + 3) * (μᶜˡ + 2) * (μᶜˡ + 1) /
                (FT(6) * qᶜˡ_abs))
end

"""
$(TYPEDSIGNATURES)

Bounds `(minimum_slope, maximum_slope)` [1/m] on the cloud PSD slope,
`λ_min = (μᶜˡ + 1) / ⟨D⟩_max` and `λ_max = (μᶜˡ + 1) / ⟨D⟩_min`, from the mean-diameter
bounds `parameters.maximum_mean_droplet_diameter` and
`parameters.minimum_mean_droplet_diameter`.

Both bounds carry the same `(μᶜˡ + 1)` factor, so what they really bound is the mean
droplet *diameter* rather than the slope itself. For the gamma PSD
``N(D) = N_0 D^{μ} e^{-λ D}``, the number-weighted mean diameter is

```math
\\langle D \\rangle = \\frac{\\int_0^∞ D\\, N(D)\\, dD}{\\int_0^∞ N(D)\\, dD}
                    = \\frac{Γ(μ + 2)}{λ\\, Γ(μ + 1)}
                    = \\frac{μ + 1}{λ},
```

using ``Γ(z + 1) = z\\, Γ(z)``, so dividing the shared ``μ + 1`` factor back out
recovers the bounding diameters exactly, whatever ``μᶜˡ`` was diagnosed. The
defaults admit mean droplet diameters of 1–40 μm.
"""
@inline function cloud_slope_bounds(μᶜˡ, parameters)
    FT = typeof(μᶜˡ)
    return ((μᶜˡ + 1) / FT(parameters.maximum_mean_droplet_diameter),
            (μᶜˡ + 1) / FT(parameters.minimum_mean_droplet_diameter))
end

"""
$(TYPEDSIGNATURES)

Bounded cloud PSD slope λᶜˡ [1/m]: [`unbounded_cloud_slope_parameter`](@ref) clamped to
[`cloud_slope_bounds`](@ref).
"""
@inline function cloud_slope_parameter(Nᶜˡ, μᶜˡ, qᶜˡ_abs, ρᴸ, parameters)
    minimum_slope, maximum_slope = cloud_slope_bounds(μᶜˡ, parameters)
    return clamp(unbounded_cloud_slope_parameter(Nᶜˡ, μᶜˡ, qᶜˡ_abs, ρᴸ),
                 minimum_slope, maximum_slope)
end

"""
$(TYPEDSIGNATURES)

Return the cloud number concentration [1/m³] adjusted for the cloud slope bounds.

When the cloud mass is too small (or too large) to support the prescribed `Nᶜˡ` at
the given `μᶜˡ`, the slope parameter hits its bounds. The number is then recomputed
from the clamped slope to maintain mass-PSD consistency, so that downstream rates
(autoconversion, immersion freezing) see a physically consistent cloud number.
"""
@inline function bounded_cloud_number(Nᶜˡ, μᶜˡ, qᶜˡ, ρ, ρᴸ, mass_scale_floor, parameters)
    FT = typeof(qᶜˡ)
    qᶜˡ_abs = max(qᶜˡ * ρ, FT(mass_scale_floor))  # absolute cloud content [kg/m³]

    unbounded_slope = unbounded_cloud_slope_parameter(Nᶜˡ, μᶜˡ, qᶜˡ_abs, ρᴸ)
    minimum_slope, maximum_slope = cloud_slope_bounds(μᶜˡ, parameters)
    λᶜˡ = clamp(unbounded_slope, minimum_slope, maximum_slope)

    # If the slope was clamped, recompute N from it to maintain
    # mass consistency: N = qᶜˡ_abs × λ^(μ+1) × 6 / (π ρ_w Γ(μ+4)/Γ(μ+1))
    # Since Γ(μ+4)/Γ(μ+1) = (μ+3)(μ+2)(μ+1), the result simplifies to:
    Nᶜˡ_bounded = qᶜˡ_abs * FT(6) * λᶜˡ^3 /
                  (FT(π) * ρᴸ * (μᶜˡ + 3) * (μᶜˡ + 2) * (μᶜˡ + 1))

    # Only adjust when clamping was needed; use per-volume [1/m³] convention
    needs_adjustment = (unbounded_slope < minimum_slope) | (unbounded_slope > maximum_slope)
    return ifelse(needs_adjustment, Nᶜˡ_bounded, Nᶜˡ)
end

Base.summary(::CloudDropletProperties) = "CloudDropletProperties"

function Base.show(io::IO, c::CloudDropletProperties)
    print(io, summary(c), "(")
    print(io, "nᶜˡ=", c.number_concentration, " m⁻³, ")
    print(io, "μᶜˡ=", round(c.shape_parameter, digits=2), ")")
end

"""
$(TYPEDSIGNATURES)

Diagnose the cloud PSD state from prognostic cloud liquid and cloud number.

The cloud number is converted from the prognostic specific number `nᶜˡ` [kg⁻¹]
to an absolute concentration, then
diagnose `μᶜˡ` via Liu-Daum, apply the slope bounds, and return the adjusted
cloud number together with the PSD correction used by immersion freezing.
"""
@inline function diagnose_cloud_dsd(p3, qᶜˡ, nᶜˡ, ρ)
    FT = typeof(qᶜˡ + nᶜˡ + ρ)
    qᶜˡ_eff = max(0, qᶜˡ)
    parameters = p3.process_rates
    floors = parameters.floors
    # The floor must be `FT`-wrapped: an untyped literal promotes `nᶜˡ_eff` and every
    # quantity derived from it to Float64 in a Float32 run, which leaves the returned
    # `nᶜˡ` inferred as `Union{Float32, Float64}` through the `ifelse` below.
    nᶜˡ_eff = max(nᶜˡ, FT(p3.minimum_number_mixing_ratio))
    Nᶜˡ = nᶜˡ_eff * ρ
    ρᴸ = parameters.liquid_water_density
    mass_scale = FT(floors.mass_scale)

    μᶜˡ = liu_daum_shape_parameter(Nᶜˡ)
    Nᶜˡ_bounded = bounded_cloud_number(Nᶜˡ, μᶜˡ, qᶜˡ_eff, ρ, ρᴸ, mass_scale, parameters)
    nᶜˡ_bounded = safe_divide(Nᶜˡ_bounded, ρ, zero(FT))

    λᶜˡ = cloud_slope_parameter(Nᶜˡ_bounded, μᶜˡ, max(qᶜˡ_eff * ρ, mass_scale), ρᴸ, parameters)

    return (; Nᶜˡ = Nᶜˡ_bounded,
              nᶜˡ = nᶜˡ_bounded,
              μᶜˡ,
              λᶜˡ)
end
