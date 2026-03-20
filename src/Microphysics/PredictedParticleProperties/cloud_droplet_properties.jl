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
    # DEPRECATED: Not used by rain_autoconversion_rate (KK2000 is threshold-free).
    # Retained for API stability. Will be removed in a future breaking release.
    autoconversion_threshold :: FT
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
    liu_daum_shape_parameter(Nc)

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
Note: `autoconversion_threshold` is **unused** — KK2000 is threshold-free.
Retained for API stability.

# Keyword Arguments

- `number_concentration`: Nc [1/m³], default 100×10⁶ (continental)
- `autoconversion_threshold`: Conversion diameter [m], default 25 μm
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
round(cloud.shape_parameter, digits=1)  # μ_c diagnosed from Nc = 100×10⁶ m⁻³

# output
8.3
```
"""
function CloudDropletProperties(FT = Oceananigans.defaults.FloatType;
                                number_concentration = 100e6,
                                autoconversion_threshold = 25e-6,
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
        FT(autoconversion_threshold),
        FT(condensation_timescale),
        FT(μ_c),
        FT(freezing_psd_correction)
    )
end

Base.summary(::CloudDropletProperties) = "CloudDropletProperties"

function Base.show(io::IO, c::CloudDropletProperties)
    print(io, summary(c), "(")
    print(io, "nᶜˡ=", c.number_concentration, " m⁻³, ")
    print(io, "μᶜ=", round(c.shape_parameter, digits=2), ")")
end
