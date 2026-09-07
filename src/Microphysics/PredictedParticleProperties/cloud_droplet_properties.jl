#####
##### Cloud Droplet Properties
#####
##### Cloud droplet properties for the P3 scheme.
#####

"""
    CloudShape{FT}

Coefficients of the Liu-Daum (2000)-type relation that diagnoses the cloud gamma PSD
shape parameter ``μ^{cl}`` from the droplet number density, together with the bounds the
diagnosis is clamped to. Evaluated by [`liu_daum_shape_parameter`](@ref):

```math
\\chi = a \\, N^{cl} + b, \\qquad
\\mu^{cl} = \\mathrm{clamp}\\!\\left(\\frac{1}{\\chi^2} - 1,\\;
                                    \\mu^{cl}_{\\min},\\; \\mu^{cl}_{\\max}\\right)
```

``\\chi`` is the relative dispersion of the droplet spectrum, and ``(a, b)`` are the
Liu-Daum regression of ``\\chi`` on droplet concentration, fit to aircraft measurements of
warm cloud droplet spectra: at fixed water content, more droplets means a narrower
spectrum, hence a larger ``μ^{cl}``. The bounds keep ``μ^{cl}`` inside the range the fit
was measured over.

The coefficient is stated here for the absolute number density in SI units [m⁻³], so
``a`` carries units of m³. The published form uses cm⁻³, hence the 10⁻⁶ difference from
the printed 5.714 × 10⁻⁴.

See the constructor for the meaning, units and defaults of each coefficient.
"""
struct CloudShape{FT}
    relative_dispersion_number_coefficient :: FT # a, multiplying Nᶜˡ [m³]
    relative_dispersion_intercept :: FT          # b, the intercept [-]
    minimum_shape_parameter :: FT                # lower bound on the diagnosed μᶜˡ [-]
    maximum_shape_parameter :: FT                # upper bound on the diagnosed μᶜˡ [-]
end

"""
$(TYPEDSIGNATURES)

Construct `CloudShape`.

# Keyword Arguments

- `relative_dispersion_number_coefficient`: ``a`` [m³], default `5.714e-10`
- `relative_dispersion_intercept`: ``b`` [-], default `0.2714`
- `minimum_shape_parameter`: ``μ^{cl}_{\\min}`` [-], default `2`
- `maximum_shape_parameter`: ``μ^{cl}_{\\max}`` [-], default `15`

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: CloudShape
CloudShape(Float64)

# output
CloudShape(a=5.714e-10 m³, b=0.2714, μᶜˡ ∈ [2.0, 15.0])
```
"""
function CloudShape(FT::DataType = Oceananigans.defaults.FloatType;
                    relative_dispersion_number_coefficient = 5.714e-10,
                    relative_dispersion_intercept = 0.2714,
                    minimum_shape_parameter = 2,
                    maximum_shape_parameter = 15)

    a = relative_dispersion_number_coefficient
    b = relative_dispersion_intercept
    μ_min = minimum_shape_parameter
    μ_max = maximum_shape_parameter

    a ≥ 0 || throw(ArgumentError("relative_dispersion_number_coefficient must be nonnegative, got $a"))
    b > 0 || throw(ArgumentError("relative_dispersion_intercept must be positive, got $b"))
    μ_min ≤ μ_max || throw(ArgumentError("minimum_shape_parameter $μ_min exceeds maximum_shape_parameter $μ_max"))

    return CloudShape(FT(a), FT(b), FT(μ_min), FT(μ_max))
end

# Allow a container built at one precision to be reused at another, so that
# `CloudDroplets(Float32; shape = CloudShape(Float64; ...))`
# keeps the configured values instead of erroring on the field types. The identity method is
# also the tie-breaker that keeps `convert` unambiguous against `Base.convert(::Type{T}, ::T)`.
Base.convert(::Type{CloudShape{FT}}, p::CloudShape) where FT =
    CloudShape(FT(p.relative_dispersion_number_coefficient), FT(p.relative_dispersion_intercept),
               FT(p.minimum_shape_parameter), FT(p.maximum_shape_parameter))

Base.convert(::Type{CloudShape{FT}}, p::CloudShape{FT}) where FT = p

Base.summary(::CloudShape) = "CloudShape"

function Base.show(io::IO, p::CloudShape)
    print(io, summary(p), "(")
    print(io, "a=", p.relative_dispersion_number_coefficient, " m³, ")
    print(io, "b=", p.relative_dispersion_intercept, ", ")
    print(io, "μᶜˡ ∈ [", p.minimum_shape_parameter, ", ", p.maximum_shape_parameter, "])")
end

# Prescribed cloud droplet parameters for warm microphysics; see the
# `CloudDroplets` constructor.
struct CloudDroplets{FT}
    number_concentration :: FT
    condensation_timescale :: FT
    # Coefficients and bounds of the Liu-Daum relative-dispersion relation. Read by every
    # path that diagnoses μᶜˡ from a local droplet number: `diagnose_cloud_dsd` and
    # `immersion_freezing_cloud_rate`, as well as the constructor below.
    shape :: CloudShape{FT}
    # Cloud gamma PSD shape parameter μᶜˡ ∈ [μᶜˡ_min, μᶜˡ_max].
    # Diagnosed from Nᶜˡ via the Liu-Daum (2000) relation in the constructor.
    # Affects the immersion freezing PSD correction.
    shape_parameter :: FT
    # PSD correction for cloud immersion freezing: C(μᶜˡ) = Γ(μᶜˡ+7)Γ(μᶜˡ+1)/Γ(μᶜˡ+4)²
    # Precomputed at construction time from `shape_parameter` for GPU compatibility.
    freezing_psd_correction :: FT
end

"""
$(TYPEDSIGNATURES)

Diagnose the cloud droplet gamma PSD shape parameter μᶜˡ from the absolute number
concentration `Nᶜˡ` [m⁻³] and the [`CloudShape`](@ref) `shape`:

```math
\\chi = a \\, N^{cl} + b, \\qquad
\\mu^{cl} = \\mathrm{clamp}\\!\\left(\\frac{1}{\\chi^2} - 1,\\;
                                    \\mu^{cl}_{\\min},\\; \\mu^{cl}_{\\max}\\right)
```

The relation is written for the absolute number density, so a specific droplet
number [kg⁻¹] would first have to be multiplied by ρ. `Nᶜˡ` here is already the
absolute density [m⁻³], so no ρ is required.

Every model path — construction-time diagnosis, prognostic `diagnose_cloud_dsd`, and the
`immersion_freezing_cloud_rate` PSD correction — passes `p3.cloud.shape`, so a
custom fit reaches all three.

# Examples

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: CloudShape,
                                                       liu_daum_shape_parameter
shape = CloudShape(Float64)
round(liu_daum_shape_parameter(100e6, shape), digits=1)  # continental default

# output
8.3
```
"""
@inline function liu_daum_shape_parameter(Nᶜˡ, shape)
    FT = typeof(float(Nᶜˡ))
    a = FT(shape.relative_dispersion_number_coefficient)
    b = FT(shape.relative_dispersion_intercept)
    # χ is the relative dispersion of the droplet spectrum; see `CloudShape`
    # for what the regression means and why the bounds are there.
    χ = a * Nᶜˡ + b
    μᶜˡ = FT(1) / χ^2 - FT(1)
    return clamp(μᶜˡ,
                 FT(shape.minimum_shape_parameter),
                 FT(shape.maximum_shape_parameter))
end

"""
$(TYPEDSIGNATURES)

Convenience wrapper evaluating [`liu_daum_shape_parameter`](@ref) with the default
[`CloudShape`](@ref).

Provided for interactive use only. No prognostic or immersion-freezing path may call it:
those must read `p3.cloud.shape` so that a configured fit is actually used.
"""
@inline liu_daum_shape_parameter(Nᶜˡ) = liu_daum_shape_parameter(Nᶜˡ, CloudShape(typeof(float(Nᶜˡ))))

"""
$(TYPEDSIGNATURES)

Construct `CloudDroplets` with prescribed parameters.

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

**Cloud DSD shape parameter (C4 fix):** `μᶜˡ` is diagnosed from Nᶜˡ via a
Liu–Daum (2000)-type relation, bounded by `shape`. Since Nᶜˡ is prescribed
here, μᶜˡ is constant too, so it is diagnosed once at construction time via
[`liu_daum_shape_parameter`](@ref) rather than every timestep. Pass `shape_parameter`
explicitly to override the diagnosis (e.g., for sensitivity studies).

Overriding `shape_parameter` sets *only* the construction-time value. The prognostic
`diagnose_cloud_dsd` and `immersion_freezing_cloud_rate` paths re-diagnose μᶜˡ from the
local droplet number and read `shape` instead, so a sensitivity study that
must move all three sets `shape`.

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
- `shape`: [`CloudShape`](@ref) holding the coefficients and bounds
  of the Liu-Daum relation, default `CloudShape(FT)`. Read by every path that
  diagnoses μᶜˡ from a local droplet number.
- `shape_parameter`: μᶜˡ for cloud gamma PSD [-], default `nothing` (diagnosed
  from Nᶜˡ via Liu-Daum relation). Pass an explicit value to override the
  construction-time diagnosis only.

# References

[Morrison and Milbrandt (2015a)](@cite Morrison2015parameterization),
[Khairoutdinov and Kogan (2000)](@cite KhairoutdinovKogan2000).

# Examples

```jldoctest
using Oceananigans, Breeze
using Breeze.Microphysics.PredictedParticleProperties: CloudDroplets
cloud = CloudDroplets()
round(cloud.shape_parameter, digits=1)  # μᶜˡ diagnosed from Nᶜˡ = 200×10⁶ m⁻³

# output
5.7
```
"""
function CloudDroplets(FT = Oceananigans.defaults.FloatType;
               number_concentration = 200e6,
               condensation_timescale = 1,
               shape = CloudShape(FT),
               shape_parameter = nothing)
    shape = convert(CloudShape{FT}, shape)
    # Convert before diagnosing so the relation is evaluated entirely in `FT`; otherwise a
    # `Float64` keyword would promote the whole calculation in a `Float32` scheme.
    Nᶜˡ = FT(number_concentration)
    # Diagnose μᶜˡ from Nᶜˡ via the Liu-Daum (2000) relation by default.
    # Since Nᶜˡ is prescribed (not predicted), μᶜˡ is also constant — it is
    # safe to evaluate the empirical relation once at construction time.
    μᶜˡ = isnothing(shape_parameter) ? liu_daum_shape_parameter(Nᶜˡ, shape) : FT(shape_parameter)
    # Pre-compute PSD correction at construction time for GPU compatibility.
    # C(μᶜˡ) = Γ(μᶜˡ+7)Γ(μᶜˡ+1)/Γ(μᶜˡ+4)² accounts for the broader-than-mean
    # volume distribution of a gamma PSD in the immersion freezing rate.
    freezing_psd_correction = psd_correction_spherical_volume(μᶜˡ)
    return CloudDroplets(Nᶜˡ, FT(condensation_timescale), shape, μᶜˡ, FT(freezing_psd_correction))
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

Base.summary(::CloudDroplets) = "CloudDroplets"

function Base.show(io::IO, c::CloudDroplets)
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

    μᶜˡ = liu_daum_shape_parameter(Nᶜˡ, p3.cloud.shape)
    Nᶜˡ_bounded = bounded_cloud_number(Nᶜˡ, μᶜˡ, qᶜˡ_eff, ρ, ρᴸ, mass_scale, parameters)
    nᶜˡ_bounded = safe_divide(Nᶜˡ_bounded, ρ, zero(FT))

    λᶜˡ = cloud_slope_parameter(Nᶜˡ_bounded, μᶜˡ, max(qᶜˡ_eff * ρ, mass_scale), ρᴸ, parameters)

    return (; Nᶜˡ = Nᶜˡ_bounded,
            nᶜˡ = nᶜˡ_bounded,
            μᶜˡ,
            λᶜˡ)
end
