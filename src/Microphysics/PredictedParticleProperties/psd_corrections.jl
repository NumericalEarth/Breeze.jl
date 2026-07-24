#####
##### PSD correction factors for spherical drop immersion freezing
#####
##### For a gamma PSD N'(D) = N₀ D^μ exp(-λD), the volume-integrated
##### freezing rate (proportional to V_drop³) gains a factor relative to
##### the mean-mass monodisperse approximation. For spherical drops the
##### correction is exactly:
#####
#####   C(μ) = Γ(μ+7) Γ(μ+1) / Γ(μ+4)²
#####
##### The per-drop freezing rate scales as V_drop ∝ D³, so the mass freezing
##### rate ∝ m × V_drop ∝ D⁶ per drop. The correction is the ratio of the
##### PSD-integrated D⁶ moment to the mean-mass approximation (D_m)⁶:
#####   C(μ) = M₆ M₀ / M₃²  where Mₖ = ∫ Dᵏ N'(D) dD
#####

export psd_correction_spherical_volume

"""
$(TYPEDSIGNATURES)

Compute the analytically exact PSD correction factor for volume-dependent
immersion freezing of spherical drops with a gamma size distribution.

For a gamma PSD N'(D) = N₀ D^μ exp(−λD), the Barklie-Gokhale (1959)
freezing rate is proportional to ⟨V_drop⟩ times number, but because the
freezing probability scales with drop volume V ∝ D³, the PSD-integrated
rate contains ⟨D³⟩ whilst the mean-mass approximation uses ⟨D⟩³. The
ratio of the two is:

```math
C(\\mu) = \\frac{\\Gamma(\\mu + 7)\\,\\Gamma(\\mu + 1)}{\\Gamma(\\mu + 4)^2}
```

Computed in log-space to avoid overflow at large μ.

Exact values:
- μ = 0: 720 × 1 / 36 = 20.0
- μ = 2: 40320 × 2 / 14400 = 5.6
- μ = 5: ≈ 2.945
- Monotonically decreasing with μ (narrower PSD → less enhancement)

# Arguments
- `mu`: Shape parameter μ of the gamma PSD [-]

# Example

```jldoctest
using Breeze.Microphysics.PredictedParticleProperties: psd_correction_spherical_volume
psd_correction_spherical_volume(0.0)

# output
20.000000000000007
```
"""
@inline function psd_correction_spherical_volume(mu)
    FT = typeof(mu)
    log_correction = loggamma(mu + FT(7)) + loggamma(mu + FT(1)) - FT(2) * loggamma(mu + FT(4))
    return exp(log_correction)
end
