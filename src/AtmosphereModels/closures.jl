using Oceananigans: defaults
using Oceananigans.Operators: Az
using Oceananigans.Units: days
using Oceananigans.TurbulenceClosures: HorizontalDivergenceScalarBiharmonicDiffusivity

@inline _area_scaled_biharmonic_divergence_damping_ν(i, j, k, grid, ℓx, ℓy, ℓz, clock, fields, λ) =
    Az(i, j, k, grid, ℓx, ℓy, ℓz)^2 / λ

"""
$(TYPEDSIGNATURES)

Return a `HorizontalDivergenceScalarBiharmonicDiffusivity` whose viscosity
is **area-scaled**:

```math
ν(λ, φ, z) \\;=\\; \\frac{A_z(λ, φ, z)^2}{\\tau}
```

where ``A_z`` is the horizontal area of each grid cell and ``τ`` is a
user-supplied `timescale`. This adapts the damping strength to local grid
resolution — fine cells (smaller ``A_z``) get smaller absolute viscosity
but the same per-`τ` decay rate at grid scale, so the closure behaves
consistently across a stretched or curvilinear grid (lat-lon, cubed
sphere, refinement zones).

The `HorizontalDivergenceFormulation` makes the closure act only on the
**divergent** component of the horizontal velocity (no effect on tracers,
no effect on the rotational mode), so it suppresses grid-scale acoustic
noise without polluting balanced/Rossby dynamics. Biharmonic (4th order in
space) means damping rate scales as ``k^4`` — grid-scale modes are hit
``\\sim 16\\times`` harder than the next-coarsest resolved mode.

# Picking `timescale`

A useful rule of thumb is that the e-folding time at grid scale is
``\\tau / π^4 ≈ τ / 97``. So `timescale = 1day` gives an e-folding time
of about 15 minutes for grid-scale modes; `timescale = 6hours` ≈ 4 minutes;
`timescale = 1hour` ≈ 37 seconds. Choose based on how aggressively you
want to suppress grid-scale noise without affecting the resolved scales.

This pattern follows
NumericalEarth.jl's `area_scaled_biharmonic_viscosity` helper, with
`HorizontalDivergenceFormulation` substituted for `HorizontalFormulation`
so that vorticity is preserved.

Example:

```jldoctest
using Breeze
using Oceananigans.Units
divergence_damping = area_scaled_biharmonic_divergence_damping(timescale=1day)
typeof(divergence_damping).name.name

# output
ScalarBiharmonicDiffusivity
```
"""
function area_scaled_biharmonic_divergence_damping(FT=defaults.FloatType; timescale=1day)
    return HorizontalDivergenceScalarBiharmonicDiffusivity(FT;
        ν             = _area_scaled_biharmonic_divergence_damping_ν,
        discrete_form = true,
        parameters    = timescale)
end
