# [Particle Properties](@id p3_particle_properties)

Ice particles in P3 span a continuum from small pristine crystals to large rimed graupel.
The mass-diameter and area-diameter relationships vary across this spectrum, depending on
particle size and riming state.

The foundational particle property relationships are from
[Morrison & Milbrandt (2015a])(@citet), Section 2.

## Mass-Diameter Relationship

The particle mass ``m(D)`` follows a piecewise power law that depends on maximum dimension ``D``,
rime fraction ``Fᶠ``, and rime density ``ρᶠ``. This formulation is given in
[Morrison2015parameterization](@citet) Equations 1-5.

### The Four Regimes

P3 defines four diameter regimes with distinct mass-diameter relationships:

**Regime 1: Small Spherical Ice** (``D < D_{th}``)

Small ice particles are assumed spherical with pure ice density
([Morrison2015parameterization](@citet) Eq. 1):

```math
m(D) = \frac{π}{6} ρᵢ D³
```

where ``ρᵢ = 900`` kg/m³ is the value used in the reference lookup tables
(pure ice is approximately 917 kg/m³).

**Regime 2: Vapor-Grown Aggregates** (``D_{th} ≤ D < D_{gr}`` or unrimed)

Larger particles follow an empirical power law based on aircraft observations
of ice crystals and aggregates ([Morrison2015parameterization](@citet) Eq. 2):

```math
m(D) = α D^β
```

where ``α = 0.0121`` kg/m^β and ``β = 1.9`` are based on observations compiled in the
supplementary material of [Morrison2015parameterization](@citet).
This relationship captures the fractal nature of aggregated crystals.

**Regime 3: Graupel** (``D_{gr} ≤ D < D_{cr}``)

When particles acquire sufficient rime, they become compact graupel
with density ``ρ_g`` ([Morrison2015parameterization](@cite) Eq. 3):

```math
m(D) = \frac{π}{6} ρ_g D³
```

The graupel density ``ρ_g`` depends on the rime fraction and rime density
([Morrison2015parameterization](@citet) Eq. 17):

```math
ρ_g = Fᶠ ρᶠ + (1 - Fᶠ) ρ_d
```

where ``ρ_d`` is the density of the deposited (vapor-grown) ice component.

**Regime 4: Partially Rimed** (``D ≥ D_{cr}``)

The largest particles have a rimed core with unrimed aggregate extensions
([Morrison2015parameterization](@citet) Eq. 4):

```math
m(D) = \frac{α}{1 - Fᶠ} D^β
```

### Threshold Diameters

The transitions between regimes occur at critical diameters determined by
equating masses ([Morrison2015parameterization](@citet) Eqs. 12-14):

**Spherical-Aggregate Threshold** ``D_{th}``:

The diameter where spherical mass equals aggregate mass:

```math
D_{th} = \left( \frac{6α}{π ρᵢ} \right)^{1/(3-β)}
```

**Aggregate-Graupel Threshold** ``D_{gr}``:

The diameter where aggregate mass equals graupel mass:

```math
D_{gr} = \left( \frac{6α}{π ρ_g} \right)^{1/(3-β)}
```

**Graupel-Partial Threshold** ``D_{cr}``:

The diameter where graupel mass equals partially rimed mass:

```math
D_{cr} = \left( \frac{6α}{π ρ_g (1 - Fᶠ)} \right)^{1/(3-β)}
```

### Deposited Ice Density

The density of the vapor-deposited (unrimed) component ``ρ_d`` is derived from
the constraint that total mass equals rime mass plus deposited mass.
From [Morrison2015parameterization](@citet) Equation 16:

```math
ρ_d = \frac{Fᶠ ρᶠ}{(β - 2) \frac{k - 1}{(1 - Fᶠ)k - 1} - (1 - Fᶠ)}
```

where ``k = (1 - Fᶠ)^{-1/(3-β)}``.

## Code Example: Mass-Diameter Relationship

```@example p3_particles
using Breeze.Microphysics.PredictedParticleProperties
using CairoMakie

# Create mass power law parameters
mass = IceMassPowerLaw()

# Compute mass for different particle sizes
D = 10 .^ range(-5, -2, length=100)  # 10 μm to 1 cm

# Unrimed ice
m_unrimed = [ice_mass(mass, 0.0, 400.0, d) for d in D]

# Moderately rimed (50% rime fraction)
m_rimed = [ice_mass(mass, 0.5, 500.0, d) for d in D]

# Plot
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [m]",
    ylabel = "Mass m [kg]",
    xscale = log10,
    yscale = log10,
    title = "Ice Particle Mass vs Diameter")

lines!(ax, D, m_unrimed, label="Unrimed (Fᶠ = 0)")
lines!(ax, D, m_rimed, label="Rimed (Fᶠ = 0.5)")

# Add reference lines for spherical ice
m_sphere = @. mass.ice_density * π / 6 * D^3
lines!(ax, D, m_sphere, linestyle=:dash, color=:gray, label="Spherical ice")

axislegend(ax, position=:lt)
fig
```

## Code Example: Regime Thresholds

```@example p3_particles
# Compute thresholds for different rime fractions
mass = IceMassPowerLaw()

println("Threshold diameters for unrimed ice (Fᶠ = 0):")
thresholds = ice_regime_thresholds(mass, 0.0, 400.0)
println("  D_th (spherical) = $(round(thresholds.spherical * 1e6, digits=1)) μm")

println("\nThreshold diameters for rimed ice (Fᶠ = 0.5, ρᶠ = 500 kg/m³):")
thresholds = ice_regime_thresholds(mass, 0.5, 500.0)
println("  D_th (spherical) = $(round(thresholds.spherical * 1e6, digits=1)) μm")
println("  D_gr (graupel)   = $(round(thresholds.graupel * 1e3, digits=2)) mm")
println("  D_cr (partial)   = $(round(thresholds.partial_rime * 1e3, digits=2)) mm")
println("  ρ_g (graupel)    = $(round(thresholds.ρ_graupel, digits=1)) kg/m³")
```

## Area-Diameter Relationship

The projected cross-sectional area ``A(D)`` determines collection rates and fall speed.
These relationships are from [Morrison2015parameterization](@citet)
Equations 6-8.

**Small Spherical Ice** (``D < D_{th}``):

```math
A(D) = \frac{π}{4} D²
```

**Nonspherical Ice** (aggregates):

```math
A(D) = γ D^σ
```

where ``γ`` and ``σ`` are empirical coefficients from
[Mitchell1996powerlaws](@citet) (see [Morrison2015parameterization](@citet) Table 1).

**Graupel**:

Reverts to spherical:

```math
A(D) = \frac{π}{4} D²
```

**Partially Rimed**:

Per official P3 code, the projected area is interpolated by particle mass between
the unrimed and graupel relationships, rather than a simple Fᶠ weighting:

```math
A(D) = A_{ur} + \frac{m_{pr} - m_{ur}}{m_{gr} - m_{ur}} \left(A_{gr} - A_{ur}\right)
```

with ``A_{ur} = γ D^σ``, ``A_{gr} = \frac{π}{4} D^2``,
``m_{ur} = α D^β``, ``m_{gr} = \frac{π}{6} ρ_g D^3``,
and ``m_{pr} = c_{sr} D^{d_{sr}}`` from the partially rimed mass law.

## Terminal Velocity

The official P3 code computes terminal velocity using the
[MitchellHeymsfield2005](@cite) Best-number drag formulation with the
regime-dependent ``m(D)`` and ``A(D)`` relationships. The resulting fall speeds
are stored in lookup tables and include the air-density correction
``(ρ₀/ρ)^{0.54}`` following [HeymsfieldEtAl2006](@cite). Regime-specific coefficients
from the P3 papers therefore do not appear as a single global power law in the
lookup tables.

!!! note "Velocity Coefficients"
    The regime-specific power-law coefficients in the literature are a compact summary
    of fall-speed behavior, but the official P3 derives fall speeds from the
    Best-number drag law and tabulates the results. See
    [Morrison2015parameterization](@citet) supplementary material for coefficient definitions.

## Particle Density

The effective density ``ρ(D)`` is defined as mass divided by the volume
of a sphere with diameter ``D``:

```math
ρ(D) = \frac{m(D)}{(π/6) D³} = \frac{6 m(D)}{π D³}
```

This definition is convenient for comparing particles of different types
and connects directly to the mass-diameter relationship.

```@example p3_particles
# Compute effective density across sizes
ρ_unrimed = @. m_unrimed / (π/6 * D^3)
ρ_rimed = @. m_rimed / (π/6 * D^3)

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [m]",
    ylabel = "Effective density ρ [kg/m³]",
    xscale = log10,
    title = "Ice Particle Density vs Diameter")

lines!(ax, D, ρ_unrimed, label="Unrimed (Fᶠ = 0)")
lines!(ax, D, ρ_rimed, label="Rimed (Fᶠ = 0.5)")
hlines!(ax, [917], linestyle=:dash, color=:gray, label="Pure ice")

axislegend(ax, position=:rt)
fig
```

## Effect of Riming

Riming dramatically affects particle properties. This is the key insight of P3 that enables
continuous evolution without discrete category conversions
([Morrison2015parameterization](@citet) Section 2d):

| Property | Unrimed Aggregate | Heavily Rimed Graupel |
|----------|-------------------|----------------------|
| Mass | ``α D^β`` | ``(π/6) ρ_g D³`` |
| Density | Low (~100 kg/m³) | High (~500 kg/m³) |
| Fall speed | Slow | Fast |
| Collection efficiency | Low | High |

```@example p3_particles
# Compare mass for different rime fractions
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [mm]",
    ylabel = "Mass m [mg]",
    title = "Effect of Riming on Particle Mass")

D_mm = range(0.1, 5, length=50)
D_m = D_mm .* 1e-3

for (Ff, label, color) in [(0.0, "Fᶠ = 0", :blue),
                            (0.25, "Fᶠ = 0.25", :green),
                            (0.5, "Fᶠ = 0.5", :orange),
                            (0.75, "Fᶠ = 0.75", :red)]
    m = [ice_mass(mass, Ff, 500.0, d) for d in D_m]
    lines!(ax, D_mm, m .* 1e6, label=label, color=color)  # Convert to mg
end

axislegend(ax, position=:lt)
fig
```

## Rime Density Parameterization

The rime density ``ρᶠ`` depends on the collection conditions during riming. The
parameterization follows [Cober and List (1993)](@cite CoberList1993) as implemented in
[Morrison2015parameterization](@citet). The rime density is computed as a function of
the impact parameter ``R_i``, which depends on droplet size, impact velocity, and temperature:

```math
ρᶠ = \begin{cases}
(0.051 + 0.114 R_i - 0.0055 R_i^2) \times 1000 & R_i \le 8 \\
611 + 72.25 (R_i - 8) & R_i > 8
\end{cases}
```

The rime density is bounded:
- ``ρ_{min} = 50`` kg/m³ is minimum rime density
- ``ρ_{max} = 900`` kg/m³ is maximum rime density

The rime density affects the graupel density ``ρ_g`` and thus the regime thresholds.
As particles rime more heavily, they become denser and more spherical.

!!! note "Official P3 implementation details"
    The Fortran scheme clamps ``R_i`` to [1, 12] before applying the Cober–List fit;
    the linear branch for ``R_i > 8`` is extended to ``R_i = 12`` so that
    ``ρᶠ = 900`` kg/m³. When riming is inactive, ``ρᶠ`` defaults to 400 kg/m³.
    The lookup tables discretize ``ρᶠ`` on an uneven grid (50, 250, 450, 650, 900 kg/m³)
    and interpolate between bins.

## Summary

The P3 mass-diameter relationship captures the full spectrum of ice particle types:

1. **Small crystals**: Dense, spherical approximation
2. **Aggregates**: Fractal structure, low density, follows ``m ∝ D^{1.9}``
3. **Graupel**: Compact, dense from riming
4. **Partially rimed**: Large aggregates with rimed cores

The transitions occur naturally through the regime thresholds, which depend only on the
predicted rime fraction and rime density—no arbitrary conversion terms required.

## References for This Section

- [Morrison2015parameterization](@cite): Primary source for m(D), A(D), V(D) relationships
- [Morrison2015part2](@cite): Validation of particle property parameterizations
- [pruppacher2010microphysics](@cite): Background on ice particle physics
