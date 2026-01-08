# [Particle Properties](@id p3_particle_properties)

Ice particles in P3 span a continuum from small pristine crystals to large rimed graupel.
The mass-diameter and area-diameter relationships vary across this spectrum, depending on
particle size and riming state.

## Mass-Diameter Relationship

The particle mass ``m(D)`` follows a piecewise power law that depends on maximum dimension ``D``,
rime fraction ``Fᶠ``, and rime density ``ρᶠ``.

### The Four Regimes

P3 defines four diameter regimes with distinct mass-diameter relationships:

**Regime 1: Small Spherical Ice** (``D < D_{th}``)

Small ice particles are assumed spherical with pure ice density:

```math
m(D) = \frac{π}{6} ρᵢ D³
```

where ``ρᵢ = 917`` kg/m³ is pure ice density.

**Regime 2: Vapor-Grown Aggregates** (``D_{th} ≤ D < D_{gr}`` or unrimed)

Larger particles follow an empirical power law based on aircraft observations
of ice crystals and aggregates:

```math
m(D) = α D^β
```

where ``α = 0.0121`` kg/m^β and ``β = 1.9`` from [MorrisonMilbrandt2016](@cite).
This relationship captures the fractal nature of aggregated crystals.

**Regime 3: Graupel** (``D_{gr} ≤ D < D_{cr}``)

When particles acquire sufficient rime, they become compact graupel
with density ``ρ_g``:

```math
m(D) = \frac{π}{6} ρ_g D³
```

The graupel density ``ρ_g`` depends on the rime fraction and rime density:

```math
ρ_g = Fᶠ ρᶠ + (1 - Fᶠ) ρ_d
```

where ``ρ_d`` is the density of the deposited (vapor-grown) ice component.

**Regime 4: Partially Rimed** (``D ≥ D_{cr}``)

The largest particles have a rimed core with unrimed aggregate extensions:

```math
m(D) = \frac{α}{1 - Fᶠ} D^β
```

### Threshold Diameters

The transitions between regimes occur at critical diameters determined by
equating masses:

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
From [Morrison2015parameterization](@cite) Equation 16:

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

**Small Spherical Ice** (``D < D_{th}``):

```math
A(D) = \frac{π}{4} D²
```

**Nonspherical Ice** (aggregates):

```math
A(D) = γ D^σ
```

where ``γ`` and ``σ`` are empirical coefficients.

**Graupel**:

Reverts to spherical:

```math
A(D) = \frac{π}{4} D²
```

**Partially Rimed**:

Weighted average of spherical and nonspherical:

```math
A(D) = Fᶠ \frac{π}{4} D² + (1 - Fᶠ) γ D^σ
```

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

Riming dramatically affects particle properties:

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

## Summary

The P3 mass-diameter relationship captures the full spectrum of ice particle types:

1. **Small crystals**: Dense, spherical approximation
2. **Aggregates**: Fractal structure, low density, follows ``m ∝ D^{1.9}``
3. **Graupel**: Compact, dense from riming
4. **Partially rimed**: Large aggregates with rimed cores

The transitions occur naturally through the regime thresholds, which depend only on the
predicted rime fraction and rime density—no arbitrary conversion terms required.

