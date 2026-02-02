# [Integral Properties](@id p3_integral_properties)

Bulk microphysical rates require population-averaged quantities computed by integrating
over the particle size distribution. P3 defines numerous integral properties organized
by physical concept.

These integrals are pre-computed and stored in lookup tables in the Fortran P3 code
(see `create_p3_lookupTable_1.f90` in the [P3-microphysics repository](https://github.com/P3-microphysics/P3-microphysics)).
The integral formulations are from:
- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Fall speed, ventilation, collection
- [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021): Sixth moment integrals for 3-moment ice
- [Morrison et al. (2025)](@cite Morrison2025complete3moment): Complete 3-moment lookup tables

## General Form

All integral properties have the form:

```math
\langle X \rangle = \frac{\int_0^∞ X(D) N'(D)\, dD}{\int_0^∞ W(D) N'(D)\, dD}
```

where ``X(D)`` is the quantity of interest and ``W(D)`` is a weighting function
(often unity or particle mass).

## Fall Speed Integrals

Terminal velocity determines sedimentation rates. P3 computes three weighted fall speeds,
corresponding to `uns`, `ums`, `uzs` in the Fortran lookup tables
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Table 3).

### Terminal Velocity Formulation

Individual particle fall speed follows the [Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005)
Best number formulation, which relates fall speed to particle mass, projected area, and air properties.
The formulation accounts for the transition from Stokes to turbulent flow regimes and includes
surface roughness effects. A density correction factor ``(ρ₀/ρ)^{0.54}`` is applied following
[Heymsfield et al. (2006)](@cite HeymsfieldEtAl2006).

For mixed-phase particles (with liquid fraction ``F^l``), the fall speed is linearly interpolated
between the ice fall speed and the rain fall speed:

```math
V(D) = F^l V_{rain}(D) + (1 - F^l) V_{ice}(D)
```

The fall speed depends on the mass-diameter and area-diameter relationships, which vary
across the four particle regimes (see [Particle Properties](@ref p3_particle_properties)).

### Number-Weighted Fall Speed

```math
V_n = \frac{\int_0^∞ V(D) N'(D)\, dD}{\int_0^∞ N'(D)\, dD}
```

This represents the average fall speed of particles and governs number flux:

```math
F_N = -V_n \cdot N
```

### Mass-Weighted Fall Speed

```math
V_m = \frac{\int_0^∞ V(D) m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

This governs mass flux:

```math
F_L = -V_m \cdot L
```

### Reflectivity-Weighted Fall Speed

For 3-moment ice, the 6th moment flux uses:

```math
V_z = \frac{\int_0^∞ V(D) D^6 N'(D)\, dD}{\int_0^∞ D^6 N'(D)\, dD}
```

```@example p3_integrals
using Breeze.Microphysics.PredictedParticleProperties

# Create a size distribution state
state = IceSizeDistributionState(Float64;
    intercept = 1e6,
    shape = 0.0,
    slope = 1000.0)

# Evaluate fall speed integrals
V_n = evaluate(NumberWeightedFallSpeed(), state)
V_m = evaluate(MassWeightedFallSpeed(), state)
V_z = evaluate(ReflectivityWeightedFallSpeed(), state)

println("Fall speed integrals:")
println("  V_n (number-weighted) = $(round(V_n, digits=3))")
println("  V_m (mass-weighted)   = $(round(V_m, digits=3))")
println("  V_z (reflectivity)    = $(round(V_z, digits=3))")
```

## Deposition/Sublimation Integrals

Vapor diffusion to/from ice particles is enhanced by air flow around falling particles.

### Ventilation Factor

The ventilation factor ``f_v`` accounts for enhanced mass transfer:

```math
f_v = a_v + b_v \text{Re}^{1/2} \text{Sc}^{1/3}
```

where:
- ``\text{Re} = V D / ν`` is the Reynolds number
- ``\text{Sc} = ν / D_v`` is the Schmidt number
- ``a_v, b_v`` are empirical coefficients from [HallPruppacher1976](@cite)

### Ventilation Integrals

P3 computes six ventilation-related integrals for different size regimes:

| Integral | Description | Size Regime |
|----------|-------------|-------------|
| `SmallIceVentilationConstant` | Constant term for small ice | D < 100 μm |
| `SmallIceVentilationReynolds` | Re-dependent term for small ice | D < 100 μm |
| `LargeIceVentilationConstant` | Constant term for large ice | D ≥ 100 μm |
| `LargeIceVentilationReynolds` | Re-dependent term for large ice | D ≥ 100 μm |
| `Ventilation` | Total ventilation integral | All sizes |
| `VentilationEnhanced` | Enhanced ventilation (large ice only) | D ≥ 100 μm |

```@example p3_integrals
# Ventilation integrals
v_basic = evaluate(Ventilation(), state)
v_enhanced = evaluate(VentilationEnhanced(), state)

println("Ventilation integrals:")
println("  Basic ventilation     = $(round(v_basic, sigdigits=3))")
println("  Enhanced (large ice)  = $(round(v_enhanced, sigdigits=3))")
```

## Bulk Property Integrals

Population-averaged properties for radiation, radar, and diagnostics.

### Effective Radius

Important for radiation parameterizations:

```math
r_{eff} = \frac{\int_0^∞ D^3 N'(D)\, dD}{2 \int_0^∞ D^2 N'(D)\, dD} = \frac{M_3}{2 M_2}
```

### Mean Diameter

Mass-weighted mean particle size:

```math
D_m = \frac{\int_0^∞ D \cdot m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

### Mean Density

Mass-weighted particle density:

```math
ρ_m = \frac{\int_0^∞ ρ(D) m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

### Reflectivity

Radar reflectivity factor (proportional to 6th moment):

```math
Z = \int_0^∞ D^6 N'(D)\, dD = N₀ \frac{Γ(μ + 7)}{λ^{μ+7}}
```

```@example p3_integrals
r_eff = evaluate(EffectiveRadius(), state)
D_m = evaluate(MeanDiameter(), state)
ρ_m = evaluate(MeanDensity(), state)
Z = evaluate(Reflectivity(), state)

println("Bulk properties:")
println("  Effective radius = $(round(r_eff * 1e6, digits=1)) μm")
println("  Mean diameter    = $(round(D_m * 1e3, digits=2)) mm")
println("  Mean density     = $(round(ρ_m, digits=1)) kg/m³")
println("  Reflectivity Z   = $(round(Z * 1e18, sigdigits=3)) mm⁶/m³")
```

## Collection Integrals

Collection processes (aggregation, riming) require integrals over collision kernels.

### Aggregation

The collection kernel for ice-ice aggregation is:

```math
K_{agg}(D_1, D_2) = E_{agg} \frac{π}{4} (D_1 + D_2)^2 |V(D_1) - V(D_2)|
```

The aggregation rate integral:

```math
I_{agg} = \int_0^∞ \int_0^∞ K_{agg}(D_1, D_2) N'(D_1) N'(D_2)\, dD_1 dD_2
```

### Ice-Cloud Collection (Riming)

```math
\frac{dq^f}{dt} = E_{ic} q^{cl} \int_0^∞ A(D) V(D) N'(D)\, dD
```

### Ice-Rain Collection

```math
I_{ir} = \int_0^∞ A(D) V(D) N'(D)\, dD
```

```@example p3_integrals
n_agg = evaluate(AggregationNumber(), state)
n_rain = evaluate(RainCollectionNumber(), state)

println("Collection integrals:")
println("  Aggregation number   = $(round(n_agg, sigdigits=3))")
println("  Rain collection      = $(round(n_rain, sigdigits=3))")
```

## Sixth Moment Integrals

For 3-moment ice ([Milbrandt et al. (2021)](@cite MilbrandtEtAl2021),
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024)),
P3 tracks the 6th moment ``Z`` which requires additional integrals
for each process affecting reflectivity. These are documented in
[Morrison et al. (2025)](@cite Morrison2025complete3moment) and stored
in the 3-moment lookup table (`p3_lookupTable_1.dat-v*_3momI`).

| Process | Integral | Physical Meaning |
|---------|----------|------------------|
| Riming | `SixthMomentRime` | Z change from rime accretion |
| Deposition | `SixthMomentDeposition` | Z change from vapor growth |
| Melting | `SixthMomentMelt1`, `SixthMomentMelt2` | Z change from melting |
| Aggregation | `SixthMomentAggregation` | Z change from aggregation |
| Shedding | `SixthMomentShedding` | Z change from liquid shedding |
| Sublimation | `SixthMomentSublimation` | Z change from sublimation |

```@example p3_integrals
# Sixth moment integrals (with nonzero liquid fraction for shedding)
state_wet = IceSizeDistributionState(Float64;
    intercept = 1e6, shape = 0.0, slope = 1000.0,
    liquid_fraction = 0.1)

z_rime = evaluate(SixthMomentRime(), state_wet)
z_dep = evaluate(SixthMomentDeposition(), state_wet)
z_agg = evaluate(SixthMomentAggregation(), state_wet)
z_shed = evaluate(SixthMomentShedding(), state_wet)

println("Sixth moment integrals:")
println("  Rime         = $(round(z_rime, sigdigits=3))")
println("  Deposition   = $(round(z_dep, sigdigits=3))")
println("  Aggregation  = $(round(z_agg, sigdigits=3))")
println("  Shedding     = $(round(z_shed, sigdigits=3))")
```

## Lambda Limiter Integrals

To prevent unphysical size distributions, P3 limits the slope parameter ``λ``
based on physical constraints.

| Integral | Purpose |
|----------|---------|
| `SmallQLambdaLimit` | Lower bound on λ (prevents unrealistically large particles) |
| `LargeQLambdaLimit` | Upper bound on λ (prevents unrealistically small particles) |

## Dependence on Distribution Parameters

Integral values depend strongly on the size distribution parameters:

```@example p3_integrals
using CairoMakie

# Vary slope parameter
λ_values = 10 .^ range(2.5, 4.5, length=20)
V_n_values = Float64[]
V_m_values = Float64[]
V_z_values = Float64[]

for λ in λ_values
    state = IceSizeDistributionState(Float64; intercept=1e6, shape=0.0, slope=λ)
    push!(V_n_values, evaluate(NumberWeightedFallSpeed(), state))
    push!(V_m_values, evaluate(MassWeightedFallSpeed(), state))
    push!(V_z_values, evaluate(ReflectivityWeightedFallSpeed(), state))
end

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Slope parameter λ [m⁻¹]",
    ylabel = "Fall speed integral",
    xscale = log10,
    title = "Fall Speed Integrals vs λ")

lines!(ax, λ_values, V_n_values, label="Vₙ (number)")
lines!(ax, λ_values, V_m_values, label="Vₘ (mass)")
lines!(ax, λ_values, V_z_values, label="Vᵤ (reflectivity)")

axislegend(ax, position=:rt)
fig
```

## Numerical Integration and Tabulation

The official P3 lookup tables are generated using fixed-bin numerical integration
(40,000 diameter bins for single-particle integrals and 1,500 for collection integrals),
with constant bin widths in diameter space.

Breeze evaluates integrals directly using Chebyshev-Gauss quadrature with
a change of variables to map ``[0, ∞)`` to a bounded interval:

```@example p3_integrals
# The evaluate function uses 64 quadrature points by default
V_n_64 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=64)
V_n_128 = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=128)

println("Quadrature convergence:")
println("  64 points:  $(V_n_64)")
println("  128 points: $(V_n_128)")
println("  Difference: $(abs(V_n_128 - V_n_64))")
```

## Tabulation

For efficiency in simulations, integrals can be pre-computed and stored in
lookup tables:

```@example p3_integrals
using Oceananigans: CPU

# Create tabulated fall speed integrals
params = TabulationParameters(Float64;
    number_of_mass_points = 10,
    number_of_rime_fraction_points = 3,
    number_of_liquid_fraction_points = 2,
    number_of_quadrature_points = 64)
fs = IceFallSpeed()
fs_tab = tabulate(fs, CPU(), params)

println("Tabulated fall speed integrals:")
println("  Table summary: $(summary(fs_tab.number_weighted))")
println("  Sample value: $(fs_tab.number_weighted.table[5, 2, 1])")
```

## Summary

P3 uses 29+ integral properties organized by concept:

| Category | Count | Purpose |
|----------|-------|---------|
| Fall speed | 3 | Sedimentation fluxes |
| Deposition | 6 | Vapor diffusion rates |
| Bulk properties | 7 | Radiation, diagnostics |
| Collection | 2 | Aggregation, riming |
| Sixth moment | 9 | 3-moment closure |
| Lambda limiter | 2 | Distribution bounds |

All integrals use the same infrastructure: define the integrand, then call
`evaluate(integral_type, state)` with optional quadrature settings.

## References for This Section

- [Morrison2015parameterization](@cite): Fall speed, ventilation, collection integrals (Table 3, Section 2)
- [HallPruppacher1976](@cite): Ventilation factor coefficients
- [MilbrandtEtAl2021](@cite): Sixth moment integrals for three-moment ice (Table 1)
- [MilbrandtEtAl2024](@cite): Updated three-moment formulation
- [Morrison2025complete3moment](@cite): Complete three-moment lookup table (29 quantities)
