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
(see [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2b for
the underlying ``V(D)`` formulation; the integrated fall speeds are stored in
`p3_lookupTable_1.dat-v*` and described in
[Morrison et al. (2025)](@cite Morrison2025complete3moment)).

### Terminal Velocity Formulation

Individual particle fall speed follows the [Mitchell and Heymsfield (2005)](@cite MitchellHeymsfield2005)
Best number formulation, which relates fall speed to particle mass, projected area, and air properties.
The formulation accounts for the transition from Stokes to turbulent flow regimes and includes
surface roughness effects. A density correction factor ``(ρ₀/ρ)^{0.54}`` is applied following
[Heymsfield et al. (2007)](@cite HeymsfieldEtAl2007).

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

## Lambda Limiter Integrals

To prevent unphysical size distributions, P3 limits the slope parameter ``λ``
based on physical constraints.

| Integral | Purpose |
|----------|---------|
| `SmallQLambdaLimit` | Lower bound on λ (prevents unrealistically large particles) |
| `LargeQLambdaLimit` | Upper bound on λ (prevents unrealistically small particles) |

## Tabulation

For efficiency in simulations, integrals are organized into three Breeze lookup-table families:

- `lookupTable_1`: fall speed, ventilation, bulk, collection, sixth-moment, and lambda-limiter integrals
- `lookupTable_2`: ice-rain and inter-category collection families
- `lookupTable_3`: three-moment diagnostic lookup for `μᶦ`, `λᶦ`, and companion fields

```@example p3_integrals
using Logging: NullLogger, with_logger

# The default constructor reads the Fortran ASCII lookup tables
# (downloaded automatically on first use).
p3 = with_logger(NullLogger()) do
    PredictedParticlePropertiesMicrophysics()
end

fs = p3.ice.fall_speed
println("Tabulated fall speed integrals from Fortran tables:")
println("  Number-weighted: $(typeof(fs.number_weighted))")
println("  Mass-weighted:   $(typeof(fs.mass_weighted))")
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

At runtime each integral is read from the corresponding Fortran ASCII lookup
table; the rain 1D tables are tabulated at startup from Chebyshev–Gauss
quadrature evaluators.

## References for This Section

- [Morrison2015parameterization](@cite): Fall speed, ventilation, collection integrals (Section 2b and Appendix C)
- [HallPruppacher1976](@cite): Ventilation factor coefficients
- [MilbrandtEtAl2021](@cite): Sixth moment integrals for three-moment ice (Table 1)
- [MilbrandtEtAl2024](@cite): Updated three-moment formulation
- [Morrison2025complete3moment](@cite): Complete three-moment lookup table (29 quantities)
