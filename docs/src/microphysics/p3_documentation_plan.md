# P3 Documentation Plan

This document outlines the comprehensive documentation needed for the Predicted Particle Properties (P3) microphysics scheme in Breeze.jl.

## Documentation Structure

```
docs/src/microphysics/
├── p3_overview.md              # Introduction and motivation
├── p3_particle_properties.md   # Mass, area, density relationships
├── p3_size_distribution.md     # Gamma PSD and parameter determination
├── p3_integral_properties.md   # Bulk properties from PSD integrals
├── p3_processes.md             # Microphysical process rates
├── p3_prognostics.md           # Prognostic variables and tendencies
└── p3_examples.md              # Worked examples and simulations
```

---

## 1. Overview (`p3_overview.md`)

### Content
- **Motivation**: Why a single ice category with predicted properties?
- **Comparison to traditional schemes**: Cloud ice / snow / graupel / hail categories
- **Key innovations**: Continuous property evolution, 3-moment ice, liquid fraction
- **History**: Morrison & Milbrandt (2015) → (2016) → Milbrandt et al. (2024)

### Equations
- None (conceptual overview)

### Code Examples
```julia
using Breeze.Microphysics.PredictedParticleProperties

# Create P3 scheme with default parameters
microphysics = PredictedParticlePropertiesMicrophysics()
```

---

## 2. Particle Properties (`p3_particle_properties.md`)

### Content
- **Mass-diameter relationship**: Piecewise m(D) across four regimes
- **Area-diameter relationship**: Projected area for fall speed
- **Density**: How particle density varies with size and riming
- **Thresholds**: D_th (spherical), D_gr (graupel), D_cr (partial rime)

### Equations

**Spherical ice** (D < D_th):
```math
m(D) = \frac{π}{6} ρᵢ D³
```

**Vapor-grown aggregates** (D ≥ D_th, unrimed or lightly rimed):
```math
m(D) = α D^β
```
where α = 0.0121 kg/m^β, β = 1.9

**Graupel** (D_gr ≤ D < D_cr):
```math
m(D) = \frac{π}{6} ρ_g D³
```

**Partially rimed** (D ≥ D_cr):
```math
m(D) = \frac{α}{1 - F^f} D^β
```

**Threshold formulas**:
```math
D_{th} = \left(\frac{6α}{π ρᵢ}\right)^{1/(3-β)}
```

### Code Examples
```julia
mass = IceMassPowerLaw()
thresholds = ice_regime_thresholds(mass, rime_fraction, rime_density)

# Compute mass at different sizes
m_small = ice_mass(mass, 0.0, 400.0, 10e-6)   # 10 μm particle
m_large = ice_mass(mass, 0.0, 400.0, 1e-3)    # 1 mm particle
```

### Figures
- m(D) and A(D) plots for different rime fractions (replicate Fig. 1 from MM2015)
- Density vs diameter for different riming states

---

## 3. Size Distribution (`p3_size_distribution.md`)

### Content
- **Gamma distribution**: N'(D) = N₀ D^μ exp(-λD)
- **Moments**: M_k = ∫ D^k N'(D) dD
- **μ-λ relationship**: Shape parameter closure
- **Lambda solver**: Determining (N₀, λ, μ) from (L, N)

### Equations

**Size distribution**:
```math
N'(D) = N₀ D^μ e^{-λD}
```

**Moments**:
```math
M_k = N₀ \frac{Γ(k + μ + 1)}{λ^{k+μ+1}}
```

**Normalization**:
```math
N = \int_0^∞ N'(D)\, dD = N₀ \frac{Γ(μ+1)}{λ^{μ+1}}
```

**μ-λ relation** (Morrison & Milbrandt 2015):
```math
μ = \text{clamp}(a λ^b - c, 0, μ_{max})
```
with a = 0.00191, b = 0.8, c = 2, μ_max = 6

**Mass content**:
```math
L = \int_0^∞ m(D) N'(D)\, dD
```

### Code Examples
```julia
# Solve for distribution parameters
L_ice = 1e-4  # kg/m³
N_ice = 1e5   # 1/m³

params = distribution_parameters(L_ice, N_ice, rime_fraction, rime_density)
# Returns (N₀, λ, μ)
```

### Figures
- Size distributions for different λ, μ values
- L/N ratio vs λ showing solver convergence

---

## 4. Integral Properties (`p3_integral_properties.md`)

### Content
- **Fall speeds**: Number-, mass-, reflectivity-weighted
- **Deposition/sublimation**: Ventilation factors
- **Bulk properties**: Effective radius, mean diameter, reflectivity
- **Collection**: Aggregation, riming kernels
- **Sixth moment**: For 3-moment scheme

### Equations

**Number-weighted fall speed**:
```math
V_n = \frac{\int_0^∞ V(D) N'(D)\, dD}{\int_0^∞ N'(D)\, dD}
```

**Mass-weighted fall speed**:
```math
V_m = \frac{\int_0^∞ V(D) m(D) N'(D)\, dD}{\int_0^∞ m(D) N'(D)\, dD}
```

**Fall speed power law**:
```math
V(D) = a_V \left(\frac{ρ_0}{ρ}\right)^{0.5} D^{b_V}
```

**Ventilation factor** (Hall & Pruppacher 1976):
```math
f_v = a_v + b_v \text{Re}^{0.5} \text{Sc}^{1/3}
```

### Code Examples
```julia
state = IceSizeDistributionState(Float64;
    intercept = 1e6, shape = 0.0, slope = 1000.0)

V_n = evaluate(NumberWeightedFallSpeed(), state)
V_m = evaluate(MassWeightedFallSpeed(), state)
```

### Figures
- Fall speed vs λ for different μ
- Comparison of V_n, V_m, V_z

---

## 5. Microphysical Processes (`p3_processes.md`)

### Content

#### Warm rain processes
- **Condensation/evaporation**: Saturation adjustment or explicit
- **Autoconversion**: Cloud → rain (Khairoutdinov-Kogan or Berry-Reinhardt)
- **Accretion**: Cloud + rain → rain
- **Rain evaporation**: Below cloud base

#### Ice nucleation
- **Heterogeneous**: INP activation
- **Homogeneous**: T < -38°C freezing
- **Secondary**: Hallett-Mossop, rime splintering

#### Vapor-ice exchange
- **Deposition**: Vapor → ice (supersaturated wrt ice)
- **Sublimation**: Ice → vapor (subsaturated)

#### Collection processes
- **Riming**: Ice + cloud → ice (rime fraction increases)
- **Ice-rain collection**: Ice + rain → ice
- **Aggregation**: Ice + ice → ice

#### Phase change
- **Melting**: Ice → rain (T > 0°C)
- **Shedding**: Liquid on ice → rain
- **Refreezing**: Liquid on ice → rime

### Equations

**Deposition rate** (per particle):
```math
\frac{dm}{dt} = 4π C f_v D_v (ρ_v - ρ_{v,i})
```

**Riming rate**:
```math
\frac{dq^f}{dt} = E_{ic} \int_0^∞ A(D) V(D) q^{cl} N'(D)\, dD
```

**Melting**:
```math
\frac{dm}{dt} = -\frac{4π C}{L_f} \left[ k_a (T - T_0) + L_v D_v (ρ_v - ρ_{v,s}) \right] f_v
```

### Code Examples
- Process tendency computations (once implemented)

---

## 6. Prognostic Variables (`p3_prognostics.md`)

### Content
- **What P3 tracks**: 9 prognostic fields
- **Tendency equations**: ∂ρq/∂t = ...
- **Coupling to dynamics**: How microphysics couples to AtmosphereModel

### Variables

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ρqᶜˡ | Cloud liquid | kg/m³ | Cloud droplet mass |
| ρqʳ | Rain | kg/m³ | Rain mass |
| ρnʳ | Rain number | 1/m³ | Raindrop number |
| ρqⁱ | Ice | kg/m³ | Total ice mass |
| ρnⁱ | Ice number | 1/m³ | Ice particle number |
| ρqᶠ | Rime | kg/m³ | Rime/frost mass |
| ρbᶠ | Rime volume | m³/m³ | Rime volume density |
| ρzⁱ | Ice reflectivity | m⁶/m³ | 6th moment |
| ρqʷⁱ | Water on ice | kg/m³ | Liquid fraction |

### Equations

**Mass tendency**:
```math
\frac{∂ρq^i}{∂t} = \text{DEP} + \text{RIM} + \text{AGG} - \text{SUB} - \text{MLT}
```

**Number tendency**:
```math
\frac{∂ρn^i}{∂t} = \text{NUC} - \text{AGG}_n - \text{MLT}_n
```

---

## 7. Examples and Simulations (`p3_examples.md`)

### Content
- **Parcel model**: Adiabatic ascent with P3 microphysics
- **1D kinematic**: Prescribed updraft, test sedimentation
- **Convective cell**: 2D/3D simulation showing ice evolution

### Simulations

1. **Ice particle explorer** (already implemented)
   - Visualize m(D), A(D), V(D) relationships
   - Show effect of riming on particle properties

2. **Adiabatic parcel**
   - Start with supersaturated air
   - Watch ice nucleation and growth
   - Track rime fraction evolution

3. **Squall line** (advanced)
   - Show graupel formation
   - Compare to traditional schemes

---

## Implementation Order

1. **p3_overview.md** - Start with motivation (half day)
2. **p3_particle_properties.md** - Core physics (1 day)
3. **p3_size_distribution.md** - PSD and solver (1 day)
4. **p3_integral_properties.md** - All integrals documented (1 day)
5. **p3_prognostics.md** - Variable definitions (half day)
6. **p3_processes.md** - Detailed process physics (2 days)
7. **p3_examples.md** - Simulations and visualization (1-2 days)

**Total estimate: 7-8 days of focused work**

---

## References

All equations should cite:
- Morrison, H., & Milbrandt, J. A. (2015). Parameterization of cloud microphysics...
- Milbrandt, J. A., & Morrison, H. (2016). ...Part III: Three-moment...
- Milbrandt, J. A., et al. (2024). ...Predicted liquid fraction...
- Hall, W. D., & Pruppacher, H. R. (1976). Ventilation...
- Khairoutdinov, M., & Kogan, Y. (2000). Autoconversion...
