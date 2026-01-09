# [Prognostic Variables and Tendencies](@id p3_prognostics)

P3 tracks 9 prognostic variables that together describe the complete microphysical
state of the atmosphere. This section documents each variable, its physical meaning,
and the tendency equations governing its evolution.

The prognostic variable formulation has evolved through the P3 papers:
- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Original 4 ice variables
- [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021): Added ``ρz^i`` for 3-moment ice
- [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction): Added ``ρq^{wi}`` for liquid fraction

Our implementation follows P3 v5.5 with all 6 ice prognostic variables.

## Variable Definitions

### Cloud Liquid

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^{cl}`` | Cloud liquid mass density | kg/m³ | Mass of cloud droplets per unit volume |

Cloud droplet **number** is prescribed (not prognostic) in the standard P3 configuration,
typically set to continental (100 cm⁻³) or marine (50 cm⁻³) values.

### Rain

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^r`` | Rain mass density | kg/m³ | Mass of raindrops per unit volume |
| ``ρn^r`` | Rain number density | m⁻³ | Number of raindrops per unit volume |

Rain follows a gamma size distribution with parameters diagnosed from the mass/number ratio.

### Ice

| Symbol | Name | Units | Description |
|--------|------|-------|-------------|
| ``ρq^i`` | Total ice mass density | kg/m³ | Total ice mass (all forms) |
| ``ρn^i`` | Ice number density | m⁻³ | Number of ice particles |
| ``ρq^f`` | Rime mass density | kg/m³ | Mass of rime (frost) on ice |
| ``ρb^f`` | Rime volume density | m³/m³ | Volume of rime per unit volume |
| ``ρz^i`` | Ice reflectivity | m⁶/m³ | 6th moment of size distribution |
| ``ρq^{wi}`` | Water on ice | kg/m³ | Liquid water coating ice particles |

## Derived Quantities

From the prognostic variables, key diagnostic properties are computed:

**Rime fraction** (mass fraction of rime):
```math
F^f = \frac{ρq^f}{ρq^i}
```

**Rime density** (density of the rime layer):
```math
ρ^f = \frac{ρq^f}{ρb^f}
```

**Liquid fraction** (mass fraction of liquid coating):
```math
F^l = \frac{ρq^{wi}}{ρq^i}
```

**Mean particle mass**:
```math
\bar{m} = \frac{ρq^i}{ρn^i}
```

## Tendency Equations

Each prognostic variable evolves according to:

```math
\frac{\partial (ρX)}{\partial t} = \text{ADV} + \text{TURB} + \text{SED} + \text{SRC}
```

where:
- **ADV**: Advection by resolved flow
- **TURB**: Subgrid turbulent transport
- **SED**: Sedimentation (gravitational settling)
- **SRC**: Microphysical source/sink terms

### Cloud Liquid Tendency

```math
\frac{\partial ρq^{cl}}{\partial t}\bigg|_{src} = \underbrace{COND}_{\text{condensation}} 
- \underbrace{EVAP}_{\text{evaporation}}
- \underbrace{AUTO}_{\text{autoconversion}}
- \underbrace{ACCR}_{\text{accretion by rain}}
- \underbrace{RIM}_{\text{riming by ice}}
- \underbrace{HOMF}_{\text{homogeneous freezing}}
```

### Rain Mass Tendency

```math
\frac{\partial ρq^r}{\partial t}\bigg|_{src} = \underbrace{AUTO}_{\text{autoconversion}}
+ \underbrace{ACCR}_{\text{accretion}}
+ \underbrace{SHED}_{\text{shedding from ice}}
+ \underbrace{MELT}_{\text{complete melting}}
- \underbrace{EVAP}_{\text{rain evaporation}}
- \underbrace{COLL}_{\text{collection by ice}}
- \underbrace{FREZ}_{\text{freezing}}
```

### Rain Number Tendency

```math
\frac{\partial ρn^r}{\partial t}\bigg|_{src} = \underbrace{AUTO_n}_{\text{autoconversion}}
+ \underbrace{SHED_n}_{\text{shedding}}
+ \underbrace{MELT_n}_{\text{melting}}
- \underbrace{EVAP_n}_{\text{evaporation}}
- \underbrace{COLL_n}_{\text{collection}}
- \underbrace{SCBK}_{\text{self-collection/breakup}}
```

### Ice Mass Tendency

```math
\frac{\partial ρq^i}{\partial t}\bigg|_{src} = \underbrace{NUC}_{\text{nucleation}}
+ \underbrace{DEP}_{\text{deposition}}
+ \underbrace{RIM}_{\text{riming}}
+ \underbrace{COLL}_{\text{rain collection}}
- \underbrace{SUB}_{\text{sublimation}}
- \underbrace{MELT}_{\text{melting}}
```

### Ice Number Tendency

```math
\frac{\partial ρn^i}{\partial t}\bigg|_{src} = \underbrace{NUC_n}_{\text{nucleation}}
+ \underbrace{SEC}_{\text{secondary production}}
- \underbrace{AGG_n}_{\text{aggregation}}
- \underbrace{MELT_n}_{\text{melting}}
```

### Rime Mass Tendency

```math
\frac{\partial ρq^f}{\partial t}\bigg|_{src} = \underbrace{RIM}_{\text{riming}}
+ \underbrace{COLL}_{\text{rain collection}}
+ \underbrace{REFR}_{\text{refreezing}}
- \underbrace{SUB_f}_{\text{sublimation}}
- \underbrace{MELT_f}_{\text{melting}}
```

### Rime Volume Tendency

```math
\frac{\partial ρb^f}{\partial t}\bigg|_{src} = \frac{1}{ρ^f}\left(\underbrace{RIM}_{\text{riming}}
+ \underbrace{COLL}_{\text{rain collection}}
+ \underbrace{REFR}_{\text{refreezing}}\right)
- \underbrace{SUB_b}_{\text{sublimation}}
- \underbrace{MELT_b}_{\text{melting}}
```

### Reflectivity Tendency (3-moment)

```math
\frac{\partial ρz^i}{\partial t}\bigg|_{src} = \underbrace{DEP_z}_{\text{deposition}}
+ \underbrace{RIM_z}_{\text{riming}}
- \underbrace{AGG_z}_{\text{aggregation}}
- \underbrace{SUB_z}_{\text{sublimation}}
- \underbrace{MELT_z}_{\text{melting}}
- \underbrace{SHED_z}_{\text{shedding}}
```

### Liquid on Ice Tendency

```math
\frac{\partial ρq^{wi}}{\partial t}\bigg|_{src} = \underbrace{MELT_{part}}_{\text{partial melting}}
- \underbrace{SHED}_{\text{shedding}}
- \underbrace{REFR}_{\text{refreezing}}
- \underbrace{EVAP_{wi}}_{\text{evaporation}}
```

## Sedimentation

Each quantity sediments at its characteristic velocity:

| Variable | Sedimentation Velocity | Flux |
|----------|----------------------|------|
| ``ρq^r`` | ``V_m^r`` | ``F_q^r = -V_m^r ρq^r`` |
| ``ρn^r`` | ``V_n^r`` | ``F_n^r = -V_n^r ρn^r`` |
| ``ρq^i`` | ``V_m^i`` | ``F_q^i = -V_m^i ρq^i`` |
| ``ρn^i`` | ``V_n^i`` | ``F_n^i = -V_n^i ρn^i`` |
| ``ρq^f`` | ``V_m^i`` | ``F_q^f = -V_m^i ρq^f`` |
| ``ρb^f`` | ``V_m^i`` | ``F_b^f = -V_m^i ρb^f`` |
| ``ρz^i`` | ``V_z^i`` | ``F_z^i = -V_z^i ρz^i`` |
| ``ρq^{wi}`` | ``V_m^i`` | ``F_q^{wi} = -V_m^i ρq^{wi}`` |

The sedimentation tendency is:

```math
\frac{\partial ρX}{\partial t}\bigg|_{sed} = -\frac{\partial F_X}{\partial z}
```

## Coupling to AtmosphereModel

In Breeze, P3 microphysics couples to `AtmosphereModel` through the microphysics interface:

```julia
# Prognostic field names
names = prognostic_field_names(microphysics)
# Returns (:ρqᶜˡ, :ρqʳ, :ρnʳ, :ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρzⁱ, :ρqʷⁱ)
```

The microphysics scheme provides:
1. **`microphysical_tendency`**: Computes source terms for all prognostic variables
2. **`compute_moisture_fractions`**: Converts prognostic densities to mixing ratios
3. **`update_microphysical_fields!`**: Updates diagnostic fields after state update

## Conservation Properties

P3 conserves:

**Total water**:
```math
\frac{d}{dt}\left( q_v + q^{cl} + q^r + q^i \right) = 0 \quad \text{(closed system)}
```

**Ice number** (modulo nucleation, aggregation, melting):
```math
\frac{dN^i}{dt} = NUC_n + SEC - AGG_n - MELT_n
```

**Energy** (through latent heat coupling):
```math
\frac{dθ}{dt} = \frac{1}{c_p Π}\left( L_v \dot{q}^{cl} + L_s \dot{q}^i + L_f \dot{q}^f \right)
```

## Numerical Considerations

### Positivity

All prognostic variables must remain non-negative. Limiters ensure:

```math
ρX^{n+1} = \max(0, ρX^n + Δt \cdot \text{tendency})
```

### Consistency

The rime fraction must satisfy ``0 ≤ F^f ≤ 1``:

```math
ρq^f ≤ ρq^i
```

Similarly for liquid fraction:

```math
ρq^{wi} ≤ ρq^i
```

### Threshold Handling

Small values below numerical thresholds are set to zero:

```julia
q_min = microphysics.minimum_mass_mixing_ratio  # Default: 1e-14 kg/kg
n_min = microphysics.minimum_number_mixing_ratio  # Default: 1e-16 1/kg
```

## Code Example

```@example p3_prognostics
using Breeze.Microphysics.PredictedParticleProperties

p3 = PredictedParticlePropertiesMicrophysics()

# Get all prognostic field names
names = prognostic_field_names(p3)
println("Prognostic fields:")
for name in names
    println("  ", name)
end
```

```@example p3_prognostics
# Access thresholds
println("\nNumerical thresholds:")
println("  Minimum mass mixing ratio: ", p3.minimum_mass_mixing_ratio, " kg/kg")
println("  Minimum number mixing ratio: ", p3.minimum_number_mixing_ratio, " 1/kg")
```

## References for This Section

- [Morrison2015parameterization](@cite): Original prognostic variables and tendencies (Section 2)
- [MilbrandtEtAl2021](@cite): Sixth moment prognostic (``ρz^i``) for three-moment ice
- [MilbrandtEtAl2025liquidfraction](@cite): Liquid fraction prognostic (``ρq^{wi}``)
- [Morrison2025complete3moment](@cite): Complete tendency equations with all six ice variables
- [MilbrandtYau2005](@cite): Multi-moment microphysics and sedimentation

