# [Microphysical Processes](@id p3_processes)

P3 includes a comprehensive set of microphysical processes governing the evolution
of cloud, rain, and ice hydrometeors. This section documents the physical formulations
and rate equations from the P3 papers.

!!! note "Implementation Status"
    The process rate formulations documented here are from the P3 papers. Our implementation
    provides the integral infrastructure for computing bulk rates (see [Integral Properties](@ref p3_integral_properties)).
    Full tendency functions for all processes are a TODO for future work.

## Process Overview

```
                        ┌─────────────┐
                        │   Vapor     │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌──────────┐     ┌──────────┐     ┌──────────┐
       │  Cloud   │     │   Ice    │     │   Rain   │
       └────┬─────┘     └────┬─────┘     └────┬─────┘
            │                │                │
            │    accretion   │    riming      │
            └───────────────►├◄───────────────┘
                             │
                        melting
                             │
                             ▼
                       ┌──────────┐
                       │   Rain   │
                       └──────────┘
```

The following subsections document processes from:
- [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization): Core process formulations
- [Milbrandt et al. (2021)](@cite MilbrandtEtAl2021): Z-tendencies for each process
- [Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction): Liquid fraction processes

## Warm Rain Processes

### Condensation and Evaporation

Cloud liquid grows by condensation when supersaturated with respect to liquid water.
The saturation adjustment approach instantaneously relaxes to saturation
([Rogers & Yau (1989)](@cite rogers1989short)):

```math
\frac{dq^{cl}}{dt} = \frac{q_v - q_{vs}(T)}{\tau_c}
```

where ``\tau_c`` is the condensation timescale (default 1 s) and ``q_{vs}`` is
saturation specific humidity.

!!! note "Explicit Supersaturation"
    The E3SM implementation of P3 includes modifications for explicit supersaturation
    evolution rather than saturation adjustment. Our implementation currently uses
    saturation adjustment for cloud liquid.

### Autoconversion

Cloud droplets grow to rain through collision-coalescence. The 
[Khairoutdinov & Kogan (2000)](@cite KhairoutdinovKogan2000)
parameterization expresses autoconversion as:

```math
\frac{dq^r}{dt}\bigg|_{auto} = 1350 \, q_{cl}^{2.47} N_c^{-1.79}
```

where ``q_{cl}`` is cloud liquid mixing ratio and ``N_c`` is cloud droplet number
concentration.

The threshold diameter for autoconversion (default 25 μm) sets the boundary between
cloud and rain.

### Accretion

Rain collects cloud droplets ([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 46):

```math
\frac{dq^r}{dt}\bigg|_{accr} = E_{rc} \frac{\pi}{4} q^{cl} \int_0^∞ D^2 V(D) N'_r(D)\, dD
```

where ``E_{rc}`` is the rain-cloud collection efficiency and ``N'_r`` is the rain
drop size distribution.

### Rain Evaporation

Below cloud base, rain evaporates in subsaturated air
([Pruppacher & Klett (2010)](@cite pruppacher2010microphysics)):

```math
\frac{dm}{dt} = 4\pi C D_v f_v (ρ_v - ρ_{vs})
```

where:
- ``C = D/2`` is the droplet capacity (spherical)
- ``D_v`` is water vapor diffusivity
- ``f_v`` is the ventilation factor
- ``ρ_v - ρ_{vs}`` is the vapor deficit

Integrated over the drop size distribution
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 47):

```math
\frac{dq^r}{dt}\bigg|_{evap} = 2\pi D_v (S - 1) \int_0^∞ D f_v N'_r(D)\, dD
```

where ``S = ρ_v/ρ_{vs}`` is the saturation ratio.

## Ice Nucleation

### Heterogeneous Nucleation

Ice nucleating particles (INPs) activate at temperatures below about -5°C.
From [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2f:

```math
\frac{dN^i}{dt}\bigg|_{het} = n_{INP}(T) \frac{d T}{dt}\bigg|_{neg}
```

where ``n_{INP}(T)`` follows parameterizations like [DeMott et al. (2010)](@cite)
or [Meyers et al. (1992)](@cite).

!!! note "INP Parameterization"
    The specific INP parameterization is configurable in P3. Our implementation
    will support multiple options when nucleation rates are fully implemented.

### Homogeneous Freezing

Cloud droplets freeze homogeneously at ``T < -38°C``
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization)):

```math
\frac{dq^i}{dt}\bigg|_{hom} = q^{cl} \quad \text{when } T < 235\,\text{K}
```

### Secondary Ice Production

#### Hallett-Mossop Process

Rime splintering produces secondary ice in the temperature range -3 to -8°C
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Section 2g):

```math
\frac{dN^i}{dt}\bigg|_{HM} = C_{HM} \frac{dq^f}{dt}
```

where ``C_{HM} \approx 350`` splinters per mg of rime.

## Vapor-Ice Exchange

### Deposition Growth

Ice particles grow by vapor deposition when ``S_i > 1`` (supersaturated wrt ice).
From [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 30:

```math
\frac{dm}{dt} = 4\pi C f_v \frac{S_i - 1}{\frac{L_s}{K_a T}\left(\frac{L_s}{R_v T} - 1\right) + \frac{R_v T}{e_{si} D_v}}
```

where:
- ``C`` is the particle capacity (shape-dependent)
- ``f_v`` is the ventilation factor
- ``L_s`` is latent heat of sublimation
- ``K_a`` is thermal conductivity of air
- ``e_{si}`` is saturation vapor pressure over ice

Integrated over the size distribution:

```math
\frac{dq^i}{dt}\bigg|_{dep} = 4\pi n_i D_v (S_i - 1) f(T, p) \int_0^∞ C(D) f_v(D) N'(D)\, dD
```

The ventilation integrals (see [Integral Properties](@ref p3_integral_properties))
compute this integral efficiently. The ventilation enhancement factor is documented
in [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Table 3.

### Sublimation

The same formulation applies for ``S_i < 1``, with mass loss rather than gain.

### Z-Tendency from Deposition

For three-moment ice ([Milbrandt et al. (2021)](@cite MilbrandtEtAl2021),
[Morrison et al. (2025)](@cite Morrison2025complete3moment)), the sixth moment
tendency from deposition/sublimation is:

```math
\frac{dZ}{dt}\bigg|_{dep} = 6 \frac{Z}{L} \frac{dL}{dt}\bigg|_{dep} \cdot \mathcal{F}_{dep}
```

where ``\mathcal{F}_{dep}`` is a correction factor from the lookup tables.

## Collection Processes

### Riming (Ice-Cloud Collection)

Ice particles collect cloud droplets
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 36):

```math
\frac{dq^f}{dt} = E_{ic} q^{cl} \int_0^∞ A(D) V(D) N'(D)\, dD
```

where ``E_{ic}`` is the ice-cloud collection efficiency (default 0.1).

Simultaneously, the rime volume increases:

```math
\frac{db^f}{dt} = \frac{1}{ρ^f} \frac{dq^f}{dt}
```

where ``ρ^f`` is the rime density, which depends on impact velocity and temperature.

#### Rime Density Parameterization

From [Heymsfield & Pflaum (1985)](@cite) as used in
[Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization):

```math
ρ^f = \min\left(917, \max\left(50, a_ρ + b_ρ \ln\left(\frac{V}{D}\right) + c_ρ T_c\right)\right)
```

where ``T_c`` is temperature in Celsius.

### Ice-Rain Collection

Ice particles can also collect raindrops
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 40):

```math
\frac{dq^f}{dt}\bigg|_{ir} = E_{ir} \int_0^∞ \int_0^∞ K(D_i, D_r) N'_i(D_i) N'_r(D_r)\, dD_i dD_r
```

where the collection kernel is:

```math
K(D_i, D_r) = \frac{\pi}{4}(D_i + D_r)^2 |V_i - V_r|
```

### Aggregation (Ice-Ice Collection)

Ice particles aggregate when they collide
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 42):

```math
\frac{dN^i}{dt}\bigg|_{agg} = -\frac{1}{2} E_{agg} \int_0^∞ \int_0^∞ K(D_1, D_2) N'(D_1) N'(D_2)\, dD_1 dD_2
```

The factor of 1/2 avoids double-counting. Mass is conserved; only number decreases.

The aggregation efficiency ``E_{agg}`` depends on temperature, with maximum efficiency
near 0°C where ice surfaces are "sticky".

## Phase Change Processes

### Melting

At ``T > 273.15`` K, ice particles melt
([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 44):

```math
\frac{dm}{dt} = -\frac{4\pi C}{L_f} \left[ K_a (T - T_0) + L_v D_v (ρ_v - ρ_{vs}) \right] f_v
```

where:
- ``L_f`` is latent heat of fusion
- ``T_0 = 273.15`` K is the melting point
- The first term is sensible heat transfer
- The second term is latent heat from vapor deposition

### Liquid Fraction During Melting

With predicted liquid fraction ([Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction)),
meltwater initially coats the ice particle (increasing ``q^{wi}``):

```math
\frac{dq^{wi}}{dt}\bigg|_{melt} = -\frac{dm_{ice}}{dt}
```

This allows tracking of wet ice particles before complete melting.

### Shedding

When liquid fraction exceeds a threshold (typically 50%), excess liquid sheds as rain
([Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction)):

```math
\frac{dq^{wi}}{dt}\bigg|_{shed} = -k_{shed} (F^l - F^l_{max}) q^i \quad \text{when } F^l > F^l_{max}
```

The shed mass converts to rain:

```math
\frac{dq^r}{dt}\bigg|_{shed} = -\frac{dq^{wi}}{dt}\bigg|_{shed}
```

### Refreezing

Liquid on ice can refreeze, converting to rime
([Milbrandt et al. (2025)](@cite MilbrandtEtAl2025liquidfraction)):

```math
\frac{dq^{wi}}{dt}\bigg|_{refreeze} = -q^{wi} / \tau_{freeze} \quad \text{when } T < 273\,\text{K}
```

```math
\frac{dq^f}{dt}\bigg|_{refreeze} = -\frac{dq^{wi}}{dt}\bigg|_{refreeze}
```

## Sedimentation

Hydrometeors fall under gravity. The flux divergence appears in the tendency:

```math
\frac{\partial ρq}{\partial t}\bigg|_{sed} = -\frac{\partial (ρq V)}{\partial z}
```

Different moments sediment at different rates
([Milbrandt & Yau (2005)](@cite MilbrandtYau2005)):

| Quantity | Sedimentation Velocity |
|----------|----------------------|
| Number ``N`` | ``V_n`` (number-weighted) |
| Mass ``L`` | ``V_m`` (mass-weighted) |
| Reflectivity ``Z`` | ``V_z`` (Z-weighted) |

This differential sedimentation causes the size distribution to evolve as particles fall.
The three velocities are computed using the fall speed integrals
(see [Integral Properties](@ref p3_integral_properties)).

For three-moment ice ([Milbrandt et al. (2021)](@cite MilbrandtEtAl2021)),
tracking ``V_z`` allows proper size sorting of precipitation particles.

## Process Summary

| Process | Affects | Key Parameter | Reference |
|---------|---------|---------------|-----------|
| Condensation | ``q^{cl}`` | Saturation timescale | [Rogers & Yau (1989)](@cite rogers1989short) |
| Autoconversion | ``q^{cl} \to q^r`` | K-K coefficients | [KhairoutdinovKogan2000](@cite) |
| Accretion | ``q^{cl} \to q^r`` | Collection efficiency | [Morrison2015parameterization](@cite) |
| Rain evaporation | ``q^r \to q_v`` | Ventilation | [Morrison2015parameterization](@cite) |
| Heterogeneous nucleation | ``N^i`` | INP concentration | [Morrison2015parameterization](@cite) |
| Homogeneous freezing | ``q^{cl} \to q^i`` | T threshold | [Morrison2015parameterization](@cite) |
| Deposition | ``q^i`` | Ventilation, ``S_i`` | [Morrison2015parameterization](@cite) |
| Sublimation | ``q^i \to q_v`` | Ventilation, ``S_i`` | [Morrison2015parameterization](@cite) |
| Riming | ``q^{cl} \to q^f`` | ``E_{ic}`` | [Morrison2015parameterization](@cite) |
| Ice-rain collection | ``q^r \to q^f`` | ``E_{ir}`` | [Morrison2015parameterization](@cite) |
| Aggregation | ``N^i`` | ``E_{agg}(T)`` | [Morrison2015parameterization](@cite) |
| Melting | ``q^i \to q^{wi} \to q^r`` | ``T > 273`` K | [Morrison2015parameterization](@cite) |
| Shedding | ``q^{wi} \to q^r`` | ``F^l_{max}`` | [MilbrandtEtAl2025liquidfraction](@cite) |
| Refreezing | ``q^{wi} \to q^f`` | ``T < 273`` K | [MilbrandtEtAl2025liquidfraction](@cite) |
| Sedimentation | All | ``V_n, V_m, V_z`` | [MilbrandtYau2005](@cite) |

## Temperature Dependence

Many processes have strong temperature dependence:

```
T < 235 K:  Homogeneous freezing (cloud → ice)
235-268 K:  Heterogeneous nucleation, deposition growth
265-273 K:  Hallett-Mossop ice multiplication  
268-273 K:  Maximum aggregation efficiency
T > 273 K:  Melting, shedding
```

## Coupling to Thermodynamics

Microphysical processes release or absorb latent heat:

```math
\frac{dT}{dt}\bigg|_{micro} = \frac{L_v}{c_p} \frac{dq^{cl}}{dt} + \frac{L_s}{c_p} \frac{dq^i}{dt} + \frac{L_f}{c_p} \frac{dq^f}{dt}
```

where:
- ``L_v \approx 2.5 \times 10^6`` J/kg (vaporization)
- ``L_s \approx 2.83 \times 10^6`` J/kg (sublimation)
- ``L_f \approx 3.34 \times 10^5`` J/kg (fusion)

## References for This Section

### Core P3 Process References
- [Morrison2015parameterization](@cite): Primary process formulations (Section 2)
- [Morrison2015part2](@cite): Process validation against observations
- [MilbrandtEtAl2021](@cite): Z-tendencies for three-moment ice
- [MilbrandtEtAl2025liquidfraction](@cite): Liquid fraction processes (shedding, refreezing)
- [Morrison2025complete3moment](@cite): Complete three-moment process rates

### Related References
- [KhairoutdinovKogan2000](@cite): Warm rain autoconversion
- [MilbrandtYau2005](@cite): Multimoment sedimentation
- [pruppacher2010microphysics](@cite): Cloud physics fundamentals
- [rogers1989short](@cite): Cloud physics textbook

