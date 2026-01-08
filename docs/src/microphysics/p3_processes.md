# [Microphysical Processes](@id p3_processes)

P3 includes a comprehensive set of microphysical processes governing the evolution
of cloud, rain, and ice hydrometeors. This section documents the physical formulations
and rate equations.

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

## Warm Rain Processes

### Condensation and Evaporation

Cloud liquid grows by condensation when supersaturated with respect to liquid water.
The saturation adjustment approach instantaneously relaxes to saturation:

```math
\frac{dq^{cl}}{dt} = \frac{q_v - q_{vs}(T)}{\tau_c}
```

where ``\tau_c`` is the condensation timescale (default 1 s) and ``q_{vs}`` is
saturation specific humidity.

For explicit microphysics, the condensation rate depends on droplet surface area:

```math
\frac{dm}{dt} = 4\pi r D_v (ρ_v - ρ_{vs})
```

### Autoconversion

Cloud droplets grow to rain through collision-coalescence. The [Khairoutdinov-Kogan](@cite KhairoutdinovKogan2000)
parameterization expresses autoconversion as:

```math
\frac{dq^r}{dt}\bigg|_{auto} = 1350 \, q_{cl}^{2.47} N_c^{-1.79}
```

where ``q_{cl}`` is cloud liquid mixing ratio and ``N_c`` is cloud droplet number
concentration.

The threshold diameter for autoconversion (default 25 μm) sets the boundary between
cloud and rain.

### Accretion

Rain collects cloud droplets:

```math
\frac{dq^r}{dt}\bigg|_{accr} = E_{rc} \frac{\pi}{4} q^{cl} \int_0^∞ D^2 V(D) N'_r(D)\, dD
```

where ``E_{rc}`` is the rain-cloud collection efficiency and ``N'_r`` is the rain
drop size distribution.

### Rain Evaporation

Below cloud base, rain evaporates in subsaturated air:

```math
\frac{dm}{dt} = 4\pi C D_v f_v (ρ_v - ρ_{vs})
```

where:
- ``C = D/2`` is the droplet capacity (spherical)
- ``D_v`` is water vapor diffusivity
- ``f_v`` is the ventilation factor
- ``ρ_v - ρ_{vs}`` is the vapor deficit

Integrated over the drop size distribution:

```math
\frac{dq^r}{dt}\bigg|_{evap} = 2\pi D_v (S - 1) \int_0^∞ D f_v N'_r(D)\, dD
```

where ``S = ρ_v/ρ_{vs}`` is the saturation ratio.

## Ice Nucleation

### Heterogeneous Nucleation

Ice nucleating particles (INPs) activate at temperatures below about -5°C:

```math
\frac{dN^i}{dt}\bigg|_{het} = n_{INP}(T) \frac{d T}{dt}\bigg|_{neg}
```

where ``n_{INP}(T)`` follows parameterizations like [DeMott et al. (2010)](@cite)
or [Meyers et al. (1992)](@cite).

### Homogeneous Freezing

Cloud droplets freeze homogeneously at ``T < -38°C``:

```math
\frac{dq^i}{dt}\bigg|_{hom} = q^{cl} \quad \text{when } T < 235\,\text{K}
```

### Secondary Ice Production

#### Hallett-Mossop Process

Rime splintering produces secondary ice in the temperature range -3 to -8°C:

```math
\frac{dN^i}{dt}\bigg|_{HM} = C_{HM} \frac{dq^f}{dt}
```

where ``C_{HM} \approx 350`` splinters per mg of rime.

## Vapor-Ice Exchange

### Deposition Growth

Ice particles grow by vapor deposition when ``S_i > 1`` (supersaturated wrt ice):

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
compute this integral efficiently.

### Sublimation

The same formulation applies for ``S_i < 1``, with mass loss rather than gain.

## Collection Processes

### Riming (Ice-Cloud Collection)

Ice particles collect cloud droplets:

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

From [Heymsfield and Pflaum (1985)](@cite):

```math
ρ^f = \min\left(917, \max\left(50, a_ρ + b_ρ \ln\left(\frac{V}{D}\right) + c_ρ T_c\right)\right)
```

where ``T_c`` is temperature in Celsius.

### Ice-Rain Collection

Ice particles can also collect raindrops:

```math
\frac{dq^f}{dt}\bigg|_{ir} = E_{ir} \int_0^∞ \int_0^∞ K(D_i, D_r) N'_i(D_i) N'_r(D_r)\, dD_i dD_r
```

where the collection kernel is:

```math
K(D_i, D_r) = \frac{\pi}{4}(D_i + D_r)^2 |V_i - V_r|
```

### Aggregation (Ice-Ice Collection)

Ice particles aggregate when they collide:

```math
\frac{dN^i}{dt}\bigg|_{agg} = -\frac{1}{2} E_{agg} \int_0^∞ \int_0^∞ K(D_1, D_2) N'(D_1) N'(D_2)\, dD_1 dD_2
```

The factor of 1/2 avoids double-counting. Mass is conserved; only number decreases.

The aggregation efficiency ``E_{agg}`` depends on temperature, with maximum efficiency
near 0°C where ice surfaces are "sticky".

## Phase Change

### Melting

At ``T > 273.15`` K, ice particles melt:

```math
\frac{dm}{dt} = -\frac{4\pi C}{L_f} \left[ K_a (T - T_0) + L_v D_v (ρ_v - ρ_{vs}) \right] f_v
```

where:
- ``L_f`` is latent heat of fusion
- ``T_0 = 273.15`` K is the melting point
- The first term is sensible heat transfer
- The second term is latent heat from vapor deposition

Meltwater initially coats the ice particle (increasing ``q^{wi}``), then sheds to rain.

### Shedding

When liquid fraction exceeds a threshold (typically 50%), excess liquid sheds as rain:

```math
\frac{dq^{wi}}{dt}\bigg|_{shed} = -k_{shed} (F^l - F^l_{max}) q^i \quad \text{when } F^l > F^l_{max}
```

The shed mass converts to rain:

```math
\frac{dq^r}{dt}\bigg|_{shed} = -\frac{dq^{wi}}{dt}\bigg|_{shed}
```

### Refreezing

Liquid on ice can refreeze, converting to rime:

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

Different moments sediment at different rates:

| Quantity | Sedimentation Velocity |
|----------|----------------------|
| Number ``N`` | ``V_n`` (number-weighted) |
| Mass ``L`` | ``V_m`` (mass-weighted) |
| Reflectivity ``Z`` | ``V_z`` (Z-weighted) |

This differential sedimentation causes the size distribution to evolve as particles fall.

## Process Summary

| Process | Affects | Key Parameter |
|---------|---------|---------------|
| Condensation | ``q^{cl}`` | Saturation timescale |
| Autoconversion | ``q^{cl} \to q^r`` | K-K coefficients |
| Accretion | ``q^{cl} \to q^r`` | Collection efficiency |
| Rain evaporation | ``q^r \to q_v`` | Ventilation |
| Heterogeneous nucleation | ``N^i`` | INP concentration |
| Homogeneous freezing | ``q^{cl} \to q^i`` | T threshold |
| Deposition | ``q^i`` | Ventilation, ``S_i`` |
| Sublimation | ``q^i \to q_v`` | Ventilation, ``S_i`` |
| Riming | ``q^{cl} \to q^f`` | ``E_{ic}`` |
| Ice-rain collection | ``q^r \to q^f`` | ``E_{ir}`` |
| Aggregation | ``N^i`` | ``E_{agg}(T)`` |
| Melting | ``q^i \to q^{wi} \to q^r`` | ``T > 273`` K |
| Shedding | ``q^{wi} \to q^r`` | ``F^l_{max}`` |
| Refreezing | ``q^{wi} \to q^f`` | ``T < 273`` K |
| Sedimentation | All | ``V_n, V_m, V_z`` |

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

