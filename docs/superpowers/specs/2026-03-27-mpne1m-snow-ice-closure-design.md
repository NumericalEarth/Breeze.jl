# MPNE1M Snow and Ice Process Closure

**Date**: 2026-03-27
**Scope**: MPNE1M (mixed-phase non-equilibrium 1-moment) scheme only

## Problem

The MPNE1M microphysics scheme lacks closure for snow and is missing sink processes for
cloud ice. Snow has a prognostic field (`ρqˢ`) with zero tendency terms — no sources, no
sinks, no sedimentation. Cloud ice only has deposition/sublimation — no autoconversion to
snow, no accretion losses. This breaks mass conservation and makes mixed-phase simulations
unrealistic.

## Design

Add full snow and ice interaction processes to match CloudMicrophysics.jl's
`BulkMicrophysicsTendencies.jl` reference implementation. The key constraint is that Breeze
uses water vapor (not total water) as the prognostic moisture variable, so wrapper functions
must use Breeze's internal thermodynamics instead of Thermodynamics.jl.

### Files Changed

1. `ext/BreezeCloudMicrophysicsExt/BreezeCloudMicrophysicsExt.jl` — new imports
2. `ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl` — 3 new wrappers
3. `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl` — extended tendencies, snow
   sedimentation, new dispatch methods

### New Imports from CloudMicrophysics.Microphysics1M

- `conv_q_icl_to_q_sno_no_supersat` — ice-to-snow autoconversion (threshold-based)
- `accretion_rain_sink` — rain sink from ice-rain collisions
- `accretion_snow_rain` — rain-snow collision rate

(`accretion` and `terminal_velocity` are already imported and dispatch on type parameters.)

### Breeze-Native Wrapper Functions

These 3 functions depend on Thermodynamics.jl in CloudMicrophysics and must be reimplemented
using Breeze's thermodynamics. They follow the established `rain_evaporation` wrapper pattern.

#### `snow_sublimation_deposition`

Mirrors `rain_evaporation` but over ice surface. Computes the ventilated Mason equation for
snow with ice-surface supersaturation.

```
snow_sublimation_deposition(snow::Snow, vel::Blk1MVelTypeSnow, aps::AirProperties,
                            q::MoistureMassFractions, qˢ, ρ, T, constants)
```

- Supersaturation: `𝒮 = supersaturation(T, ρ, q, constants, PlanarIceSurface())`
- Diffusional growth factor: `G = diffusional_growth_factor_ice(aps, T, constants)` (exists)
- Size distribution: `n₀ = get_n0(pdf, qˢ, ρ)`, `λ⁻¹ = lambda_inverse(pdf, mass, qˢ, ρ)`
- Mean velocity: `v₀ = get_v0(vel)`  (note: `get_v0` for snow takes no ρ argument)
- Rate: `4π n₀/ρ * 𝒮 * G * λ⁻¹² * ventilation`
- Guard: only active when `qˢ > ϵ_numerics`
- Returns signed rate: positive = deposition (vapor→snow), negative = sublimation (snow→vapor)
- Unlike rain evaporation, both signs are physical — snow can grow by deposition

#### `snow_melting`

Sensible-heat-driven melting of snow to rain above freezing.

```
snow_melting(snow::Snow, vel::Blk1MVelTypeSnow, aps::AirProperties,
             qˢ, ρ, T, constants)
```

- Latent heat of fusion: `ℒf = ice_latent_heat(T, constants) - liquid_latent_heat(T, constants)`
- Rate: `4π n₀/ρ * K_therm/ℒf * (T - T_freeze) * λ⁻¹² * ventilation`
- Guard: only active when `qˢ > ϵ_numerics` AND `T > T_freeze`
- Always non-negative (melting only)

#### `warm_accretion_melt_factor`

Branchless thermal melt ratio for warm accretion (liquid+snow riming and rain+snow
collisions in warm conditions).

```
warm_accretion_melt_factor(snow::Snow, T, constants)
```

- `α = ifelse(T <= T_freeze, 0, cˡ / ℒf * (T - T_freeze))`
- `cˡ = constants.liquid.heat_capacity`
- `ℒf = ice_latent_heat(T, constants) - liquid_latent_heat(T, constants)`

### Extended `mpne1m_tendencies`

#### New Process Rates

| Symbol | Function | Physics |
|--------|----------|---------|
| `Sᵃᶜⁿᵛⁱˢ` | `conv_q_icl_to_q_sno_no_supersat(snow.acnv1M, qᶜⁱ, true)` | Ice → snow autoconversion |
| `Sᵃᶜᶜˡˢ` | `accretion(cloud_liquid, snow, vel.snow, ce, qᶜˡ, qˢ, ρ)` | Liquid + snow |
| `Sᵃᶜᶜⁱˢ` | `accretion(cloud_ice, snow, vel.snow, ce, qᶜⁱ, qˢ, ρ)` | Ice + snow |
| `Sᵃᶜᶜⁱʳ` | `accretion(cloud_ice, rain, vel.rain, ce, qᶜⁱ, qʳ, ρ)` | Ice + rain → snow |
| `Sᵃᶜᶜʳⁱ` | `accretion_rain_sink(rain, cloud_ice, vel.rain, ce, qᶜⁱ, qʳ, ρ)` | Rain sink from ice-rain |
| `Sʳˢ` | `accretion_snow_rain(snow, rain, vel.snow, vel.rain, ce, qˢ, qʳ, ρ)` | Rain→snow (cold) |
| `Sˢʳ` | `accretion_snow_rain(rain, snow, vel.rain, vel.snow, ce, qʳ, qˢ, ρ)` | Snow→rain (warm) |
| `Sˢᵘᵇˡ` | `snow_sublimation_deposition(...)` | Snow ↔ vapor |
| `Sᵐᵉˡᵗ` | `snow_melting(...)` | Snow → rain |
| `α` | `warm_accretion_melt_factor(...)` | Thermal melt ratio |

Rate limiters (paralleling existing patterns):
- `Sˢᵘᵇˡ`: sublimation bounded by `max(Sˢᵘᵇˡ, -max(0, qˢ) / τⁿᵘᵐ)`
- `Sᵐᵉˡᵗ`: bounded by `min(Sᵐᵉˡᵗ, max(0, qˢ) / τⁿᵘᵐ)`

#### Temperature Routing

All routing uses branchless `ifelse` with `T_freeze` from the `Snow` parameter struct, following
`BulkMicrophysicsTendencies.jl` exactly.

**Liquid + snow accretion:**
- Cold (`T < T_freeze`): riming — liquid loses `Sᵃᶜᶜˡˢ`, snow gains `Sᵃᶜᶜˡˢ`
- Warm (`T >= T_freeze`): shedding — liquid loses `Sᵃᶜᶜˡˢ`, rain gains `Sᵃᶜᶜˡˢ`
  - Plus thermal melt: snow loses `α * Sᵃᶜᶜˡˢ`, rain gains `α * Sᵃᶜᶜˡˢ`

**Rain + snow collisions:**
- Cold: rain freezes to snow — rain loses `Sʳˢ`, snow gains `Sʳˢ`
- Warm: snow melts to rain — snow loses `Sˢʳ`, rain gains `Sˢʳ`
  - Plus thermal melt: snow loses `α * Sʳˢ`, rain gains `α * Sʳˢ`

#### Updated Tendency Equations

```
ρqᵛ_phys  = ρ * (−Sᶜᵒⁿᵈ − Sᵈᵉᵖ − Sᵉᵛᵃᵖ − Sˢᵘᵇˡ)
ρqᶜˡ_phys = ρ * ( Sᶜᵒⁿᵈ − Sᵃᶜⁿᵛ − Sᵃᶜᶜ − Sᵃᶜᶜˡˢ)
ρqᶜⁱ_phys = ρ * ( Sᵈᵉᵖ − Sᵃᶜⁿᵛⁱˢ − Sᵃᶜᶜⁱˢ − Sᵃᶜᶜⁱʳ)
ρqʳ_phys  = ρ * ( Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ − Sᵃᶜᶜʳⁱ + Sᵐᵉˡᵗ
                  + ifelse(warm, Sᵃᶜᶜˡˢ + α*Sᵃᶜᶜˡˢ + Sˢʳ + α*Sʳˢ, 0)
                  − ifelse(cold, Sʳˢ, 0))
ρqˢ_phys  = ρ * ( Sᵃᶜⁿᵛⁱˢ + Sᵃᶜᶜⁱˢ + Sᵃᶜᶜⁱʳ + Sᵃᶜᶜʳⁱ + Sˢᵘᵇˡ − Sᵐᵉˡᵗ
                  + ifelse(cold, Sᵃᶜᶜˡˢ + Sʳˢ, 0)
                  − ifelse(warm, α*Sᵃᶜᶜˡˢ + Sˢʳ + α*Sʳˢ, 0))
```

**Conservation**: Sum of all five ρq_phys = 0 by construction. Every source rate appears
exactly once as a positive term in one species and once as a negative term in another.

#### Negative Moisture Correction

The existing 4-species correction is unchanged. No correction is added for snow. The
physical rate limiters on sublimation and melting provide sufficient guards.

The existing correction routing remains:
- `δᵛ` → cloud liquid (condensation partner)
- `δᶜˡ` → rain (collection partner)
- `δᶜⁱ` → vapor (deposition partner)
- `δʳ` → vapor (evaporation partner)

Snow tendencies participate in the physics but snow's negative moisture correction is
handled by existing rate limiters, not the correction mechanism.

### Snow Sedimentation

#### Field Materialization

Add `wˢ` as a `ZFaceField` alongside existing `wʳ`, with the same boundary condition setup
(`bottom=nothing` to preserve kernel-set values).

#### Terminal Velocity Computation

In `update_microphysical_auxiliaries!` for `MixedPhase1M`, add:
```
𝕎ˢ = terminal_velocity(categories.snow, categories.hydrometeor_velocities.snow, ρ, ℳ.qˢ)
wˢ = -𝕎ˢ  # negative = downward
wˢ₀ = bottom_terminal_velocity(bμp.precipitation_boundary_condition, wˢ)
μ.wˢ[i, j, k] = ifelse(k == 1, wˢ₀, wˢ)
```

#### Sedimentation Transport

Add velocity dispatch:
```
AM.microphysical_velocities(bμp::OMCM, μ, ::Val{:ρqˢ}) = (u=zf, v=zf, w=μ.wˢ)
```

This hooks snow into the same advection-based sedimentation framework as rain.

### Tendency Dispatch

Add `microphysical_tendency(bμp::MPNE1M, ::Val{:ρqˢ}, ...)` that calls
`mpne1m_tendencies` and returns `G.ρqˢ`. Update `mpne1m_tendencies` return to include `ρqˢ`.

Remove the TODO comment on line 758.

### Water Vapor Pitfall

Breeze uses `qᵛ` (water vapor) as the prognostic moisture variable, not `qᵗ` (total water).
The new snow processes affect vapor through:
- `Sˢᵘᵇˡ` — snow sublimation/deposition (direct vapor ↔ snow exchange)

All other new processes (autoconversion, accretion, melting, rain-snow collisions) exchange
mass between condensate species only — they do not touch vapor. The vapor tendency equation
correctly includes `-Sˢᵘᵇˡ` as the only new vapor term.

The `snow_sublimation_deposition` wrapper computes supersaturation over ice using Breeze's
`supersaturation(T, ρ, q, constants, PlanarIceSurface())`, which takes the
`MoistureMassFractions` struct that already contains the correct vapor fraction.
