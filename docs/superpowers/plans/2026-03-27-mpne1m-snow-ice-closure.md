# MPNE1M Snow and Ice Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete snow microphysics and ice sink processes to the MPNE1M scheme, achieving mass closure for all five moisture species.

**Architecture:** Extend three files in the CloudMicrophysics extension: add new CloudMicrophysics.jl function imports, write Breeze-native wrapper functions for snow sublimation/deposition, melting, and thermal melt factor (mirroring the established `rain_evaporation` pattern), then extend `mpne1m_tendencies` with 10 new process rates and temperature-routed tendency equations. Snow sedimentation uses a new `wˢ` face field hooked into the existing advection framework.

**Tech Stack:** Julia, Breeze.jl, CloudMicrophysics.jl (Microphysics1M), Oceananigans.jl

**Spec:** `docs/superpowers/specs/2026-03-27-mpne1m-snow-ice-closure-design.md`

---

### Task 1: Add New Imports

**Files:**
- Modify: `ext/BreezeCloudMicrophysicsExt/BreezeCloudMicrophysicsExt.jl:17-20`

- [ ] **Step 1: Add the three new Microphysics1M imports**

In `ext/BreezeCloudMicrophysicsExt/BreezeCloudMicrophysicsExt.jl`, replace the existing import block:

```julia
using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    accretion,
    terminal_velocity
```

with:

```julia
using CloudMicrophysics.Microphysics1M:
    conv_q_lcl_to_q_rai,
    conv_q_icl_to_q_sno_no_supersat,
    accretion,
    accretion_rain_sink,
    accretion_snow_rain,
    terminal_velocity
```

- [ ] **Step 2: Commit**

```bash
git add ext/BreezeCloudMicrophysicsExt/BreezeCloudMicrophysicsExt.jl
git commit -m "Import snow/ice microphysics functions from CloudMicrophysics"
```

---

### Task 2: Add `snow_sublimation_deposition` Wrapper

**Files:**
- Modify: `ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl` (insert after `rain_evaporation` at line 139)

- [ ] **Step 1: Add the wrapper function**

Insert the following after the `rain_evaporation` function (after line 139) in
`ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl`:

```julia
#####
##### Snow sublimation/deposition (TRANSLATION: uses Breeze thermodynamics over ice surface)
#####

"""
    snow_sublimation_deposition(snow_params, vel, aps, q, qˢ, ρ, T, constants)

Compute the snow sublimation/deposition rate (dqˢ/dt).

Positive values mean deposition (vapor → snow), negative means sublimation (snow → vapor).
Unlike rain evaporation, both signs are physical for snow.

This is a translation of `CloudMicrophysics.Microphysics1M.evaporation_sublimation`
for snow that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `snow_params`: Snow microphysics parameters (pdf, mass, vent)
- `vel`: Snow terminal velocity parameters
- `aps`: Air properties (kinematic viscosity, vapor diffusivity, thermal conductivity)
- `q`: `MoistureMassFractions` containing vapor, liquid, and ice mass fractions
- `qˢ`: Snow specific humidity
- `ρ`: Air density
- `T`: Temperature
- `constants`: Breeze ThermodynamicConstants

# Returns
Rate of change of snow specific humidity (positive = deposition, negative = sublimation)
"""
@inline function snow_sublimation_deposition(
    (; pdf, mass, vent)::Snow{FT},
    vel::Blk1MVelTypeSnow{FT},
    aps::AirProperties{FT},
    q::MoistureMassFractions{FT},
    qˢ::FT,
    ρ::FT,
    T::FT,
    constants,
) where {FT}
    (; ν_air, D_vapor) = aps
    (; χv, ve, Δv) = vel
    (; r0) = mass
    aᵥ = vent.a
    bᵥ = vent.b

    # Supersaturation over ice (𝒮 > 0 → deposition, 𝒮 < 0 → sublimation)
    𝒮 = supersaturation(T, ρ, q, constants, PlanarIceSurface())

    G = diffusional_growth_factor_ice(aps, T, constants)
    n₀ = get_n0(pdf, qˢ, ρ)
    v₀ = get_v0(vel, ρ)
    λ⁻¹ = lambda_inverse(pdf, mass, qˢ, ρ)

    # Ventilated sublimation/deposition rate from Mason equation
    base_rate = 4π * n₀ / ρ * 𝒮 * G * λ⁻¹^2

    # Ventilation correction terms
    Sc = ν_air / D_vapor
    Re = 2v₀ * χv / ν_air * λ⁻¹
    size_factor = (r0 / λ⁻¹)^((ve + Δv) / 2)
    gamma_factor = Γ((ve + Δv + 5) / 2)

    ventilation = aᵥ + bᵥ * cbrt(Sc) * sqrt(Re) / size_factor * gamma_factor

    rate = base_rate * ventilation

    # Both sublimation (𝒮 < 0) and deposition (𝒮 > 0) are physical for snow
    has_snow = qˢ > ϵ_numerics(FT)
    return ifelse(has_snow, rate, zero(FT))
end
```

- [ ] **Step 2: Commit**

```bash
git add ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl
git commit -m "Add snow sublimation/deposition Breeze wrapper"
```

---

### Task 3: Add `snow_melting` Wrapper

**Files:**
- Modify: `ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl` (insert after `snow_sublimation_deposition`)

- [ ] **Step 1: Add the wrapper function**

Insert after `snow_sublimation_deposition`:

```julia
#####
##### Snow melting (TRANSLATION: uses Breeze thermodynamics for latent heat of fusion)
#####

"""
    snow_melting(snow_params, vel, aps, qˢ, ρ, T, constants)

Compute the snow melting rate (dqˢ/dt due to melting, always non-negative).

Sensible-heat-driven melting: heat from warm air (T > T_freeze) melts snow to rain.
The rate is proportional to (T - T_freeze) and includes ventilation corrections.

This is a translation of `CloudMicrophysics.Microphysics1M.snow_melt`
that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `snow_params`: Snow microphysics parameters (T_freeze, pdf, mass, vent)
- `vel`: Snow terminal velocity parameters
- `aps`: Air properties (kinematic viscosity, vapor diffusivity, thermal conductivity)
- `qˢ`: Snow specific humidity
- `ρ`: Air density
- `T`: Temperature
- `constants`: Breeze ThermodynamicConstants

# Returns
Rate of snow mass lost to melting [kg/kg/s] (always non-negative)
"""
@inline function snow_melting(
    (; T_freeze, pdf, mass, vent)::Snow{FT},
    vel::Blk1MVelTypeSnow{FT},
    aps::AirProperties{FT},
    qˢ::FT,
    ρ::FT,
    T::FT,
    constants,
) where {FT}
    (; ν_air, D_vapor, K_therm) = aps
    (; χv, ve, Δv) = vel
    (; r0) = mass
    aᵥ = vent.a
    bᵥ = vent.b

    # Latent heat of fusion: ℒⁱ(vapor→ice) - ℒˡ(vapor→liquid) = ℒf(liquid→ice)
    ℒf = ice_latent_heat(T, constants) - liquid_latent_heat(T, constants)

    n₀ = get_n0(pdf, qˢ, ρ)
    v₀ = get_v0(vel, ρ)
    λ⁻¹ = lambda_inverse(pdf, mass, qˢ, ρ)

    # Sensible-heat-driven melting rate
    base_rate = 4π * n₀ / ρ * K_therm / ℒf * (T - T_freeze) * λ⁻¹^2

    # Ventilation correction terms
    Sc = ν_air / D_vapor
    Re = 2v₀ * χv / ν_air * λ⁻¹
    size_factor = (r0 / λ⁻¹)^((ve + Δv) / 2)
    gamma_factor = Γ((ve + Δv + 5) / 2)

    ventilation = aᵥ + bᵥ * cbrt(Sc) * sqrt(Re) / size_factor * gamma_factor

    melt_rate = base_rate * ventilation

    # Only melt when snow exists and temperature is above freezing
    melting = (qˢ > ϵ_numerics(FT)) & (T > T_freeze)
    return ifelse(melting, melt_rate, zero(FT))
end
```

- [ ] **Step 2: Commit**

```bash
git add ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl
git commit -m "Add snow melting Breeze wrapper"
```

---

### Task 4: Add `warm_accretion_melt_factor` Wrapper

**Files:**
- Modify: `ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl` (insert after `snow_melting`)

- [ ] **Step 1: Add the wrapper function**

Insert after `snow_melting`:

```julia
#####
##### Warm accretion melt factor (TRANSLATION: uses Breeze thermodynamics for Lf and cv_l)
#####

"""
    warm_accretion_melt_factor(snow_params, T, constants)

Compute the thermal melt factor for warm accretion processes.

When cloud liquid or rain collides with snow above freezing, the sensible heat
carried by the warm hydrometeor melts additional snow. The factor ``α`` gives
the mass ratio of melted snow to accreted warm hydrometeor mass:

``α = cˡ (T - T_{freeze}) / ℒf``

This is a translation of `CloudMicrophysics.BulkMicrophysicsTendencies.warm_accretion_melt_factor`
that uses Breeze's internal thermodynamics instead of Thermodynamics.jl.

# Arguments
- `snow_params`: Snow parameters (contains T_freeze)
- `T`: Temperature
- `constants`: Breeze ThermodynamicConstants

# Returns
Thermal melt factor α (zero when T <= T_freeze)
"""
@inline function warm_accretion_melt_factor(
    (; T_freeze)::Snow{FT},
    T::FT,
    constants,
) where {FT}
    cˡ = constants.liquid.heat_capacity
    ℒf = ice_latent_heat(T, constants) - liquid_latent_heat(T, constants)
    ΔT = T - T_freeze
    return ifelse(T <= T_freeze, zero(FT), cˡ / ℒf * ΔT)
end
```

- [ ] **Step 2: Commit**

```bash
git add ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl
git commit -m "Add warm accretion melt factor Breeze wrapper"
```

---

### Task 5: Add Snow Sedimentation Infrastructure

**Files:**
- Modify: `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl:177-181` (velocity dispatch)
- Modify: `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl:300-319` (field materialization)
- Modify: `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl:352-375` (terminal velocity computation)

- [ ] **Step 1: Add snow sedimentation velocity dispatch**

In `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl`, after line 181 (the rain
velocity dispatch), add the snow velocity dispatch:

```julia
# Snow sedimentation: snow falls with terminal velocity (mixed-phase schemes only)
@inline AM.microphysical_velocities(bμp::MixedPhase1M, μ, ::Val{:ρqˢ}) = (u=zf, v=zf, w=μ.wˢ)
```

- [ ] **Step 2: Add `wˢ` face field materialization**

In `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl`, replace the
`materialize_microphysical_fields` function (lines 300-319) with:

```julia
function AM.materialize_microphysical_fields(bμp::OneMomentLiquidRain, grid, bcs)
    if bμp isa WP1M
        center_names = (warm_phase_field_names..., :qᵉ)
    elseif bμp isa WPNE1M
        center_names = (:ρqᶜˡ, warm_phase_field_names...)
    elseif bμp isa MP1M
        center_names = (warm_phase_field_names..., ice_phase_field_names..., :qᵉ)
    elseif bμp isa MPNE1M
        center_names = (:ρqᶜˡ, :ρqᶜⁱ, warm_phase_field_names..., ice_phase_field_names...)
    end

    center_fields = center_field_tuple(grid, center_names...)

    # Precipitation terminal velocities (negative = downward)
    # bottom = nothing ensures the kernel-set value is preserved during fill_halo_regions!
    face_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wʳ = ZFaceField(grid; boundary_conditions=face_bcs)

    if bμp isa MixedPhase1M
        wˢ_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
        wˢ = ZFaceField(grid; boundary_conditions=wˢ_bcs)
        return (; zip(center_names, center_fields)..., wʳ, wˢ)
    end

    return (; zip(center_names, center_fields)..., wʳ)
end
```

- [ ] **Step 3: Add snow terminal velocity to `update_microphysical_auxiliaries!`**

In `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl`, replace the
`update_microphysical_auxiliaries!` method for `MixedPhase1M` (lines 352-375) with:

```julia
# Mixed-phase one-moment schemes
@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, bμp::MixedPhase1M, ℳ::MixedPhaseOneMomentState, ρ, 𝒰, constants)
    # State fields
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.qᶜⁱ[i, j, k] = ℳ.qᶜⁱ
    @inbounds μ.qʳ[i, j, k] = ℳ.qʳ
    @inbounds μ.qˢ[i, j, k] = ℳ.qˢ

    # Vapor from thermodynamic state
    @inbounds μ.qᵛ[i, j, k] = 𝒰.moisture_mass_fractions.vapor

    # Derived: total liquid and ice
    @inbounds μ.qˡ[i, j, k] = ℳ.qᶜˡ + ℳ.qʳ
    @inbounds μ.qⁱ[i, j, k] = ℳ.qᶜⁱ + ℳ.qˢ

    # Terminal velocities with bottom boundary condition
    categories = bμp.categories

    # Rain terminal velocity
    𝕎 = terminal_velocity(categories.rain, categories.hydrometeor_velocities.rain, ρ, ℳ.qʳ)
    wʳ = -𝕎 # negative = downward
    wʳ₀ = bottom_terminal_velocity(bμp.precipitation_boundary_condition, wʳ)
    @inbounds μ.wʳ[i, j, k] = ifelse(k == 1, wʳ₀, wʳ)

    # Snow terminal velocity
    𝕎ˢ = terminal_velocity(categories.snow, categories.hydrometeor_velocities.snow, ρ, ℳ.qˢ)
    wˢ = -𝕎ˢ # negative = downward
    wˢ₀ = bottom_terminal_velocity(bμp.precipitation_boundary_condition, wˢ)
    @inbounds μ.wˢ[i, j, k] = ifelse(k == 1, wˢ₀, wˢ)

    return nothing
end
```

- [ ] **Step 4: Commit**

```bash
git add ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl
git commit -m "Add snow sedimentation infrastructure (wˢ field, terminal velocity, dispatch)"
```

---

### Task 6: Extend `mpne1m_tendencies` With Full Snow and Ice Processes

**Files:**
- Modify: `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl:675-764`

- [ ] **Step 1: Update the comment block and function**

In `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl`, replace the comment block
(lines 675-688) and the entire `mpne1m_tendencies` function (lines 690-746) with:

```julia
#
#   dqⁱ/dt = (qᵛ - qᵛ⁺ⁱ) / (Γⁱ τⁱ)
#
# where qᵛ⁺ⁱ is the saturation specific humidity over ice, τⁱ is the ice relaxation
# timescale, and Γⁱ is the thermodynamic adjustment factor using ice latent heat.
#
# `ice_thermodynamic_adjustment_factor` and `deposition_rate` are defined in `Breeze.Microphysics`
# so they can be shared by multiple bulk microphysics schemes.
#
#   ρqᵛ:  −Sᶜᵒⁿᵈ − Sᵈᵉᵖ − Sᵉᵛᵃᵖ − Sˢᵘᵇˡ
#   ρqᶜˡ: +Sᶜᵒⁿᵈ − Sᵃᶜⁿᵛ − Sᵃᶜᶜ − Sᵃᶜᶜˡˢ
#   ρqᶜⁱ: +Sᵈᵉᵖ − Sᵃᶜⁿᵛⁱˢ − Sᵃᶜᶜⁱˢ − Sᵃᶜᶜⁱʳ
#   ρqʳ:  +Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ − Sᵃᶜᶜʳⁱ + Sᵐᵉˡᵗ + T-routed(Sᵃᶜᶜˡˢ, Sʳˢ, Sˢʳ, α)
#   ρqˢ:  +Sᵃᶜⁿᵛⁱˢ + Sᵃᶜᶜⁱˢ + Sᵃᶜᶜⁱʳ + Sᵃᶜᶜʳⁱ + Sˢᵘᵇˡ − Sᵐᵉˡᵗ + T-routed(Sᵃᶜᶜˡˢ, Sʳˢ, Sˢʳ, α)
#####

@inline function mpne1m_tendencies(bμp::MPNE1M, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    categories = bμp.categories
    τᶜˡ = liquid_relaxation_timescale(bμp.cloud_formation, categories)
    τᶜⁱ = ice_relaxation_timescale(bμp.cloud_formation, categories)
    qᶜˡ = ℳ.qᶜˡ
    qᶜⁱ = ℳ.qᶜⁱ
    qʳ = ℳ.qʳ
    qˢ = ℳ.qˢ

    T = temperature(𝒰, constants)
    q = 𝒰.moisture_mass_fractions
    qᵛ = q.vapor

    # Condensation: vapor ↔ cloud liquid
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    Sᶜᵒⁿᵈ = condensation_rate(qᵛ, qᵛ⁺, qᶜˡ, T, ρ, q, τᶜˡ, constants)
    Sᶜᵒⁿᵈ = ifelse(isnan(Sᶜᵒⁿᵈ), zero(Sᶜᵒⁿᵈ), Sᶜᵒⁿᵈ)

    # Deposition: vapor ↔ cloud ice
    qᵛ⁺ⁱ = saturation_specific_humidity(T, ρ, constants, PlanarIceSurface())
    Sᵈᵉᵖ = deposition_rate(qᵛ, qᵛ⁺ⁱ, qᶜⁱ, T, ρ, q, τᶜⁱ, constants)
    Sᵈᵉᵖ = ifelse(isnan(Sᵈᵉᵖ), zero(Sᵈᵉᵖ), Sᵈᵉᵖ)

    # Evaporation: rain → vapor (Sᵉᵛᵃᵖ < 0 when rain evaporates)
    Sᵉᵛᵃᵖ = rain_evaporation(categories.rain,
                             categories.hydrometeor_velocities.rain,
                             categories.air_properties,
                             q, qʳ, ρ, T, constants)
    Sᵉᵛᵃᵖ = max(Sᵉᵛᵃᵖ, -max(0, qʳ) / τⁿᵘᵐ)

    # Snow sublimation/deposition: snow ↔ vapor (positive = deposition)
    Sˢᵘᵇˡ = snow_sublimation_deposition(categories.snow,
                                        categories.hydrometeor_velocities.snow,
                                        categories.air_properties,
                                        q, qˢ, ρ, T, constants)
    Sˢᵘᵇˡ = max(Sˢᵘᵇˡ, -max(0, qˢ) / τⁿᵘᵐ)

    # Snow melting: snow → rain (always non-negative)
    Sᵐᵉˡᵗ = snow_melting(categories.snow,
                         categories.hydrometeor_velocities.snow,
                         categories.air_properties,
                         qˢ, ρ, T, constants)
    Sᵐᵉˡᵗ = min(Sᵐᵉˡᵗ, max(0, qˢ) / τⁿᵘᵐ)

    # Collection: cloud liquid → rain (does not involve vapor)
    Sᵃᶜⁿᵛ = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ)
    Sᵃᶜᶜ = accretion(categories.cloud_liquid, categories.rain,
                     categories.hydrometeor_velocities.rain, categories.collisions,
                     qᶜˡ, qʳ, ρ)

    # Ice → snow autoconversion
    Sᵃᶜⁿᵛⁱˢ = conv_q_icl_to_q_sno_no_supersat(categories.snow.acnv1M, qᶜⁱ, true)

    # Accretion: cloud liquid + snow
    Sᵃᶜᶜˡˢ = accretion(categories.cloud_liquid, categories.snow,
                       categories.hydrometeor_velocities.snow, categories.collisions,
                       qᶜˡ, qˢ, ρ)

    # Accretion: cloud ice + snow → snow
    Sᵃᶜᶜⁱˢ = accretion(categories.cloud_ice, categories.snow,
                       categories.hydrometeor_velocities.snow, categories.collisions,
                       qᶜⁱ, qˢ, ρ)

    # Accretion: cloud ice + rain → snow (ice sink)
    Sᵃᶜᶜⁱʳ = accretion(categories.cloud_ice, categories.rain,
                       categories.hydrometeor_velocities.rain, categories.collisions,
                       qᶜⁱ, qʳ, ρ)

    # Rain sink from ice-rain collisions (rain sink, forms snow)
    Sᵃᶜᶜʳⁱ = accretion_rain_sink(categories.rain, categories.cloud_ice,
                                 categories.hydrometeor_velocities.rain, categories.collisions,
                                 qᶜⁱ, qʳ, ρ)

    # Rain-snow collisions (computed for both cold and warm pathways)
    Sʳˢ = accretion_snow_rain(categories.snow, categories.rain,
                             categories.hydrometeor_velocities.snow,
                             categories.hydrometeor_velocities.rain,
                             categories.collisions, qˢ, qʳ, ρ)
    Sˢʳ = accretion_snow_rain(categories.rain, categories.snow,
                             categories.hydrometeor_velocities.rain,
                             categories.hydrometeor_velocities.snow,
                             categories.collisions, qʳ, qˢ, ρ)

    # Thermal melt factor for warm accretion
    α = warm_accretion_melt_factor(categories.snow, T, constants)

    # Temperature routing (branchless)
    is_warm = T >= categories.snow.T_freeze

    # Physics tendencies — conserved by construction: sum of all five = 0
    ρqᵛ_phys  = ρ * (-Sᶜᵒⁿᵈ - Sᵈᵉᵖ - Sᵉᵛᵃᵖ - Sˢᵘᵇˡ)
    ρqᶜˡ_phys = ρ * ( Sᶜᵒⁿᵈ - Sᵃᶜⁿᵛ - Sᵃᶜᶜ - Sᵃᶜᶜˡˢ)
    ρqᶜⁱ_phys = ρ * ( Sᵈᵉᵖ - Sᵃᶜⁿᵛⁱˢ - Sᵃᶜᶜⁱˢ - Sᵃᶜᶜⁱʳ)
    ρqʳ_phys  = ρ * ( Sᵃᶜⁿᵛ + Sᵃᶜᶜ + Sᵉᵛᵃᵖ - Sᵃᶜᶜʳⁱ + Sᵐᵉˡᵗ
                     + ifelse(is_warm, Sᵃᶜᶜˡˢ + α * Sᵃᶜᶜˡˢ + Sˢʳ + α * Sʳˢ, zero(T))
                     - ifelse(is_warm, zero(T), Sʳˢ))
    ρqˢ_phys  = ρ * ( Sᵃᶜⁿᵛⁱˢ + Sᵃᶜᶜⁱˢ + Sᵃᶜᶜⁱʳ + Sᵃᶜᶜʳⁱ + Sˢᵘᵇˡ - Sᵐᵉˡᵗ
                     + ifelse(is_warm, zero(T), Sᵃᶜᶜˡˢ + Sʳˢ)
                     - ifelse(is_warm, α * Sᵃᶜᶜˡˢ + Sˢʳ + α * Sʳˢ, zero(T)))

    # Numerical relaxation guards — conserved by routing each correction to its exchange partner.
    # When q < 0, replace with -ρq/τ and route the delta to the coupled tracer:
    #   v→cl (condensation), cl→r (collection), ci→v (deposition), r→v (evaporation).
    # This preserves ρqᵛ + ρqᶜˡ + ρqᶜⁱ + ρqʳ + ρqˢ = 0 regardless of which guards fire.
    # Snow has no correction — rate limiters on sublimation and melting suffice.
    δᵛ  = ifelse(qᵛ  >= 0, zero(ρqᵛ_phys),  -ρ * qᵛ  / τⁿᵘᵐ - ρqᵛ_phys)
    δᶜˡ = ifelse(qᶜˡ >= 0, zero(ρqᶜˡ_phys), -ρ * qᶜˡ / τᶜˡ  - ρqᶜˡ_phys)
    δᶜⁱ = ifelse(qᶜⁱ >= 0, zero(ρqᶜⁱ_phys), -ρ * qᶜⁱ / τᶜⁱ  - ρqᶜⁱ_phys)
    δʳ  = ifelse(qʳ  >= 0, zero(ρqʳ_phys),  -ρ * qʳ  / τⁿᵘᵐ - ρqʳ_phys)

    ρqᵛ  = ρqᵛ_phys  + δᵛ  - δᶜⁱ - δʳ
    ρqᶜˡ = ρqᶜˡ_phys + δᶜˡ - δᵛ
    ρqᶜⁱ = ρqᶜⁱ_phys + δᶜⁱ
    ρqʳ  = ρqʳ_phys  + δʳ  - δᶜˡ
    ρqˢ  = ρqˢ_phys

    return (; ρqᵛ, ρqᶜˡ, ρqᶜⁱ, ρqʳ, ρqˢ)
end
```

- [ ] **Step 2: Commit**

```bash
git add ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl
git commit -m "Extend mpne1m_tendencies with full snow and ice processes"
```

---

### Task 7: Add `:ρqˢ` Tendency Dispatch and Clean Up TODO

**Files:**
- Modify: `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl` (after existing dispatch methods, around line 764)

- [ ] **Step 1: Remove TODO comment and add snow tendency dispatch**

In `ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl`, replace the `:ρqᶜⁱ` dispatch
method (the one with the TODO comment) with:

```julia
@inline function AM.microphysical_tendency(bμp::MPNE1M, ::Val{:ρqᶜⁱ}, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    return mpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqᶜⁱ
end
```

Then, after the `:ρqʳ` dispatch method, add the new `:ρqˢ` dispatch:

```julia
@inline function AM.microphysical_tendency(bμp::MPNE1M, ::Val{:ρqˢ}, ρ, ℳ::MixedPhaseOneMomentState, 𝒰, constants)
    return mpne1m_tendencies(bμp, ρ, ℳ, 𝒰, constants).ρqˢ
end
```

- [ ] **Step 2: Commit**

```bash
git add ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl
git commit -m "Add snow tendency dispatch and remove ice TODO"
```

---

### Task 8: Tests

**Files:**
- Modify: `test/cloud_microphysics_1M.jl` (add new test sets at the end, before the closing of the file)

- [ ] **Step 1: Add snow sedimentation field and velocity tests**

At the end of `test/cloud_microphysics_1M.jl` (before the final line), add:

```julia
@testset "Mixed-phase non-equilibrium snow field materialization [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Snow terminal velocity field should exist
    @test haskey(model.microphysical_fields, :wˢ)

    # Snow sedimentation velocity dispatch
    μ = model.microphysical_fields
    vel_snow = microphysical_velocities(microphysics, μ, Val(:ρqˢ))
    @test vel_snow !== nothing
    @test haskey(vel_snow, :w)

    # Other tracers still have correct dispatch
    vel_rain = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain !== nothing
    vel_cloud = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud === nothing
end

@testset "MPNE1M snow processes time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 10
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Cold, supersaturated conditions → cloud ice should form via deposition
    set!(model; θ=260, qᵗ=FT(0.010))

    # Run for a few relaxation timescales
    τ = FT(1) / microphysics.cloud_formation.ice.rate
    simulation = Simulation(model; Δt=τ/5, stop_time=10τ, verbose=false)
    run!(simulation)

    # Cloud ice should have formed from deposition
    qᶜⁱ_max = maximum(model.microphysical_fields.qᶜⁱ)
    @test qᶜⁱ_max > FT(1e-6)

    # Snow should have formed from ice autoconversion
    qˢ_max = maximum(model.microphysical_fields.qˢ)
    @test qˢ_max > FT(0)

    # Model should complete without errors (all tendencies computed)
    @test model.clock.iteration > 0
end
```

- [ ] **Step 2: Run the tests**

Run:
```bash
cd /home/x-kcheng2/Aeolus/Breeze.jl3
julia --project -e 'using Pkg; Pkg.test("Breeze"; test_args=["cloud_microphysics_1M"])'
```

Expected: all tests pass, including the two new test sets.

- [ ] **Step 3: Commit**

```bash
git add test/cloud_microphysics_1M.jl
git commit -m "Add tests for MPNE1M snow sedimentation, tendencies, and time-stepping"
```

---

### Task 9: Verify Trailing Whitespace and Final Newlines

**Files:**
- All modified files

- [ ] **Step 1: Check for trailing whitespace and trailing blank lines**

Run:
```bash
cd /home/x-kcheng2/Aeolus/Breeze.jl3
# Check trailing whitespace
grep -rn ' $' ext/BreezeCloudMicrophysicsExt/BreezeCloudMicrophysicsExt.jl \
                ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl \
                ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl \
                test/cloud_microphysics_1M.jl

# Check files end with exactly one newline (no trailing blank lines)
for f in ext/BreezeCloudMicrophysicsExt/BreezeCloudMicrophysicsExt.jl \
         ext/BreezeCloudMicrophysicsExt/cloud_microphysics_translations.jl \
         ext/BreezeCloudMicrophysicsExt/one_moment_microphysics.jl \
         test/cloud_microphysics_1M.jl; do
    tail -c 2 "$f" | xxd | head -1
done
```

Expected: no trailing whitespace found, each file ends with `0a` (single newline).

- [ ] **Step 2: Fix any issues found and commit**

If issues are found, fix them and commit:
```bash
git add -A
git commit -m "Fix trailing whitespace"
```
