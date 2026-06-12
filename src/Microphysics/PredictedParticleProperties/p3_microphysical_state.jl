using Oceananigans: CenterField
using Oceananigans.Fields: ZeroField
using Oceananigans.Operators: ℑzᵃᵃᶜ
using DocStringExtensions: TYPEDSIGNATURES

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze.Thermodynamics: MoistureMassFractions

using Breeze: Microphysics

const P3 = PredictedParticlePropertiesMicrophysics

#####
##### P3MicrophysicalState
#####

"""
    P3MicrophysicalState{FT} <: AbstractMicrophysicalState{FT}

Microphysical state for P3 (Predicted Particle Properties) microphysics.

Contains the local mixing ratios and number concentrations needed to compute
tendencies for cloud liquid, rain, ice, rime, and predicted liquid fraction.

# Fields
$(TYPEDFIELDS)
"""
struct P3MicrophysicalState{FT} <: AbstractMicrophysicalState{FT}
    "Cloud liquid mixing ratio [kg/kg]"
    qᶜˡ :: FT
    "Cloud number concentration [1/kg]"
    nᶜˡ :: FT
    "Rain mixing ratio [kg/kg]"
    qʳ  :: FT
    "Rain number concentration [1/kg]"
    nʳ  :: FT
    "Ice mixing ratio [kg/kg]"
    qⁱ  :: FT
    "Ice number concentration [1/kg]"
    nⁱ  :: FT
    "Rime mass mixing ratio [kg/kg]"
    qᶠ  :: FT
    "Rime volume [m³/kg]"
    bᶠ  :: FT
    "Ice sixth moment [m⁶/kg]"
    zⁱ  :: FT
    "Liquid water on ice mixing ratio [kg/kg]"
    qʷⁱ :: FT
    "Predicted supersaturation [kg/kg] (Grabowski & Morrison 2008)"
    sˢᵃᵗ :: FT
    "Unactivated aerosol number concentration [1/kg] (zero when no aerosol prognostic)"
    nᵃ  :: FT
    "Cell-center vertical velocity [m/s] — drives the dynamic supersaturation forcing A_w in `coupled_saturation_adjustment_rates`"
    w   :: FT
end

@inline function P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                      qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ)
    FT = typeof(sˢᵃᵗ)
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, zero(FT), zero(FT))
end

@inline function P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                      qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, nᵃ)
    FT = typeof(sˢᵃᵗ)
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ,
                                qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, nᵃ, zero(FT))
end

#####
##### Configuration probes
#####

"""
$(TYPEDSIGNATURES)

Whether `p3` is configured to run the 3-moment ice path (carries the
reflectivity/sixth moment `ρz̃ⁱ`).
"""
@inline is_three_moment_ice(p3::P3) = !isnothing(three_moment_shape_table(p3))

# Initial aerosol number for the prognostic-ρnᵃ path. Returns the total per-mass
# aerosol count from the AerosolMode parameters; the host (parcel set!,
# AtmosphereModel set_microphysical_initial_state!) is responsible for scaling
# by air density when assigning to `ρnᵃ` if a per-volume value is required.
# The prescribed-Nᶜ path returns 0, matching the framework default.
@inline AM.initial_aerosol_number(p3::P3) = _initial_aerosol_number(p3.aerosol)
@inline _initial_aerosol_number(::Nothing) = 0
@inline _initial_aerosol_number(aerosol::AerosolActivation) = sum_aerosol_number(aerosol)

#####
##### Prognostic field names
#####

# The 3-moment ice switch must be resolvable to a constant tuple at compile time,
# otherwise the resulting Union return type forces the generic GPU
# `extract_microphysical_prognostics` recursion to allocate.
#
# We dispatch on the *type* of `three_moment_shape_table(p3)` — it is `Nothing` in
# 2-moment mode and a concrete table type in 3-moment mode, so the compiler folds
# the helper down to a static tuple per concrete P3 type.
#
# The `predict_supersaturation` flag is value-only (a Bool field of
# `ProcessRateParameters`) and is not type-dispatchable without restructuring,
# so `ρsˢᵃᵗ` stays in the prognostic list. Its tendency is gated to zero in
# `tendency_ρsˢᵃᵗ` when the flag is off, so the only cost of always carrying it
# is one advected/integrated tracer.

@inline z̃_prognostic_names(::Nothing) = ()
@inline z̃_prognostic_names(_) = (:ρz̃ⁱ,)

# Aerosol depletion is enabled (and `ρnᵃ` advected) iff `p3.aerosol` is a
# concrete `AerosolActivation`; in the prescribed-Nᶜ path no aerosol field is needed.
@inline aerosol_prognostic_names(::Nothing) = ()
@inline aerosol_prognostic_names(_) = (:ρnᵃ,)

"""
$(TYPEDSIGNATURES)

Return prognostic field names for the P3 scheme.

- Cloud: ρqᶜˡ, ρnᶜˡ
- Rain: ρqʳ, ρnʳ
- Ice (always): ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ
- Ice (3-moment only): ρz̃ⁱ
- Supersaturation: ρsˢᵃᵗ (tendency = 0 when `predict_supersaturation = false`)
- Aerosol (only when `aerosol::AerosolActivation` is set): ρnᵃ
"""
@inline function AM.prognostic_field_names(p3::P3)
    cloud_names = (:ρqᶜˡ, :ρnᶜˡ)
    rain_names = (:ρqʳ, :ρnʳ)
    ice_names = (:ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
    z_names = z̃_prognostic_names(three_moment_shape_table(p3))
    ssat_names = (:ρsˢᵃᵗ,)
    aero_names = aerosol_prognostic_names(p3.aerosol)

    return tuple(cloud_names..., rain_names..., ice_names...,
                 z_names..., ssat_names..., aero_names...)
end

"""
$(TYPEDSIGNATURES)

Effective cloud droplet number concentration [kg⁻¹] seen by P3's process rates.

In the prescribed-Nᶜ path (`p3.aerosol === nothing`, matching Fortran
`log_predictNc = .false.`), `nc` is always `nccnst_2` at every microphysics call
and is not advected by the dynamical core. This helper returns the prescribed
value so that downstream rates (CCN activation, condensation efficiency,
autoconversion, immersion freezing) use the scheme-level parameter rather than
the unused, drifting prognostic field.

In the prognostic path (aerosol activation enabled), it returns the advected
per-mass number `μ.ρnᶜˡ / ρ` as usual.
"""
@inline effective_cloud_droplet_number(p3::P3, ρnᶜˡ, ρ) =
    isnothing(p3.aerosol) ? p3.cloud.number_concentration / ρ : ρnᶜˡ / ρ

#####
##### Moisture prognostic name
#####

"""
$(TYPEDSIGNATURES)

P3 is a non-equilibrium scheme: vapor (`qᵛ`) is the prognostic moisture variable.
"""
AM.moisture_prognostic_name(::P3) = :ρqᵛ

"""
$(TYPEDSIGNATURES)

Convert total moisture to the prognostic moisture variable for P3.

For P3, the prognostic moisture is vapor: `qᵛ = qᵗ - qᶜˡ - qʳ - qⁱ - qʷⁱ`.

This helper is used by parcel-style paths that still carry total moisture.
"""
@inline function AM.specific_prognostic_moisture_from_total(::P3, qᵗ, ℳ::P3MicrophysicalState)
    return max(0, qᵗ - ℳ.qᶜˡ - ℳ.qʳ - ℳ.qⁱ - ℳ.qʷⁱ)
end

@inline function AM.specific_prognostic_moisture_from_total(::P3, qᵗ, μ_fields::NamedTuple, ρ)
    return qᵗ - μ_fields.ρqᶜˡ / ρ - μ_fields.ρqʳ / ρ - μ_fields.ρqⁱ / ρ - μ_fields.ρqʷⁱ / ρ
end

#####
##### Materialize microphysical fields
#####

"""
$(TYPEDSIGNATURES)

Create prognostic and diagnostic fields for P3 microphysics.

The P3 scheme requires the following fields on `grid`:

**Prognostic (density-weighted):**
- `ρqᶜˡ`, `ρnᶜˡ`: Cloud liquid mass and number densities
- `ρqʳ`, `ρnʳ`: Rain mass and number densities
- `ρqⁱ`, `ρnⁱ`: Ice mass and number densities
- `ρqᶠ`, `ρbᶠ`: Rime mass and volume densities
- `ρz̃ⁱ`: Advected square-root sixth moment density, where `z̃ⁱ = sqrt(zⁱ nⁱ)`
- `ρqʷⁱ`: Liquid water on ice mass density

**Diagnostic:**
- `qᵛ`: Vapor specific humidity (mirrors the prognostic vapor field)
"""
function AM.materialize_microphysical_fields(::P3, grid, bcs)
    # Create all prognostic fields
    ρqᶜˡ = CenterField(grid)  # Cloud liquid
    ρnᶜˡ = CenterField(grid)  # Cloud number
    ρqʳ  = CenterField(grid)  # Rain mass
    ρnʳ  = CenterField(grid)  # Rain number
    ρqⁱ  = CenterField(grid)  # Ice mass
    ρnⁱ  = CenterField(grid)  # Ice number
    ρqᶠ  = CenterField(grid)  # Rime mass
    ρbᶠ  = CenterField(grid)  # Rime volume
    ρz̃ⁱ = CenterField(grid)  # Advected square-root sixth moment
    ρqʷⁱ = CenterField(grid)  # Liquid on ice
    ρsˢᵃᵗ = CenterField(grid) # Predicted supersaturation
    ρnᵃ  = CenterField(grid)  # Unactivated aerosol number density [1/m³]

    # Diagnostic mixing ratio / number-concentration fields
    # (updated each step in update_microphysical_auxiliaries!, matching the Kessler pattern)
    qᶜˡ = CenterField(grid)  # Cloud liquid specific humidity [kg/kg]
    nᶜˡ = CenterField(grid)  # Cloud number concentration [kg⁻¹]
    qʳ  = CenterField(grid)  # Rain specific humidity [kg/kg]
    nʳ  = CenterField(grid)  # Rain number concentration [kg⁻¹]
    qⁱ  = CenterField(grid)  # Ice specific humidity [kg/kg]
    nⁱ  = CenterField(grid)  # Ice number concentration [kg⁻¹]
    qᶠ  = CenterField(grid)  # Rime mass mixing ratio [kg/kg]
    bᶠ  = CenterField(grid)  # Rime volume [m³/kg]
    zⁱ  = CenterField(grid)  # Ice sixth moment [m⁶/kg]
    z̃ⁱ  = CenterField(grid)  # Advected square-root sixth moment √(zⁱ nⁱ)
    qʷⁱ = CenterField(grid)  # Liquid water on ice [kg/kg]
    sˢᵃᵗ = CenterField(grid) # Supersaturation [kg/kg]
    nᵃ   = CenterField(grid) # Unactivated aerosol [kg⁻¹]

    # Diagnostic field for vapor
    qᵛ = CenterField(grid)

    # Sedimentation velocity fields (pre-computed during update_state!)
    wʳ  = CenterField(grid)  # Rain mass-weighted terminal velocity
    wʳₙ = CenterField(grid)  # Rain number-weighted terminal velocity
    wⁱ  = CenterField(grid)  # Ice mass-weighted terminal velocity
    wⁱₙ = CenterField(grid)  # Ice number-weighted terminal velocity
    wⁱ_z = CenterField(grid) # Ice reflectivity-weighted terminal velocity

    # Microphysical tendency cache (written in update_microphysical_auxiliaries!, read by
    # grid_microphysical_tendency). Storing the microphysics-only contribution avoids 10×
    # redundant compute_p3_process_rates calls — one per prognostic field per grid point.
    cache_ρqᶜˡ = CenterField(grid)
    cache_ρnᶜˡ = CenterField(grid)
    cache_ρqʳ  = CenterField(grid)
    cache_ρnʳ  = CenterField(grid)
    cache_ρqⁱ  = CenterField(grid)
    cache_ρnⁱ  = CenterField(grid)
    cache_ρqᶠ  = CenterField(grid)
    cache_ρbᶠ  = CenterField(grid)
    cache_ρz̃ⁱ = CenterField(grid)
    cache_ρqʷⁱ = CenterField(grid)
    cache_ρsˢᵃᵗ = CenterField(grid)
    cache_ρqᵛ  = CenterField(grid)
    cache_ρnᵃ  = CenterField(grid)

    return (; ρqᶜˡ, ρnᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρz̃ⁱ, ρqʷⁱ, ρsˢᵃᵗ, ρnᵃ,
              qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, z̃ⁱ, qʷⁱ, sˢᵃᵗ, nᵃ, qᵛ,
              wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z,
              cache_ρqᶜˡ, cache_ρnᶜˡ, cache_ρqʳ, cache_ρnʳ, cache_ρqⁱ, cache_ρnⁱ,
              cache_ρqᶠ, cache_ρbᶠ, cache_ρz̃ⁱ, cache_ρqʷⁱ, cache_ρsˢᵃᵗ, cache_ρqᵛ, cache_ρnᵃ)
end

#####
##### Gridless MicrophysicalState construction
#####
#
# P3 is a non-equilibrium scheme: all condensate comes from prognostic fields μ.

"""
$(TYPEDSIGNATURES)

Build a [`P3MicrophysicalState`](@ref) from density-weighted prognostic variables.

P3 is a non-equilibrium scheme, so all cloud and precipitation variables come
from the prognostic fields `μ`, not from the thermodynamic state `𝒰`.
"""
# Compile-time NamedTuple field lookup with a default — used so that the gridless
# `microphysical_state` path works whether or not `μ` carries the optional `ρz̃ⁱ`
# (3-moment ice) and `ρsˢᵃᵗ` (predicted supersaturation) fields.
@generated function get_or_default(μ::NamedTuple{names}, ::Val{key}, default) where {names, key}
    return key in names ? :(μ.$key) : :(default)
end

@inline vertical_velocity(velocities, FT) = FT(velocities.w)

# Interpolate a face-located w field to a cell center.
# All call sites pass face fields (or ZeroField placeholders); no scalar fallback needed.
@inline interpolate_w_to_center(grid, i, j, k, w_field, FT) = FT(ℑzᵃᵃᶜ(i, j, k, grid, w_field))

@inline function AM.microphysical_state(p3::P3, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    nᶜˡ = effective_cloud_droplet_number(p3, μ.ρnᶜˡ, ρ)
    qʳ  = μ.ρqʳ / ρ
    nʳ  = μ.ρnʳ / ρ
    qⁱ  = μ.ρqⁱ / ρ
    nⁱ  = μ.ρnⁱ / ρ
    # Fortran advects z̃ = √(z·N) and converts to physical z at microphysics entry:
    #   where (nitot > 0) zitot = zitot**2 / nitot; elsewhere zitot = 0
    # ρz̃ⁱ stores the advected variable z̃; convert to physical z = z̃²/N for internal use.
    # In 2-moment mode ρz̃ⁱ is absent from `μ`; treat it as 0 (zⁱ then collapses to 0).
    FT = typeof(ρ)
    z̃ⁱ  = get_or_default(μ, Val(:ρz̃ⁱ), zero(FT)) / ρ
    zⁱ  = ifelse(nⁱ > FT(1e-20), z̃ⁱ^2 / nⁱ, zero(FT))
    qʷⁱ = μ.ρqʷⁱ / ρ
    rime_state = consistent_rime_state(p3, qⁱ, μ.ρqᶠ / ρ, μ.ρbᶠ / ρ, qʷⁱ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    # ρsˢᵃᵗ is absent unless predicted supersaturation is enabled; default to 0.
    sˢᵃᵗ = get_or_default(μ, Val(:ρsˢᵃᵗ), zero(FT)) / ρ
    # ρnᵃ is absent unless prognostic-aerosol path is enabled; default to 0.
    nᵃ = get_or_default(μ, Val(:ρnᵃ), zero(FT)) / ρ
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, nᵃ, vertical_velocity(velocities, FT))
end

# Disambiguation for P3 with Nothing or empty microphysical fields
@inline AM.microphysical_state(::P3, ρ, ::Nothing, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))
@inline AM.microphysical_state(::P3, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))

@inline function AM.grid_microphysical_state(i, j, k, grid, p3::P3, μ, ρ, 𝒰, velocities)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        nᶜˡ = effective_cloud_droplet_number(p3, μ.ρnᶜˡ[i, j, k], ρ)
        qʳ  = μ.ρqʳ[i, j, k] / ρ
        nʳ  = μ.ρnʳ[i, j, k] / ρ
        qⁱ  = μ.ρqⁱ[i, j, k] / ρ
        nⁱ  = μ.ρnⁱ[i, j, k] / ρ
        FT = typeof(ρ)
        z̃ⁱ  = μ.ρz̃ⁱ[i, j, k] / ρ
        zⁱ  = ifelse(nⁱ > FT(1e-20), z̃ⁱ^2 / nⁱ, zero(FT))
        qʷⁱ = μ.ρqʷⁱ[i, j, k] / ρ
    end
    rime_state = consistent_rime_state(p3, qⁱ, @inbounds(μ.ρqᶠ[i, j, k]) / ρ, @inbounds(μ.ρbᶠ[i, j, k]) / ρ, qʷⁱ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    sˢᵃᵗ = @inbounds μ.ρsˢᵃᵗ[i, j, k] / ρ
    nᵃ   = @inbounds μ.ρnᵃ[i, j, k] / ρ
    w = interpolate_w_to_center(grid, i, j, k, velocities.w, FT)
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, nᵃ, w)
end

# Clamp the prognostic `ρz̃ⁱ` to its physically consistent maximum
# (Fortran-parity `apply_mui_bounds_to_zi`). Without this writeback, sources
# accumulate ρz̃ⁱ above the physical bound while tendency math sees only the
# bounded zⁱ — there is no restoring force and the prognostic drifts.
# 2-moment (no `three_moment_shape_table`) is a compile-time no-op.
@inline clamp_ice_sixth_moment!(μ, i, j, k, p3::P3, ρ, ℳ) =
    clamp_ice_sixth_moment_dispatch(three_moment_shape_table(p3), μ, i, j, k, p3, ρ, ℳ)

@inline clamp_ice_sixth_moment_dispatch(::Nothing, μ, i, j, k, p3::P3, ρ, ℳ) = ℳ

@inline function clamp_ice_sixth_moment_dispatch(::P3ThreeMomentShapeTable,
                                                   μ, i, j, k, p3::P3, ρ, ℳ)
    FT = typeof(ρ)
    qⁱ_total = total_ice_mass(ℳ.qⁱ, ℳ.qʷⁱ)
    qⁱ_safe = max(qⁱ_total, FT(1e-20))
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ, ℳ.qʷⁱ)
    Fˡ = liquid_fraction_on_ice(ℳ.qⁱ, ℳ.qʷⁱ)
    μ_ice = compute_ice_shape_parameter(p3, qⁱ_safe, ℳ.nⁱ, ℳ.zⁱ,
                                         rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    zⁱ_bounded = bound_ice_sixth_moment(p3, qⁱ_safe, ℳ.nⁱ, ℳ.zⁱ,
                                         rime_state.Fᶠ, Fˡ, rime_state.ρᶠ, μ_ice)
    @inbounds μ.ρz̃ⁱ[i, j, k] = ρ * sqrt(max(0, zⁱ_bounded * ℳ.nⁱ))
    return P3MicrophysicalState(ℳ.qᶜˡ, ℳ.nᶜˡ, ℳ.qʳ, ℳ.nʳ, ℳ.qⁱ, ℳ.nⁱ,
                                 ℳ.qᶠ, ℳ.bᶠ, zⁱ_bounded, ℳ.qʷⁱ, ℳ.sˢᵃᵗ, ℳ.nᵃ, ℳ.w)
end

# GPU-compatible update_microphysical_fields! for P3.
# Bypasses the generic extract_microphysical_prognostics which uses runtime Symbol
# dispatch that GPU compilers cannot resolve. Instead, directly constructs
# P3MicrophysicalState from @inbounds field access and delegates to
# update_microphysical_auxiliaries!.
@inline function AM.update_microphysical_fields!(μ, i, j, k, grid, p3::P3, ρ, 𝒰, constants)
    @inbounds begin
        # TODO: thread real velocities here once AM.update_microphysical_fields!
        # signature carries them. ℳ.w == 0 is acceptable in this auxiliary path
        # because downstream update_microphysical_auxiliaries! does not consume w.
        velocities = (u = ZeroField(), v = ZeroField(), w = ZeroField())
        ℳ_raw = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, 𝒰, velocities)
        ℳ = clamp_ice_sixth_moment!(μ, i, j, k, p3, ρ, ℳ_raw)
        AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, p3, ℳ, ρ, 𝒰, constants)
    end
    return nothing
end

#####
##### Update microphysical auxiliary fields
#####

"""
$(TYPEDSIGNATURES)

Update diagnostic microphysical fields after state update.

After the moisture refactor, vapor is the prognostic moisture variable.
The diagnostic `qᵛ` field is updated from the thermodynamic state.
"""
# Lightweight diagnostics update — called from the thermodynamic variables kernel.
# Only writes basic specific quantities and vapor. The heavy computation (terminal
# velocities, process rates, tendency cache) is deferred to microphysics_model_update!
# via a SEPARATE kernel launch, avoiding GPU compilation failure from force-inlining
# ~1000 lines of P3 physics into the thermodynamic kernel.
@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, p3::P3, ℳ::P3MicrophysicalState, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k]  = 𝒰.moisture_mass_fractions.vapor
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.nᶜˡ[i, j, k] = ℳ.nᶜˡ
    @inbounds μ.qʳ[i, j, k]  = ℳ.qʳ
    @inbounds μ.nʳ[i, j, k]  = ℳ.nʳ
    @inbounds μ.qⁱ[i, j, k]  = ℳ.qⁱ
    @inbounds μ.nⁱ[i, j, k]  = ℳ.nⁱ
    @inbounds μ.qᶠ[i, j, k]  = ℳ.qᶠ
    @inbounds μ.bᶠ[i, j, k]  = ℳ.bᶠ
    @inbounds μ.zⁱ[i, j, k]  = ℳ.zⁱ
    @inbounds μ.z̃ⁱ[i, j, k]  = μ.ρz̃ⁱ[i, j, k] / ρ
    @inbounds μ.qʷⁱ[i, j, k] = ℳ.qʷⁱ
    @inbounds μ.sˢᵃᵗ[i, j, k] = ℳ.sˢᵃᵗ
    @inbounds μ.nᵃ[i, j, k]   = ℳ.nᵃ

    return nothing
end

# GPU-safe return struct for ice properties (NamedTuples require jl_f_tuple on GPU).
struct P3IceProps{FT}
    qᶠ :: FT
    bᶠ :: FT
    Fᶠ :: FT
    Fˡ :: FT
    ρᶠ :: FT
    qⁱ_total :: FT
    # D10 impose_max_Ni cap mirrored from compute_p3_process_rates so the PSD
    # (μ_ice, zⁱ_bounded) and the tabulated Z tendency use the same nⁱ that the
    # rate = N × m_table × env decomposition inside the process rates was built with.
    nⁱ :: FT
    μ_ice :: FT
    μ_cloud :: FT
    Nᶜ :: FT
    zⁱ_bounded :: FT
    D_v :: FT
    nu :: FT
    λ_r :: FT
end

# GPU-safe return struct for the full P3 computation (NamedTuples require jl_f_tuple on GPU).
struct P3CacheResult{FT}
    wʳ :: FT; wʳₙ :: FT; wⁱ :: FT; wⁱₙ :: FT; wⁱ_z :: FT
    c_qcl :: FT; c_ncl :: FT; c_qr :: FT; c_nr :: FT
    c_qi :: FT; c_ni :: FT; c_qf :: FT; c_bf :: FT
    c_zi :: FT; c_qwi :: FT; c_ss :: FT; c_qv :: FT
    c_na :: FT
end

@inline function z̃ⁱ_tendency(nⁱ, zⁱ, tendency_ρz_phys, tendency_ρn)
    FT = typeof(nⁱ + zⁱ + tendency_ρz_phys + tendency_ρn)
    z_times_n = zⁱ * nⁱ
    existing_distribution = (zⁱ > 0) & (nⁱ > 0) & (z_times_n > 0)

    regularized_z_times_n = max(z_times_n, eps(FT)^2)
    z̃ = sqrt(regularized_z_times_n)
    existing_tendency = (nⁱ * tendency_ρz_phys + zⁱ * tendency_ρn) / (2 * z̃)

    # At ice initiation z=n=0, d(sqrt(zn))/dt is sqrt(dz/dt * dn/dt).
    # This is the one-sided limit for simultaneous positive Z and N sources.
    source_z_tendency = max(0, tendency_ρz_phys)
    source_n_tendency = max(0, tendency_ρn)
    source_tendency = sqrt(source_z_tendency * source_n_tendency)

    return ifelse(existing_distribution, existing_tendency, source_tendency)
end

@inline function z̃ⁱ_tendency(nⁱ, zⁱ, tendency_ρz_phys, tendency_ρn,
                              ρz̃ⁱ, sink_limiting_timescale)
    raw_tendency = z̃ⁱ_tendency(nⁱ, zⁱ, tendency_ρz_phys, tendency_ρn)
    available_ρz̃ = max(0, ρz̃ⁱ)
    maximum_sink = available_ρz̃ / sink_limiting_timescale
    return max(raw_tendency, -maximum_sink)
end

# All P3 physics in a single @noinline function returning a concrete struct.
# compute_p3_process_rates (also @noinline) handles the heavy rates.
# All operations are scalar — no array access.
@noinline function _p3_scalar_compute(p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    props = p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
    Fᶠ = props.Fᶠ
    ρᶠ = props.ρᶠ

    # Rain terminal velocities — fused call shares λ_r, ρ_correction, log10(λ_r)
    # across the two 1D table lookups (mass- and number-weighted).
    vᵣ = rain_terminal_velocities(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    wʳ   = vᵣ.mass_weighted
    wʳₙ  = vᵣ.number_weighted
    # Fortran parity: after impose_max_Ni (microphy_p3.f90:2812/4390/4937) the nitot
    # array is capped in place, so all downstream math — process rates, terminal
    # velocities, Z tendency, reflectivity — sees the same value. Mirror that here by
    # using props.nⁱ (= min(ℳ.nⁱ, max_Ni/ρ)) wherever Fortran would see capped nitot.
    # Fortran indexes the ice fall-speed lookup with qitot (= dry + liquid-on-ice);
    # the table's q-norm axis is total ice mass per particle.
    qⁱ_total = total_ice_mass(ℳ.qⁱ, ℳ.qʷⁱ)
    # Fused call: shares m̄, ρ_correction, log(m̄), and the 5D interpolation indices
    # across mass-, number-, and reflectivity-weighted fall speeds.
    vᵢ = ice_terminal_velocities(p3, qⁱ_total, props.nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=props.Fˡ, μ=props.μ_ice)
    wⁱ, wⁱₙ, wⁱ_z = vᵢ.mass_weighted, vᵢ.number_weighted, vᵢ.reflectivity_weighted

    # Process rates (heavy, @noinline — compiled as a separate GPU function).
    # Passing `props` lets compute_p3_process_rates skip the redundant
    # rain_slope_parameter / consistent_rime_state / qⁱ_total / Fˡ recomputation.
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props)

    # Tendency extraction
    c_qcl = tendency_ρqᶜˡ(rates, ρ)
    # Prescribed-Nᶜ path: nc is a scheme parameter (not advected); tendency = 0.
    c_ncl = isnothing(p3.aerosol) ? zero(typeof(ρ)) :
            tendency_ρnᶜˡ(rates, ρ, props.Nᶜ, ℳ.qᶜˡ, p3)
    c_qr  = tendency_ρqʳ(rates, ρ)
    c_nr  = tendency_ρnʳ(rates, ρ, props.nⁱ, ℳ.qⁱ, ℳ.nʳ, ℳ.qʳ, p3)
    c_qi  = tendency_ρqⁱ(rates, ρ)
    c_ni  = tendency_ρnⁱ(rates, ρ)
    c_qf  = tendency_ρqᶠ(rates, ρ, Fᶠ)
    c_bf  = tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, ℳ.qⁱ, p3.process_rates)
    # Sixth moment tendency: use tabulated path when ice_integrals table exists, analytic otherwise.
    # Direct call avoids dynamic dispatch on ice_integrals_table(p3) return type in @noinline.
    tendency_ρz_phys = p3_ice_sixth_moment_tendency(ice_integrals_table(p3), p3, rates, ρ, ℳ, props)
    ρz̃ⁱ = ρ * sqrt(max(0, ℳ.zⁱ * props.nⁱ))
    c_zi = z̃ⁱ_tendency(props.nⁱ, props.zⁱ_bounded, tendency_ρz_phys, c_ni,
                        ρz̃ⁱ, p3.process_rates.sink_limiting_timescale)
    c_qwi = tendency_ρqʷⁱ(rates, ρ)
    c_ss  = tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates)
    c_qv  = tendency_ρqᵛ(rates, ρ)
    # Aerosol depletion: every activated cloud droplet removes one from ρnᵃ.
    # Zero in the prescribed-Nᶜ path (rates.ccn_activation_number is 0 there).
    c_na  = tendency_ρnᵃ(rates, ρ)

    FT = typeof(ρ)
    return P3CacheResult{FT}(wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z,
                              c_qcl, c_ncl, c_qr, c_nr, c_qi, c_ni, c_qf, c_bf, c_zi, c_qwi, c_ss, c_qv, c_na)
end

# Kernel entry point: reads OffsetArrays → calls @noinline scalar compute → writes OffsetArrays.
# Keeping array access in the kernel (inlined) and physics in @noinline (separate compilation)
# prevents the GPU compiler from seeing the full P3 physics + OffsetArray access together.
@inline function p3_compute_and_cache!(μ, i, j, k, grid, p3::P3, ρ, 𝒰, constants, velocities)
    @inbounds begin
        ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, 𝒰, velocities)
    end

    r = _p3_scalar_compute(p3, ρ, ℳ, 𝒰, constants)

    @inbounds begin
        μ.wʳ[i, j, k]   = -r.wʳ
        μ.wʳₙ[i, j, k]  = -r.wʳₙ
        μ.wⁱ[i, j, k]   = -r.wⁱ
        μ.wⁱₙ[i, j, k]  = -r.wⁱₙ
        μ.wⁱ_z[i, j, k] = -r.wⁱ_z
        μ.cache_ρqᶜˡ[i, j, k] = r.c_qcl
        μ.cache_ρnᶜˡ[i, j, k] = r.c_ncl
        μ.cache_ρqʳ[i, j, k]  = r.c_qr
        μ.cache_ρnʳ[i, j, k]  = r.c_nr
        μ.cache_ρqⁱ[i, j, k]  = r.c_qi
        μ.cache_ρnⁱ[i, j, k]  = r.c_ni
        μ.cache_ρqᶠ[i, j, k]  = r.c_qf
        μ.cache_ρbᶠ[i, j, k]  = r.c_bf
        μ.cache_ρz̃ⁱ[i, j, k] = r.c_zi
        μ.cache_ρqʷⁱ[i, j, k] = r.c_qwi
        μ.cache_ρsˢᵃᵗ[i, j, k] = r.c_ss
        μ.cache_ρqᵛ[i, j, k]  = r.c_qv
        μ.cache_ρnᵃ[i, j, k]  = r.c_na
    end

    return nothing
end

#####
##### Moisture fractions (state-based)
#####

"""
$(TYPEDSIGNATURES)

Compute moisture mass fractions from P3 microphysical state.

After the moisture refactor, the first argument `qᵛ` is the prognostic
vapor specific humidity (not total moisture). Returns `MoistureMassFractions`
with vapor, liquid (cloud + rain + liquid on ice), and ice components.
"""
@inline function AM.moisture_fractions(::P3, ℳ::P3MicrophysicalState, qᵛ)
    # Total liquid = cloud + rain + liquid on ice
    qˡ = ℳ.qᶜˡ + ℳ.qʳ + ℳ.qʷⁱ

    # Ice (frozen fraction)
    qⁱ = ℳ.qⁱ

    return MoistureMassFractions(qᵛ, qˡ, qⁱ)
end

#####
##### Microphysical velocities (sedimentation)
#####
#
# Terminal velocities are pre-computed in update_microphysical_auxiliaries!
# and stored in diagnostic fields. microphysical_velocities returns NamedTuples
# compatible with Oceananigans' sum_of_velocities.

@inline AM.microphysical_velocities(::P3, μ, name) = nothing  # Default: no sedimentation

@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρnᶜˡ}) = nothing

# Rain mass: mass-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqʳ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wʳ)

# Rain number: number-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρnʳ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wʳₙ)

# Ice mass: mass-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

# Ice number: number-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρnⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱₙ)

# Rime mass: same as ice mass (rime falls with ice)
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqᶠ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

# Rime volume: same as ice mass
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρbᶠ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

# Ice square-root sixth moment: reflectivity-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρz̃ⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ_z)

# Liquid on ice: same as ice mass
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqʷⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

#####
##### Microphysical tendencies
#####
#
# Two paths:
#   1. Grid-based (AtmosphereModel): grid_microphysical_tendency reads from the cache
#      fields populated by update_microphysical_auxiliaries! — one compute_p3_process_rates
#      call per grid point serves all 10 P3 fields.
#   2. Gridless (ParcelModel): microphysical_tendency builds state and computes rates directly.

# Helper to compute P3 rates and extract ice properties from ℳ
@inline function p3_ice_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    FT = typeof(ρ)
    qⁱ_raw = total_ice_mass(ℳ.qⁱ, ℳ.qʷⁱ)
    has_ice_mass = qⁱ_raw > FT(1e-20)
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ, ℳ.qʷⁱ)
    qⁱ_total = max(qⁱ_raw, FT(1e-20))
    Fˡ = liquid_fraction_on_ice(ℳ.qⁱ, ℳ.qʷⁱ)
    nⁱ_global = min(ℳ.nⁱ, p3.process_rates.maximum_ice_number_density / ρ)
    μ_for_limiter = compute_ice_shape_parameter(p3, qⁱ_total, nⁱ_global, ℳ.zⁱ,
                                                rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    nⁱ_bounded = bounded_ice_number(p3, qⁱ_total, nⁱ_global, rime_state.Fᶠ, Fˡ,
                                    rime_state.ρᶠ, μ_for_limiter)
    nⁱ = ifelse(has_ice_mass, nⁱ_bounded, FT(0))
    μ_ice = compute_ice_shape_parameter(p3, qⁱ_total, nⁱ, ℳ.zⁱ, rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    zⁱ_bounded = bound_ice_sixth_moment(p3, qⁱ_total, nⁱ, ℳ.zⁱ, rime_state.Fᶠ, Fˡ, rime_state.ρᶠ, μ_ice)
    T = temperature(𝒰, constants)
    P = 𝒰.reference_pressure
    transport = air_transport_properties(T, P)
    λ_r = rain_slope_parameter(ℳ.qʳ, ℳ.nʳ, p3.process_rates)
    return P3IceProps{FT}(rime_state.qᶠ, rime_state.bᶠ, rime_state.Fᶠ, Fˡ,
                          rime_state.ρᶠ, qⁱ_total, nⁱ, μ_ice, cloud.μ_c, cloud.Nᶜ,
                          zⁱ_bounded, transport.D_v, transport.nu, λ_r)
end

@inline function p3_rates_and_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    # Build ice properties first, then reuse them when computing rates to avoid
    # the redundant rain_slope_parameter / consistent_rime_state / qⁱ_total / Fˡ
    # calls inside compute_p3_process_rates.
    props = p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props)
    return rates, props
end

@inline function p3_ice_sixth_moment_tendency(::Nothing, p3, rates, ρ, ℳ::P3MicrophysicalState, props::P3IceProps)
    return tendency_ρzⁱ(rates, ρ, props.qⁱ_total, props.nⁱ, props.zⁱ_bounded, p3.process_rates)
end

@inline function p3_ice_sixth_moment_tendency(ice_table::P3IceIntegralsTable, p3, rates, ρ, ℳ::P3MicrophysicalState, props::P3IceProps)
    # The fully tabulated Z-tendency overload represents Fortran's dormant
    # log_full3mom branch. Runtime P3 v5.5 uses the active hybrid path: group-1
    # processes reconstruct Z with fixed μ over the same safety timescale used
    # by process-rate limiting, while group-2 sources initialize new ice moments
    # analytically.
    return active_ice_sixth_moment_tendency(ice_table, p3, rates, ρ,
                                            ℳ.qⁱ, ℳ.qʷⁱ, props.nⁱ, props.qᶠ,
                                            props.bᶠ, props.zⁱ_bounded,
                                            props.μ_ice, zero(typeof(ρ)))
end
