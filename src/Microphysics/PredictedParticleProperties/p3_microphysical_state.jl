using Oceananigans: CenterField, Field
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, NormalFlow
using Oceananigans.Fields: ZeroField, ZFaceField
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: ℑzᵃᵃᶜ
using DocStringExtensions: TYPEDSIGNATURES

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

using Breeze.Thermodynamics: MoistureMassFractions, mixture_heat_capacity

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
    "Cell-center vertical velocity [m/s] (retained for the common microphysical-state interface)"
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

# Initial aerosol reservoir for the prognostic-ρnᵃ path.
#
# P3's aerosol distribution is specified per unit mass: `AerosolMode.number_mixing_ratio` is
# [kg⁻¹], so `sum_aerosol_number` is [kg⁻¹] and so are `activated_number` and
# `total_activated_number`. That is the basis the activation cap in
# `prognostic_ccn_activation_rate` compares against, since `nᶜˡ = ρnᶜˡ/ρ` and `nᵃ = ρnᵃ/ρ`
# are both per unit mass, and it is the basis `tendency_ρnᶜˡ` assumes when it multiplies
# `ncnuc` by ρ.
#
# The prognostic field `ρnᵃ` therefore holds ρ nᵃ [m⁻³], and seeding it means multiplying by
# air density here. Skipping the multiplication makes the diagnosed `nᵃ = ρnᵃ / ρ`
# proportional to `1 / ρ` instead of equal to the configured number mixing ratio.
# The prescribed-Nᶜ path has no `ρnᵃ` field, and returns 0 to match the framework default.
@inline AM.initial_aerosol_number(p3::P3) = initial_aerosol_number(p3.aerosol)
@inline initial_aerosol_number(::Nothing) = 0
@inline initial_aerosol_number(aerosol::AerosolActivation) = sum_aerosol_number(aerosol)
@inline AM.initial_aerosol_number_density(p3::P3, ρ) = ρ * AM.initial_aerosol_number(p3)

#####
##### Prognostic field names
#####

# The 3-moment ice switch must be resolvable to a constant tuple at compile time,
# otherwise the resulting Union return type forces the generic GPU
# `extract_microphysical_prognostics` recursion to allocate.
#
# We dispatch on the *type* of `three_moment_shape_table(p3)` — it is `Nothing` in
# 2-moment mode and a concrete table type in 3-moment mode, so the compiler folds
# the helper down to a static tuple per concrete P3 type. The value of
# `predict_supersaturation` is carried by `ProcessRateParameters`' second type parameter
# for the same reason.
#
# Every switch here gates allocation as well as transport: the fields a configuration
# does not use are never created (see `materialize_microphysical_fields`), so an
# unguarded read would be a missing-property error rather than a silent zero.

@inline z̃_prognostic_names(::Nothing) = ()
@inline z̃_prognostic_names(_) = (:ρz̃ⁱ,)

@inline supersaturation_prognostic_names(::ProcessRateParameters{FT, false}) where FT = ()
@inline supersaturation_prognostic_names(::ProcessRateParameters{FT, true}) where FT = (:ρsˢᵃᵗ,)

# Droplet number and aerosol depletion are prognostic iff `p3.aerosol` is a concrete
# `AerosolActivation`. In the prescribed-Nᶜ path (Fortran `log_predictNc = .false.`) `nc`
# is the scheme parameter `p3.cloud.number_concentration` at every microphysics call, so
# no rate reads `ρnᶜˡ` or `ρnᵃ`. Advecting them would integrate transport unrelated to the
# number the physics uses, and `materialize_microphysical_fields` does not even allocate
# them in that path.
@inline cloud_prognostic_names(::Nothing) = (:ρqᶜˡ,)
@inline cloud_prognostic_names(_) = (:ρqᶜˡ, :ρnᶜˡ)

@inline aerosol_prognostic_names(::Nothing) = ()
@inline aerosol_prognostic_names(_) = (:ρnᵃ,)

"""
$(TYPEDSIGNATURES)

Return prognostic field names for the P3 scheme.

- Cloud mass (always): ρqᶜˡ
- Cloud number (only when `aerosol::AerosolActivation` is set): ρnᶜˡ
- Rain: ρqʳ, ρnʳ
- Ice (always): ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ
- Ice (3-moment only): ρz̃ⁱ
- Supersaturation (only when `predict_supersaturation = true`): ρsˢᵃᵗ
- Aerosol (only when `aerosol::AerosolActivation` is set): ρnᵃ
"""
@inline function AM.prognostic_field_names(p3::P3)
    cloud_names = cloud_prognostic_names(p3.aerosol)
    rain_names = (:ρqʳ, :ρnʳ)
    ice_names = (:ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
    z_names = z̃_prognostic_names(three_moment_shape_table(p3))
    ssat_names = supersaturation_prognostic_names(p3.process_rates)
    aero_names = aerosol_prognostic_names(p3.aerosol)

    return tuple(cloud_names..., rain_names..., ice_names...,
                 z_names..., ssat_names..., aero_names...)
end

# True condensate partial densities that contribute to total air mass. Number moments,
# supersaturation, and aerosol are not masses. Rime mass is already contained in total
# ice mass, so including ρqᶠ would count it twice; ρbᶠ and ρz̃ⁱ are ice properties.
@inline AM.condensate_field_names(::P3) = (:ρqᶜˡ, :ρqʳ, :ρqⁱ, :ρqʷⁱ)

#####
##### Negative moisture correction
#####
#
# The advection operator is not positive-definite, so any of P3's prognostic
# densities can come back negative from a stage update. Without this repair the
# negative values persist: the process rates `clamp_positive` what they read, but
# `total_condensate_density` (and through it the total density, buoyancy, and the
# diagnosed thermodynamic state) keeps seeing the raw negative mass.

AM.negative_moisture_correction(p3::P3) = p3.negative_moisture_correction

# Species-borrowing chain, ordered so that each species borrows from the next and the
# last borrows from vapor: ρqʷⁱ ← ρqⁱ ← ρqʳ ← ρqᶜˡ ← ρqᵛ. A negative liquid-on-ice
# deficit is covered by the ice mass that carries it (implied refreezing), negative ice
# by rain (implied freezing), and the warm-phase tail matches the 1- and 2-moment
# schemes. Rime mass and volume are *components* of the ice state rather than
# independent water reservoirs, so they are repaired by clamping instead of borrowing.
#
# Borrowing searches the whole lighter-species tail, so an empty immediate donor does
# not prevent a deficit from reaching available water farther down the chain.
@inline AM.correction_moisture_fields(::P3, μ) = (μ.ρqʷⁱ, μ.ρqⁱ, μ.ρqʳ, μ.ρqᶜˡ)

# Fields that must vanish with the mass they describe. Ice number, rime mass, rime
# volume, and the advected sixth moment are all properties of the ice population, so
# zeroing them when `ρqⁱ` is gone destroys no water. Liquid on ice is deliberately not
# paired with `ρqⁱ`: it is real water, and the whole-particle clip in
# `_p3_phase2_rates` (Fˡ > 0.99) already sheds it to rain when the dry ice mass is gone.
@inline AM.correction_number_mass_pairs(p3::P3, μ) =
    (cloud_number_correction_pairs(p3.aerosol, μ)...,
     (μ.ρnʳ, μ.ρqʳ), (μ.ρnⁱ, μ.ρqⁱ),
     (μ.ρqᶠ, μ.ρqⁱ), (μ.ρbᶠ, μ.ρqⁱ),
     z̃ⁱ_correction_pairs(three_moment_shape_table(p3), μ)...)

# Fields clamped to zero rather than borrowed against, because they carry no water:
# the number moments, the rime properties, the sixth moment, and the unactivated
# aerosol count. `ρsˢᵃᵗ` is excluded — subsaturation is legitimately negative.
@inline AM.correction_number_fields(p3::P3, μ) =
    (cloud_number_correction_fields(p3.aerosol, μ)...,
     μ.ρnʳ, μ.ρnⁱ, μ.ρqᶠ, μ.ρbᶠ,
     z̃ⁱ_correction_fields(three_moment_shape_table(p3), μ)...,
     aerosol_correction_fields(p3.aerosol, μ)...)

# Same compile-time switches as `prognostic_field_names`: dispatch on the *type* of the
# 3-moment table and of the aerosol container so each tuple folds to a constant. The
# prescribed-Nᶜ path has no `ρnᶜˡ`/`ρnᵃ` fields at all, so there is nothing to repair.
@inline z̃ⁱ_correction_pairs(::Nothing, μ) = ()
@inline z̃ⁱ_correction_pairs(_, μ) = ((μ.ρz̃ⁱ, μ.ρqⁱ),)

@inline z̃ⁱ_correction_fields(::Nothing, μ) = ()
@inline z̃ⁱ_correction_fields(_, μ) = (μ.ρz̃ⁱ,)

@inline cloud_number_correction_pairs(::Nothing, μ) = ()
@inline cloud_number_correction_pairs(_, μ) = ((μ.ρnᶜˡ, μ.ρqᶜˡ),)

@inline cloud_number_correction_fields(::Nothing, μ) = ()
@inline cloud_number_correction_fields(_, μ) = (μ.ρnᶜˡ,)

@inline aerosol_correction_fields(::Nothing, μ) = ()
@inline aerosol_correction_fields(_, μ) = (μ.ρnᵃ,)

"""
$(TYPEDSIGNATURES)

Effective cloud droplet number concentration [kg⁻¹] seen by P3's process rates.

In the prescribed-Nᶜ path (`p3.aerosol === nothing`, matching Fortran
`log_predictNc = .false.`), `nc` is always `nccnst_2` at every microphysics call, so this
helper returns that prescribed value and ignores its `ρnᶜˡ` argument. Droplet number is
not a state variable in that configuration: `prognostic_field_names` omits `ρnᶜˡ` and
`materialize_microphysical_fields` does not allocate it.

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
- `ρqᶜˡ`: Cloud liquid mass density
- `ρqʳ`, `ρnʳ`: Rain mass and number densities
- `ρqⁱ`, `ρnⁱ`: Ice mass and number densities
- `ρqᶠ`, `ρbᶠ`: Rime mass and volume densities
- `ρz̃ⁱ`: Advected square-root sixth moment density, where `z̃ⁱ = sqrt(zⁱ nⁱ)`
- `ρqʷⁱ`: Liquid water on ice mass density
- `ρnᶜˡ`, `ρnᵃ`: Cloud number and unactivated aerosol number densities, allocated only
  when `p3.aerosol isa AerosolActivation`. The prescribed-Nᶜ path (Fortran
  `log_predictNc = .false.`) takes droplet number from `p3.cloud.number_concentration`,
  so neither field exists there and neither is advected.

**Diagnostic:**
- `qᵛ`: Vapor specific humidity (mirrors the prognostic vapor field)

**Sedimentation velocities** (`wᶜˡ`, `wᶜˡₙ`, `wʳ`, `wʳₙ`, `wⁱ`, `wⁱₙ`, `wⁱ_z`, `wⁱ_z̃`):
z-Face fields, because the scalar flux divergence consumes them as advecting velocities at
(Center, Center, Face). The surface face carries the precipitation flux out of the domain
unless `precipitation_boundary_condition = ImpenetrableBoundaryCondition()`; the top face
is held at zero so nothing sediments in from above the model top.
"""
function AM.materialize_microphysical_fields(p3::P3, grid, bcs)
    # Create all prognostic fields
    ρqᶜˡ = CenterField(grid)  # Cloud liquid
    ρqʳ  = CenterField(grid)  # Rain mass
    ρnʳ  = CenterField(grid)  # Rain number
    ρqⁱ  = CenterField(grid)  # Ice mass
    ρnⁱ  = CenterField(grid)  # Ice number
    ρqᶠ  = CenterField(grid)  # Rime mass
    ρbᶠ  = CenterField(grid)  # Rime volume
    ρqʷⁱ = CenterField(grid)  # Liquid on ice

    # Diagnostic mixing ratio / number-concentration fields
    # (updated each step in update_microphysical_auxiliaries!, matching the Kessler pattern)
    qᶜˡ = CenterField(grid)  # Cloud liquid specific humidity [kg/kg]
    qʳ  = CenterField(grid)  # Rain specific humidity [kg/kg]
    nʳ  = CenterField(grid)  # Rain number concentration [kg⁻¹]
    qⁱ  = CenterField(grid)  # Ice specific humidity [kg/kg]
    nⁱ  = CenterField(grid)  # Ice number concentration [kg⁻¹]
    qᶠ  = CenterField(grid)  # Rime mass mixing ratio [kg/kg]
    bᶠ  = CenterField(grid)  # Rime volume [m³/kg]
    qʷⁱ = CenterField(grid)  # Liquid water on ice [kg/kg]

    # Diagnostic field for vapor
    qᵛ = CenterField(grid)

    # Sedimentation velocity fields (pre-computed once per RK-stage tendency evaluation).
    # These are *advecting* velocities: the scalar flux divergence reads them at
    # (Center, Center, Face) via `Az_qᶜᶜᶠ(i, j, k, grid, w) = Azᶜᶜᶠ(i, j, k, grid) * w[i, j, k]`,
    # so they must live at z-Faces. `bottom = nothing` leaves the kernel-written surface
    # value untouched by `fill_halo_regions!` (the surface boundary condition is applied in
    # `write_p3_fall_speeds!` instead), while the default impenetrable top boundary holds
    # `w[i, j, Nz+1] = 0` so no precipitation falls in through the model top.
    face_bcs = FieldBoundaryConditions(grid, (Center(), Center(), Face()); bottom=nothing)
    wᶜˡ = ZFaceField(grid; boundary_conditions=face_bcs) # Cloud mass-weighted terminal velocity
    wᶜˡₙ = ZFaceField(grid; boundary_conditions=face_bcs) # Cloud number-weighted terminal velocity
    wʳ  = ZFaceField(grid; boundary_conditions=face_bcs)  # Rain mass-weighted terminal velocity
    wʳₙ = ZFaceField(grid; boundary_conditions=face_bcs) # Rain number-weighted terminal velocity
    wⁱ  = ZFaceField(grid; boundary_conditions=face_bcs)  # Ice mass-weighted terminal velocity
    wⁱₙ = ZFaceField(grid; boundary_conditions=face_bcs) # Ice number-weighted terminal velocity
    wⁱ_z = ZFaceField(grid; boundary_conditions=face_bcs) # Ice reflectivity-weighted terminal velocity

    # Microphysical tendency cache (written once per RK-stage tendency evaluation,
    # then added to G). Storing the microphysics-only contribution avoids one
    # compute_p3_process_rates call per prognostic field.
    cache_ρqᶜˡ = CenterField(grid)
    cache_ρqʳ  = CenterField(grid)
    cache_ρnʳ  = CenterField(grid)
    cache_ρqⁱ  = CenterField(grid)
    cache_ρnⁱ  = CenterField(grid)
    cache_ρqᶠ  = CenterField(grid)
    cache_ρbᶠ  = CenterField(grid)
    cache_ρqʷⁱ = CenterField(grid)
    cache_ρqᵛ  = CenterField(grid)

    # Hallett–Mossop uses the temperature at the lowest active atmospheric cell.
    # Store one value per column rather than assuming that local k=1 is active.
    surface_temperature = Field{Center, Center, Nothing}(grid)

    fields = (; ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ,
                qᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ, qᵛ,
                wᶜˡ, wᶜˡₙ, wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z,
                cache_ρqᶜˡ, cache_ρqʳ, cache_ρnʳ, cache_ρqⁱ, cache_ρnⁱ,
                cache_ρqᶠ, cache_ρbᶠ, cache_ρqʷⁱ, cache_ρqᵛ,
                surface_temperature)

    return merge(fields,
                 aerosol_activation_fields(p3.aerosol, grid),
                 ice_sixth_moment_fields(three_moment_shape_table(p3), grid, face_bcs),
                 supersaturation_fields(p3.process_rates, grid))
end

# Optional field groups. Each switch gates allocation, not just transport, so a
# configuration never carries memory for state it does not use. All three dispatch on a
# *type* (`Nothing` / a concrete table / `ProcessRateParameters{FT, PS}`) so the
# merged NamedTuple is a compile-time constant, which lets the read sites fold their
# guards away.

# Droplet number and unactivated aerosol. The prescribed-Nᶜ path (Fortran
# `log_predictNc = .false.`) takes `nc` from `p3.cloud.number_concentration` at every call
# and never reads `ρnᶜˡ` or `ρnᵃ`.
@inline aerosol_activation_fields(::Nothing, grid) = (;)

@inline aerosol_activation_fields(_, grid) =
    (; ρnᶜˡ = CenterField(grid),        # Cloud number density [1/m³]
       ρnᵃ = CenterField(grid),         # Unactivated aerosol number density [1/m³]
       nᶜˡ = CenterField(grid),         # Cloud number concentration [kg⁻¹]
       nᵃ = CenterField(grid),          # Unactivated aerosol [kg⁻¹]
       cache_ρnᶜˡ = CenterField(grid),
       cache_ρnᵃ = CenterField(grid))

# Ice sixth moment. In 2-moment mode `zⁱ` collapses to zero, so the moment, its advected
# square root, and the sqrt-moment fall speed that transports it are all dead weight.
# `wⁱ_z` stays: it is the reflectivity-weighted fall speed diagnostic, not `ρz̃ⁱ`'s
# advecting velocity.
@inline ice_sixth_moment_fields(::Nothing, grid, face_bcs) = (;)

@inline ice_sixth_moment_fields(_, grid, face_bcs) =
    (; ρz̃ⁱ = CenterField(grid),        # Advected square-root sixth moment
       zⁱ = CenterField(grid),          # Ice sixth moment [m⁶/kg]
       z̃ⁱ = CenterField(grid),          # √(zⁱ nⁱ)
       cache_ρz̃ⁱ = CenterField(grid),
       wⁱ_z̃ = ZFaceField(grid; boundary_conditions=face_bcs))

# The sqrt-moment fall speed needs a stage-local halo fill, so
# `prepare_microphysical_tendencies!` splices it into the velocity tuple.
@inline sqrt_moment_velocities(::Nothing, μ) = ()
@inline sqrt_moment_velocities(_, μ) = (μ.wⁱ_z̃,)

# Predicted supersaturation (Fortran `log_predictSsat`, `.false.` by default). With the
# switch off every rate that would touch `sˢᵃᵗ` is gated to zero, so the prognostic
# carries no information.
@inline supersaturation_fields(::ProcessRateParameters{FT, false}, grid) where FT = (;)

@inline supersaturation_fields(::ProcessRateParameters{FT, true}, grid) where FT =
    (; ρsˢᵃᵗ = CenterField(grid), sˢᵃᵗ = CenterField(grid),
       cache_ρsˢᵃᵗ = CenterField(grid))

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
    # ρnᶜˡ is absent unless the aerosol-activation path is enabled; the prescribed-Nᶜ
    # branch of `effective_cloud_droplet_number` ignores the value it is handed.
    nᶜˡ = effective_cloud_droplet_number(p3, get_or_default(μ, Val(:ρnᶜˡ), 0 * ρ), ρ)
    qʳ  = μ.ρqʳ / ρ
    nʳ  = μ.ρnʳ / ρ
    qⁱ  = μ.ρqⁱ / ρ
    nⁱ  = μ.ρnⁱ / ρ
    # Fortran advects z̃ = √(z·N) and converts to physical z at microphysics entry:
    #   where (nitot > 0) zitot = zitot**2 / nitot; elsewhere zitot = 0
    # ρz̃ⁱ stores the advected variable z̃; convert to physical z = z̃²/N for internal use.
    # In 2-moment mode ρz̃ⁱ is absent from `μ`; treat it as 0 (zⁱ then collapses to 0).
    FT = typeof(ρ)
    z̃ⁱ  = get_or_default(μ, Val(:ρz̃ⁱ), 0 * ρ) / ρ
    zⁱ  = ifelse(nⁱ > FT(1e-20), z̃ⁱ^2 / nⁱ, 0 * z̃ⁱ)
    qʷⁱ = μ.ρqʷⁱ / ρ
    rime_state = consistent_rime_state(p3, qⁱ, μ.ρqᶠ / ρ, μ.ρbᶠ / ρ, qʷⁱ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    # ρsˢᵃᵗ is absent unless predicted supersaturation is enabled; default to 0.
    sˢᵃᵗ = get_or_default(μ, Val(:ρsˢᵃᵗ), 0 * ρ) / ρ
    # ρnᵃ is absent unless prognostic-aerosol path is enabled; default to 0.
    nᵃ = get_or_default(μ, Val(:ρnᵃ), 0 * ρ) / ρ
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, nᵃ, vertical_velocity(velocities, FT))
end

# Disambiguation for P3 with Nothing or empty microphysical fields
@inline AM.microphysical_state(::P3, ρ, ::Nothing, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))
@inline AM.microphysical_state(::P3, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))

# Apply the same rime-state writeback to parcel prognostics that
# `AM.update_microphysical_fields!` applies to grid fields.
@inline function AM.postprocess_microphysical_prognostics(p3::P3, μ::NamedTuple, ρ)
    qⁱ = μ.ρqⁱ / ρ
    qᶠ = μ.ρqᶠ / ρ
    bᶠ = μ.ρbᶠ / ρ
    qʷⁱ = μ.ρqʷⁱ / ρ
    rime_state = consistent_rime_state(p3, qⁱ, qᶠ, bᶠ, qʷⁱ)
    return merge(μ, (; ρqᶠ = ρ * rime_state.qᶠ,
                       ρbᶠ = ρ * rime_state.bᶠ))
end

# Droplet number and unactivated aerosol on the grid. Dispatch on the *type* of
# `p3.aerosol` rather than indexing behind a runtime branch, because in the prescribed-Nᶜ
# path the `ρnᶜˡ`/`ρnᵃ` fields do not exist at all.
@inline grid_cloud_droplet_number(p3::P3, ::Nothing, μ, i, j, k, ρ) =
    p3.cloud.number_concentration / ρ
@inline grid_cloud_droplet_number(p3::P3, _, μ, i, j, k, ρ) =
    @inbounds μ.ρnᶜˡ[i, j, k] / ρ

@inline grid_aerosol_number(::Nothing, μ, i, j, k, ρ) = 0 * ρ
@inline grid_aerosol_number(_, μ, i, j, k, ρ) = @inbounds μ.ρnᵃ[i, j, k] / ρ

# Same for the optional sixth moment and supersaturation prognostics: absent in 2-moment
# mode and with predicted supersaturation off, where both collapse to zero anyway.
@inline grid_sixth_moment(::Nothing, μ, i, j, k, ρ) = 0 * ρ
@inline grid_sixth_moment(_, μ, i, j, k, ρ) = @inbounds μ.ρz̃ⁱ[i, j, k] / ρ

@inline grid_supersaturation(::ProcessRateParameters{FT, false}, μ, i, j, k, ρ) where FT = 0 * ρ
@inline grid_supersaturation(::ProcessRateParameters{FT, true}, μ, i, j, k, ρ) where FT =
    @inbounds μ.ρsˢᵃᵗ[i, j, k] / ρ

@inline function AM.grid_microphysical_state(i, j, k, grid, p3::P3, μ, ρ, 𝒰, velocities)
    nᶜˡ = grid_cloud_droplet_number(p3, p3.aerosol, μ, i, j, k, ρ)
    @inbounds begin
        qᶜˡ = μ.ρqᶜˡ[i, j, k] / ρ
        qʳ  = μ.ρqʳ[i, j, k] / ρ
        nʳ  = μ.ρnʳ[i, j, k] / ρ
        qⁱ  = μ.ρqⁱ[i, j, k] / ρ
        nⁱ  = μ.ρnⁱ[i, j, k] / ρ
        qʷⁱ = μ.ρqʷⁱ[i, j, k] / ρ
    end
    FT = typeof(ρ)
    z̃ⁱ  = grid_sixth_moment(three_moment_shape_table(p3), μ, i, j, k, ρ)
    zⁱ  = ifelse(nⁱ > FT(1e-20), z̃ⁱ^2 / nⁱ, 0 * z̃ⁱ)
    rime_state = consistent_rime_state(p3, qⁱ, @inbounds(μ.ρqᶠ[i, j, k]) / ρ, @inbounds(μ.ρbᶠ[i, j, k]) / ρ, qʷⁱ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    sˢᵃᵗ = grid_supersaturation(p3.process_rates, μ, i, j, k, ρ)
    nᵃ   = grid_aerosol_number(p3.aerosol, μ, i, j, k, ρ)
    w = interpolate_w_to_center(grid, i, j, k, velocities.w, FT)
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, nᵃ, w)
end

# Clamp the prognostic `ρz̃ⁱ` to its physically consistent maximum
# (Fortran-parity `apply_mui_bounds_to_zi`). Without this writeback, sources
# accumulate ρz̃ⁱ above the physical bound while tendency math sees only the
# bounded zⁱ — there is no restoring force and the prognostic drifts.
# 2-moment (no `three_moment_shape_table`) is a compile-time no-op.
struct P3IceMomentBounds{FT}
    qⁱ_total :: FT
    nⁱ_diagnostic :: FT
    nⁱ :: FT
    μ_ice :: FT
    ρ_mean :: FT
    zⁱ :: FT
end

@inline function p3_ice_moment_bounds(p3::P3, ρ, qⁱ_raw, nⁱ_raw, zⁱ,
                                      Fᶠ, Fˡ, ρᶠ)
    FT = typeof(ρ)
    has_ice_mass = qⁱ_raw > FT(1e-20)
    qⁱ_total = max(qⁱ_raw, FT(1e-20))
    nⁱ_global = min(clamp_positive(nⁱ_raw),
                    p3.process_rates.maximum_ice_number_density / ρ)
    nⁱ_diagnostic = max(nⁱ_global, p3.minimum_number_mixing_ratio)
    μ_ice = compute_ice_shape_parameter(p3, qⁱ_total, nⁱ_diagnostic, zⁱ,
                                        Fᶠ, Fˡ, ρᶠ)
    ρ_mean = ice_mean_density(p3, qⁱ_total, nⁱ_diagnostic, zⁱ,
                              Fᶠ, Fˡ, ρᶠ, μ_ice)
    nⁱ_bounded = bounded_ice_number(p3, qⁱ_total, nⁱ_diagnostic,
                                    Fᶠ, Fˡ, ρᶠ, μ_ice)
    nⁱ = ifelse(has_ice_mass, nⁱ_bounded, FT(0))
    zⁱ_bounded = bound_ice_sixth_moment_with_density(qⁱ_total, nⁱ, zⁱ,
                                                     ρ_mean)
    return P3IceMomentBounds{FT}(qⁱ_total, nⁱ_diagnostic, nⁱ, μ_ice,
                                 ρ_mean, zⁱ_bounded)
end

# Write the consistent rime state back to the prognostic densities. `grid_microphysical_state`
# already passes `ρqᶠ`/`ρbᶠ` through `consistent_rime_state`, so `ℳ.qᶠ`/`ℳ.bᶠ` are what every
# process rate sees, and (through `μ.qᶠ`/`μ.bᶠ`) what the scalar transport operators advect:
# the correction zeroes `qᶠ` when the rime volume vanishes, zeroes both below the rime-mass
# floor, and caps `qᶠ` at the dry ice mass.
#
# Without this writeback the prognostic keeps the uncorrected mass while the advected specific
# field reports the corrected one, so hidden rime receives no transport while the ice carrying
# it moves away, then reappears once the correction stops firing. This is the same drift that
# `clamp_ice_sixth_moment!` prevents for `ρz̃ⁱ`.
#
# Rime mass is a *component* of the ice mass rather than an independent water reservoir
# (`condensate_field_names` excludes `ρqᶠ`), so the clip moves the rime fraction Fᶠ and never
# the total water or the total density.
@inline function clamp_rime_state!(μ, i, j, k, ρ, ℳ::P3MicrophysicalState)
    @inbounds μ.ρqᶠ[i, j, k] = ρ * ℳ.qᶠ
    @inbounds μ.ρbᶠ[i, j, k] = ρ * ℳ.bᶠ
    return nothing
end

@inline clamp_ice_sixth_moment!(μ, i, j, k, p3::P3, ρ, ℳ) =
    clamp_ice_sixth_moment_dispatch(three_moment_shape_table(p3), μ, i, j, k, p3, ρ, ℳ)

@inline clamp_ice_sixth_moment_dispatch(::Nothing, μ, i, j, k, p3::P3, ρ, ℳ) = ℳ

@inline function clamp_ice_sixth_moment_dispatch(::P3ThreeMomentShapeTable,
                                                   μ, i, j, k, p3::P3, ρ, ℳ)
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)
    qⁱ_total = total_ice_mass(ℳ.qⁱ, qʷⁱ)
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(ℳ.qⁱ, qʷⁱ)
    bounds = p3_ice_moment_bounds(p3, ρ, qⁱ_total, ℳ.nⁱ, ℳ.zⁱ,
                                  rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    zⁱ_bounded = bounds.zⁱ
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
        clamp_rime_state!(μ, i, j, k, ρ, ℳ_raw)
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
# Only writes basic specific quantities and vapor. Terminal velocities are deferred to
# prepare_microphysical_tendencies! before sedimentation. Process-rate caches are filled
# in compute_microphysical_tendencies! from the current state with an adiabatic-only driver,
# avoiding GPU compilation failure from force-inlining ~1000 lines of P3 physics into
# the thermodynamic kernel.
@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, p3::P3, ℳ::P3MicrophysicalState, ρ, 𝒰, constants)
    @inbounds μ.qᵛ[i, j, k]  = 𝒰.moisture_mass_fractions.vapor
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.qʳ[i, j, k]  = ℳ.qʳ
    @inbounds μ.nʳ[i, j, k]  = ℳ.nʳ
    @inbounds μ.qⁱ[i, j, k]  = ℳ.qⁱ
    @inbounds μ.nⁱ[i, j, k]  = ℳ.nⁱ
    @inbounds μ.qᶠ[i, j, k]  = ℳ.qᶠ
    @inbounds μ.bᶠ[i, j, k]  = ℳ.bᶠ
    @inbounds μ.qʷⁱ[i, j, k] = ℳ.qʷⁱ
    write_cloud_number_diagnostics!(μ, i, j, k, p3.aerosol, ℳ)
    write_sixth_moment_diagnostics!(μ, i, j, k, three_moment_shape_table(p3), ρ, ℳ)
    write_supersaturation_diagnostic!(μ, i, j, k, p3.process_rates, ℳ)

    return nothing
end

# Diagnostics for the optional prognostic groups. Each specific field is what
# `compute_tendencies!` advects to assemble the matching ∂(ρx)/∂t, so it has to equal
# `ρx / ρ` — which `ℳ.nᶜˡ` does in the aerosol-activation path, where the field exists.
# Configurations without the prognostic have no field to write.
@inline write_cloud_number_diagnostics!(μ, i, j, k, ::Nothing, ℳ) = nothing

@inline function write_cloud_number_diagnostics!(μ, i, j, k, _, ℳ)
    @inbounds μ.nᶜˡ[i, j, k] = ℳ.nᶜˡ
    @inbounds μ.nᵃ[i, j, k]  = ℳ.nᵃ
    return nothing
end

@inline write_sixth_moment_diagnostics!(μ, i, j, k, ::Nothing, ρ, ℳ) = nothing

@inline function write_sixth_moment_diagnostics!(μ, i, j, k, _, ρ, ℳ)
    @inbounds μ.zⁱ[i, j, k] = ℳ.zⁱ
    @inbounds μ.z̃ⁱ[i, j, k] = μ.ρz̃ⁱ[i, j, k] / ρ
    return nothing
end

@inline write_supersaturation_diagnostic!(
    μ, i, j, k, ::ProcessRateParameters{FT, false}, ℳ
) where FT = nothing

@inline function write_supersaturation_diagnostic!(
    μ, i, j, k, ::ProcessRateParameters{FT, true}, ℳ
) where FT
    @inbounds μ.sˢᵃᵗ[i, j, k] = ℳ.sˢᵃᵗ
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
    # Number and Table-3 mean density diagnosed before the lambda limiter. Fortran
    # retains these values for volume-equivalent diameter and sixth-moment bounds.
    nⁱ_diagnostic :: FT
    ρ_mean :: FT
    μ_ice :: FT
    μ_cloud :: FT
    λ_cloud :: FT
    Nᶜ :: FT
    zⁱ_bounded :: FT
    D_v :: FT
    nu :: FT
    λ_r :: FT
end

# GPU-safe return structs (NamedTuples require jl_f_tuple on GPU).
struct P3FallSpeedResult{FT}
    wᶜˡ :: FT; wᶜˡₙ :: FT; wʳ :: FT; wʳₙ :: FT; wⁱ :: FT; wⁱₙ :: FT; wⁱ_z :: FT
end

struct P3TendencyCacheResult{FT}
    c_qcl :: FT; c_ncl :: FT; c_qr :: FT; c_nr :: FT
    c_qi :: FT; c_ni :: FT; c_qf :: FT; c_bf :: FT
    c_zi :: FT; c_qwi :: FT; c_ss :: FT; c_qv :: FT
    c_na :: FT
end

struct P3CacheResult{FT}
    wᶜˡ :: FT; wᶜˡₙ :: FT; wʳ :: FT; wʳₙ :: FT; wⁱ :: FT; wⁱₙ :: FT; wⁱ_z :: FT
    c_qcl :: FT; c_ncl :: FT; c_qr :: FT; c_nr :: FT
    c_qi :: FT; c_ni :: FT; c_qf :: FT; c_bf :: FT
    c_zi :: FT; c_qwi :: FT; c_ss :: FT; c_qv :: FT
    c_na :: FT
end

@inline function z̃ⁱ_tendency(nⁱ, zⁱ, tendency_ρz_phys, tendency_ρn)
    FT = typeof(nⁱ + zⁱ + tendency_ρz_phys + tendency_ρn)
    z_times_n = zⁱ * nⁱ
    existing_distribution = (zⁱ > 0) & (nⁱ > 0) & (z_times_n > 0)

    z̃ = sqrt(max(0, z_times_n))
    numerator = nⁱ * tendency_ρz_phys + zⁱ * tendency_ρn
    existing_tendency = safe_divide(numerator, 2 * z̃, zero(FT))

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

# Terminal velocities must be available before scalar tendency assembly, while
# process rates need the resolved host tendencies assembled during that step.
# Keep both computations scalar and return concrete structs for GPU compilation.
@noinline function p3_fall_speed_compute(p3::P3, ρ, ℳ::P3MicrophysicalState,
                                          props::P3IceProps, constants)
    Fᶠ = props.Fᶠ
    ρᶠ = props.ρᶠ

    # Cloud terminal velocities — Fortran sediments cloud mass and number with
    # DSD-integrated Stokes velocities in sedimentation_liquid(liq_type = 1).
    vᶜ = cloud_terminal_velocities(p3, ℳ.qᶜˡ, ρ, props.nu, props.μ_cloud, props.λ_cloud,
                                   constants)
    wᶜˡ = vᶜ.mass_weighted
    wᶜˡₙ = vᶜ.number_weighted

    # Rain terminal velocities — fused call shares λ_r, ρ_correction, log10(λ_r)
    # across the two 1D table lookups (mass- and number-weighted).
    vᵣ = rain_terminal_velocities(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    wʳ   = vᵣ.mass_weighted
    wʳₙ  = vᵣ.number_weighted
    # Fortran parity: after impose_max_Ni (microphy_p3.f90:2812/4390/4937) the nitot
    # array is capped in place, so all downstream math — process rates, terminal
    # velocities, Z tendency, reflectivity — sees the same value. Mirror that here by
    # using props.nⁱ (= min(ℳ.nⁱ, max_Ni/ρ)) wherever Fortran would see capped nitot.
    # Fortran indexes the ice fall-speed lookup with qitot. `props.qⁱ_total`
    # includes liquid coating only when that prognostic mode is active, so a
    # stale restart coating cannot affect the non-liquid-fraction fast branch.
    qⁱ_total = props.qⁱ_total
    # Fused call: shares m̄, ρ_correction, log(m̄), and the 5D interpolation indices
    # across mass-, number-, and reflectivity-weighted fall speeds.
    vᵢ = ice_terminal_velocities(p3, qⁱ_total, props.nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=props.Fˡ, μ=props.μ_ice)
    wⁱ, wⁱₙ, wⁱ_z = vᵢ.mass_weighted, vᵢ.number_weighted, vᵢ.reflectivity_weighted

    FT = typeof(ρ)
    return P3FallSpeedResult{FT}(wᶜˡ, wᶜˡₙ, wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z)
end

@noinline function p3_tendency_compute(p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰,
                                        constants, props::P3IceProps,
                                        surface_temperature, temperature_tendency,
                                        vapor_tendency)
    Fᶠ = props.Fᶠ
    ρᶠ = props.ρᶠ
    # Process rates (heavy, @noinline — compiled as a separate GPU function).
    # Passing `props` lets compute_p3_process_rates skip the redundant
    # rain_slope_parameter / consistent_rime_state / qⁱ_total / Fˡ recomputation.
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, props,
                                     surface_temperature, temperature_tendency,
                                     vapor_tendency)

    # Tendency extraction
    c_qcl = tendency_ρqᶜˡ(rates, ρ)
    # Prescribed-Nᶜ path: nc is a scheme parameter (not advected); tendency = 0.
    c_ncl = isnothing(p3.aerosol) ? zero(typeof(ρ)) :
            tendency_ρnᶜˡ(rates, ρ, props.Nᶜ, ℳ.qᶜˡ, p3)
    c_qr  = tendency_ρqʳ(rates, ρ, p3.process_rates)
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
    c_qwi = tendency_ρqʷⁱ(rates, ρ, p3.process_rates)
    c_ss  = tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates)
    c_qv  = tendency_ρqᵛ(rates, ρ)
    # Aerosol depletion: every activated cloud droplet removes one from ρnᵃ.
    # Zero in the prescribed-Nᶜ path (rates.ccn_activation_number is 0 there).
    c_na  = tendency_ρnᵃ(rates, ρ)

    FT = typeof(ρ)
    return P3TendencyCacheResult{FT}(c_qcl, c_ncl, c_qr, c_nr, c_qi, c_ni,
                                     c_qf, c_bf, c_zi, c_qwi, c_ss, c_qv, c_na)
end


# Adiabatic temperature tendency from the grid or parcel vertical velocity, used as
# P3's external thermodynamic forcing on both paths. The accompanying resolved-vapor
# tendency is zero. On the Eulerian grid path this deliberately omits resolved
# transport, turbulent mixing, radiation, and user forcing from the diffusional-growth
# driver; see `_p3_compute_tendency_cache_kernel!`.
@inline function p3_adiabatic_temperature_tendency(ℳ::P3MicrophysicalState, 𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(𝒰.moisture_mass_fractions, constants)
    return -constants.gravitational_acceleration * ℳ.w / cᵖᵐ
end

# Combined scalar helper retained for the gridless/test path.
@noinline function _p3_scalar_compute(p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    surface_temperature = temperature(𝒰, constants)
    temperature_tendency = p3_adiabatic_temperature_tendency(ℳ, 𝒰, constants)
    return _p3_scalar_compute(p3, ρ, ℳ, 𝒰, constants, surface_temperature,
                              temperature_tendency, zero(ρ))
end


@noinline function _p3_scalar_compute(p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants,
                                      surface_temperature)
    return _p3_scalar_compute(p3, ρ, ℳ, 𝒰, constants, surface_temperature,
                              zero(ρ), zero(ρ))
end


@noinline function _p3_scalar_compute(p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants,
                                      surface_temperature, temperature_tendency,
                                      vapor_tendency)
    props = p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
    velocities = p3_fall_speed_compute(p3, ρ, ℳ, props, constants)
    tendencies = p3_tendency_compute(p3, ρ, ℳ, 𝒰, constants, props,
                                      surface_temperature, temperature_tendency,
                                      vapor_tendency)
    FT = typeof(ρ)
    return P3CacheResult{FT}(velocities.wᶜˡ, velocities.wᶜˡₙ,
                              velocities.wʳ, velocities.wʳₙ,
                              velocities.wⁱ, velocities.wⁱₙ, velocities.wⁱ_z,
                              tendencies.c_qcl, tendencies.c_ncl,
                              tendencies.c_qr, tendencies.c_nr,
                              tendencies.c_qi, tendencies.c_ni,
                              tendencies.c_qf, tendencies.c_bf,
                              tendencies.c_zi, tendencies.c_qwi,
                              tendencies.c_ss, tendencies.c_qv, tendencies.c_na)
end


#####
##### Surface precipitation boundary condition
#####
#
# The fall-speed fields are at (Center, Center, Face), so index `k = 1` is the bottom
# face of the domain and carries the surface precipitation flux. `nothing` (the default)
# keeps the diagnosed fall speed there, so precipitation leaves the domain through an open
# surface. An `ImpenetrableBoundaryCondition` zeroes it, so precipitation instead
# accumulates in the lowest cell. Mirrors `bottom_terminal_velocity` in the one-moment
# scheme; dispatch is on the boundary-condition *type*, so it folds to a constant per
# concrete P3 type and stays GPU-safe.
#
# TODO: Use the lowest *active* face of each column rather than `k = 1` so the condition
# also applies over an immersed bottom. `compute_p3_surface_temperature!` already performs
# that column scan for Hallett-Mossop; the one-moment scheme has the same limitation.

const P3ImpenetrableBoundaryCondition = BoundaryCondition{<:NormalFlow, Nothing}

@inline bottom_fall_speed_factor(::Nothing, FT) = one(FT)
@inline bottom_fall_speed_factor(::P3ImpenetrableBoundaryCondition, FT) = zero(FT)

@inline function write_p3_fall_speeds!(μ, i, j, k, p3::P3,
                                       result::P3FallSpeedResult{FT}) where FT
    # `k` indexes the bottom face of cell `k`. Sedimentation is always downward, so the
    # donor cell for that face is cell `k` itself and the fall speed diagnosed at centre
    # `k` is the upwind velocity there. The top face (`k = Nz+1`) is outside the `:xyz`
    # launch region and is held at zero by the impenetrable top boundary condition.
    surface = ifelse(k == 1,
                     bottom_fall_speed_factor(p3.precipitation_boundary_condition, FT),
                     one(FT))
    @inbounds begin
        μ.wᶜˡ[i, j, k]  = -surface * result.wᶜˡ
        μ.wᶜˡₙ[i, j, k] = -surface * result.wᶜˡₙ
        μ.wʳ[i, j, k]   = -surface * result.wʳ
        μ.wʳₙ[i, j, k]  = -surface * result.wʳₙ
        μ.wⁱ[i, j, k]   = -surface * result.wⁱ
        μ.wⁱₙ[i, j, k]  = -surface * result.wⁱₙ
        μ.wⁱ_z[i, j, k] = -surface * result.wⁱ_z
    end
    write_p3_sqrt_moment_fall_speed!(μ, i, j, k, three_moment_shape_table(p3),
                                     surface, result)
    return nothing
end


@inline function write_p3_tendency_cache!(μ, i, j, k, p3::P3, result::P3TendencyCacheResult)
    @inbounds begin
        μ.cache_ρqᶜˡ[i, j, k] = result.c_qcl
        μ.cache_ρqʳ[i, j, k]  = result.c_qr
        μ.cache_ρnʳ[i, j, k]  = result.c_nr
        μ.cache_ρqⁱ[i, j, k]  = result.c_qi
        μ.cache_ρnⁱ[i, j, k]  = result.c_ni
        μ.cache_ρqᶠ[i, j, k]  = result.c_qf
        μ.cache_ρbᶠ[i, j, k]  = result.c_bf
        μ.cache_ρqʷⁱ[i, j, k] = result.c_qwi
        μ.cache_ρqᵛ[i, j, k]  = result.c_qv
    end
    write_p3_cloud_number_cache!(μ, i, j, k, p3.aerosol, result)
    write_p3_sixth_moment_cache!(μ, i, j, k, three_moment_shape_table(p3), result)
    write_p3_supersaturation_cache!(μ, i, j, k, p3.process_rates, result)
    return nothing
end

# Configurations without a prognostic have no cache to fill and no `Gⁿ` slot to add it to.
# The corresponding `result` entries are zero there anyway.
@inline write_p3_cloud_number_cache!(μ, i, j, k, ::Nothing, result) = nothing

@inline function write_p3_cloud_number_cache!(μ, i, j, k, _, result)
    @inbounds μ.cache_ρnᶜˡ[i, j, k] = result.c_ncl
    @inbounds μ.cache_ρnᵃ[i, j, k] = result.c_na
    return nothing
end

@inline write_p3_sixth_moment_cache!(μ, i, j, k, ::Nothing, result) = nothing

@inline function write_p3_sixth_moment_cache!(μ, i, j, k, _, result)
    @inbounds μ.cache_ρz̃ⁱ[i, j, k] = result.c_zi
    return nothing
end

@inline write_p3_supersaturation_cache!(
    μ, i, j, k, ::ProcessRateParameters{FT, false}, result
) where FT = nothing

@inline function write_p3_supersaturation_cache!(
    μ, i, j, k, ::ProcessRateParameters{FT, true}, result
) where FT
    @inbounds μ.cache_ρsˢᵃᵗ[i, j, k] = result.c_ss
    return nothing
end

# `wⁱ_z̃` is `ρz̃ⁱ`'s advecting velocity, so it exists only in 3-moment mode. The
# characteristic is the mean of the Z- and N-weighted speeds (see `microphysical_velocities`).
@inline write_p3_sqrt_moment_fall_speed!(μ, i, j, k, ::Nothing, surface, result) = nothing

@inline function write_p3_sqrt_moment_fall_speed!(μ, i, j, k, _, surface, result)
    @inbounds μ.wⁱ_z̃[i, j, k] = -surface * (result.wⁱ_z + result.wⁱₙ) / 2
    return nothing
end


@inline function p3_compute_fall_speeds!(μ, i, j, k, grid, p3::P3, ρ, 𝒰,
                                          constants, velocities)
    ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, 𝒰, velocities)
    props = p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
    result = p3_fall_speed_compute(p3, ρ, ℳ, props, constants)
    return write_p3_fall_speeds!(μ, i, j, k, p3, result)
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
# Terminal velocities are pre-computed in prepare_microphysical_tendencies!
# and stored in diagnostic fields. microphysical_velocities returns NamedTuples
# compatible with Oceananigans' sum_of_velocities.

@inline AM.microphysical_velocities(::P3, μ, name) = nothing  # Default: no sedimentation

# Cloud mass: mass-weighted Stokes fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqᶜˡ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wᶜˡ)

# Cloud number: number-weighted Stokes fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρnᶜˡ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wᶜˡₙ)

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

# Ice square-root sixth moment. From d√(ZN) = ½√(N/Z)dZ + ½√(Z/N)dN,
# the local sedimentation characteristic is the mean of the Z- and N-weighted
# particle velocities. Keeping it in the normal scalar path also preserves the
# configured nonlinear and implicit advection semantics.
# TODO: Replace this characteristic approximation with coupled Z- and N-flux
# divergences once the host tracer interface can assemble one tendency from two
# moment fluxes; one velocity is not exact for independently size-sorted profiles.
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρz̃ⁱ}) =
    (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ_z̃)

# Liquid on ice: same as ice mass
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqʷⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

#####
##### Microphysical tendencies
#####
#
# Two paths:
#   1. Grid-based (AtmosphereModel): the fused driver fills the cache once from the
#      current state using the adiabatic-only forcing and adds every cached tendency to G.
#   2. Gridless (ParcelModel): microphysical_tendency builds state and computes rates directly.

# Helper to compute P3 rates and extract ice properties from ℳ
@inline function p3_ice_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    FT = typeof(ρ)
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)
    qⁱ_raw = total_ice_mass(ℳ.qⁱ, qʷⁱ)
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ, qʷⁱ)
    Fˡ = liquid_fraction_on_ice(ℳ.qⁱ, qʷⁱ)
    bounds = p3_ice_moment_bounds(p3, ρ, qⁱ_raw, ℳ.nⁱ, ℳ.zⁱ,
                                  rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    T = temperature(𝒰, constants)
    P = p3_air_pressure(𝒰, constants)
    transport = air_transport_properties(T, P)
    λ_r = rain_slope_parameter(ℳ.qʳ, ℳ.nʳ, p3.process_rates)
    return P3IceProps{FT}(rime_state.qᶠ, rime_state.bᶠ, rime_state.Fᶠ, Fˡ,
                          rime_state.ρᶠ, bounds.qⁱ_total, bounds.nⁱ,
                          bounds.nⁱ_diagnostic, bounds.ρ_mean, bounds.μ_ice,
                          cloud.μ_c, cloud.λ_c, cloud.Nᶜ, bounds.zⁱ,
                          transport.D_v, transport.nu, λ_r)
end

@inline function p3_rates_and_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    # Build ice properties first, then reuse them when computing rates to avoid
    # the redundant rain_slope_parameter / consistent_rime_state / qⁱ_total / Fˡ
    # calls inside compute_p3_process_rates.
    props = p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
    surface_temperature = temperature(𝒰, constants)
    temperature_tendency = p3_adiabatic_temperature_tendency(ℳ, 𝒰, constants)
    rates = compute_p3_process_rates(
        p3, ρ, ℳ, 𝒰, constants, props, surface_temperature,
        temperature_tendency, zero(ρ))
    return rates, props
end

@inline function cloud_shape_before_homogeneous_freezing(p3, rates, ρ,
                                                          ℳ::P3MicrophysicalState,
                                                          props::P3IceProps)
    FT = typeof(ρ)
    τ = p3.process_rates.sink_limiting_timescale

    # Remove the homogeneous-freezing sink from the assembled cloud tendency to
    # recover the residual cloud reservoir immediately before that process.
    cloud_mass_tendency = tendency_ρqᶜˡ(rates, ρ) / ρ +
                          rates.cloud_homogeneous_mass
    cloud_mass = max(0, ℳ.qᶜˡ + cloud_mass_tendency * τ)

    cloud_number_tendency = cloud_number_tendency_before_homogeneous_freezing(
        p3, ρ, ℳ.qᶜˡ, props.Nᶜ,
        rates.ccn_activation_mass, rates.ccn_activation_number,
        rates.autoconversion, rates.accretion, rates.cloud_self_collection,
        rates.cloud_riming_number, rates.cloud_freezing_number,
        rates.cloud_warm_collection_number)
    prognostic_cloud_number = max(0, props.Nᶜ / ρ + cloud_number_tendency * τ)
    prescribed_cloud_number = p3.cloud.number_concentration / ρ
    cloud_number = ifelse(isnothing(p3.aerosol), prescribed_cloud_number,
                          prognostic_cloud_number)

    residual_cloud = diagnose_cloud_dsd(p3, cloud_mass, cloud_number, ρ)
    has_residual_cloud = cloud_mass >= p3.minimum_mass_mixing_ratio
    return ifelse(has_residual_cloud, residual_cloud.μ_c, props.μ_cloud)
end

@inline function p3_ice_sixth_moment_tendency(::Nothing, p3, rates, ρ, ℳ::P3MicrophysicalState, props::P3IceProps)
    μ_cloud = cloud_shape_before_homogeneous_freezing(p3, rates, ρ, ℳ, props)
    return tendency_ρzⁱ(rates, ρ, props.qⁱ_total, props.nⁱ, props.zⁱ_bounded,
                         p3.process_rates, zero(typeof(ρ)), μ_cloud)
end

@inline function p3_ice_sixth_moment_tendency(ice_table::P3IceIntegralsTable, p3, rates, ρ, ℳ::P3MicrophysicalState, props::P3IceProps)
    # The fully tabulated Z-tendency overload represents Fortran's dormant
    # log_full3mom branch. Runtime P3 v5.5 uses the active hybrid path: group-1
    # processes reconstruct Z with fixed μ over the same safety timescale used
    # by process-rate limiting, while group-2 sources initialize new ice moments
    # analytically.
    μ_cloud = cloud_shape_before_homogeneous_freezing(p3, rates, ρ, ℳ, props)
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)
    return active_ice_sixth_moment_tendency(ice_table, p3, rates, ρ,
                                            ℳ.qⁱ, qʷⁱ, props.nⁱ, props.qᶠ,
                                            props.bᶠ, props.zⁱ_bounded,
                                            props.μ_ice, zero(typeof(ρ)), μ_cloud)
end
