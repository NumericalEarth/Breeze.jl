using Oceananigans: CenterField, Field
using Oceananigans.BoundaryConditions: BoundaryCondition, FieldBoundaryConditions, NormalFlow
using Oceananigans.Fields: ZeroField, ZFaceField
using Oceananigans.Grids: Center, Face
using Oceananigans.Operators: ℑzᵃᵃᶜ

using Breeze.AtmosphereModels: AtmosphereModels as AM
using Breeze.AtmosphereModels: AbstractMicrophysicalState

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
    "Liquid water on ice mixing ratio [kg/kg]"
    qʷⁱ :: FT
    "Liquid supersaturation [kg/kg] (Grabowski & Morrison 2008)"
    sᵛ⁺ˡ :: FT
    "Unactivated aerosol number concentration [1/kg] (zero when no aerosol prognostic)"
    nᵃ  :: FT
    "Cell-center vertical velocity [m/s] (retained for the common microphysical-state interface)"
    w   :: FT
end

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
# The prescribed-Nᶜˡ path has no `ρnᵃ` field, and returns 0 to match the framework default.
@inline AM.initial_aerosol_number(p3::P3) = initial_aerosol_number(p3.aerosol)
@inline initial_aerosol_number(::Nothing) = 0
@inline initial_aerosol_number(aerosol::AerosolActivation) = sum_aerosol_number(aerosol)
@inline AM.initial_aerosol_number_density(p3::P3, ρ) = ρ * AM.initial_aerosol_number(p3)

#####
##### Prognostic field names
#####

# Each optional-field switch must be resolvable to a constant tuple at compile time,
# otherwise the resulting Union return type forces the generic GPU
# `extract_microphysical_prognostics` recursion to allocate.
#
# We therefore dispatch on the *type* of the optional container, so the compiler folds
# the helper down to a static tuple per concrete P3 type. The value of
# `predict_supersaturation` is carried by `ProcessRate`' second type parameter
# for the same reason.
#
# Every switch here gates allocation as well as transport: the fields a configuration
# does not use are never created (see `materialize_microphysical_fields`), so an
# unguarded read would be a missing-property error rather than a silent zero.

@inline supersaturation_prognostic_names(::ProcessRate{FT, false}) where FT = ()
@inline supersaturation_prognostic_names(::ProcessRate{FT, true}) where FT = (:ρsᵛ⁺ˡ,)

# Droplet number and aerosol depletion are prognostic iff `p3.aerosol` is a concrete
# `AerosolActivation`. In the prescribed-Nᶜˡ path, `nᶜˡ`
# is the scheme parameter `p3.cloud.number_concentration` at every microphysics call, so
# no rate reads `ρnᶜˡ` or `ρnᵃ`. Advecting them would integrate transport unrelated to the
# number the physics uses, and `materialize_microphysical_fields` does not even allocate
# them in that path.
@inline cloud_prognostic_names(::Nothing) = (:ρqᶜˡ,)
@inline cloud_prognostic_names(_) = (:ρqᶜˡ, :ρnᶜˡ)

@inline aerosol_prognostic_names(::Nothing) = ()
@inline aerosol_prognostic_names(_) = (:ρnᵃ,)

@inline AM.aerosol_field_names(p3::P3) = aerosol_prognostic_names(p3.aerosol)

"""
$(TYPEDSIGNATURES)

Return prognostic field names for the P3 scheme.

- Cloud mass (always): ρqᶜˡ
- Cloud number (only when `aerosol::AerosolActivation` is set): ρnᶜˡ
- Rain: ρqʳ, ρnʳ
- Ice (always): ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ
- Liquid supersaturation (only when `predict_supersaturation = true`): ρsᵛ⁺ˡ
- Aerosol (only when `aerosol::AerosolActivation` is set): ρnᵃ
"""
@inline function AM.prognostic_field_names(p3::P3)
    cloud_names = cloud_prognostic_names(p3.aerosol)
    rain_names = (:ρqʳ, :ρnʳ)
    ice_names = (:ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
    supersaturation_names = supersaturation_prognostic_names(p3.process_rates)
    aero_names = AM.aerosol_field_names(p3)

    return tuple(cloud_names..., rain_names..., ice_names...,
                 supersaturation_names..., aero_names...)
end

# True condensate partial densities that contribute to total air mass. Number moments,
# supersaturation, and aerosol are not masses. Rime mass is already contained in total
# ice mass, so including ρqᶠ would count it twice; ρbᶠ is an ice property.
@inline AM.condensate_field_names(::P3) = (:ρqᶜˡ, :ρqʳ, :ρqⁱ, :ρqʷⁱ)

#####
##### Negative moisture correction
#####
#
# The advection operator is not positive-definite, so any of P3's prognostic
# densities can come back negative from a stage update. Without this repair the
# negative values persist: the process rates clamp what they read at zero, but
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

# Fields that must vanish with the mass they describe. Ice number, rime mass, and rime
# volume are all properties of the ice population, so zeroing them when `ρqⁱ` is gone
# destroys no water. Liquid on ice is deliberately not paired with `ρqⁱ`: it is real
# water, and the whole-particle clip in `p3_phase2_rates` (Fˡ > 0.99) already sheds it
# to rain when the dry ice mass is gone.
@inline AM.correction_number_mass_pairs(p3::P3, μ) =
    (cloud_number_correction_pairs(p3.aerosol, μ)...,
     (μ.ρnʳ, μ.ρqʳ), (μ.ρnⁱ, μ.ρqⁱ),
     (μ.ρqᶠ, μ.ρqⁱ), (μ.ρbᶠ, μ.ρqⁱ))

# Fields clamped to zero rather than borrowed against, because they carry no water:
# the number moments, the rime properties, and the unactivated aerosol count.
# `ρsᵛ⁺ˡ` is excluded because liquid supersaturation may be negative.
@inline AM.correction_number_fields(p3::P3, μ) =
    (cloud_number_correction_fields(p3.aerosol, μ)...,
     μ.ρnʳ, μ.ρnⁱ, μ.ρqᶠ, μ.ρbᶠ,
     aerosol_correction_fields(p3.aerosol, μ)...)

# Same compile-time switch as `prognostic_field_names`: dispatch on the *type* of the
# aerosol container so each tuple folds to a constant. The prescribed-Nᶜˡ path has no
# `ρnᶜˡ`/`ρnᵃ` fields at all, so there is nothing to repair.
@inline cloud_number_correction_pairs(::Nothing, μ) = ()
@inline cloud_number_correction_pairs(_, μ) = ((μ.ρnᶜˡ, μ.ρqᶜˡ),)

@inline cloud_number_correction_fields(::Nothing, μ) = ()
@inline cloud_number_correction_fields(_, μ) = (μ.ρnᶜˡ,)

@inline aerosol_correction_fields(::Nothing, μ) = ()
@inline aerosol_correction_fields(_, μ) = (μ.ρnᵃ,)

"""
$(TYPEDSIGNATURES)

Effective cloud droplet number concentration [kg⁻¹] seen by P3's process rates.

In the prescribed-Nᶜˡ path (`p3.aerosol === nothing`), the droplet number is always
`p3.cloud.number_concentration` at every microphysics call, so this helper returns
that prescribed value and ignores its `ρnᶜˡ` argument. Droplet number is
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
- `ρqʷⁱ`: Liquid water on ice mass density
- `ρnᶜˡ`, `ρnᵃ`: Cloud number and unactivated aerosol number densities, allocated only
when `p3.aerosol isa AerosolActivation`. The prescribed-Nᶜˡ path takes droplet
  number from `p3.cloud.number_concentration`, so neither field exists there and
  neither is advected.

**Diagnostic:**
- `qᵛ`: Vapor specific humidity (mirrors the prognostic vapor field)

**Sedimentation velocities** (`wᶜˡ`, `wᶜˡₙ`, `wʳ`, `wʳₙ`, `wⁱ`, `wⁱₙ`):
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

    # Hallett–Mossop uses the temperature at the lowest active atmospheric cell.
    # Store one value per column rather than assuming that local k=1 is active.
    surface_temperature = Field{Center, Center, Nothing}(grid)

    fields = (; ρqᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ,
                qᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ, qᵛ,
                wᶜˡ, wᶜˡₙ, wʳ, wʳₙ, wⁱ, wⁱₙ,
                surface_temperature)

    return merge(fields,
                 aerosol_activation_fields(p3.aerosol, grid),
                 supersaturation_fields(p3.process_rates, grid))
end

# Optional field groups. Each switch gates allocation, not just transport, so a
# configuration never carries memory for state it does not use. Both dispatch on a
# *type* (`Nothing` / `ProcessRate{FT, PS}`) so the merged NamedTuple is a
# compile-time constant, which lets the read sites fold their guards away.

# Droplet number and unactivated aerosol. The prescribed-Nᶜˡ path takes the droplet
# number from `p3.cloud.number_concentration` at every call and never reads `ρnᶜˡ`
# or `ρnᵃ`.
@inline aerosol_activation_fields(::Nothing, grid) = (;)

@inline aerosol_activation_fields(_, grid) =
    (; ρnᶜˡ = CenterField(grid),        # Cloud number density [1/m³]
       ρnᵃ = CenterField(grid),         # Unactivated aerosol number density [1/m³]
       nᶜˡ = CenterField(grid),         # Cloud number concentration [kg⁻¹]
       nᵃ = CenterField(grid))          # Unactivated aerosol [kg⁻¹]

# Predicted supersaturation, off by default. With the switch off every rate that
# would touch `sᵛ⁺ˡ` is gated to zero, so the prognostic carries no information.
@inline supersaturation_fields(::ProcessRate{FT, false}, grid) where FT = (;)

@inline supersaturation_fields(::ProcessRate{FT, true}, grid) where FT =
    (; ρsᵛ⁺ˡ = CenterField(grid), sᵛ⁺ˡ = CenterField(grid))

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
# `microphysical_state` path works whether or not `μ` carries the optional
# `ρnᶜˡ`/`ρnᵃ` (prognostic aerosol) and `ρsᵛ⁺ˡ` (predicted supersaturation) fields.
@generated function get_or_default(μ::NamedTuple{names}, ::Val{key}, default) where {names, key}
    return key in names ? :(μ.$key) : :(default)
end

@inline vertical_velocity(velocities, FT) = FT(velocities.w)

# Interpolate a face-located w field to a cell center.
# All call sites pass face fields (or ZeroField placeholders); no scalar fallback needed.
@inline interpolate_w_to_center(grid, i, j, k, w_field, FT) = FT(ℑzᵃᵃᶜ(i, j, k, grid, w_field))

@inline function AM.microphysical_state(p3::P3, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    # ρnᶜˡ is absent unless the aerosol-activation path is enabled; the prescribed-Nᶜˡ
    # branch of `effective_cloud_droplet_number` ignores the value it is handed.
    nᶜˡ = effective_cloud_droplet_number(p3, get_or_default(μ, Val(:ρnᶜˡ), 0 * ρ), ρ)
    qʳ  = μ.ρqʳ / ρ
    nʳ  = μ.ρnʳ / ρ
    qⁱ  = μ.ρqⁱ / ρ
    nⁱ  = μ.ρnⁱ / ρ
    FT = typeof(ρ)
    qʷⁱ = μ.ρqʷⁱ / ρ
    rime_state = consistent_rime_state(p3, qⁱ, μ.ρqᶠ / ρ, μ.ρbᶠ / ρ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    # ρsᵛ⁺ˡ is absent unless predicted supersaturation is enabled; default to 0.
    sᵛ⁺ˡ = get_or_default(μ, Val(:ρsᵛ⁺ˡ), 0 * ρ) / ρ
    # ρnᵃ is absent unless prognostic-aerosol path is enabled; default to 0.
    nᵃ = get_or_default(μ, Val(:ρnᵃ), 0 * ρ) / ρ
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ, sᵛ⁺ˡ, nᵃ,
                                vertical_velocity(velocities, FT))
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
    rime_state = consistent_rime_state(p3, qⁱ, qᶠ, bᶠ)
    return merge(μ, (; ρqᶠ = ρ * rime_state.qᶠ,
                       ρbᶠ = ρ * rime_state.bᶠ))
end

# Droplet number and unactivated aerosol on the grid. Dispatch on the *type* of
# `p3.aerosol` rather than indexing behind a runtime branch, because in the prescribed-Nᶜˡ
# path the `ρnᶜˡ`/`ρnᵃ` fields do not exist at all.
@inline grid_cloud_droplet_number(p3::P3, ::Nothing, μ, i, j, k, ρ) =
    p3.cloud.number_concentration / ρ
@inline grid_cloud_droplet_number(p3::P3, _, μ, i, j, k, ρ) =
    @inbounds μ.ρnᶜˡ[i, j, k] / ρ

@inline grid_aerosol_number(::Nothing, μ, i, j, k, ρ) = 0 * ρ
@inline grid_aerosol_number(_, μ, i, j, k, ρ) = @inbounds μ.ρnᵃ[i, j, k] / ρ

# Same for the optional supersaturation prognostic: absent with prediction
# disabled, where it collapses to zero anyway.
@inline grid_supersaturation(::ProcessRate{FT, false}, μ, i, j, k, ρ) where FT = 0 * ρ
@inline grid_supersaturation(::ProcessRate{FT, true}, μ, i, j, k, ρ) where FT =
    @inbounds μ.ρsᵛ⁺ˡ[i, j, k] / ρ

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
    rime_state = consistent_rime_state(p3, qⁱ, @inbounds(μ.ρqᶠ[i, j, k]) / ρ, @inbounds(μ.ρbᶠ[i, j, k]) / ρ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    sᵛ⁺ˡ = grid_supersaturation(p3.process_rates, μ, i, j, k, ρ)
    nᵃ   = grid_aerosol_number(p3.aerosol, μ, i, j, k, ρ)
    w = interpolate_w_to_center(grid, i, j, k, velocities.w, FT)
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ, sᵛ⁺ˡ, nᵃ, w)
end

# Bounded ice moments plus the Table-1 bracket (`prep`) of the diagnostic population, which
# the λ-limiter and the mean-density read share.
struct P3IceMoments{FT, P}
    qⁱ_total :: FT
    nⁱ_diagnostic :: FT
    nⁱ :: FT
    prep :: P
end

struct P3IceMomentBounds{FT}
    qⁱ_total :: FT
    nⁱ_diagnostic :: FT
    nⁱ :: FT
    ρ_mean :: FT
end

@inline function p3_ice_moments(p3::P3, ρ, qⁱ_raw, nⁱ_raw, Fᶠ, Fˡ, ρᶠ)
    FT = typeof(ρ)
    floors = p3.process_rates.floors
    mass_scale = FT(floors.mass_scale)
    has_ice_mass = qⁱ_raw > mass_scale
    qⁱ_total = max(qⁱ_raw, mass_scale)
    nⁱ_global = min(max(0, nⁱ_raw),
                    p3.process_rates.maximum_ice_number_density / ρ)
    nⁱ_diagnostic = max(nⁱ_global, p3.minimum_number_mixing_ratio)
    limiter = p3.ice.lambda_limiter
    prep = diagnostic_ice_bracket(limiter, qⁱ_total, nⁱ_diagnostic, Fᶠ, Fˡ, ρᶠ, floors)
    nⁱ_bounded = bounded_ice_number(limiter, prep, qⁱ_total, nⁱ_diagnostic, floors)
    nⁱ = ifelse(has_ice_mass, nⁱ_bounded, FT(0))
    return P3IceMoments{FT, typeof(prep)}(qⁱ_total, nⁱ_diagnostic, nⁱ, prep)
end

@inline function p3_ice_moment_bounds(p3::P3, ρ, qⁱ_raw, nⁱ_raw, Fᶠ, Fˡ, ρᶠ)
    FT = typeof(ρ)
    moments = p3_ice_moments(p3, ρ, qⁱ_raw, nⁱ_raw, Fᶠ, Fˡ, ρᶠ)
    ρ_mean = ice_mean_density(p3.ice.bulk_properties, moments.prep)
    return P3IceMomentBounds{FT}(moments.qⁱ_total, moments.nⁱ_diagnostic,
                                 moments.nⁱ, ρ_mean)
end

# Write the consistent rime state back to the prognostic densities. `grid_microphysical_state`
# already passes `ρqᶠ`/`ρbᶠ` through `consistent_rime_state`, so `ℳ.qᶠ`/`ℳ.bᶠ` are what every
# process rate sees, and (through `μ.qᶠ`/`μ.bᶠ`) what the scalar transport operators advect:
# the correction zeroes `qᶠ` when the rime volume vanishes, zeroes both below the rime-mass
# floor, and caps `qᶠ` at the dry ice mass.
#
# Without this writeback the prognostic keeps the uncorrected mass while the advected specific
# field reports the corrected one, so hidden rime receives no transport while the ice carrying
# it moves away, then reappears once the correction stops firing.
#
# Rime mass is a *component* of the ice mass rather than an independent water reservoir
# (`condensate_field_names` excludes `ρqᶠ`), so the clip moves the rime fraction Fᶠ and never
# the total water or the total density.
@inline function clamp_rime_state!(μ, i, j, k, ρ, ℳ::P3MicrophysicalState)
    @inbounds μ.ρqᶠ[i, j, k] = ρ * ℳ.qᶠ
    @inbounds μ.ρbᶠ[i, j, k] = ρ * ℳ.bᶠ
    return nothing
end

# GPU-compatible update_microphysical_fields! for P3.
# Bypasses the generic extract_microphysical_prognostics which uses runtime Symbol
# dispatch that GPU compilers cannot resolve. Instead, directly constructs
# P3MicrophysicalState from @inbounds field access and delegates to
# update_microphysical_auxiliaries!.
@inline function AM.update_microphysical_fields!(μ, i, j, k, grid, p3::P3, ρ, 𝒰, constants)
    @inbounds begin
        # TODO: thread real velocities here once AM.update_microphysical_fields!
        # signature carries them. ℳ.w == 0 is acceptable in this auxiliary path because
        # nothing downstream of it consumes w: the auxiliary writes are copies of the
        # prognostic state, and the fall-speed path reads only the size distributions.
        # `w` enters P3 solely through the adiabatic temperature tendency and CCN
        # activation, both of which live in the tendency kernel.
        velocities = (u = ZeroField(), v = ZeroField(), w = ZeroField())
        ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, 𝒰, velocities)
        clamp_rime_state!(μ, i, j, k, ρ, ℳ)
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

Writes the specific quantities `qᶜˡ`, `qʳ`, `nʳ`, `qⁱ`, `nⁱ`, `qᶠ`, `bᶠ`, `qʷⁱ`, the
optional number and supersaturation diagnostics, and the six z-Face terminal velocities
that sedimentation advects with.

After the moisture refactor, vapor is the prognostic moisture variable.
The diagnostic `qᵛ` field is updated from the thermodynamic state.
"""
# Called from the thermodynamic variables kernel, so every auxiliary field P3 owns is
# established by `update_state!` and carries the same time level as the prognostics it was
# built from. That matters for the terminal velocities in particular: they are diagnostics
# a user can output, and they are also the advecting velocities sedimentation reads, so
# computing them anywhere else would leave them one stage behind the rest of the state.
#
# The process rates are the exception, and legitimately so: they are tendencies, not state.
# `compute_microphysical_tendencies!` evaluates them from the current state with an
# adiabatic-only driver, which also keeps ~1000 lines of P3 process physics out of this
# kernel. The fall-speed path is small by comparison.
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
    write_supersaturation_diagnostic!(μ, i, j, k, p3.process_rates, ℳ)
    p3_compute_fall_speeds!(μ, i, j, k, p3, ρ, ℳ, 𝒰, constants)

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

@inline write_supersaturation_diagnostic!(
    μ, i, j, k, ::ProcessRate{FT, false}, ℳ
) where FT = nothing

@inline function write_supersaturation_diagnostic!(
    μ, i, j, k, ::ProcessRate{FT, true}, ℳ
) where FT
    @inbounds μ.sᵛ⁺ˡ[i, j, k] = ℳ.sᵛ⁺ˡ
    return nothing
end

# GPU-safe property structs (NamedTuples require jl_f_tuple on GPU).
struct P3CoreIceProps{FT, P}
    qᶠ :: FT
    bᶠ :: FT
    Fᶠ :: FT
    Fˡ :: FT
    ρᶠ :: FT
    qⁱ_total :: FT
    # impose_max_Ni cap mirrored from compute_p3_process_rates so the tabulated
    # rate decomposition uses the same nⁱ throughout the process path.
    nⁱ :: FT
    # Number diagnosed before the lambda limiter; the process payload combines it
    # with mean density to build the volume-equivalent diameter.
    nⁱ_diagnostic :: FT
    # Table-1 bracket of the diagnostic population, for the mean-density read.
    prep :: P
end

struct P3FallSpeedProps{FT}
    Fᶠ :: FT
    Fˡ :: FT
    ρᶠ :: FT
    qⁱ_total :: FT
    nⁱ :: FT
    μᶜˡ :: FT
    λᶜˡ :: FT
    ν :: FT
end

struct P3ProcessProps{FT}
    qᶠ :: FT
    bᶠ :: FT
    Fᶠ :: FT
    Fˡ :: FT
    ρᶠ :: FT
    qⁱ_total :: FT
    nⁱ :: FT
    nⁱ_diagnostic :: FT
    ρ_mean :: FT
    Nᶜˡ :: FT
    λʳ :: FT
end

# GPU-safe return structs (NamedTuples require jl_f_tuple on GPU).
struct P3FallSpeedResult{FT}
    wᶜˡ :: FT; wᶜˡₙ :: FT; wʳ :: FT; wʳₙ :: FT; wⁱ :: FT; wⁱₙ :: FT
end

struct P3TendencyResult{FT}
    tendency_ρqᶜˡ :: FT; tendency_ρnᶜˡ :: FT
    tendency_ρqʳ :: FT; tendency_ρnʳ :: FT
    tendency_ρqⁱ :: FT; tendency_ρnⁱ :: FT
    tendency_ρqᶠ :: FT; tendency_ρbᶠ :: FT
    tendency_ρqʷⁱ :: FT; tendency_ρsᵛ⁺ˡ :: FT
    tendency_ρqᵛ :: FT; tendency_ρnᵃ :: FT
end

# Terminal velocities must be available before scalar tendency assembly, while
# process rates need the resolved host tendencies assembled during that step.
# Keep both computations scalar and return concrete structs for GPU compilation.
@inline function p3_fall_speed_compute(p3::P3, ρ, ℳ::P3MicrophysicalState,
                                       properties::P3FallSpeedProps, constants)
    Fᶠ = properties.Fᶠ
    ρᶠ = properties.ρᶠ

    # Cloud terminal velocities — cloud mass and number sediment with DSD-integrated
    # Stokes velocities.
    vᶜ = cloud_terminal_velocities(p3, ℳ.qᶜˡ, ρ, properties.ν, properties.μᶜˡ, properties.λᶜˡ,
                                   constants)
    wᶜˡ = vᶜ.mass_weighted
    wᶜˡₙ = vᶜ.number_weighted

    # Rain terminal velocities — fused call shares λ_r, ρ_correction, log10(λ_r)
    # across the two 1D table lookups (mass- and number-weighted).
    vᵣ = rain_terminal_velocities(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    wʳ   = vᵣ.mass_weighted
    wʳₙ  = vᵣ.number_weighted
    # The global ice-number cap must be seen consistently by all downstream math —
    # process rates and terminal velocities alike — so use
    # properties.nⁱ (= min(ℳ.nⁱ, Nⁱ_max/ρ)) rather than the raw prognostic here.
    # The ice fall-speed lookup is indexed with the total ice mass.
    # `properties.qⁱ_total` includes the liquid coating only when that prognostic
    # mode is active, so a stale restart coating cannot affect the
    # non-liquid-fraction fast branch.
    qⁱ_total = properties.qⁱ_total
    # Fused call: shares m̄, ρ_correction, log(m̄), and the 4D interpolation indices
    # across mass- and number-weighted fall speeds.
    vᵢ = ice_terminal_velocities(p3, qⁱ_total, properties.nⁱ, Fᶠ, ρᶠ, ρ;
                                 Fˡ = properties.Fˡ)
    wⁱ, wⁱₙ = vᵢ.mass_weighted, vᵢ.number_weighted

    FT = typeof(ρ)
    return P3FallSpeedResult{FT}(wᶜˡ, wᶜˡₙ, wʳ, wʳₙ, wⁱ, wⁱₙ)
end

@inline function p3_tendency_compute(p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰,
                                     constants, properties::P3ProcessProps,
                                     surface_temperature, temperature_tendency,
                                     vapor_tendency)
    Fᶠ = properties.Fᶠ
    ρᶠ = properties.ρᶠ
    # Passing `properties` lets compute_p3_process_rates skip the redundant
    # rain_slope_parameter / consistent_rime_state / qⁱ_total / Fˡ recomputation.
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, properties,
                                     surface_temperature, temperature_tendency,
                                     vapor_tendency)

    # Tendency extraction
    cloud_mass_tendency = tendency_ρqᶜˡ(rates, ρ)
    # Prescribed-Nᶜˡ path: nc is a scheme parameter (not advected); tendency = 0.
    cloud_number_tendency = isnothing(p3.aerosol) ? zero(typeof(ρ)) :
                            tendency_ρnᶜˡ(rates, ρ, properties.Nᶜˡ, ℳ.qᶜˡ, p3)
    rain_mass_tendency = tendency_ρqʳ(rates, ρ, p3.process_rates)
    rain_number_tendency = tendency_ρnʳ(rates, ρ, p3)
    ice_mass_tendency = tendency_ρqⁱ(rates, ρ)
    ice_number_tendency = tendency_ρnⁱ(rates, ρ)
    rime_mass_tendency = tendency_ρqᶠ(rates, ρ, Fᶠ)
    rime_volume_tendency = tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, ℳ.qⁱ, p3.process_rates)
    coating_mass_tendency = tendency_ρqʷⁱ(rates, ρ, p3.process_rates)
    supersaturation_tendency = tendency_ρsᵛ⁺ˡ(rates, ρ, p3.process_rates)
    vapor_mass_tendency = tendency_ρqᵛ(rates, ρ)
    # Aerosol depletion: every activated cloud droplet removes one from ρnᵃ.
    # Zero in the prescribed-Nᶜˡ path (rates.ccn_activation_number is 0 there).
    aerosol_number_tendency = tendency_ρnᵃ(rates, ρ)

    FT = typeof(ρ)
    return P3TendencyResult{FT}(cloud_mass_tendency, cloud_number_tendency,
                                     rain_mass_tendency, rain_number_tendency,
                                     ice_mass_tendency, ice_number_tendency,
                                     rime_mass_tendency, rime_volume_tendency,
                                     coating_mass_tendency, supersaturation_tendency,
                                     vapor_mass_tendency, aerosol_number_tendency)
end

# Adiabatic temperature tendency from the grid or parcel vertical velocity, used as
# P3's external thermodynamic forcing on both paths. The accompanying resolved-vapor
# tendency is zero. On the Eulerian grid path this deliberately omits resolved
# transport, turbulent mixing, radiation, and user forcing from the diffusional-growth
# driver; see `_p3_add_tendencies_kernel!`.
@inline function p3_adiabatic_temperature_tendency(ℳ::P3MicrophysicalState, 𝒰, constants)
    cᵖᵐ = mixture_heat_capacity(𝒰.moisture_mass_fractions, constants)
    return -constants.gravitational_acceleration * ℳ.w / cᵖᵐ
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
    end
    return nothing
end

# `G` is the reduced tuple from `p3_tendency_fields`. The `+=` lands on top of the
# advection, diffusion and forcing the scalar kernels have already written.
@inline function add_p3_tendencies!(G, i, j, k, p3::P3, result::P3TendencyResult)
    @inbounds begin
        G.ρqᶜˡ[i, j, k] += result.tendency_ρqᶜˡ
        G.ρqʳ[i, j, k]  += result.tendency_ρqʳ
        G.ρnʳ[i, j, k]  += result.tendency_ρnʳ
        G.ρqⁱ[i, j, k]  += result.tendency_ρqⁱ
        G.ρnⁱ[i, j, k]  += result.tendency_ρnⁱ
        G.ρqᶠ[i, j, k]  += result.tendency_ρqᶠ
        G.ρbᶠ[i, j, k]  += result.tendency_ρbᶠ
        G.ρqʷⁱ[i, j, k] += result.tendency_ρqʷⁱ
        G.ρqᵛ[i, j, k]  += result.tendency_ρqᵛ
    end
    add_p3_cloud_number_tendencies!(G, i, j, k, p3.aerosol, result)
    add_p3_supersaturation_tendency!(G, i, j, k, p3.process_rates, result)
    return nothing
end

# Configurations without a prognostic have no `Gⁿ` slot to add to.
@inline add_p3_cloud_number_tendencies!(G, i, j, k, ::Nothing, result) = nothing

@inline function add_p3_cloud_number_tendencies!(G, i, j, k, _, result)
    @inbounds G.ρnᶜˡ[i, j, k] += result.tendency_ρnᶜˡ
    @inbounds G.ρnᵃ[i, j, k] += result.tendency_ρnᵃ
    return nothing
end

@inline add_p3_supersaturation_tendency!(
    G, i, j, k, ::ProcessRate{FT, false}, result
) where FT = nothing

@inline function add_p3_supersaturation_tendency!(
    G, i, j, k, ::ProcessRate{FT, true}, result
) where FT
    @inbounds G.ρsᵛ⁺ˡ[i, j, k] += result.tendency_ρsᵛ⁺ˡ
    return nothing
end

# The caller supplies `ℳ` rather than a grid and velocities: `update_microphysical_auxiliaries!`
# has already built the state for this cell, so rebuilding it here would repeat the whole
# `grid_microphysical_state` read for nothing.
@inline function p3_compute_fall_speeds!(μ, i, j, k, p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    properties = p3_fall_speed_properties(p3, ρ, ℳ, 𝒰, constants)
    result = p3_fall_speed_compute(p3, ρ, ℳ, properties, constants)
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
# Terminal velocities are diagnosed in update_microphysical_auxiliaries! and stored in
# diagnostic fields, so they are current whenever `update_state!` has run.
# microphysical_velocities returns NamedTuples compatible with Oceananigans'
# sum_of_velocities.

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

# Liquid on ice: same as ice mass
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρqʷⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ)

#####
##### Microphysical tendencies
#####
#
# Two paths, both evaluating the process-rate bundle once per cell:
#   1. Grid-based (AtmosphereModel): the fused driver builds the state with the
#      adiabatic-only forcing and adds every tendency straight into Gⁿ.
#   2. Gridless (ParcelModel): microphysical_tendencies builds state and computes rates directly.

# Ice properties shared within each specialized consumer payload.
@inline function p3_core_ice_properties(p3, ρ, ℳ::P3MicrophysicalState)
    FT = typeof(ρ)
    qʷⁱ = active_liquid_on_ice(p3, ℳ.qʷⁱ)
    qⁱ_raw = total_ice_mass(ℳ.qⁱ, qʷⁱ)
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ)
    Fˡ = liquid_fraction_on_ice(ℳ.qⁱ, qʷⁱ, p3.process_rates.floors)
    moments = p3_ice_moments(p3, ρ, qⁱ_raw, ℳ.nⁱ,
                             rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    return P3CoreIceProps{FT, typeof(moments.prep)}(rime_state.qᶠ, rime_state.bᶠ,
                                                    rime_state.Fᶠ, Fˡ, rime_state.ρᶠ,
                                                    moments.qⁱ_total, moments.nⁱ,
                                                    moments.nⁱ_diagnostic, moments.prep)
end

@inline function p3_fall_speed_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    FT = typeof(ρ)
    ice = p3_core_ice_properties(p3, ρ, ℳ)
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    T = temperature(𝒰, constants)
    P = air_pressure(𝒰, constants)
    ν = air_kinematic_viscosity(T, P, constants)
    return P3FallSpeedProps{FT}(ice.Fᶠ, ice.Fˡ, ice.ρᶠ, ice.qⁱ_total, ice.nⁱ,
                                cloud.μᶜˡ, cloud.λᶜˡ, ν)
end

@inline function p3_process_properties(p3, ρ, ℳ::P3MicrophysicalState)
    FT = typeof(ρ)
    ice = p3_core_ice_properties(p3, ρ, ℳ)
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    λʳ = rain_slope_parameter(ℳ.qʳ, ℳ.nʳ, p3.process_rates)
    ρ_mean = ice_mean_density(p3.ice.bulk_properties, ice.prep)
    return P3ProcessProps{FT}(ice.qᶠ, ice.bᶠ, ice.Fᶠ, ice.Fˡ, ice.ρᶠ,
                              ice.qⁱ_total, ice.nⁱ, ice.nⁱ_diagnostic, ρ_mean,
                              cloud.Nᶜˡ, λʳ)
end

# Every P3 tendency from one process-rate evaluation, for callers with no grid. With no
# column below, the local temperature stands in for the surface temperature.
@inline function p3_state_tendencies(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    # Build process properties once, then reuse them when computing rates to avoid
    # redundant rain-slope, consistent-rime-state, total-ice-mass, and liquid-fraction
    # diagnoses inside compute_p3_process_rates.
    properties = p3_process_properties(p3, ρ, ℳ)
    surface_temperature = temperature(𝒰, constants)
    temperature_tendency = p3_adiabatic_temperature_tendency(ℳ, 𝒰, constants)
    return p3_tendency_compute(p3, ρ, ℳ, 𝒰, constants, properties,
                               surface_temperature, temperature_tendency, zero(ρ))
end

# Name-to-slot map for `P3TendencyResult`. Names P3 does not evolve (`:s`, `:qᵗ`, host
# tracers) take the zero fallback. Leave the result type unparameterized in every method,
# or the fallback becomes ambiguous with the named slots.
@inline p3_tendency_component(result::P3TendencyResult, ::Val) = zero(result.tendency_ρqᵛ)
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρqᶜˡ}) = result.tendency_ρqᶜˡ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρnᶜˡ}) = result.tendency_ρnᶜˡ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρqʳ})  = result.tendency_ρqʳ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρnʳ})  = result.tendency_ρnʳ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρqⁱ})  = result.tendency_ρqⁱ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρnⁱ})  = result.tendency_ρnⁱ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρqᶠ})  = result.tendency_ρqᶠ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρbᶠ})  = result.tendency_ρbᶠ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρqʷⁱ}) = result.tendency_ρqʷⁱ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρsᵛ⁺ˡ}) = result.tendency_ρsᵛ⁺ˡ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρqᵛ})  = result.tendency_ρqᵛ
@inline p3_tendency_component(result::P3TendencyResult, ::Val{:ρnᵃ})  = result.tendency_ρnᵃ
