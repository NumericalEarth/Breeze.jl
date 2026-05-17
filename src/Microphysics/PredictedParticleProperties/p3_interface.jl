#####
##### Microphysics interface implementation for P3
#####
##### These functions integrate the P3 scheme with AtmosphereModel,
##### allowing it to be used as a drop-in microphysics scheme.
#####
##### This file follows the MicrophysicalState abstraction pattern:
##### - P3MicrophysicalState encapsulates local microphysical variables
##### - Gridless microphysical_state(p3, ρ, μ, 𝒰) builds the state
##### - State-based microphysical_tendency(p3, name, ρ, ℳ, 𝒰, constants) computes tendencies
#####

using Oceananigans: CenterField
using Oceananigans.Fields: ZeroField
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

@inline _z̃_prognostic_names(::Nothing) = ()
@inline _z̃_prognostic_names(_) = (:ρz̃ⁱ,)

"""
$(TYPEDSIGNATURES)

Return prognostic field names for the P3 scheme.

The 2-moment ice path advects 10 fields; enabling 3-moment ice adds `ρz̃ⁱ`:

- Cloud: ρqᶜˡ, ρnᶜˡ
- Rain: ρqʳ, ρnʳ
- Ice (always): ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρqʷⁱ
- Ice (3-moment only): ρz̃ⁱ
- Supersaturation: ρsˢᵃᵗ (tendency = 0 when `predict_supersaturation = false`)
"""
@inline function AM.prognostic_field_names(p3::P3)
    cloud_names = (:ρqᶜˡ, :ρnᶜˡ)
    rain_names = (:ρqʳ, :ρnʳ)
    ice_names = (:ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρqʷⁱ)
    z_names = _z̃_prognostic_names(three_moment_shape_table(p3))
    ssat_names = (:ρsˢᵃᵗ,)

    return tuple(cloud_names..., rain_names..., ice_names..., z_names..., ssat_names...)
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

    return (; ρqᶜˡ, ρnᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρz̃ⁱ, ρqʷⁱ, ρsˢᵃᵗ,
              qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, z̃ⁱ, qʷⁱ, sˢᵃᵗ, qᵛ,
              wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z,
              cache_ρqᶜˡ, cache_ρnᶜˡ, cache_ρqʳ, cache_ρnʳ, cache_ρqⁱ, cache_ρnⁱ,
              cache_ρqᶠ, cache_ρbᶠ, cache_ρz̃ⁱ, cache_ρqʷⁱ, cache_ρsˢᵃᵗ, cache_ρqᵛ)
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
@generated function _nt_get(μ::NamedTuple{names}, ::Val{key}, default) where {names, key}
    return key in names ? :(μ.$key) : :(default)
end

@inline function AM.microphysical_state(p3::P3, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    nᶜˡ = effective_cloud_droplet_number(p3, μ.ρnᶜˡ, ρ)
    qʳ  = μ.ρqʳ / ρ
    nʳ  = μ.ρnʳ / ρ
    qⁱ  = μ.ρqⁱ / ρ
    nⁱ  = μ.ρnⁱ / ρ
    # M13: Fortran advects z̃ = √(z·N) and converts to physical z at microphysics entry:
    #   where (nitot > 0) zitot = zitot**2 / nitot; elsewhere zitot = 0
    # ρz̃ⁱ stores the advected variable z̃; convert to physical z = z̃²/N for internal use.
    # In 2-moment mode ρz̃ⁱ is absent from `μ`; treat it as 0 (zⁱ then collapses to 0).
    FT = typeof(ρ)
    z̃ⁱ  = _nt_get(μ, Val(:ρz̃ⁱ), zero(FT)) / ρ
    zⁱ  = ifelse(nⁱ > FT(1e-20), z̃ⁱ^2 / nⁱ, zero(FT))
    qʷⁱ = μ.ρqʷⁱ / ρ
    rime_state = consistent_rime_state(p3, qⁱ, μ.ρqᶠ / ρ, μ.ρbᶠ / ρ, qʷⁱ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    # ρsˢᵃᵗ is absent unless predicted supersaturation is enabled; default to 0.
    sˢᵃᵗ = _nt_get(μ, Val(:ρsˢᵃᵗ), zero(FT)) / ρ
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ)
end

# Disambiguation for P3 with Nothing or empty microphysical fields
@inline AM.microphysical_state(::P3, ρ, ::Nothing, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))
@inline AM.microphysical_state(::P3, ρ, ::NamedTuple{(), Tuple{}}, 𝒰, velocities) = AM.NothingMicrophysicalState(typeof(ρ))

# GPU-compatible OffsetArray 3D indexing.
#
# OffsetArrays.jl's N-dim getindex/setindex! has code paths (checkbounds, axes,
# parent, getproperty) that GPUCompiler rejects. Even CuDeviceArray's 1D getindex
# has dispatch complexity (arrayref → @boundscheck → throw_boundserror) that the
# GPU compiler flags when the kernel is too complex to fully inline.
#
# These @generated overrides produce zero-dispatch code at compile time:
# - GPU path (CuDeviceArray): getfield + unsafe_load/unsafe_store! on the raw
#   LLVMPtr — no method dispatch, no bounds checking, no throw paths
# - CPU path (Array etc.): delegates to the parent's own 3D indexing
using OffsetArrays: OffsetArray
@generated function Base.getindex(A::OffsetArray{T, 3, AA}, i::Int, j::Int, k::Int) where {T, AA<:AbstractArray{T, 3}}
    if hasfield(AA, :dims) && hasfield(AA, :ptr)
        # GPU path: getfield + Base.unsafe_load on LLVMPtr — minimal dispatch
        align = Base.datatype_alignment(T)
        quote
            o = getfield(A, :offsets)
            p = getfield(A, :parent)
            ii = i - getfield(o, 1)
            jj = j - getfield(o, 2)
            kk = k - getfield(o, 3)
            d = getfield(p, :dims)
            n1 = getfield(d, 1)
            n2 = getfield(d, 2)
            lin = ii + n1 * ((jj - 1) + n2 * (kk - 1))
            Base.unsafe_load(getfield(p, :ptr), lin, $(Val(align)))
        end
    else
        # CPU path: regular indexing on parent
        quote
            o = getfield(A, :offsets)
            p = getfield(A, :parent)
            @inbounds p[i - getfield(o, 1), j - getfield(o, 2), k - getfield(o, 3)]
        end
    end
end

@generated function Base.setindex!(A::OffsetArray{T, 3, AA}, val, i::Int, j::Int, k::Int) where {T, AA<:AbstractArray{T, 3}}
    if hasfield(AA, :dims) && hasfield(AA, :ptr)
        # GPU path: getfield + Base.unsafe_store! on LLVMPtr — minimal dispatch
        align = Base.datatype_alignment(T)
        quote
            o = getfield(A, :offsets)
            p = getfield(A, :parent)
            ii = i - getfield(o, 1)
            jj = j - getfield(o, 2)
            kk = k - getfield(o, 3)
            d = getfield(p, :dims)
            n1 = getfield(d, 1)
            n2 = getfield(d, 2)
            lin = ii + n1 * ((jj - 1) + n2 * (kk - 1))
            Base.unsafe_store!(getfield(p, :ptr), convert($T, val), lin, $(Val(align)))
        end
    else
        # CPU path: regular indexing on parent
        quote
            o = getfield(A, :offsets)
            p = getfield(A, :parent)
            @inbounds p[i - getfield(o, 1), j - getfield(o, 2), k - getfield(o, 3)] = val
        end
    end
end

# GPU-compatible grid_microphysical_state: directly access fields with @inbounds
# to bypass both the generic symbol-based NamedTuple extraction and OffsetArrays'
# bounds checking that the GPU compiler cannot compile.
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
    return P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ)
end

# GPU-compatible update_microphysical_fields! for P3.
# Bypasses the generic extract_microphysical_prognostics which uses runtime Symbol
# dispatch that GPU compilers cannot resolve. Instead, directly constructs
# P3MicrophysicalState from @inbounds field access and delegates to
# update_microphysical_auxiliaries!.
@inline function AM.update_microphysical_fields!(μ, i, j, k, grid, p3::P3, ρ, 𝒰, constants)
    @inbounds begin
        ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, 𝒰, (; u=zero(ρ), v=zero(ρ), w=zero(ρ)))
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
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ, ℳ.qʷⁱ)

    @inbounds μ.qᵛ[i, j, k]  = 𝒰.moisture_mass_fractions.vapor
    @inbounds μ.qᶜˡ[i, j, k] = ℳ.qᶜˡ
    @inbounds μ.nᶜˡ[i, j, k] = ℳ.nᶜˡ
    @inbounds μ.qʳ[i, j, k]  = ℳ.qʳ
    @inbounds μ.nʳ[i, j, k]  = ℳ.nʳ
    @inbounds μ.qⁱ[i, j, k]  = ℳ.qⁱ
    @inbounds μ.nⁱ[i, j, k]  = ℳ.nⁱ
    @inbounds μ.qᶠ[i, j, k]  = rime_state.qᶠ
    @inbounds μ.bᶠ[i, j, k]  = rime_state.bᶠ
    @inbounds μ.zⁱ[i, j, k]  = ℳ.zⁱ
    @inbounds μ.z̃ⁱ[i, j, k]  = μ.ρz̃ⁱ[i, j, k] / ρ
    @inbounds μ.qʷⁱ[i, j, k] = ℳ.qʷⁱ
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
    μ_ice :: FT
    μ_cloud :: FT
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
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    Fᶠ = props.Fᶠ
    ρᶠ = props.ρᶠ

    # Rain terminal velocities (separate functions — the rain fall-speed table is 1D)
    wʳ   = rain_terminal_velocity_mass_weighted(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    wʳₙ  = rain_terminal_velocity_number_weighted(p3, ℳ.qʳ, ℳ.nʳ, ρ)
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

    # Process rates (heavy, @noinline — compiled as a separate GPU function)
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

    # Tendency extraction
    c_qcl = tendency_ρqᶜˡ(rates, ρ)
    # Prescribed-Nᶜ path: nc is a scheme parameter (not advected); tendency = 0.
    c_ncl = isnothing(p3.aerosol) ? zero(typeof(ρ)) :
            tendency_ρnᶜˡ(rates, ρ, cloud.Nᶜ, ℳ.qᶜˡ, p3.process_rates)
    c_qr  = tendency_ρqʳ(rates, ρ)
    c_nr  = tendency_ρnʳ(rates, ρ, props.nⁱ, ℳ.qⁱ, ℳ.nʳ, ℳ.qʳ, p3.process_rates)
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

    FT = typeof(ρ)
    return P3CacheResult{FT}(wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z,
                              c_qcl, c_ncl, c_qr, c_nr, c_qi, c_ni, c_qf, c_bf, c_zi, c_qwi, c_ss, c_qv)
end

# Kernel entry point: reads OffsetArrays → calls @noinline scalar compute → writes OffsetArrays.
# Keeping array access in the kernel (inlined) and physics in @noinline (separate compilation)
# prevents the GPU compiler from seeing the full P3 physics + OffsetArray access together.
@inline function _p3_compute_and_cache!(μ, i, j, k, grid, p3::P3, ρ, 𝒰, constants)
    @inbounds begin
        ℳ = AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ, 𝒰, (; u=zero(ρ), v=zero(ρ), w=zero(ρ)))
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
    nⁱ_raw = min(ℳ.nⁱ, p3.process_rates.maximum_ice_number_density / ρ)
    nⁱ = ifelse(has_ice_mass, nⁱ_raw, FT(0))
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ, ℳ.qʷⁱ)
    qⁱ_total = max(qⁱ_raw, FT(1e-20))
    Fˡ = liquid_fraction_on_ice(ℳ.qⁱ, ℳ.qʷⁱ)
    μ_ice = compute_ice_shape_parameter(p3, qⁱ_total, nⁱ, ℳ.zⁱ, rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    zⁱ_bounded = bound_ice_sixth_moment(p3, qⁱ_total, nⁱ, ℳ.zⁱ, rime_state.Fᶠ, Fˡ, rime_state.ρᶠ, μ_ice)
    T = temperature(𝒰, constants)
    P = 𝒰.reference_pressure
    transport = air_transport_properties(T, P)
    λ_r = rain_slope_parameter(ℳ.qʳ, ℳ.nʳ, p3.process_rates)
    return P3IceProps{FT}(rime_state.qᶠ, rime_state.bᶠ, rime_state.Fᶠ, Fˡ,
                          rime_state.ρᶠ, qⁱ_total, nⁱ, μ_ice, cloud.μ_c,
                          zⁱ_bounded, transport.D_v, transport.nu, λ_r)
end

@inline function p3_rates_and_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    # Compute all process rates from microphysical state ℳ and thermodynamic state 𝒰
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    return rates, p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
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

"""
$(TYPEDSIGNATURES)

Cloud number tendency: gains from activation and loses proportionally with cloud sinks.

In the prescribed-Nᶜ path (`p3.aerosol === nothing`), `nc` is a scheme-level
parameter (Fortran `nccnst_2`), not a prognostic. The `ρnᶜˡ` field is still
allocated but carries no physical meaning, so the microphysical tendency is
zero and the field remains at its initial value.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    isnothing(p3.aerosol) && return zero(ρ)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    return tendency_ρnᶜˡ(rates, ρ, cloud.Nᶜ, ℳ.qᶜˡ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Cloud liquid tendency: loses mass to autoconversion, accretion, and riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶜˡ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Rain mass tendency: gains from autoconversion, accretion, melting, shedding; loses to evaporation, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʳ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Rain number tendency: gains from autoconversion, melting, shedding; loses to self-collection, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnʳ(rates, ρ, props.nⁱ, ℳ.qⁱ, ℳ.nʳ, ℳ.qʳ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Ice mass tendency: gains from deposition, riming, refreezing; loses to melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqⁱ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Ice number tendency: loses from melting and aggregation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnⁱ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Rime mass tendency: gains from cloud/rain riming, refreezing; loses proportionally with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶠ(rates, ρ, props.Fᶠ)
end

"""
$(TYPEDSIGNATURES)

Rime volume tendency: gains from new rime; loses with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρbᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρbᶠ(rates, ρ, props.Fᶠ, props.ρᶠ, ℳ.qⁱ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Ice sixth moment tendency: changes with deposition, melting, riming, and nucleation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρz̃ⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    # M13: Convert physical z tendency to advected z̃ = √(z·N) tendency
    tendency_ρz_phys = p3_ice_sixth_moment_tendency(ice_integrals_table(p3), p3, rates, ρ, ℳ, props)
    tendency_ρn = tendency_ρnⁱ(rates, ρ)
    ρz̃ⁱ = ρ * sqrt(max(0, ℳ.zⁱ * props.nⁱ))
    return z̃ⁱ_tendency(props.nⁱ, props.zⁱ_bounded, tendency_ρz_phys, tendency_ρn,
                        ρz̃ⁱ, p3.process_rates.sink_limiting_timescale)
end

"""
$(TYPEDSIGNATURES)

Liquid on ice tendency: loses from shedding and refreezing.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʷⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʷⁱ(rates, ρ)
end

"""
$(TYPEDSIGNATURES)

Supersaturation tendency: zero when predict_supersaturation = false.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρsˢᵃᵗ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates)
end

"""
$(TYPEDSIGNATURES)

Vapor tendency: loses from condensation, deposition, nucleation; gains from evaporation, sublimation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᵛ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᵛ(rates, ρ)
end

# Fallback for any unhandled field names - return zero tendency
@inline AM.microphysical_tendency(::P3, name, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) = zero(ρ)

#####
##### Thermodynamic state adjustment
#####

"""
$(TYPEDSIGNATURES)

Apply saturation adjustment for P3.

P3 is a non-equilibrium scheme - cloud formation and dissipation are handled
by explicit process rates, not instantaneous saturation adjustment.
Therefore, this function returns the state unchanged.
"""
@inline AM.maybe_adjust_thermodynamic_state(𝒰, ::P3, qᵛ, constants) = 𝒰

#####
##### Model update
#####

"""
$(TYPEDSIGNATURES)

Apply P3 model update during state update phase.

Launches a separate GPU kernel to compute terminal velocities, process rates,
and tendency cache fields. This heavy computation is split out of the thermodynamic
variables kernel to avoid overwhelming the GPU compiler with force-inlined P3 physics
(~1000 lines of code). The lighter `update_microphysical_auxiliaries!` only writes
basic diagnostic quantities in the thermodynamic kernel.
"""
function AM.microphysics_model_update!(p3::P3, model)
    grid = model.grid
    arch = grid.architecture
    μ = model.microphysical_fields
    ρ_field = AM.dynamics_density(model.dynamics)
    constants = model.thermodynamic_constants

    launch!(arch, grid, :xyz,
            _p3_compute_and_cache_kernel!,
            μ, model.formulation, model.dynamics, grid, constants, p3, ρ_field)

    return nothing
end

using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

@kernel function _p3_compute_and_cache_kernel!(μ, formulation, dynamics, grid, constants, p3, ρ_field)
    i, j, k = @index(Global, NTuple)

    @inbounds ρ = ρ_field[i, j, k]

    # Reconstruct thermodynamic state (same as in the thermodynamic kernel)
    ρqᵛᵉ = μ.qᵛ[i, j, k] * ρ  # qᵛ was already written by update_microphysical_auxiliaries!
    qᵛᵉ = μ.qᵛ[i, j, k]
    q = AM.moisture_fractions(p3, AM.grid_microphysical_state(i, j, k, grid, p3, μ, ρ,
            nothing, (; u=zero(ρ), v=zero(ρ), w=zero(ρ))), qᵛᵉ)
    𝒰₀ = AM.diagnose_thermodynamic_state(i, j, k, grid, formulation, dynamics, q)
    𝒰 = AM.maybe_adjust_thermodynamic_state(𝒰₀, p3, qᵛᵉ, constants)

    _p3_compute_and_cache!(μ, i, j, k, grid, p3, ρ, 𝒰, constants)
end

#####
##### Fused tendency override (fast path for AtmosphereModel)
#####
#
# `microphysics_model_update!` already wrote every cell's microphysics contribution
# into the `cache_ρ*` fields. The fused override simply `+=`s those cached values
# into `Gⁿ` in a single kernel launch after the dynamics tendency kernels run.
# The state-based `microphysical_tendency` methods above remain the gridless
# fallback used by ParcelModels.

@kernel function _add_p3_base_tendencies_kernel!(Gρqᵛ, Gρqᶜˡ, Gρnᶜˡ, Gρqʳ, Gρnʳ,
                                                 Gρqⁱ, Gρnⁱ, Gρqᶠ, Gρbᶠ,
                                                 Gρqʷⁱ, Gρsˢᵃᵗ, μ)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Gρqᵛ[i, j, k]   += μ.cache_ρqᵛ[i, j, k]
        Gρqᶜˡ[i, j, k]  += μ.cache_ρqᶜˡ[i, j, k]
        Gρnᶜˡ[i, j, k]  += μ.cache_ρnᶜˡ[i, j, k]
        Gρqʳ[i, j, k]   += μ.cache_ρqʳ[i, j, k]
        Gρnʳ[i, j, k]   += μ.cache_ρnʳ[i, j, k]
        Gρqⁱ[i, j, k]   += μ.cache_ρqⁱ[i, j, k]
        Gρnⁱ[i, j, k]   += μ.cache_ρnⁱ[i, j, k]
        Gρqᶠ[i, j, k]   += μ.cache_ρqᶠ[i, j, k]
        Gρbᶠ[i, j, k]   += μ.cache_ρbᶠ[i, j, k]
        Gρqʷⁱ[i, j, k]  += μ.cache_ρqʷⁱ[i, j, k]
        Gρsˢᵃᵗ[i, j, k] += μ.cache_ρsˢᵃᵗ[i, j, k]
    end
end

@kernel function _add_p3_z̃ⁱ_tendency_kernel!(Gρz̃ⁱ, cache_ρz̃ⁱ)
    i, j, k = @index(Global, NTuple)
    @inbounds Gρz̃ⁱ[i, j, k] += cache_ρz̃ⁱ[i, j, k]
end

function AM.compute_microphysical_tendencies!(p3::P3, model)
    grid = model.grid
    arch = grid.architecture
    G = model.timestepper.Gⁿ
    μ = model.microphysical_fields

    launch!(arch, grid, :xyz, _add_p3_base_tendencies_kernel!,
            G.ρqᵛ, G.ρqᶜˡ, G.ρnᶜˡ, G.ρqʳ, G.ρnʳ,
            G.ρqⁱ, G.ρnⁱ, G.ρqᶠ, G.ρbᶠ, G.ρqʷⁱ, G.ρsˢᵃᵗ, μ)

    if is_three_moment_ice(p3)
        launch!(arch, grid, :xyz, _add_p3_z̃ⁱ_tendency_kernel!, G.ρz̃ⁱ, μ.cache_ρz̃ⁱ)
    end

    return nothing
end

#####
##### Number concentration diagnostic
#####
#
# P3 carries prognostic number-density fields for cloud liquid, rain, and ice,
# so `number_concentration` just hands the requested field back. This keeps the
# diagnostic interface uniform with `OneMomentCloudMicrophysics` and
# `TwoMomentCloudMicrophysics`.

Microphysics.number_concentration(model, ::P3, ::Val{:rain}) =
    get(model.microphysical_fields, :ρnʳ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val{:cloud_liquid}) =
    get(model.microphysical_fields, :ρnᶜˡ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val{:ice}) =
    get(model.microphysical_fields, :ρnⁱ, nothing)

Microphysics.number_concentration(model, ::P3, ::Val) = nothing
