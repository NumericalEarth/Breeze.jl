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
    "Predicted supersaturation [kg/kg] (H10: Grabowski & Morrison 2008)"
    sˢᵃᵗ :: FT
end

#####
##### Prognostic field names
#####

"""
$(TYPEDSIGNATURES)

Return prognostic field names for the P3 scheme.

P3 v5.5 with 3-moment ice and predicted liquid fraction has 11 prognostic fields:
- Cloud: ρqᶜˡ, ρnᶜˡ
- Rain: ρqʳ, ρnʳ
- Ice: ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρzⁱ, ρqʷⁱ
- Supersaturation: ρsˢᵃᵗ (H10: Grabowski & Morrison 2008, inactive by default)
"""
function AM.prognostic_field_names(::P3)
    cloud_names = (:ρqᶜˡ, :ρnᶜˡ)
    rain_names = (:ρqʳ, :ρnʳ)
    ice_names = (:ρqⁱ, :ρnⁱ, :ρqᶠ, :ρbᶠ, :ρzⁱ, :ρqʷⁱ)
    # H10: supersaturation (always allocated; tendency = 0 when predict_supersaturation = false)
    ssat_names = (:ρsˢᵃᵗ,)

    return tuple(cloud_names..., rain_names..., ice_names..., ssat_names...)
end

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
- `ρzⁱ`: Ice sixth moment (reflectivity) density
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
    ρzⁱ  = CenterField(grid)  # Ice 6th moment
    ρqʷⁱ = CenterField(grid)  # Liquid on ice
    ρsˢᵃᵗ = CenterField(grid) # Predicted supersaturation (H10)

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
    cache_ρzⁱ  = CenterField(grid)
    cache_ρqʷⁱ = CenterField(grid)
    cache_ρsˢᵃᵗ = CenterField(grid)
    cache_ρqᵛ  = CenterField(grid)

    return (; ρqᶜˡ, ρnᶜˡ, ρqʳ, ρnʳ, ρqⁱ, ρnⁱ, ρqᶠ, ρbᶠ, ρzⁱ, ρqʷⁱ, ρsˢᵃᵗ,
              qᶜˡ, nᶜˡ, qʳ, nʳ, qⁱ, nⁱ, qᶠ, bᶠ, zⁱ, qʷⁱ, sˢᵃᵗ, qᵛ,
              wʳ, wʳₙ, wⁱ, wⁱₙ, wⁱ_z,
              cache_ρqᶜˡ, cache_ρnᶜˡ, cache_ρqʳ, cache_ρnʳ, cache_ρqⁱ, cache_ρnⁱ,
              cache_ρqᶠ, cache_ρbᶠ, cache_ρzⁱ, cache_ρqʷⁱ, cache_ρsˢᵃᵗ, cache_ρqᵛ)
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
@inline function AM.microphysical_state(p3::P3, ρ, μ, 𝒰, velocities)
    qᶜˡ = μ.ρqᶜˡ / ρ
    nᶜˡ = μ.ρnᶜˡ / ρ
    qʳ  = μ.ρqʳ / ρ
    nʳ  = μ.ρnʳ / ρ
    qⁱ  = μ.ρqⁱ / ρ
    nⁱ  = μ.ρnⁱ / ρ
    # M13: Fortran advects z̃ = √(z·N) and converts to physical z at microphysics entry:
    #   where (nitot > 0) zitot = zitot**2 / nitot; elsewhere zitot = 0
    # ρzⁱ stores the advected variable z̃; convert to physical z = z̃²/N for internal use.
    FT = typeof(ρ)
    z̃ⁱ  = μ.ρzⁱ / ρ
    zⁱ  = ifelse(nⁱ > FT(1e-20), z̃ⁱ^2 / nⁱ, zero(FT))
    qʷⁱ = μ.ρqʷⁱ / ρ
    rime_state = consistent_rime_state(p3, qⁱ, μ.ρqᶠ / ρ, μ.ρbᶠ / ρ, qʷⁱ)
    qᶠ  = rime_state.qᶠ
    bᶠ  = rime_state.bᶠ
    sˢᵃᵗ = μ.ρsˢᵃᵗ / ρ
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
using Oceananigans.Grids: OffsetArray
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
        nᶜˡ = μ.ρnᶜˡ[i, j, k] / ρ
        qʳ  = μ.ρqʳ[i, j, k] / ρ
        nʳ  = μ.ρnʳ[i, j, k] / ρ
        qⁱ  = μ.ρqⁱ[i, j, k] / ρ
        nⁱ  = μ.ρnⁱ[i, j, k] / ρ
        FT = typeof(ρ)
        z̃ⁱ  = μ.ρzⁱ[i, j, k] / ρ
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

# All P3 physics in a single @noinline function returning a concrete struct.
# compute_p3_process_rates (also @noinline) handles the heavy rates.
# All operations are scalar — no array access.
@noinline function _p3_scalar_compute(p3::P3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    props = p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    Fᶠ = props.Fᶠ
    ρᶠ = props.ρᶠ

    # Terminal velocities (individual calls avoid NamedTuple from ice_terminal_velocities)
    wʳ   = rain_terminal_velocity_mass_weighted(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    wʳₙ  = rain_terminal_velocity_number_weighted(p3, ℳ.qʳ, ℳ.nʳ, ρ)
    wⁱ   = ice_terminal_velocity_mass_weighted(p3, ℳ.qⁱ, ℳ.nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=props.Fˡ, μ=props.μ_ice)
    wⁱₙ  = ice_terminal_velocity_number_weighted(p3, ℳ.qⁱ, ℳ.nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=props.Fˡ, μ=props.μ_ice)
    wⁱ_z = ice_terminal_velocity_reflectivity_weighted(p3, ℳ.qⁱ, ℳ.nⁱ, Fᶠ, ρᶠ, ρ; Fˡ=props.Fˡ, μ=props.μ_ice)

    # Process rates (heavy, @noinline — compiled as a separate GPU function)
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)

    # Tendency extraction
    c_qcl = tendency_ρqᶜˡ(rates, ρ)
    c_ncl = tendency_ρnᶜˡ(rates, ρ, cloud.Nᶜ, ℳ.qᶜˡ, p3.process_rates)
    c_qr  = tendency_ρqʳ(rates, ρ)
    c_nr  = tendency_ρnʳ(rates, ρ, ℳ.nⁱ, ℳ.qⁱ, ℳ.nʳ, ℳ.qʳ, p3.process_rates)
    c_qi  = tendency_ρqⁱ(rates, ρ)
    c_ni  = tendency_ρnⁱ(rates, ρ)
    c_qf  = tendency_ρqᶠ(rates, ρ, Fᶠ)
    c_bf  = tendency_ρbᶠ(rates, ρ, Fᶠ, ρᶠ, ℳ.qⁱ, p3.process_rates)
    # Sixth moment tendency: use tabulated path when table_1 exists, analytic otherwise.
    # Direct call avoids dynamic dispatch on lookup_table_1(p3) return type in @noinline.
    tendency_ρz_phys = p3_ice_sixth_moment_tendency(lookup_table_1(p3), p3, rates, ρ, ℳ, props)
    z_phys = props.zⁱ_bounded
    FT = typeof(ρ)
    z̃ = sqrt(max(z_phys * ℳ.nⁱ, FT(1e-30)))
    c_zi = (ℳ.nⁱ * tendency_ρz_phys + z_phys * c_ni) / (2 * z̃)
    c_qwi = tendency_ρqʷⁱ(rates, ρ)
    c_ss  = tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates)
    c_qv  = tendency_ρqᵛ(rates, ρ)

    # GPU NaN guard: replace any NaN output with zero.
    # Oceananigans' table interpolator uses Base.unsafe_trunc(Int, fractional_idx),
    # which is undefined behavior when fractional_idx is NaN. If any upstream
    # intermediate becomes NaN on GPU (e.g., from FMA rounding differences),
    # it cascades through the table lookup into velocities and tendencies.
    # Zero is physically correct for the near-zero hydrometeor states where
    # these GPU-specific NaN values appear.
    wʳ   = ifelse(isnan(wʳ), zero(FT), wʳ)
    wʳₙ  = ifelse(isnan(wʳₙ), zero(FT), wʳₙ)
    wⁱ   = ifelse(isnan(wⁱ), zero(FT), wⁱ)
    wⁱₙ  = ifelse(isnan(wⁱₙ), zero(FT), wⁱₙ)
    wⁱ_z = ifelse(isnan(wⁱ_z), zero(FT), wⁱ_z)
    c_qcl = ifelse(isnan(c_qcl), zero(FT), c_qcl)
    c_ncl = ifelse(isnan(c_ncl), zero(FT), c_ncl)
    c_qr  = ifelse(isnan(c_qr),  zero(FT), c_qr)
    c_nr  = ifelse(isnan(c_nr),   zero(FT), c_nr)
    c_qi  = ifelse(isnan(c_qi),   zero(FT), c_qi)
    c_ni  = ifelse(isnan(c_ni),   zero(FT), c_ni)
    c_qf  = ifelse(isnan(c_qf),   zero(FT), c_qf)
    c_bf  = ifelse(isnan(c_bf),   zero(FT), c_bf)
    c_zi  = ifelse(isnan(c_zi),   zero(FT), c_zi)
    c_qwi = ifelse(isnan(c_qwi),  zero(FT), c_qwi)
    c_ss  = ifelse(isnan(c_ss),   zero(FT), c_ss)
    c_qv  = ifelse(isnan(c_qv),   zero(FT), c_qv)

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
        μ.cache_ρzⁱ[i, j, k]  = r.c_zi
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

# Ice reflectivity: reflectivity-weighted fall speed
@inline AM.microphysical_velocities(::P3, μ, ::Val{:ρzⁱ}) = (; u = ZeroField(), v = ZeroField(), w = μ.wⁱ_z)

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
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    rime_state = consistent_rime_state(p3, ℳ.qⁱ, ℳ.qᶠ, ℳ.bᶠ, ℳ.qʷⁱ)
    qⁱ_total = max(total_ice_mass(ℳ.qⁱ, ℳ.qʷⁱ), FT(1e-20))
    Fˡ = liquid_fraction_on_ice(ℳ.qⁱ, ℳ.qʷⁱ)
    μ_ice = compute_ice_shape_parameter(p3, qⁱ_total, ℳ.nⁱ, ℳ.zⁱ, rime_state.Fᶠ, Fˡ, rime_state.ρᶠ)
    zⁱ_bounded = bound_ice_sixth_moment(p3, qⁱ_total, ℳ.nⁱ, ℳ.zⁱ, rime_state.Fᶠ, Fˡ, rime_state.ρᶠ, μ_ice)
    T = temperature(𝒰, constants)
    P = 𝒰.reference_pressure
    transport = air_transport_properties(T, P)
    λ_r = rain_slope_parameter(ℳ.qʳ, ℳ.nʳ, p3.process_rates)
    return P3IceProps{FT}(rime_state.qᶠ, rime_state.bᶠ, rime_state.Fᶠ, Fˡ,
                          rime_state.ρᶠ, qⁱ_total, μ_ice, cloud.μ_c,
                          zⁱ_bounded, transport.D_v, transport.nu, λ_r)
end

@inline function p3_rates_and_properties(p3, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    # Compute all process rates from microphysical state ℳ and thermodynamic state 𝒰
    rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants)
    return rates, p3_ice_properties(p3, ρ, ℳ, 𝒰, constants)
end

@inline function p3_ice_sixth_moment_tendency(::Nothing, p3, rates, ρ, ℳ::P3MicrophysicalState, props::P3IceProps)
    return tendency_ρzⁱ(rates, ρ, props.qⁱ_total, ℳ.nⁱ, props.zⁱ_bounded, p3.process_rates, props.μ_cloud)
end

@inline function p3_ice_sixth_moment_tendency(::P3LookupTable1, p3, rates, ρ, ℳ::P3MicrophysicalState, props::P3IceProps)
    return tendency_ρzⁱ(rates, ρ, props.qⁱ_total, ℳ.nⁱ, props.zⁱ_bounded,
                        props.Fᶠ, props.Fˡ, props.ρᶠ, p3,
                        props.nu, props.D_v, props.μ_ice, props.μ_cloud, props.λ_r)
end

"""
Cloud number tendency: gains from activation and loses proportionally with cloud sinks.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    cloud = diagnose_cloud_dsd(p3, ℳ.qᶜˡ, ℳ.nᶜˡ, ρ)
    return tendency_ρnᶜˡ(rates, ρ, cloud.Nᶜ, ℳ.qᶜˡ, p3.process_rates)
end

"""
Cloud liquid tendency: loses mass to autoconversion, accretion, and riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶜˡ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶜˡ(rates, ρ)
end

"""
Rain mass tendency: gains from autoconversion, accretion, melting, shedding; loses to evaporation, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʳ(rates, ρ)
end

"""
Rain number tendency: gains from autoconversion, melting, shedding; loses to self-collection, riming.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnʳ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnʳ(rates, ρ, ℳ.nⁱ, ℳ.qⁱ, ℳ.nʳ, ℳ.qʳ, p3.process_rates)
end

"""
Ice mass tendency: gains from deposition, riming, refreezing; loses to melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqⁱ(rates, ρ)
end

"""
Ice number tendency: loses from melting and aggregation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρnⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρnⁱ(rates, ρ)
end

"""
Rime mass tendency: gains from cloud/rain riming, refreezing; loses proportionally with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᶠ(rates, ρ, props.Fᶠ)
end

"""
Rime volume tendency: gains from new rime; loses with melting.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρbᶠ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρbᶠ(rates, ρ, props.Fᶠ, props.ρᶠ, ℳ.qⁱ, p3.process_rates)
end

"""
Ice sixth moment tendency: changes with deposition, melting, riming, and nucleation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρzⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, props = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    # M13: Convert physical z tendency to advected z̃ = √(z·N) tendency
    FT = typeof(ρ)
    tendency_ρz_phys = p3_ice_sixth_moment_tendency(lookup_table_1(p3), p3, rates, ρ, ℳ, props)
    tendency_ρn = tendency_ρnⁱ(rates, ρ)
    z_phys = props.zⁱ_bounded
    z̃ = sqrt(max(z_phys * ℳ.nⁱ, FT(1e-30)))
    return (ℳ.nⁱ * tendency_ρz_phys + z_phys * tendency_ρn) / (2 * z̃)
end

"""
Liquid on ice tendency: loses from shedding and refreezing.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqʷⁱ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqʷⁱ(rates, ρ)
end

"""
Supersaturation tendency (H10): zero when predict_supersaturation = false.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρsˢᵃᵗ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρsˢᵃᵗ(rates, ρ, p3.process_rates)
end

"""
Vapor tendency: loses from condensation, deposition, nucleation; gains from evaporation, sublimation.
"""
@inline function AM.microphysical_tendency(p3::P3, ::Val{:ρqᵛ}, ρ, ℳ::P3MicrophysicalState, 𝒰, constants)
    rates, _ = p3_rates_and_properties(p3, ρ, ℳ, 𝒰, constants)
    return tendency_ρqᵛ(rates, ρ)
end

# Fallback for any unhandled field names - return zero tendency
@inline AM.microphysical_tendency(::P3, name, ρ, ℳ::P3MicrophysicalState, 𝒰, constants) = zero(ρ)

#####
##### Grid-indexed tendency overrides (fast path for AtmosphereModel)
#####
#
# These overrides read from the tendency cache populated by update_microphysical_auxiliaries!,
# bypassing recomputation of compute_p3_process_rates for each P3 prognostic field.
# The microphysical_tendency methods above remain the gridless fallback for ParcelModels.

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqᶜˡ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqᶜˡ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρnᶜˡ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρnᶜˡ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqʳ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqʳ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρnʳ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρnʳ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρnⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρnⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqᶠ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqᶠ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρbᶠ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρbᶠ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρzⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρzⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqʷⁱ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqʷⁱ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρsˢᵃᵗ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρsˢᵃᵗ[i, j, k]

@inline AM.grid_microphysical_tendency(i, j, k, grid, ::P3, ::Val{:ρqᵛ}, ρ, fields, 𝒰, constants, velocities) =
    @inbounds fields.cache_ρqᵛ[i, j, k]

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
