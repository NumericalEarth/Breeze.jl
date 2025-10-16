using ..Thermodynamics:
    AtmosphereThermodynamics,
    ReferenceStateConstants,
    saturation_specific_humidity
using Oceananigans.Architectures: AbstractArchitecture, CPU, on_architecture, architecture
using Oceananigans.Fields: ZeroField, interior
using Oceananigans.Grids: AbstractGrid
using Oceananigans: fields

"""
    Microphysics1MParameters(cloud_params)

Wrapper owning the CloudMicrophysics parameter bundle required by the
Morrison-Milbrandt one-moment scheme. This type prevents CloudMicrophysics
symbols from leaking through Breeze’s public interface while still enabling
downstream configuration via `getproperty`.
"""
struct Microphysics1MParameters{Params}
    cloud :: Params
end

Microphysics1MParameters(params::Microphysics1MParameters) = params
Microphysics1MParameters(params::CMP.ParametersType) = Microphysics1MParameters{typeof(params)}(params)
Microphysics1MParameters(params::NamedTuple) = Microphysics1MParameters{typeof(params)}(params)

Base.getproperty(p::Microphysics1MParameters, name::Symbol) =
    name === :cloud ? getfield(p, :cloud) : getproperty(getfield(p, :cloud), name)

Base.eltype(p::Microphysics1MParameters) = eltype(p.cloud)

"""
    Microphysics1MCache(arch, FT; workspace = NamedTuple())

Architecture-aware scratch space used by `Microphysics1M` to stage intermediate
arrays.
"""
struct Microphysics1MCache{FT,
                           Arc <: AbstractArchitecture,
                           VelocityArray,
                           TendenciesTuple,
                           DiagnosticsTuple,
                           ScalarRef,
                           UpdatedArray}
    rain_velocity :: VelocityArray
    snow_velocity :: VelocityArray
    tendencies :: TendenciesTuple
    diagnostics :: DiagnosticsTuple
    surface_precipitation_flux :: ScalarRef
    column_is_updated :: UpdatedArray
end

function Microphysics1MCache(arch::Arc, ::Type{FT}, grid::AbstractGrid; workspace = NamedTuple()) where {FT, Arc <: AbstractArchitecture}
    rain_velocity = on_architecture(arch, zeros(FT, grid.Nx, grid.Ny, grid.Nz + 1))
    snow_velocity = on_architecture(arch, zeros(FT, grid.Nx, grid.Ny, grid.Nz + 1))
    center_shape = (grid.Nx, grid.Ny, grid.Nz)
    center_array() = on_architecture(arch, zeros(FT, center_shape...))

    tendencies = (
        ρq_liq = center_array(),
        ρq_ice = center_array(),
        ρq_rai = center_array(),
        ρq_sno = center_array(),
        ρe_tot = center_array(),
        ρq_tot = center_array(),
    )

    diagnostics = (
        latent_condensation = center_array(),
        latent_deposition = center_array(),
        latent_freezing = center_array(),
        latent_melting = center_array(),
        latent_evaporation = center_array(),
        water_residual = center_array(),
    )

    surface_precipitation_flux = Ref(zero(FT))
    updated = on_architecture(arch, fill(false, center_shape...))

    return Microphysics1MCache{FT,
                               Arc,
                               typeof(rain_velocity),
                               typeof(tendencies),
                               typeof(diagnostics),
                               typeof(surface_precipitation_flux),
                               typeof(updated)}(rain_velocity,
                                                snow_velocity,
                                                tendencies,
                                                diagnostics,
                                                surface_precipitation_flux,
                                                updated)
end

"""
    Microphysics1M(; thermodynamics, reference_state, parameters, architecture = CPU(),
                     cache_builder = Microphysics1MCache)

Construct the Breeze wrapper for the Morrison-Milbrandt 2015 one-moment bulk
microphysics scheme provided by `CloudMicrophysics.jl`.

# Arguments
- `thermodynamics`: `AtmosphereThermodynamics` instance used by the host model.
- `reference_state`: `ReferenceStateConstants` consistent with the model grid.
- `parameters`: Microphysics1MParameters (or a value convertible to it) holding
  the CloudMicrophysics parameter bundle, typically the NamedTuple assembled by
  `CloudMicrophysics.Parameters.microphys_1m_parameters`.
- `architecture`: Oceananigans architecture describing the execution device.
- `cache_builder`: callable returning a `Microphysics1MCache` when provided with
  `(architecture, FT)`. Override this to supply custom scratch buffers.
"""
struct Microphysics1M{FT, ParamType, ThermoType, RefType, GridType, ArcType, CacheType} <: AbstractMicrophysics
    parameters :: ParamType
    thermodynamics :: ThermoType
    reference_state :: RefType
    grid :: GridType
    cache :: CacheType
    architecture :: ArcType
end

Base.eltype(::Microphysics1M{FT}) where {FT} = FT

function Microphysics1M(; thermodynamics::AtmosphereThermodynamics{FT},
                         reference_state::ReferenceStateConstants,
                         parameters,
                         grid::AbstractGrid,
                         architecture::AbstractArchitecture = CPU(),
                         cache_builder = Microphysics1MCache) where {FT}

    parameters === nothing &&
        throw(ArgumentError("Microphysics1M requires CloudMicrophysics parameters; pass the NamedTuple returned by CloudMicrophysics.Parameters.microphys_1m_parameters."))

    mp = Microphysics1MParameters(parameters)
    ref = _promote_reference_state(reference_state, FT)
    cache = cache_builder(architecture, FT, grid)

    return Microphysics1M{FT,
                          typeof(mp),
                          typeof(thermodynamics),
                          typeof(ref),
                          typeof(grid),
                          typeof(architecture),
                          typeof(cache)}(mp,
                                         thermodynamics,
                                         ref,
                                         grid,
                                         cache,
                                         architecture)
end

## (Removed) constants for legacy helper APIs

## (Removed) Column structs: we compute pointwise and write directly to caches

@inline _field_data(x) = interior(x)
@inline _field_data(x::AbstractArray) = x

@inline function reset_microphysics_cache!(cache)
    for array in values(cache.tendencies)
        fill!(array, zero(eltype(array)))
    end
    for array in values(cache.diagnostics)
        fill!(array, zero(eltype(array)))
    end
    fill!(cache.column_is_updated, false)
    cache.surface_precipitation_flux[] = zero(typeof(cache.surface_precipitation_flux[]))
    return nothing
end

@inline latent_constants(mp::Microphysics1M{FT}) where {FT} =
    (convert(FT, mp.thermodynamics.liquid.latent_heat),
     convert(FT, mp.thermodynamics.solid.latent_heat),
     convert(FT, latent_heat_fusion(mp.thermodynamics)))

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

@inline clip(q::FT) where {FT<:Real} = ifelse(q > zero(FT), q, zero(FT))

@inline function limit_available(q::FT, dt, n::Int) where {FT<:Real}
    q_pos = clip(q)
    dt_ft = convert(FT, dt)
    denom = dt_ft * FT(n)
    return denom == zero(FT) ? zero(FT) : q_pos / denom
end

@inline function triangle_inequality_limiter(force, limit)
    f, ℓ = promote(force, limit)
    f = max(zero(f), f)
    ℓ = max(zero(ℓ), ℓ)
    return f == zero(f) ? zero(f) : f + ℓ - sqrt(f * f + ℓ * ℓ)
end

@inline q_vapor(q_tot, q_liq, q_ice, q_rain, q_snow) =
    q_tot - q_liq - q_ice - q_rain - q_snow

@inline liquid_heat_capacity(thermo::AtmosphereThermodynamics) =
    thermo.liquid.heat_capacity

@inline latent_heat_fusion(thermo::AtmosphereThermodynamics) =
    thermo.solid.latent_heat - thermo.liquid.latent_heat

@inline function fusion_factor(mp::Microphysics1M{FT}, T::FT) where {FT}
    cv_l = convert(FT, liquid_heat_capacity(mp.thermodynamics))
    L_f = convert(FT, latent_heat_fusion(mp.thermodynamics))
    return (L_f == zero(FT)) ? zero(FT) : cv_l / L_f * (T - mp.parameters.ps.T_freeze)
end

# ------------------------------------------------------------------
# Microphysical process rates
# ------------------------------------------------------------------

"""
    cloud_liquid_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)

Return the net tendency (kg kg⁻¹ s⁻¹) for cloud liquid water produced by
supersaturation-driven condensation/evaporation. Positive values indicate vapor
condensing into cloud droplets; negative values correspond to evaporation.
"""
@inline function cloud_liquid_condensation_rate(
    mp::Microphysics1M{FT},
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
    q_rain::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    qv = q_vapor(q_tot, q_liq, q_ice, q_rain, q_snow)
    q_sat = convert(FT, saturation_specific_humidity(T, ρ, mp.thermodynamics, mp.thermodynamics.liquid))

    raw = qv + q_liq > FT(0) ?
        CMNe.conv_q_vap_to_q_liq_ice_MM2015(
            mp.parameters.cl,
            mp.thermodynamics,
            q_tot,
            q_liq,
            q_ice,
            q_rain,
            q_snow,
            ρ,
            T,
        ) :
        FT(0)

    return raw > FT(0) ?
        triangle_inequality_limiter(raw, limit_available(qv - q_sat, dt, 2)) :
        -triangle_inequality_limiter(-raw, limit_available(q_liq, dt, 2))
end

"""
    cloud_ice_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)

Return the net tendency (kg kg⁻¹ s⁻¹) for cloud ice resulting from deposition
or sublimation. Supersaturation relative to ice yields positive tendencies;
evaporation produces negative values. Ice formation is suppressed above the
snow freezing temperature.
"""
@inline function cloud_ice_condensation_rate(
    mp::Microphysics1M{FT},
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
    q_rain::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    qv = q_vapor(q_tot, q_liq, q_ice, q_rain, q_snow)
    q_sat = convert(FT, saturation_specific_humidity(T, ρ, mp.thermodynamics, mp.thermodynamics.solid))

    raw = qv + q_ice > FT(0) ?
        CMNe.conv_q_vap_to_q_liq_ice_MM2015(
            mp.parameters.ci,
            mp.thermodynamics,
            q_tot,
            q_liq,
            q_ice,
            q_rain,
            q_snow,
            ρ,
            T,
        ) :
        FT(0)

    if T > mp.parameters.ps.T_freeze && raw > FT(0)
        raw = FT(0)
    end

    return raw > FT(0) ?
        triangle_inequality_limiter(raw, limit_available(qv - q_sat, dt, 2)) :
        -triangle_inequality_limiter(-raw, limit_available(q_ice, dt, 2))
end

"""
    autoconversion_liquid_to_rain_rate(mp, q_liq, ρ, dt)

One-moment warm-rain autoconversion mass flux transferring cloud liquid water
into rain water. Returns a `NamedTuple(liq=-S, rain=S)` suitable for direct
accumulation into tracer tendencies.
"""
@inline function autoconversion_liquid_to_rain_rate(
    mp::Microphysics1M{FT},
    q_liq::FT,
    ρ::FT,
    dt::FT,
) where {FT}

    raw = mp.parameters.Ndp <= FT(0) ?
        CM1.conv_q_liq_to_q_rai(mp.parameters.pr.acnv1M, q_liq, true) :
        CM2.conv_q_liq_to_q_rai(mp.parameters.var, q_liq, ρ, mp.parameters.Ndp)

    rate = triangle_inequality_limiter(clip(raw), limit_available(q_liq, dt, 5))
    return (liq = -rate, rain = rate)
end

"""
    autoconversion_ice_to_snow_rate(mp, q_ice, dt)

Cold-phase autoconversion mapping cloud ice onto the snow category. Returns
`NamedTuple(ice=-S, snow=S)`.
"""
@inline function autoconversion_ice_to_snow_rate(
    mp::Microphysics1M{FT},
    q_ice::FT,
    dt::FT,
) where {FT}

    raw = CM1.conv_q_ice_to_q_sno_no_supersat(mp.parameters.ps.acnv1M, q_ice, true)
    rate = triangle_inequality_limiter(clip(raw), limit_available(q_ice, dt, 5))
    return (ice = -rate, snow = rate)
end

@inline function accretion_cloud_rain_rate(
    mp::Microphysics1M{FT},
    q_liq::FT,
    q_rain::FT,
    ρ::FT,
    dt::FT,
) where {FT}

    raw = CM1.accretion(mp.parameters.cl, mp.parameters.pr, mp.parameters.tv.rain, mp.parameters.ce, q_liq, q_rain, ρ)
    rate = triangle_inequality_limiter(clip(raw), limit_available(q_liq, dt, 5))
    return (liq = -rate, rain = rate)
end

@inline function accretion_ice_snow_rate(
    mp::Microphysics1M{FT},
    q_ice::FT,
    q_snow::FT,
    ρ::FT,
    dt::FT,
) where {FT}

    raw = CM1.accretion(mp.parameters.ci, mp.parameters.ps, mp.parameters.tv.snow, mp.parameters.ce, q_ice, q_snow, ρ)
    rate = triangle_inequality_limiter(clip(raw), limit_available(q_ice, dt, 5))
    return (ice = -rate, snow = rate)
end

@inline function accretion_cloud_snow_rate(
    mp::Microphysics1M{FT},
    q_liq::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    raw = CM1.accretion(mp.parameters.cl, mp.parameters.ps, mp.parameters.tv.snow, mp.parameters.ce, q_liq, q_snow, ρ)
    rate = triangle_inequality_limiter(clip(raw), limit_available(q_liq, dt, 5))

    if T < mp.parameters.ps.T_freeze
        snow_rate = rate
        rain_rate = zero(FT)
    else
        α = max(zero(FT), fusion_factor(mp, T))
        snow_rate = -triangle_inequality_limiter(rate * α, limit_available(q_snow, dt, 5))
        rain_rate = rate - snow_rate
    end

    return (liq = -rate, snow = snow_rate, rain = rain_rate)
end

@inline function accretion_ice_rain_to_snow_rate(
    mp::Microphysics1M{FT},
    q_ice::FT,
    q_rain::FT,
    ρ::FT,
    dt::FT,
) where {FT}

    raw = CM1.accretion(mp.parameters.ci, mp.parameters.pr, mp.parameters.tv.rain, mp.parameters.ce, q_ice, q_rain, ρ)
    rate = triangle_inequality_limiter(clip(raw), limit_available(q_ice, dt, 5))
    return (ice = -rate, snow = rate)
end

@inline function rain_sink_from_ice_rate(
    mp::Microphysics1M{FT},
    q_ice::FT,
    q_rain::FT,
    ρ::FT,
    dt::FT,
) where {FT}

    raw = CM1.accretion_rain_sink(mp.parameters.pr, mp.parameters.ci, mp.parameters.tv.rain, mp.parameters.ce, q_ice, q_rain, ρ)
    rate = triangle_inequality_limiter(clip(raw), limit_available(q_rain, dt, 5))
    return (rain = -rate, snow = rate)
end

@inline function accretion_snow_rain_rate(
    mp::Microphysics1M{FT},
    q_rain::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    if T < mp.parameters.ps.T_freeze
        raw = CM1.accretion_snow_rain(
            mp.parameters.ps,
            mp.parameters.pr,
            mp.parameters.tv.rain,
            mp.parameters.tv.snow,
            mp.parameters.ce,
            q_snow,
            q_rain,
            ρ,
        )
        rate = triangle_inequality_limiter(clip(raw), limit_available(q_rain, dt, 5))
    else
        raw = CM1.accretion_snow_rain(
            mp.parameters.pr,
            mp.parameters.ps,
            mp.parameters.tv.snow,
            mp.parameters.tv.rain,
            mp.parameters.ce,
            q_rain,
            q_snow,
            ρ,
        )
        rate = -triangle_inequality_limiter(clip(raw), limit_available(q_snow, dt, 5))
    end

    return (snow = rate, rain = -rate)
end

@inline function rain_evaporation_rate(
    mp::Microphysics1M{FT},
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
    q_rain::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    rps = (mp.parameters.pr, mp.parameters.tv.rain, mp.parameters.aps, mp.thermodynamics)
    raw = CM1.evaporation_sublimation(rps..., q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T)
    rate = -triangle_inequality_limiter(-raw, limit_available(q_rain, dt, 5))
    return (rain = rate)
end

@inline function snow_melt_rate(
    mp::Microphysics1M{FT},
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    raw = CM1.snow_melt(mp.parameters.ps, mp.parameters.tv.snow, mp.parameters.aps, mp.thermodynamics, q_snow, ρ, T)
    rate = triangle_inequality_limiter(clip(raw), limit_available(q_snow, dt, 5))
    return (snow = -rate, rain = rate)
end

@inline function snow_deposition_rate(
    mp::Microphysics1M{FT},
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
    q_rain::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    sps = (mp.parameters.ps, mp.parameters.tv.snow, mp.parameters.aps, mp.thermodynamics)
    raw = CM1.evaporation_sublimation(sps..., q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T)

    if raw > FT(0)
        rate = triangle_inequality_limiter(raw, limit_available(q_vapor(q_tot, q_liq, q_ice, q_rain, q_snow), dt, 5))
    else
        rate = -triangle_inequality_limiter(-raw, limit_available(q_snow, dt, 5))
    end

    return (snow = rate)
end

# ------------------------------------------------------------------
# Aggregated process suites
# ------------------------------------------------------------------

"""
    precipitation_source_rates(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)

Return the net source terms for the four hydrometeor categories produced by
autoconversion and accretion processes. The returned `NamedTuple` reports mass
mixing ratio tendencies (per second) for `(liq, ice, rain, snow)`, suitable for
direct accumulation into tracer tendency storage.
"""
@inline function precipitation_source_rates(
    mp::Microphysics1M{FT},
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
    q_rain::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    liq = zero(FT)
    ice = zero(FT)
    rain = zero(FT)
    snow = zero(FT)

    for contrib in (
        autoconversion_liquid_to_rain_rate(mp, q_liq, ρ, dt),
        autoconversion_ice_to_snow_rate(mp, q_ice, dt),
        accretion_cloud_rain_rate(mp, q_liq, q_rain, ρ, dt),
        accretion_ice_snow_rate(mp, q_ice, q_snow, ρ, dt),
        accretion_cloud_snow_rate(mp, q_liq, q_snow, ρ, T, dt),
        accretion_ice_rain_to_snow_rate(mp, q_ice, q_rain, ρ, dt),
        rain_sink_from_ice_rate(mp, q_ice, q_rain, ρ, dt),
        accretion_snow_rain_rate(mp, q_rain, q_snow, ρ, T, dt),
    )
        hasproperty(contrib, :liq) && (liq += getproperty(contrib, :liq))
        hasproperty(contrib, :ice) && (ice += getproperty(contrib, :ice))
        hasproperty(contrib, :rain) && (rain += getproperty(contrib, :rain))
        hasproperty(contrib, :snow) && (snow += getproperty(contrib, :snow))
    end

    return (liq = liq, ice = ice, rain = rain, snow = snow)
end

"""
    precipitation_sink_rates(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)

Return the net sink/source contributions associated with rain evaporation,
snow melting, and snow deposition/sublimation. The result is a `NamedTuple`
over `(liq, ice, rain, snow)` mirroring the structure of
`precipitation_source_rates`.
"""
@inline function precipitation_sink_rates(
    mp::Microphysics1M{FT},
    q_tot::FT,
    q_liq::FT,
    q_ice::FT,
    q_rain::FT,
    q_snow::FT,
    ρ::FT,
    T::FT,
    dt::FT,
) where {FT}

    liq = zero(FT)
    ice = zero(FT)
    rain = zero(FT)
    snow = zero(FT)

    for contrib in (
        rain_evaporation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt),
        snow_melt_rate(mp, q_snow, ρ, T, dt),
        snow_deposition_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt),
    )
        hasproperty(contrib, :liq) && (liq += getproperty(contrib, :liq))
        hasproperty(contrib, :ice) && (ice += getproperty(contrib, :ice))
        hasproperty(contrib, :rain) && (rain += getproperty(contrib, :rain))
        hasproperty(contrib, :snow) && (snow += getproperty(contrib, :snow))
    end

    return (liq = liq, ice = ice, rain = rain, snow = snow)
end

# ------------------------------------------------------------------
# OceanBioME-style update hook
# ------------------------------------------------------------------

@inline function _extract_dt(clock, ::Type{FT}) where {FT}
    candidates = (:last_Δt, :last_stage_Δt, :Δt, :dt)
    for name in candidates
        if hasproperty(clock, name)
            dt_val = getproperty(clock, name)
            dt_ft = convert(FT, dt_val)
            return isfinite(dt_ft) ? dt_ft : one(FT)
        end
    end
    return one(FT)
end

function update_microphysics_state!(microphysics::Microphysics1M, model)
    grid = model.grid
    clock = model.clock
    model_fields = fields(model)
    FT = eltype(microphysics)
    cache = microphysics.cache

    reset_microphysics_cache!(cache)

    state_arrays = (
        density = _field_data(getproperty(model_fields, :density)),
        ρq_tot = _field_data(getproperty(model_fields, :ρq_tot)),
        ρq_liq = _field_data(getproperty(model_fields, :ρq_liq)),
        ρq_ice = _field_data(getproperty(model_fields, :ρq_ice)),
        ρq_rai = _field_data(getproperty(model_fields, :ρq_rai)),
        ρq_sno = _field_data(getproperty(model_fields, :ρq_sno)),
        temperature = _field_data(getproperty(model_fields, :temperature)),
    )

    Lv, Ls, Lf = latent_constants(microphysics)
    dt_ft = _extract_dt(clock, FT)

    # Column totals removed; diagnostics are pointwise in cache

    Nx, Ny, Nz = size(cache.column_is_updated)

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ρi = convert(FT, state_arrays.density[i, j, k])
        # Positive density assumed

        invρ = inv(ρi)
        q_tot = convert(FT, state_arrays.ρq_tot[i, j, k]) * invρ
        q_liq = convert(FT, state_arrays.ρq_liq[i, j, k]) * invρ
        q_ice = convert(FT, state_arrays.ρq_ice[i, j, k]) * invρ
        q_rain = convert(FT, state_arrays.ρq_rai[i, j, k]) * invρ
        q_snow = convert(FT, state_arrays.ρq_sno[i, j, k]) * invρ
        Ti = convert(FT, state_arrays.temperature[i, j, k])

        # Phase-change tendencies (kg kg⁻¹ s⁻¹)
        cond_liq = cloud_liquid_condensation_rate(microphysics, q_tot, q_liq, q_ice, q_rain, q_snow, ρi, Ti, dt_ft)
        cond_ice = cloud_ice_condensation_rate(microphysics, q_tot, q_liq, q_ice, q_rain, q_snow, ρi, Ti, dt_ft)

        liq_rate = cond_liq
        ice_rate = cond_ice
        rain_rate = zero(FT)
        snow_rate = zero(FT)

        # Energy and latent diagnostics
        energy_specific = Lv * cond_liq + Ls * cond_ice
        latent_cond = ρi * Lv * cond_liq
        latent_dep = ρi * Ls * cond_ice
        latent_freeze = zero(FT)
        latent_melt = zero(FT)
        latent_evap = zero(FT)

        # Warm/cold autoconversion and accretion processes
        auto_lr = autoconversion_liquid_to_rain_rate(microphysics, q_liq, ρi, dt_ft)
        liq_rate += auto_lr.liq; rain_rate += auto_lr.rain

        auto_is = autoconversion_ice_to_snow_rate(microphysics, q_ice, dt_ft)
        ice_rate += auto_is.ice; snow_rate += auto_is.snow

        acc_lr = accretion_cloud_rain_rate(microphysics, q_liq, q_rain, ρi, dt_ft)
        liq_rate += acc_lr.liq; rain_rate += acc_lr.rain

        acc_is = accretion_ice_snow_rate(microphysics, q_ice, q_snow, ρi, dt_ft)
        ice_rate += acc_is.ice; snow_rate += acc_is.snow

        acc_ls = accretion_cloud_snow_rate(microphysics, q_liq, q_snow, ρi, Ti, dt_ft)
        liq_rate += acc_ls.liq; snow_rate += acc_ls.snow; rain_rate += acc_ls.rain
        energy_specific += Lf * acc_ls.snow
        if acc_ls.snow >= zero(FT)
            latent_freeze += ρi * Lf * acc_ls.snow
        else
            latent_melt += ρi * Lf * acc_ls.snow
        end

        acc_ir = accretion_ice_rain_to_snow_rate(microphysics, q_ice, q_rain, ρi, dt_ft)
        ice_rate += acc_ir.ice; snow_rate += acc_ir.snow

        rain_sink = rain_sink_from_ice_rate(microphysics, q_ice, q_rain, ρi, dt_ft)
        rain_rate += rain_sink.rain; snow_rate += rain_sink.snow
        energy_specific += Lf * rain_sink.snow
        latent_freeze += ρi * Lf * rain_sink.snow

        acc_sr = accretion_snow_rain_rate(microphysics, q_rain, q_snow, ρi, Ti, dt_ft)
        snow_rate += acc_sr.snow; rain_rate += acc_sr.rain
        energy_specific += Lf * acc_sr.snow
        if acc_sr.snow >= zero(FT)
            latent_freeze += ρi * Lf * acc_sr.snow
        else
            latent_melt += ρi * Lf * acc_sr.snow
        end

        rain_evap = rain_evaporation_rate(microphysics, q_tot, q_liq, q_ice, q_rain, q_snow, ρi, Ti, dt_ft)
        rain_rate += rain_evap.rain
        energy_specific += Lv * rain_evap.rain
        latent_evap += ρi * Lv * rain_evap.rain

        snow_melt = snow_melt_rate(microphysics, q_snow, ρi, Ti, dt_ft)
        snow_rate += snow_melt.snow; rain_rate += snow_melt.rain
        energy_specific += Lf * snow_melt.snow
        latent_melt += ρi * Lf * snow_melt.snow

        snow_dep = snow_deposition_rate(microphysics, q_tot, q_liq, q_ice, q_rain, q_snow, ρi, Ti, dt_ft)
        snow_rate += snow_dep.snow
        energy_specific += Ls * snow_dep.snow
        latent_dep += ρi * Ls * snow_dep.snow

        # Mass and diagnostics accumulation
        mass_sum = liq_rate + ice_rate + rain_rate + snow_rate
        water_residual = ρi * mass_sum

        cache.tendencies.ρq_liq[i, j, k] = ρi * liq_rate
        cache.tendencies.ρq_ice[i, j, k] = ρi * ice_rate
        cache.tendencies.ρq_rai[i, j, k] = ρi * rain_rate
        cache.tendencies.ρq_sno[i, j, k] = ρi * snow_rate
        cache.tendencies.ρe_tot[i, j, k] = ρi * energy_specific
        cache.tendencies.ρq_tot[i, j, k] = water_residual

        cache.diagnostics.latent_condensation[i, j, k] = latent_cond
        cache.diagnostics.latent_deposition[i, j, k] = latent_dep
        cache.diagnostics.latent_freezing[i, j, k] = latent_freeze
        cache.diagnostics.latent_melting[i, j, k] = latent_melt
        cache.diagnostics.latent_evaporation[i, j, k] = latent_evap
        cache.diagnostics.water_residual[i, j, k] = water_residual

        # no column totals

        cache.column_is_updated[i, j, k] = true
    end

    # Update drift velocities and surface precipitation flux
    update_microphysics_drift_velocity!(microphysics, model_fields)
    surface_precip_flux = surface_precipitation_flux(microphysics, model_fields)
    cache.surface_precipitation_flux[] = convert(FT, surface_precip_flux)

    return nothing
end


@inline function tendency_from_cache(microphysics::Microphysics1M, name::Symbol, i, j, k)
    cache = microphysics.cache
    @boundscheck cache.column_is_updated[i, j, k] || throw(ArgumentError("Microphysics1M column tendencies are stale. Call `update_microphysics_state!` before requesting transitions."))
    return getproperty(cache.tendencies, name)[i, j, k]
end

@inline _sym(::Val{S}) where {S} = S

# Functor-centric transition: single source of truth
@inline (microphysics::Microphysics1M)(i, j, k, grid, val_tracer_name::Val, clock, model_fields) =
    tendency_from_cache(microphysics, _sym(val_tracer_name), i, j, k)

# (Removed adapters and transition wrappers; functor is the only entry point)

function microphysics_auxiliary_fields(microphysics::Microphysics1M)
    diagnostics = microphysics.cache.diagnostics
    return (
        latent_condensation = diagnostics.latent_condensation,
        latent_deposition = diagnostics.latent_deposition,
        latent_freezing = diagnostics.latent_freezing,
        latent_melting = diagnostics.latent_melting,
        latent_evaporation = diagnostics.latent_evaporation,
        water_residual = diagnostics.water_residual,
        surface_precipitation_flux = microphysics.cache.surface_precipitation_flux[],
    )
end

## (Removed) _column_result helper

@inline (microphysics::Microphysics1M)(::Val{:ρq_liq}, args...) = throw(ArgumentError("Functor-style API is not supported; use microphysics_transition and update_microphysics_state!"))
@inline (microphysics::Microphysics1M)(::Val{:ρq_ice}, args...) = throw(ArgumentError("Functor-style API is not supported; use microphysics_transition and update_microphysics_state!"))
@inline (microphysics::Microphysics1M)(::Val{:ρq_rai}, args...) = throw(ArgumentError("Functor-style API is not supported; use microphysics_transition and update_microphysics_state!"))
@inline (microphysics::Microphysics1M)(::Val{:ρq_sno}, args...) = throw(ArgumentError("Functor-style API is not supported; use microphysics_transition and update_microphysics_state!"))
@inline (microphysics::Microphysics1M)(::Val{:ρq_tot}, args...) = throw(ArgumentError("Functor-style API is not supported; use microphysics_transition and update_microphysics_state!"))
@inline (microphysics::Microphysics1M)(::Val{:ρe_tot}, args...) = throw(ArgumentError("Functor-style API is not supported; use microphysics_transition and update_microphysics_state!"))

Base.summary(m::Microphysics1M{FT}) where {FT} =
    "Microphysics1M{$FT}(architecture = $(nameof(typeof(m.architecture))))"

function Base.show(io::IO, m::Microphysics1M)
    print(io, summary(m), ":\n",
          "├── parameters: ", nameof(typeof(m.parameters)), "\n",
          "├── thermodynamics: ", summary(m.thermodynamics), "\n",
          "├── reference_state: ReferenceStateConstants\n",
          "├── grid: ", nameof(typeof(m.grid)), "\n",
          "└── cache: ", nameof(typeof(m.cache)))
end

_promote_reference_state(reference_state::ReferenceStateConstants{FT}, ::Type{FT}) where {FT} = reference_state

function _promote_reference_state(reference_state::ReferenceStateConstants, ::Type{FT}) where {FT}
    return ReferenceStateConstants(FT;
                                   base_pressure = reference_state.base_pressure,
                                   potential_temperature = reference_state.reference_potential_temperature)
end

function update_microphysics_drift_velocity!(microphysics::Microphysics1M, state)
    # Face-centered, device-agnostic broadcast terminal velocities
    FT = eltype(microphysics)
    rv = interior(microphysics.cache.rain_velocity)
    sv = interior(microphysics.cache.snow_velocity)

    ρc      = _field_data(state.density)
    ρq_rai_c = _field_data(state.ρq_rai)
    ρq_sno_c = _field_data(state.ρq_sno)

    Nx, Ny, Nz = size(ρc)

    # Compute face-centered densities and (ρq) using simple vertical averaging
    # Bottom face uses first center value (one-sided), interior faces use average
    @inbounds begin
        # bottom face k=1 from cell k=1
        rv[:, :, 1] .= CM1.terminal_velocity.(microphysics.parameters.pr,
                                              microphysics.parameters.tv.rain,
                                              ρc[:, :, 1],
                                              ρq_rai_c[:, :, 1] ./ ρc[:, :, 1])
        sv[:, :, 1] .= CM1.terminal_velocity.(microphysics.parameters.ps,
                                              microphysics.parameters.tv.snow,
                                              ρc[:, :, 1],
                                              ρq_sno_c[:, :, 1] ./ ρc[:, :, 1])

        if Nz > 1
            # interior faces k=2..Nz from average of adjacent centers k-1,k
            ρf      = (ρc[:, :, 1:Nz-1]      .+ ρc[:, :, 2:Nz])      .* (FT(0.5))
            ρq_rai_f = (ρq_rai_c[:, :, 1:Nz-1] .+ ρq_rai_c[:, :, 2:Nz]) .* (FT(0.5))
            ρq_sno_f = (ρq_sno_c[:, :, 1:Nz-1] .+ ρq_sno_c[:, :, 2:Nz]) .* (FT(0.5))

            rv[:, :, 2:Nz] .= CM1.terminal_velocity.(microphysics.parameters.pr,
                                                     microphysics.parameters.tv.rain,
                                                     ρf, ρq_rai_f ./ ρf)
            sv[:, :, 2:Nz] .= CM1.terminal_velocity.(microphysics.parameters.ps,
                                                     microphysics.parameters.tv.snow,
                                                     ρf, ρq_sno_f ./ ρf)
        end

        # top face k=Nz+1 zero
        rv[:, :, Nz + 1] .= zero(FT)
        sv[:, :, Nz + 1] .= zero(FT)
    end

    return nothing
end

function surface_precipitation_flux(microphysics::Microphysics1M, state)
    FT = eltype(microphysics)
    rain_velocity = microphysics.cache.rain_velocity
    snow_velocity = microphysics.cache.snow_velocity
    ρq_rai = _field_data(state.ρq_rai)
    ρq_sno = _field_data(state.ρq_sno)

    bottom = rain_velocity[:, :, 1] .* ρq_rai[:, :, 1] .+
             snow_velocity[:, :, 1] .* ρq_sno[:, :, 1]

    return sum(bottom) / FT(size(bottom, 1) * size(bottom, 2))
end

@inline _has_key(container::NamedTuple, key::Symbol) = hasproperty(container, key)
@inline _has_key(container::AbstractDict, key::Symbol) = haskey(container, key)
@inline _has_key(container, key::Symbol) = hasproperty(container, key)

function _assert_required(required::Tuple, container, label)
    missing = Tuple(key for key in required if !_has_key(container, key))
    isempty(missing) && return nothing

    requirement = join(string.(missing), ", ")
    throw(ArgumentError("Missing $(label) entries required by Microphysics1M: $(requirement)."))
end


function microphysics_drift_velocity(microphysics::Microphysics1M, ::Val{:ρq_rai})
    zero_field = ZeroField()
    return (u = zero_field, v = zero_field, w = microphysics.cache.rain_velocity)
end

function microphysics_drift_velocity(microphysics::Microphysics1M, ::Val{:ρq_sno})
    zero_field = ZeroField()
    return (u = zero_field, v = zero_field, w = microphysics.cache.snow_velocity)
end
