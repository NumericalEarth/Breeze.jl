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
struct Microphysics1MCache{RV,
                           SV,
                           TND,}
    rain_velocity :: RV
    snow_velocity :: SV
    tendencies :: TND
end

function Microphysics1MCache(grid::AbstractGrid; workspace = NamedTuple())
    rain_velocity = ZFaceField(grid)
    snow_velocity = ZFaceField(grid)
    tendencies = (
        ρq_liq = CenterField(grid),
        ρq_ice = CenterField(grid),
        ρq_rai = CenterField(grid),
        ρq_sno = CenterField(grid),
        ρe_tot = CenterField(grid),
    )
    
    return Microphysics1MCache(rain_velocity,
                               snow_velocity,
                               tendencies)
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
struct Microphysics1M{PS, TH, RS} <: AbstractMicrophysics
    parameters :: PS
    thermodynamics :: TH
    reference_state :: RS
end

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

    return Microphysics1M(mp, thermodynamics, ref)
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
    FT = eltype(grid)
    cache = microphysics.cache

    reset_microphysics_cache!(cache)

    # tracer fields
    state_arrays = (
        ρ = _field_data(getproperty(model_fields, :ρ)),
        ρq_tot = _field_data(getproperty(model_fields, :ρq_tot)),
        ρq_liq = _field_data(getproperty(model_fields, :ρq_liq)),
        ρq_ice = _field_data(getproperty(model_fields, :ρq_ice)),
        ρq_rai = _field_data(getproperty(model_fields, :ρq_rai)),
        ρq_sno = _field_data(getproperty(model_fields, :ρq_sno)),
        T = _field_data(getproperty(model_fields, :temperature)),
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
