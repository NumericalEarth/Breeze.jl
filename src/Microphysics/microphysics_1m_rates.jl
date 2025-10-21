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