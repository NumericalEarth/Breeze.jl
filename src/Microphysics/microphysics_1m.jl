import CloudMicrophysics.MicrophysicsNonEq as CMNe
import CloudMicrophysics.Microphysics1M as CM1
import Thermodynamics as TD

struct Microphysics1M{FT} <: AbstractMicrophysics
end

required_microphysics_tracers(::Microphysics1M) = (
    :ρe_tot, # total moist static energy
    :ρq_tot, # total non-precipitating water
    :ρq_liq, # liquid non-precipitating water
    :ρq_ice, # ice non-precipitating water
    :ρq_rai, # precipitating rain
    :ρq_sno, # precipitating snow
)

# total moist static energy
function (::Microphysics1M)(::Val{:ρe_tot}, x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
end

# total water (non-precipitating and precipitating)
function (::Microphysics1M)(::Val{:ρq_tot}, x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
end

# non-precipitating hydrometeors (liquid and ice)
function (::Microphysics1M)(::Val{:ρq_liq}, x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
end

function (::Microphysics1M)(::Val{:ρq_ice}, x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
end

# precipitating hydrometeors (rain and snow)
function (::Microphysics1M)(::Val{:ρq_rai}, x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
end

function (::Microphysics1M)(::Val{:ρq_sno}, x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
end

# condensation of vapor to liquid: q_vap -> q_liq
function condensation_vapor_to_liquid(x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
    
    FT = typeof(ρ)
    ρq_vap = ρq_tot - ρq_liq - ρq_ice - ρq_rai - ρq_sno
    T = TD.air_temperature(thp, ts)
    p_sat_liq = TD.saturation_vapor_pressure(thp, T, TD.Liquid())
    q_sat_liq = TD.q_vap_from_p_vap(thp, T, ρ, p_sat_liq)

    if ρq_vap + ρq_liq > FT(0)
        S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(
            cm.liq,
            thp,
            ρq_tot / ρ,
            ρq_liq / ρ,
            ρq_ice / ρ,
            ρq_rai / ρ,
            ρq_sno / ρ,
            ρ,
            T,
        )
    else
        S = FT(0)
    end

    return ifelse(
        S > FT(0),
        triangle_inequality_limiter(S, limit(ρq_vap / ρ - q_sat_liq, dt, 2)),
        -triangle_inequality_limiter(abs(S), limit(ρq_liq / ρ, dt, 2)),
    )
end

# condensation of vapor to ice: q_vap -> q_ice
function condensation_vapor_to_ice(x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args

    FT = typeof(ρ)
    ρq_vap = ρq_tot - ρq_liq - ρq_ice - ρq_rai - ρq_sno
    T = TD.air_temperature(thp, ts)
    p_sat_ice = TD.saturation_vapor_pressure(thp, T, TD.Ice())
    q_sat_ice = TD.q_vap_from_p_vap(thp, T, ρ, p_sat_ice)

    if ρq_vap + ρq_ice > FT(0)
        S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(
            cm.ice,
            thp,
            ρq_tot / ρ,
            ρq_liq / ρ,
            ρq_ice / ρ,
            ρq_rai / ρ,
            ρq_sno / ρ,
            ρ,
            T,
        )
    else
        S = FT(0)
    end

    # Additional condition to avoid creating ice in conditions above freezing
    # Representing the lack of INPs in warm temperatures
    if T > thp.T_freeze && S > FT(0)
        S = FT(0)
    end

    return ifelse(
        S > FT(0),
        triangle_inequality_limiter(S, limit(ρq_vap / ρ - q_sat_ice, dt, 2)),
        -triangle_inequality_limiter(abs(S), limit(ρq_ice / ρ, dt, 2)),
    )
end

# rain autoconversion: q_liq -> q_rain
function autoconversion_liquid_to_rain(x, y, z, t, args...)
    (; ρ, dt) = args

    # mp ?????????????
    # qₗ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????
    # qᵣ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????

    FT = typeof(ρ)
    S = ifelse(
        mp.Ndp <= 0,
        CM1.conv_q_liq_to_q_rai(mp.pr.acnv1M, ρq_liq / ρ, true),
        CM2.conv_q_liq_to_q_rai(mp.var, ρq_liq / ρ, ρ, mp.Ndp),
    )
    S = triangle_inequality_limiter(Sᵖ, limit(ρq_liq / ρ, dt, 5))

    return S
    # @. Sq_liq -= Sᵖ
    # @. Sq_rai += Sᵖ
end

# snow autoconversion assuming no supersaturation: q_ice -> q_snow
function autoconversion_ice_to_snow(x, y, z, t, args...)
    (; ρ, dt) = args

    # mp ?????????????
    # qᵢ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Ice())) ?????????????
    # qₛ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????

    S = triangle_inequality_limiter(
        CM1.conv_q_ice_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ, true),
        limit(qᵢ, dt, 5),
    )

    return S
    # @. Sqᵢᵖ -= Sᵖ
    # @. Sqₛᵖ += Sᵖ
end

# accretion: q_liq + q_rain -> q_rain
function accretion_liquid_and_rain_to_rain(x, y, z, t, args...)
    (; ρ, dt) = args

    # mp ?????????????
    # qₗ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????
    # qᵣ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????

    S = triangle_inequality_limiter(
        CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ, qᵣ, ρ),
        limit(qₗ, dt, 5),
    )

    return S
    # @. Sqₗᵖ -= Sᵖ
    # @. Sqᵣᵖ += Sᵖ
end

# accretion: q_ice + q_snow -> q_snow
function accretion_ice_and_snow_to_snow(x, y, z, t, args...)
    (; ρ, dt) = args

    # mp ?????????????
    # qᵢ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Ice())) ?????????????
    # qₛ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????

    S = triangle_inequality_limiter(
        CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ, qₛ, ρ),
        limit(qᵢ, dt, 5),
    )

    return S
    # @. Sqᵢᵖ -= Sᵖ
    # @. Sqₛᵖ += Sᵖ
end

# accretion: q_liq + q_sno -> q_sno or q_rai
function accretion_liquid_and_snow_to_snow_or_rain(x, y, z, t, args...)
    (; ρ, thp, ts, dt) = args

    # sink of cloud water via accretion cloud water + snow
    # mp ?????????????
    # qₗ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????
    # qₛ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????
    # Tₐ = TD.air_temperature(thp, ts) ?????????????

    Sᵖ = triangle_inequality_limiter(
        CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ, qₛ, ρ),
        limit(qₗ, dt, 5),
    )
    # if T < T_freeze cloud droplets freeze to become snow
    # else the snow melts and both cloud water and snow become rain
    α(thp, ts) = TD.Parameters.cv_l(thp) / TD.latent_heat_fusion(thp, ts) * (Tₐ(thp, ts) - mp.ps.T_freeze)
    S_snow = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        Sᵖ,
        FT(-1) * triangle_inequality_limiter(Sᵖ * α(thp, ts), limit(qₛ, dt, 5)),
    )

    return S, S_snow
    # @. Sqₛᵖ += Sᵖ_snow
    # @. Sqₗᵖ -= Sᵖ
    # @. Sqᵣᵖ += ifelse(Tₐ(thp, ts) < mp.ps.T_freeze, FT(0), Sᵖ - Sᵖ_snow)
end

# accretion: q_ice + q_rai -> q_sno
function accretion_ice_and_rain_to_snow(x, y, z, t, args...)
    (; ρ, dt) = args

    # mp ?????????????
    # qᵢ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Ice())) ?????????????
    # qᵣ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????

    # sink of ice via accretion cloud ice - rain
    S_from_ice = triangle_inequality_limiter(
        CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ, qᵣ, ρ),
        limit(qᵢ, dt, 5),
    )

    # sink of rain via accretion cloud ice - rain
    S_from_rain = triangle_inequality_limiter(
        CM1.accretion_rain_sink(cmp.pr, cmp.ci, cmp.tv.rain, cmp.ce, qᵢ, qᵣ, ρ),
        limit(qᵣ, dt, 5),
    )

    return S_from_ice, S_from_rain
    # @. Sqᵢᵖ -= S_from_ice
    # @. Sqₛᵖ += S_from_ice
    # @. Sqᵣᵖ -= S_from_rain
    # @. Sqₛᵖ += S_from_rain
end

# accretion: q_rai + q_sno -> q_rai or q_sno
function accretion_rain_and_snow_to_rain_or_snow(x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
    
    # mp ?????????????
    # qᵣ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????
    # qₛ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Ice())) ?????????????
    # Tₐ = TD.air_temperature(thp, ts) ?????????????

    S = ifelse(
        Tₐ(thp, ts) < mp.ps.T_freeze,
        triangle_inequality_limiter(
            CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₛ, qᵣ, ρ),
            limit(qᵣ, dt, 5),
        ),
        -triangle_inequality_limiter(
            CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qᵣ, qₛ, ρ),
            limit(qₛ, dt, 5),
        ),
    )

    return S
    # @. Sqₛᵖ += Sᵖ
    # @. Sqᵣᵖ -= Sᵖ
    # @. Sqₛᵖ += ifelse(Tₐ(thp, ts) < mp.ps.T_freeze, FT(0), Sᵖ)
end

# evaporation: q_rai -> q_vap
function evaporation_rain_to_vapor(x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args
    
    # rps = (cm.pr, cm.tv.rain, cm.aps, thp) ?????????????
    # Tₐ = TD.air_temperature(thp, ts) ?????????????
    # qᵣ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????

    S = -triangle_inequality_limiter(
        -CM1.evaporation_sublimation(rps..., ρq_tot / ρ, ρq_liq / ρ, ρq_ice / ρ, ρq_rai / ρ, ρq_sno / ρ, ρ, Tₐ(thp, ts)),
        limit(qᵣ, dt, 5),
    )

    return S
    # @. Sqᵣᵖ += Sᵖ
end

# melting: q_sno -> q_rai
function melting_snow_to_rain(x, y, z, t, args...)
    (; ρ, ts, dt, cm, thp) = args
    
    # sps = (cm.ps, cm.tv.snow, cm.aps, thp)
    # Tₐ = TD.air_temperature(thp, ts) ?????????????
    # qₛ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Ice())) ?????????????

    S = triangle_inequality_limiter(
        CM1.snow_melt(sps..., qₛ, ρ, Tₐ(thp, ts)),
        limit(qₛ, dt, 5),
    )

    return S
    # @. Sqᵣᵖ += Sᵖ
    # @. Sqₛᵖ -= Sᵖ
end

# deposition/sublimation: q_vap <-> q_sno
function deposition_sublimation_vapor_to_snow(x, y, z, t, args...)
    (; ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ts, dt, cm, thp) = args

    # sps = (cm.ps, cm.tv.snow, cm.aps, thp)
    # Tₐ = TD.air_temperature(thp, ts) ?????????????
    # qᵥ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Liquid())) ?????????????
    # qₛ = TD.q_vap_from_p_vap(thp, Tₐ, ρ, TD.saturation_vapor_pressure(thp, Tₐ, TD.Ice())) ?????????????

    FT = typeof(ρ)
    S = CM1.evaporation_sublimation(sps..., 
                                    ρq_tot / ρ , 
                                    ρq_liq / ρ, 
                                    ρq_ice / ρ, 
                                    ρq_rai / ρ, 
                                    ρq_sno / ρ, 
                                    ρ, 
                                    Tₐ(thp, ts))
    S = ifelse(
        S > FT(0),
        triangle_inequality_limiter(S, limit(qᵥ(thp, ts), dt, 5)),
        -triangle_inequality_limiter(FT(-1) * S, limit(qₛ, dt, 5)),
    )

    return S
    # @. Sqₛᵖ += Sᵖ
end

function sedimentation_velocities_rain(x, y, z, t, args...)
    (; ρ, ρq_rai, cm) = args

    # cmp = some parmeters from CAP / cm

    w_sed = CM1.terminal_velocity(
        cmp.pr,
        cmp.tv.rain,
        ρ,
        max(zero(ρ), ρq_rai / ρ),
    )
    return w_sed
end

function sedimentation_velocities_snow(x, y, z, t, args...)
    (; ρ, ρq_sno, cm) = args
    
    # cmp = some parmeters from CAP / cm

    w_sed = CM1.terminal_velocity(
        cmp.ps,
        cmp.tv.snow,
        ρ,
        max(zero(ρ), ρq_sno / ρ),
    )
    return w_sed
end

function sedimentation_velocities_liquid(x, y, z, t, args...)
    (; ρ, ρq_liq, cm) = args
    
    #cmc = some parmeters from CAP / cm
    
    w_sed = CMNe.terminal_velocity(
        cmc.liquid,
        cmc.Ch2022.rain,
        ρ,
        max(zero(ρ), ρq_liq / ρ),
    )
    return w_sed
end

function sedimentation_velocities_ice(x, y, z, t, args...)
    (; ρ, ρq_ice, cm) = args
        
    #cmc = some parmeters from CAP / cm
    
    w_sed = CMNe.terminal_velocity(
        cmc.ice,
        cmc.Ch2022.small_ice,
        ρ,
        max(zero(ρ), ρq_ice / ρ),
    )
    return w_sed
end


# # ------------------------------------------------------------------------------------------------ #
# # CLIMA microphysics interface
# # source for non-precipitating hydrometeors
# """
#     cloud_sources(cm_params, thp, qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ, dt)

#  - cm_params - CloudMicrophysics parameters struct for cloud water or ice condensate
#  - thp - Thermodynamics parameters struct
#  - qₜ - total specific humidity
#  - qₗ - liquid specific humidity
#  - qᵢ - ice specific humidity
#  - qᵣ - rain specific humidity
#  - qₛ - snow specific humidity
#  - ρ - air density
#  - Tₐ - air temperature
#  - dt - model time step

# Returns the condensation/evaporation or deposition/sublimation rate for
# non-equilibrium Morrison and Milbrandt 2015 cloud formation.
# """
# # function cloud_sources(
# #     cm_params::CMP.CloudLiquid{FT},
# #     thp,
# #     qₜ,
# #     qₗ,
# #     qᵢ,
# #     qᵣ,
# #     qₛ,
# #     ρ,
# #     T,
# #     dt,
# # ) where {FT}

# #     qᵥ = qₜ - qₗ - qᵢ - qᵣ - qₛ
# #     qₛₗ = TD.q_vap_from_p_vap(
# #         thp,
# #         T,
# #         ρ,
# #         TD.saturation_vapor_pressure(thp, T, TD.Liquid()),
# #     )

# #     if qᵥ + qₗ > FT(0)
# #         S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(
# #             cm_params,
# #             thp,
# #             qₜ,
# #             qₗ,
# #             qᵢ,
# #             qᵣ,
# #             qₛ,
# #             ρ,
# #             T,
# #         )
# #     else
# #         S = FT(0)
# #     end

# #     return ifelse(
# #         S > FT(0),
# #         triangle_inequality_limiter(S, limit(qᵥ - qₛₗ, dt, 2)),
# #         -triangle_inequality_limiter(abs(S), limit(qₗ, dt, 2)),
# #     )
# # end

# # function cloud_sources(
# #     cm_params::CMP.CloudIce{FT},
# #     thp,
# #     qₜ,
# #     qₗ,
# #     qᵢ,
# #     qᵣ,
# #     qₛ,
# #     ρ,
# #     T,
# #     dt,
# # ) where {FT}

# #     qᵥ = qₜ - qₗ - qᵢ - qᵣ - qₛ

# #     qₛᵢ = TD.q_vap_from_p_vap(
# #         thp,
# #         T,
# #         ρ,
# #         TD.saturation_vapor_pressure(thp, T, TD.Ice()),
# #     )

# #     if qᵥ + qᵢ > FT(0)
# #         S = CMNe.conv_q_vap_to_q_liq_ice_MM2015(
# #             cm_params,
# #             thp,
# #             qₜ,
# #             qₗ,
# #             qᵢ,
# #             qᵣ,
# #             qₛ,
# #             ρ,
# #             T,
# #         )
# #     else
# #         S = FT(0)
# #     end

# #     # Additional condition to avoid creating ice in conditions above freezing
# #     # Representing the lack of INPs in warm temperatures
# #     if T > thp.T_freeze && S > FT(0)
# #         S = FT(0)
# #     end

# #     return ifelse(
# #         S > FT(0),
# #         triangle_inequality_limiter(S, limit(qᵥ - qₛᵢ, dt, 2)),
# #         -triangle_inequality_limiter(abs(S), limit(qᵢ, dt, 2)),
# #     )
# # end

# # function cloud_condensate_tendency!(
# #     Yₜ,
# #     Y,
# #     p,
# #     ::NonEquilMoistModel,
# #     ::Microphysics1Moment,
# #     _,
# # )
# #     (; ᶜts) = p.precomputed
# #     (; params, dt) = p
# #     FT = eltype(params)
# #     thp = CAP.thermodynamics_params(params)
# #     cmc = CAP.microphysics_cloud_params(params)

# #     Tₐ = @. lazy(TD.air_temperature(thp, ᶜts))

# #     @. Yₜ.c.ρq_liq +=
# #         Y.c.ρ * cloud_sources(
# #             cmc.liquid,
# #             thp,
# #             specific(Y.c.ρq_tot, Y.c.ρ),
# #             specific(Y.c.ρq_liq, Y.c.ρ),
# #             specific(Y.c.ρq_ice, Y.c.ρ),
# #             specific(Y.c.ρq_rai, Y.c.ρ),
# #             specific(Y.c.ρq_sno, Y.c.ρ),
# #             Y.c.ρ,
# #             Tₐ,
# #             dt,
# #         )
# #     @. Yₜ.c.ρq_ice +=
# #         Y.c.ρ * cloud_sources(
# #             cmc.ice,
# #             thp,
# #             specific(Y.c.ρq_tot, Y.c.ρ),
# #             specific(Y.c.ρq_liq, Y.c.ρ),
# #             specific(Y.c.ρq_ice, Y.c.ρ),
# #             specific(Y.c.ρq_rai, Y.c.ρ),
# #             specific(Y.c.ρq_sno, Y.c.ρ),
# #             Y.c.ρ,
# #             Tₐ,
# #             dt,
# #         )
# # end

# # sources for precipitating hydrometeors
# # function compute_precipitation_sources!(
# #     Sᵖ,
# #     Sᵖ_snow,
# #     Sqₗᵖ,
# #     Sqᵢᵖ,
# #     Sqᵣᵖ,
# #     Sqₛᵖ,
# #     ρ,
# #     qₜ,
# #     qₗ,
# #     qᵢ,
# #     qᵣ,
# #     qₛ,
# #     ts,
# #     dt,
# #     mp,
# #     thp,
# # )
# #     FT = eltype(thp)
# #     @. Sqₗᵖ = FT(0)
# #     @. Sqᵢᵖ = FT(0)
# #     @. Sqᵣᵖ = FT(0)
# #     @. Sqₛᵖ = FT(0)

#     # #! format: off
#     # # rain autoconversion: q_liq -> q_rain
#     # @. Sᵖ = ifelse(
#     #     mp.Ndp <= 0,
#     #     CM1.conv_q_liq_to_q_rai(mp.pr.acnv1M, qₗ, true),
#     #     CM2.conv_q_liq_to_q_rai(mp.var, qₗ, ρ, mp.Ndp),
#     # )
#     # @. Sᵖ = triangle_inequality_limiter(Sᵖ, limit(qₗ, dt, 5))
#     # @. Sqₗᵖ -= Sᵖ
#     # @. Sqᵣᵖ += Sᵖ

#     # # snow autoconversion assuming no supersaturation: q_ice -> q_snow
#     # @. Sᵖ = triangle_inequality_limiter(
#     #     CM1.conv_q_ice_to_q_sno_no_supersat(mp.ps.acnv1M, qᵢ, true),
#     #     limit(qᵢ, dt, 5),
#     # )
#     # @. Sqᵢᵖ -= Sᵖ
#     # @. Sqₛᵖ += Sᵖ

#     # # accretion: q_liq + q_rain -> q_rain
#     # @. Sᵖ = triangle_inequality_limiter(
#     #     CM1.accretion(mp.cl, mp.pr, mp.tv.rain, mp.ce, qₗ, qᵣ, ρ),
#     #     limit(qₗ, dt, 5),
#     # )
#     # @. Sqₗᵖ -= Sᵖ
#     # @. Sqᵣᵖ += Sᵖ

#     # # accretion: q_ice + q_snow -> q_snow
#     # @. Sᵖ = triangle_inequality_limiter(
#     #     CM1.accretion(mp.ci, mp.ps, mp.tv.snow, mp.ce, qᵢ, qₛ, ρ),
#     #     limit(qᵢ, dt, 5),
#     # )
#     # @. Sqᵢᵖ -= Sᵖ
#     # @. Sqₛᵖ += Sᵖ

#     # # accretion: q_liq + q_sno -> q_sno or q_rai
#     # # sink of cloud water via accretion cloud water + snow
#     # @. Sᵖ = triangle_inequality_limiter(
#     #     CM1.accretion(mp.cl, mp.ps, mp.tv.snow, mp.ce, qₗ, qₛ, ρ),
#     #     limit(qₗ, dt, 5),
#     # )
#     # # if T < T_freeze cloud droplets freeze to become snow
#     # # else the snow melts and both cloud water and snow become rain
#     # α(thp, ts) = TD.Parameters.cv_l(thp) / TD.latent_heat_fusion(thp, ts) * (Tₐ(thp, ts) - mp.ps.T_freeze)
#     # @. Sᵖ_snow = ifelse(
#     #     Tₐ(thp, ts) < mp.ps.T_freeze,
#     #     Sᵖ,
#     #     FT(-1) * triangle_inequality_limiter(Sᵖ * α(thp, ts), limit(qₛ, dt, 5)),
#     # )
#     # @. Sqₛᵖ += Sᵖ_snow
#     # @. Sqₗᵖ -= Sᵖ
#     # @. Sqᵣᵖ += ifelse(Tₐ(thp, ts) < mp.ps.T_freeze, FT(0), Sᵖ - Sᵖ_snow)

#     # # accretion: q_ice + q_rai -> q_sno
#     # @. Sᵖ = triangle_inequality_limiter(
#     #     CM1.accretion(mp.ci, mp.pr, mp.tv.rain, mp.ce, qᵢ, qᵣ, ρ),
#     #     limit(qᵢ, dt, 5),
#     # )
#     # @. Sqᵢᵖ -= Sᵖ
#     # @. Sqₛᵖ += Sᵖ
#     # # sink of rain via accretion cloud ice - rain
#     # @. Sᵖ = triangle_inequality_limiter(
#     #     CM1.accretion_rain_sink(mp.pr, mp.ci, mp.tv.rain, mp.ce, qᵢ, qᵣ, ρ),
#     #     limit(qᵣ, dt, 5),
#     # )
#     # @. Sqᵣᵖ -= Sᵖ
#     # @. Sqₛᵖ += Sᵖ

#     # # accretion: q_rai + q_sno -> q_rai or q_sno
#     # @. Sᵖ = ifelse(
#     #     Tₐ(thp, ts) < mp.ps.T_freeze,
#     #     triangle_inequality_limiter(
#     #         CM1.accretion_snow_rain(mp.ps, mp.pr, mp.tv.rain, mp.tv.snow, mp.ce, qₛ, qᵣ, ρ),
#     #         limit(qᵣ, dt, 5),
#     #     ),
#     #     -triangle_inequality_limiter(
#     #         CM1.accretion_snow_rain(mp.pr, mp.ps, mp.tv.snow, mp.tv.rain, mp.ce, qᵣ, qₛ, ρ),
#     #         limit(qₛ, dt, 5),
#     #     ),
#     # )
#     # @. Sqₛᵖ += Sᵖ
#     # @. Sqᵣᵖ -= Sᵖ
#     # #! format: on
# # end

# # """
# #     compute_precipitation_sinks!(Sᵖ, Sqᵣᵖ, Sqₛᵖ, ρ, qₜ, qₗ, qᵢ, qᵣ, qₛ, ts, dt, mp, thp)

# #  - Sᵖ - a temporary containter to help compute precipitation source terms
# #  - Sqᵣᵖ, Sqₛᵖ - cached storage for precipitation source terms
# #  - ρ - air density
# #  - qₜ, qₗ, qᵢ, qᵣ, qₛ - total water, cloud liquid and ice, rain and snow specific humidities
# #  - ts - thermodynamic state (see td package for details)
# #  - dt - model time step
# #  - thp, cmp - structs with thermodynamic and microphysics parameters

# # Returns the q source terms due to precipitation sinks from the 1-moment scheme.
# # The specific humidity source terms are defined as Δmᵢ / (m_dry + m_tot)
# # where i stands for total, rain or snow.
# # Also returns the total energy source term due to the microphysics processes.
# # """
# # # function compute_precipitation_sinks!(
# # #     Sᵖ,
# # #     Sqᵣᵖ,
# # #     Sqₛᵖ,
# # #     ρ,
# # #     qₜ,
# # #     qₗ,
# # #     qᵢ,
# # #     qᵣ,
# # #     qₛ,
# # #     ts,
# # #     dt,
# # #     mp,
# # #     thp,
# # # )
# #     FT = eltype(thp)
# #     sps = (mp.ps, mp.tv.snow, mp.aps, thp)
# #     rps = (mp.pr, mp.tv.rain, mp.aps, thp)

# #     # #! format: off
# #     # # evaporation: q_rai -> q_vap
# #     # @. Sᵖ = -triangle_inequality_limiter(
# #     #     -CM1.evaporation_sublimation(rps..., qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ(thp, ts)),
# #     #     limit(qᵣ, dt, 5),
# #     # )
# #     # @. Sqᵣᵖ += Sᵖ

# #     # # melting: q_sno -> q_rai
# #     # @. Sᵖ = triangle_inequality_limiter(
# #     #     CM1.snow_melt(sps..., qₛ, ρ, Tₐ(thp, ts)),
# #     #     limit(qₛ, dt, 5),
# #     # )
# #     # @. Sqᵣᵖ += Sᵖ
# #     # @. Sqₛᵖ -= Sᵖ

# #     # # deposition/sublimation: q_vap <-> q_sno
# #     # @. Sᵖ = CM1.evaporation_sublimation(sps..., qₜ, qₗ, qᵢ, qᵣ, qₛ, ρ, Tₐ(thp, ts))
# #     # @. Sᵖ = ifelse(
# #     #     Sᵖ > FT(0),
# #     #     triangle_inequality_limiter(Sᵖ, limit(qᵥ(thp, ts), dt, 5)),
# #     #     -triangle_inequality_limiter(FT(-1) * Sᵖ, limit(qₛ, dt, 5)),
# #     # )
# #     # @. Sqₛᵖ += Sᵖ
# #     # #! format: on
# # # end

# # # cached precipitation velocities
# # function set_precipitation_velocities!(
# #     Y,
# #     p,
# #     moisture_model::NonEquilMoistModel,
# #     microphysics_model::Microphysics1Moment,
# #     _,
# # )
# #     (; ᶜwₗ, ᶜwᵢ, ᶜwᵣ, ᶜwₛ, ᶜwₜqₜ, ᶜwₕhₜ, ᶜts, ᶜu) = p.precomputed
# #     (; ᶜΦ) = p.core
# #     cmc = CAP.microphysics_cloud_params(p.params)
# #     cmp = CAP.microphysics_1m_params(p.params)
# #     thp = CAP.thermodynamics_params(p.params)

# #     # # compute the precipitation terminal velocity [m/s]
# #     # @. ᶜwᵣ = CM1.terminal_velocity(
# #     #     cmp.pr,
# #     #     cmp.tv.rain,
# #     #     Y.c.ρ,
# #     #     max(zero(Y.c.ρ), Y.c.ρq_rai / Y.c.ρ),
# #     # )
# #     # @. ᶜwₛ = CM1.terminal_velocity(
# #     #     cmp.ps,
# #     #     cmp.tv.snow,
# #     #     Y.c.ρ,
# #     #     max(zero(Y.c.ρ), Y.c.ρq_sno / Y.c.ρ),
# #     # )
# #     # compute sedimentation velocity for cloud condensate [m/s]
# #     # @. ᶜwₗ = CMNe.terminal_velocity(
# #     #     cmc.liquid,
# #     #     cmc.Ch2022.rain,
# #     #     Y.c.ρ,
# #     #     max(zero(Y.c.ρ), Y.c.ρq_liq / Y.c.ρ),
# #     # )
# #     # @. ᶜwᵢ = CMNe.terminal_velocity(
# #     #     cmc.ice,
# #     #     cmc.Ch2022.small_ice,
# #     #     Y.c.ρ,
# #     #     max(zero(Y.c.ρ), Y.c.ρq_ice / Y.c.ρ),
# #     # )

# #     # compute their contributions to energy and total water advection
# #     @. ᶜwₜqₜ =
# #         Geometry.WVector(
# #             ᶜwₗ * Y.c.ρq_liq +
# #             ᶜwᵢ * Y.c.ρq_ice +
# #             ᶜwᵣ * Y.c.ρq_rai +
# #             ᶜwₛ * Y.c.ρq_sno,
# #         ) / Y.c.ρ

# #     @. ᶜwₕhₜ =
# #         Geometry.WVector(
# #             ᶜwₗ * Y.c.ρq_liq * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₗ, ᶜu))) +
# #             ᶜwᵢ * Y.c.ρq_ice * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵢ, ᶜu))) +
# #             ᶜwᵣ * Y.c.ρq_rai * (Iₗ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwᵣ, ᶜu))) +
# #             ᶜwₛ * Y.c.ρq_sno * (Iᵢ(thp, ᶜts) + ᶜΦ + $(Kin(ᶜwₛ, ᶜu))),
# #         ) / Y.c.ρ
# #     return nothing
# # end

# # function precipitation_tendency!(
# #     Yₜ,
# #     Y,
# #     p,
# #     t,
# #     ::NonEquilMoistModel,
# #     microphysics_model::Microphysics1Moment,
# #     _,
# # )
# #     (; turbconv_model) = p.atmos
# #     (; ᶜSqₗᵖ, ᶜSqᵢᵖ, ᶜSqᵣᵖ, ᶜSqₛᵖ) = p.precomputed

# #     # Update grid mean tendencies
# #     @. Yₜ.c.ρq_liq += Y.c.ρ * ᶜSqₗᵖ
# #     @. Yₜ.c.ρq_ice += Y.c.ρ * ᶜSqᵢᵖ
# #     @. Yₜ.c.ρq_rai += Y.c.ρ * ᶜSqᵣᵖ
# #     @. Yₜ.c.ρq_sno += Y.c.ρ * ᶜSqₛᵖ

# #     return nothing
# # end

# # function vertical_transport_sedimentation(
# #     ᶜρ,
# #     ᶜw,
# #     ᶜχ,
# #     ᶠJ,
# # )
# #     ᶜJ = Fields.local_geometry_field(axes(ᶜρ)).J
# #     return @. lazy(
# #         -(ᶜprecipdivᵥ(ᶠinterp(ᶜρ * ᶜJ) / ᶠJ * ᶠright_bias(Geometry.WVector(-(ᶜw)) * ᶜχ))),
# #     )
# # end

# # vertical_advection(ᶠu³, ᶜχ, ::Val{:none}) =
# #     @. lazy(-(ᶜadvdivᵥ(ᶠu³ * ᶠinterp(ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
# # vertical_advection(ᶠu³, ᶜχ, ::Val{:first_order}) =
# #     @. lazy(-(ᶜadvdivᵥ(ᶠupwind1(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))
# # vertical_advection(ᶠu³, ᶜχ, ::Val{:third_order}) =
# #     @. lazy(-(ᶜadvdivᵥ(ᶠupwind3(ᶠu³, ᶜχ)) - ᶜχ * ᶜadvdivᵥ(ᶠu³)))

# # function implicit_vertical_advection_tendency!(Yₜ, Y, p, t)
# #     (; moisture_model, turbconv_model, rayleigh_sponge, microphysics_model) =
# #         p.atmos
# #     (; dt) = p
# #     n = n_mass_flux_subdomains(turbconv_model)
# #     ᶜJ = Fields.local_geometry_field(axes(Y.c)).J
# #     ᶠJ = Fields.local_geometry_field(axes(Y.f)).J
# #     (; ᶠgradᵥ_ᶜΦ) = p.core
# #     (; ᶠu³, ᶜp, ᶜts) = p.precomputed
# #     thermo_params = CAP.thermodynamics_params(p.params)
# #     ᶜh_tot = @. lazy(
# #         TD.total_specific_enthalpy(
# #             thermo_params,
# #             ᶜts,
# #             specific(Y.c.ρe_tot, Y.c.ρ),
# #         ),
# #     )

# #     @. Yₜ.c.ρ -= ᶜdivᵥ(ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠu³)

# #     # Central vertical advection of active tracers (e_tot and q_tot)
# #     vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜh_tot, dt, Val(:none))
# #     @. Yₜ.c.ρe_tot += vtt
# #     if !(moisture_model isa DryModel)
# #         ᶜq_tot = @. lazy(specific(Y.c.ρq_tot, Y.c.ρ))
# #         vtt = vertical_transport(Y.c.ρ, ᶠu³, ᶜq_tot, dt, Val(:none))
# #         @. Yₜ.c.ρq_tot += vtt
# #     end

# #     # Vertical advection of passive tracers with the mean flow
# #     # is done in the explicit tendency.
# #     # Here we add the vertical advection with precipitation terminal velocity
# #     # using downward biasing and free outflow bottom boundary condition
# #     if moisture_model isa NonEquilMoistModel
# #         (; ᶜwₗ, ᶜwᵢ) = p.precomputed
# #         @. Yₜ.c.ρq_liq -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwₗ)) * specific(Y.c.ρq_liq, Y.c.ρ),
# #             ),
# #         )
# #         @. Yₜ.c.ρq_ice -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwᵢ)) * specific(Y.c.ρq_ice, Y.c.ρ),
# #             ),
# #         )
# #     end
# #     if microphysics_model isa Microphysics1Moment
# #         (; ᶜwᵣ, ᶜwₛ) = p.precomputed
# #         @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, Y.c.ρ),
# #             ),
# #         )
# #         @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwₛ)) * specific(Y.c.ρq_sno, Y.c.ρ),
# #             ),
# #         )
# #     end
# #     if microphysics_model isa Microphysics2Moment
# #         (; ᶜwₙₗ, ᶜwₙᵣ, ᶜwᵣ, ᶜwₛ) = p.precomputed
# #         @. Yₜ.c.ρn_liq -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwₙₗ)) * specific(Y.c.ρn_liq, Y.c.ρ),
# #             ),
# #         )
# #         @. Yₜ.c.ρn_rai -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwₙᵣ)) * specific(Y.c.ρn_rai, Y.c.ρ),
# #             ),
# #         )
# #         @. Yₜ.c.ρq_rai -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwᵣ)) * specific(Y.c.ρq_rai, Y.c.ρ),
# #             ),
# #         )
# #         @. Yₜ.c.ρq_sno -= ᶜprecipdivᵥ(
# #             ᶠinterp(Y.c.ρ * ᶜJ) / ᶠJ * ᶠright_bias(
# #                 Geometry.WVector(-(ᶜwₛ)) * specific(Y.c.ρq_sno, Y.c.ρ),
# #             ),
# #         )
# #     end
# #     if microphysics_model isa Microphysics2MomentP3
# #         (; ρ, ρn_ice, ρq_rim, ρb_rim) = Y.c
# #         ᶜwnᵢ = @. lazy(Geometry.WVector(p.precomputed.ᶜwnᵢ))
# #         ᶜwᵢ = @. lazy(Geometry.WVector(p.precomputed.ᶜwᵢ))
# #         ᶠρ = @. lazy(ᶠinterp(ρ * ᶜJ) / ᶠJ)

# #         # Note: `ρq_ice` is handled above, in `moisture_model isa NonEquilMoistModel`
# #         @. Yₜ.c.ρn_ice -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwnᵢ * specific(ρn_ice, ρ)))
# #         @. Yₜ.c.ρq_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwᵢ * specific(ρq_rim, ρ)))
# #         @. Yₜ.c.ρb_rim -= ᶜprecipdivᵥ(ᶠρ * ᶠright_bias(- ᶜwᵢ * specific(ρb_rim, ρ)))
# #     end

# #     # TODO - decide if this needs to be explicit or implicit
# #     #vertical_advection_of_water_tendency!(Yₜ, Y, p, t)

# #     @. Yₜ.f.u₃ -= ᶠgradᵥ(ᶜp) / ᶠinterp(Y.c.ρ) + ᶠgradᵥ_ᶜΦ

# #     if rayleigh_sponge isa RayleighSponge
# #         ᶠz = Fields.coordinate_field(Y.f).z
# #         zmax = z_max(axes(Y.f))
# #         rs = rayleigh_sponge
# #         @. Yₜ.f.u₃ -= β_rayleigh_w(rs, ᶠz, zmax) * Y.f.u₃
# #         if turbconv_model isa PrognosticEDMFX
# #             for j in 1:n
# #                 @. Yₜ.f.sgsʲs.:($$j).u₃ -=
# #                     β_rayleigh_w(rs, ᶠz, zmax) * Y.f.sgsʲs.:($$j).u₃
# #             end
# #         end
# #     end
# #     return nothing
# # end