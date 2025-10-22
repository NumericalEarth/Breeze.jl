struct Microphysics1M{FT, TH, SV} <: AbstractMicrophysics
    thermodynamics::TH
    dt::FT # Required for numerical stability
    sinking_velocities:: SV
end

function Microphysics1M(grid; thermodynamics)
    sinking_velocities = setup_velocity_fields(sedimentation_speeds, grid, true)
    Microphysics1M(thermodynamics, grid.dt, sinking_velocities)
end

required_microphysics_tracers(::Microphysics1M) = (:ρq_tot, :ρq_liq, :ρq_ice, :ρq_rai, :ρq_sno, :ρe_tot)
required_microphysics_auxiliary_fields(::Microphysics1M) = (:rho, :PAR)

@inline ρq_vapor(ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno) = ρq_tot - ρq_liq - ρq_ice - ρq_rai - ρq_sno
@inline specific(ρ, values...) = (value/ρ for value in values)

@inline function (mp::Microphysics1M)(::Val{:ρq_liq}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dt = mp.dt
    q_tot, q_liq, q_ice, q_rain, q_snow = specific(ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    T = air_temperature(mp.thermodynamics, ρe_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    
    cond_vapor_liquid = cloud_liquid_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    auto_liquid_rain = autoconversion_liquid_to_rain_rate(mp, q_liq, ρ, dt)
    acc_cloud_rain = accretion_cloud_rain_rate(mp, q_liq, q_rain, ρ, dt)
    acc_cloud_snow = accretion_cloud_snow_rate(mp, q_liq, q_snow, ρ, T, dt)

    return ρ * (cond_vapor_liquid.liq + auto_liquid_rain.liq + acc_cloud_rain.liq + acc_cloud_snow.liq)
end

@inline function (mp::Microphysics1M)(::Val{:ρq_ice}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dt = mp.dt
    q_tot, q_liq, q_ice, q_rain, q_snow = specific(ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    T = air_temperature(mp.thermodynamics, ρe_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)

    cond_vapor_ice = cloud_ice_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    auto_ice_snow = autoconversion_ice_to_snow_rate(mp, q_ice, dt)
    acc_ice_snow = accretion_ice_snow_rate(mp, q_ice, q_snow, ρ, dt)
    acc_ice_rain_snow = accretion_ice_rain_to_snow_rate(mp, q_ice, q_rain, ρ, dt)

    return ρ * (cond_vapor_ice.ice + auto_ice_snow.ice + acc_ice_snow.ice + acc_ice_rain_snow.ice)
end

@inline function (mp::Microphysics1M)(::Val{:ρq_rai}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dt = mp.dt
    q_tot, q_liq, q_ice, q_rain, q_snow = specific(ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    T = air_temperature(mp.thermodynamics, ρe_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)

    auto_liquid_rain = autoconversion_liquid_to_rain_rate(mp, q_liq, ρ, dt)
    acc_cloud_rain = accretion_cloud_rain_rate(mp, q_liq, q_rain, ρ, dt)
    acc_snow_rain = accretion_snow_rain_rate(mp, q_rain, q_snow, ρ, T, dt)
    acc_cloud_snow = accretion_cloud_snow_rate(mp, q_liq, q_snow, ρ, T, dt)
    sink_ice_rain = rain_sink_from_ice_rate(mp, q_ice, q_rain, ρ, dt)
    evaporation_rain = rain_evaporation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    melt_snow_rain = snow_melt_rate(mp, q_snow, ρ, T, dt)

    return ρ * (auto_liquid_rain.rain + acc_cloud_rain.rain + acc_cloud_snow.rain + acc_ice_rain_snow.rain + acc_snow_rain.rain + sink_ice_rain.rain + evaporation_rain.rain + melt_snow_rain.rain)
end

@inline function (mp::Microphysics1M)(::Val{:ρq_sno}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR) 
    dt = mp.dt
    q_tot, q_liq, q_ice, q_rain, q_snow = specific(ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    T = air_temperature(mp.thermodynamics, ρe_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)

    auto_ice_snow = autoconversion_ice_to_snow_rate(microphysics, q_ice, dt_ft)
    acc_ice_snow = accretion_ice_snow_rate(microphysics, q_ice, q_snow, ρi, dt_ft)
    acc_cloud = accretion_cloud_snow_rate(microphysics, q_liq, q_snow, ρi, Ti, dt_ft)
    acc_ice_rain_snow = accretion_ice_rain_to_snow_rate(microphysics, q_ice, q_rain, ρi, dt_ft)
    acc_snow_rain = accretion_snow_rain_rate(microphysics, q_rain, q_snow, ρi, Ti, dt_ft)
    melt_snow_rain = snow_melt_rate(microphysics, q_snow, ρi, Ti, dt_ft)
    deposition_vapor_snow = snow_deposition_rate(microphysics, q_tot, q_liq, q_ice, q_rain, q_snow, ρi, Ti, dt_ft)

    return ρ * (auto_ice_snow.snow + acc_ice_snow.snow + acc_cloud.snow + acc_ice_rain_snow.snow + acc_snow_rain.snow + melt_snow_rain.snow + deposition_vapor_snow.snow)
end

@inline function (mp::Microphysics1M)(::Val{:ρq_tot}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dρ_liq = mp(:ρq_liq, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dρ_ice = mp(:ρq_ice, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dρ_rai = mp(:ρq_rai, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dρ_sno = mp(:ρq_sno, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    return dq_liq + dq_ice + dq_rai + dq_sno
end

@inline function (mp::Microphysics1M)(::Val{:ρe_tot}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dt = mp.dt
    q_tot, q_liq, q_ice, q_rain, q_snow = specific(ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    T = air_temperature(mp.thermodynamics, ρe_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    Lv = latent_heat_vapor(mp.thermodynamics, T)
    Ls = latent_heat_sublimation(mp.thermodynamics, T)
    Lf = latent_heat_fusion(mp.thermodynamics, T)

    cond_vapor_liquid = cloud_liquid_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    cond_vapor_ice = cloud_ice_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    acc_cloud_snow = accretion_cloud_snow_rate(mp, q_liq, q_snow, ρ, T, dt)
    rain_sink_ice = rain_sink_from_ice_rate(mp, q_ice, q_rain, ρ, T, dt)
    acc_snow_rain = accretion_snow_rain_rate(mp, q_rain, q_snow, ρ, T, dt)
    rain_evap = rain_evaporation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    snow_melt = snow_melt_rate(mp, q_snow, ρ, T, dt)
    snow_dep = snow_deposition_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)

    l_vapor_liquid = Lv * cond_vapor_liquid.liq
    l_cond_ice = Ls * cond_vapor_ice.ice
    l_acc_cloud_snow = Lf * acc_cloud_snow.snow
    l_rain_sink_ice = Lf * rain_sink_ice.snow
    l_acc_snow_rain = Lf * acc_snow_rain.snow
    l_rain_evap = Lv * rain_evap.rain
    l_snow_melt = Lf * snow_melt.snow
    l_snow_dep = Ls * snow_dep.snow

    return ρ * (l_vapor_liquid + l_cond_ice + l_acc_cloud_snow + l_rain_sink_ice + l_acc_snow_rain + l_rain_evap + l_snow_melt + l_snow_dep)
end

function update_microphysics_state!(mp::Microphysics1M, model)

end

function update_tendencies!(mp::Microphysics1M, model)

end

function update_microphysics_drift_velocity!(microphysics::Microphysics1M, state)
    # extract the fields from the state

    # interpolate relevant fields to the faces

    # compute the sinking velocities
    # kernel launch here with all the necessary arguments i, j, k, etc. SUCKS


end

function update_drift_velocity_kernel!()

end

function compute_drift_velocity_kernel!(:Val{:ρq_rai}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)

end

function compute_drift_velocity_kernel!(:Val{:ρq_sno}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)

end
