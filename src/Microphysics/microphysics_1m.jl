struct Microphysics1M{TH, SV, PAR} <: AbstractMicrophysics
    thermodynamics::TH
    sedimentation_velocities:: SV
    parameters:: PAR
end

function Microphysics1M(grid; thermodynamics, parameters)
    sedimentation_velocities = (; ρq_rai = ZFaceField(grid), ρq_sno = ZFaceField(grid))
    Microphysics1M(thermodynamics, sedimentation_velocities, parameters)
end

required_microphysics_tracers(::Microphysics1M) = (:ρq_tot, :ρq_liq, :ρq_ice, :ρq_rai, :ρq_sno, :ρe_tot)
required_microphysics_auxiliary_fields(::Microphysics1M) = (:rho, :PAR)

@inline specific(ρ, values...) = (value/ρ for value in values)

@inline function (mp::Microphysics1M)(::Val{:ρq_liq}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dt = PAR.dt
    q_tot, q_liq, q_ice, q_rain, q_snow = specific(ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    T = air_temperature(mp.thermodynamics, ρe_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    
    cond_vapor_liquid = cloud_liquid_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    auto_liquid_rain = autoconversion_liquid_to_rain_rate(mp, q_liq, ρ, dt)
    acc_cloud_rain = accretion_cloud_rain_rate(mp, q_liq, q_rain, ρ, dt)
    acc_cloud_snow = accretion_cloud_snow_rate(mp, q_liq, q_snow, ρ, T, dt)

    return ρ * (cond_vapor_liquid.liq + auto_liquid_rain.liq + acc_cloud_rain.liq + acc_cloud_snow.liq)
end

@inline function (mp::Microphysics1M)(::Val{:ρq_ice}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dt = PAR.dt
    q_tot, q_liq, q_ice, q_rain, q_snow = specific(ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)
    T = air_temperature(mp.thermodynamics, ρe_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno)

    cond_vapor_ice = cloud_ice_condensation_rate(mp, q_tot, q_liq, q_ice, q_rain, q_snow, ρ, T, dt)
    auto_ice_snow = autoconversion_ice_to_snow_rate(mp, q_ice, dt)
    acc_ice_snow = accretion_ice_snow_rate(mp, q_ice, q_snow, ρ, dt)
    acc_ice_rain_snow = accretion_ice_rain_to_snow_rate(mp, q_ice, q_rain, ρ, dt)

    return ρ * (cond_vapor_ice.ice + auto_ice_snow.ice + acc_ice_snow.ice + acc_ice_rain_snow.ice)
end

@inline function (mp::Microphysics1M)(::Val{:ρq_rai}, x, y, z, t, ρ, ρq_tot, ρq_liq, ρq_ice, ρq_rai, ρq_sno, ρe_tot, PAR)
    dt = PAR.dt
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
    dt = PAR.dt
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
    dt = PAR.dt
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

@inline microphysics_sedimentation_velocity(mp::Microphysics1M, ::Val{:ρq_rai}) =
    (; u = ZeroField(), v = ZeroField(), w = mp.sedimentation_velocities.ρq_rai)

@inline microphysics_sedimentation_velocity(mp::Microphysics1M, ::Val{:ρq_sno}) =
    (; u = ZeroField(), v = ZeroField(), w = mp.sedimentation_velocities.ρq_sno)

function update_microphysics_state!(mp::Microphysics1M, model, kernel_parameters; active_cells_map = nothing)
    w_velocities             = mp.sedimentation_velocities
    arch                     = architecture(model.grid)
    grid                     = model.grid
    density                  = model.density
    parameters               = mp.parameters

    exclude_periphery = true
    for tracer_name in sedimenting_tracers(mp)
        tracer_field = model.tracers[tracer_name]
        #tracer_w_velocity_bc = mp.sedimentation_velocities[tracer_name].boundary_conditions.immersed
        w_kernel_args = tuple(Val(tracer_name), density, tracer_field, parameters)
        #w_kernel_args = tuple(Val(tracer_name), density, tracer_field, parameters, tracer_w_velocity_bc)
        launch!(arch, grid, kernel_parameters, compute_sedimentation_velocity!, 
                w_velocities[tracer_name], grid, w_kernel_args;
                active_cells_map, exclude_periphery)
        fill_halo_regions!(w_velocities[tracer_name])
    end

    return nothing
end

@kernel function compute_sedimentation_velocity!(w_sed, grid, args) 
    i, j, k = @index(Global, NTuple)
    @inbounds w_sed[i, j, k] = w_sedimentation_velocity(i, j, k, grid, args...)
end

@inline function w_sedimentation_velocity(i, j, k, grid, microphysics::Microphysics1M, ::Val{:ρq_rai}, ρ, ρq_rai)
    ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    ρq_raiᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρq_rai)
    return CM1.terminal_velocity(microphysics.parameters.pr, microphysics.parameters.tv.rain, ρᶜᶜᶠ, ρq_raiᶜᶜᶠ/ ρᶜᶜᶠ)
end

@inline function w_sedimentation_velocity(i, j, k, grid, microphysics::Microphysics1M, ::Val{:ρq_sno}, ρ, ρq_sno)
    ρᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
    ρq_snoᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρq_sno)
    return CM1.terminal_velocity(microphysics.parameters.ps, microphysics.parameters.tv.snow, ρᶜᶜᶠ, ρq_snoᶜᶜᶠ/ ρᶜᶜᶠ)
end
