@inline microphysics_drift_velocity(mp::Microphysics1M, ::small_detritus) = microphysics_drift_velocity(mp, Val(:sPOM))
@inline microphysics_drift_velocity(mp::Microphysics1M, ::large_detritus) = microphysics_drift_velocity(mp, Val(:bPOM))

@inline function microphysics_drift_velocity(mp::Microphysics1M, ::Val{tracer_name}) where tracer_name
    if tracer_name in keys(mp.sinking_velocities)
        return (u = ZeroField(), v = ZeroField(), w = mp.sinking_velocities[tracer_name])
    else
        return (u = ZeroField(), v = ZeroField(), w = ZeroField())
    end
end