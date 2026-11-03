#####
##### Approximate radiation updates between solves
#####
##### Between two solves of the radiative transfer equations the surface temperature changes and the
##### sun moves. Following Hogan and Bozzo (2015), each time step between solves updates the fluxes
##### for both, so that the surface and the atmosphere exchange the same radiative fluxes at every
##### step. The solve uses the cosine of the solar zenith angle averaged over the sunlit part of the
##### radiation interval (Hogan and Hirahara, 2015), so that a sun rising between solves gets a
##### shortwave transmittance.
#####

using Oceananigans.Utils: IterationInterval, TimeInterval

"""
$(TYPEDEF)

The state of the approximate updates of a [`RadiativeTransferModel`](@ref)'s fluxes between solves
([Hogan and Bozzo, 2015](@cite HoganBozzo2015)). Each time step between solves

- shifts the longwave fluxes by the change in the fourth power of the surface temperature `T₀`
  times their response to `T₀⁴`, computed at each solve by repeating the longwave solve with a
  warmer surface;
- scales the shortwave fluxes of the last solve by `μ₀ / μ̄₀`, where `μ₀` is the current cosine of
  the solar zenith angle and `μ̄₀` its mean over the sunlit part of the radiation interval, with
  which the solve computed the slant path of the direct beam
  ([Hogan and Hirahara, 2015](@cite HoganHirahara2015)).

# Fields
$(TYPEDFIELDS)
"""
struct ApproximateRadiationUpdates{F, C}
    "Response of the upwelling longwave flux to `T₀⁴` (W m⁻² K⁻⁴)"
    upwelling_longwave_response :: F
    "Response of the downwelling longwave flux to `T₀⁴` (W m⁻² K⁻⁴, signed like the flux)"
    downwelling_longwave_response :: F
    "Upwelling shortwave flux of the last solve, at `μ̄₀`"
    solved_upwelling_shortwave_flux :: F
    "Downwelling shortwave flux of the last solve, at `μ̄₀`"
    solved_downwelling_shortwave_flux :: F
    "Per-column `μ̄₀` of the last solve"
    solved_cos_zenith :: C
    "Per-column cosine of the solar zenith angle"
    cos_zenith :: C
    "Per-column count of sunlit samples of the radiation interval"
    sunlit_samples :: C
    "Per-column surface temperature of the last update of the longwave fluxes"
    surface_temperature :: C
end

"""
$(TYPEDSIGNATURES)

The [`ApproximateRadiationUpdates`](@ref) of a radiative transfer model on `grid` solved on
`schedule`, or `nothing` when the model solves every iteration and so has no steps between solves.
"""
function ApproximateRadiationUpdates(grid, schedule, solar_position)
    schedule isa IterationInterval && schedule.interval == 1 && return nothing

    Nx, Ny, _ = size(grid)
    column_array() = on_architecture(architecture(grid), zeros(eltype(grid), Nx * Ny))
    cos_zenith = column_array()
    initialize_cos_zenith!(cos_zenith, solar_position)

    return ApproximateRadiationUpdates(ZFaceField(grid), ZFaceField(grid), ZFaceField(grid), ZFaceField(grid),
                                       column_array(), cos_zenith, column_array(), column_array())
end

#####
##### The cosine of the solar zenith angle for a solve
#####

"""
$(TYPEDSIGNATURES)

The length of the interval between solves on `schedule`, or zero when the schedule does not fix it
(or, before the first time step, the time step is not yet known).
"""
radiation_interval(schedule::TimeInterval, clock) = schedule.interval
radiation_interval(schedule::IterationInterval, clock) = ifelse(isfinite(clock.last_Δt), schedule.interval * clock.last_Δt, zero(clock.last_Δt))
radiation_interval(schedule, clock) = 0

time_after(t::Number, seconds) = t + seconds
time_after(t::AbstractDateTime, seconds) = t + Millisecond(round(Int, 1000seconds))

# Samples of the radiation interval in the mean cosine of the solar zenith angle
const sunlit_samples_per_interval = 8

"""
$(TYPEDSIGNATURES)

Fill the per-column `cos_zenith` with the cosine of the solar zenith angle with which `rtm` solves
at the time of `clock`: its mean over the sunlit part of the radiation interval when `rtm` updates
its fluxes between solves, and its current value otherwise.
"""
update_cos_zenith!(cos_zenith, rtm::RadiativeTransferModel, grid, clock) =
    update_solve_cos_zenith!(cos_zenith, rtm.approximate_updates, rtm, grid, clock)

update_solve_cos_zenith!(cos_zenith, ::Nothing, rtm, grid, clock) =
    update_cos_zenith!(cos_zenith, rtm.solar_position, grid, clock)

function update_solve_cos_zenith!(μ̄₀, updates::ApproximateRadiationUpdates, rtm, grid, clock)
    μ₀ = updates.cos_zenith
    sunlit_samples = updates.sunlit_samples
    Δt = radiation_interval(rtm.schedule, clock) / sunlit_samples_per_interval

    fill!(μ̄₀, 0)
    fill!(sunlit_samples, 0)

    for m in 1:sunlit_samples_per_interval
        sample_clock = (; time = time_after(clock.time, (m - 1/2) * Δt))
        update_cos_zenith!(μ₀, rtm.solar_position, grid, sample_clock)
        μ̄₀ .+= μ₀
        sunlit_samples .+= μ₀ .> 0
    end

    μ̄₀ ./= max.(sunlit_samples, 1)
    copyto!(updates.solved_cos_zenith, μ̄₀)

    return nothing
end

#####
##### Updates
#####

"""
$(TYPEDSIGNATURES)

Whether a solve of `rtm` also computes the response of its longwave fluxes to the surface temperature:
only when `rtm` updates its fluxes between solves and its surface temperature can change.
"""
solves_longwave_response(rtm) = !isnothing(rtm.approximate_updates) &&
                                !(rtm.surface_radiation.surface_temperature isa ConstantField)

# The warming of the surface in the second longwave solve, from which the response follows
const surface_temperature_increment = 1

"""
$(TYPEDSIGNATURES)

Complete a solve of `rtm` and compute its flux divergence. When `rtm` updates its fluxes between
solves, record the shortwave fluxes of the solve and turn the longwave fluxes of the second
longwave solve, which the solve left in the longwave responses, into the responses to `T₀⁴`; then
bring the shortwave fluxes to the current sun.
"""
complete_radiation_solve!(rtm, model) = complete_radiation_solve!(rtm.approximate_updates, rtm, model)
complete_radiation_solve!(::Nothing, rtm, model) = compute_radiation_flux_divergence!(rtm, model.grid)

function complete_radiation_solve!(updates::ApproximateRadiationUpdates, rtm, model)
    grid = model.grid
    T₀ = rtm.surface_radiation.surface_temperature
    launch!(architecture(grid), grid, :xy, _record_radiation_solve!,
            updates.upwelling_longwave_response, updates.downwelling_longwave_response,
            updates.solved_upwelling_shortwave_flux, updates.solved_downwelling_shortwave_flux,
            updates.surface_temperature,
            rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            T₀, solves_longwave_response(rtm), grid)

    update_radiation_between_solves!(rtm, model)

    return nothing
end

@kernel function _record_radiation_solve!(R_lw_up, R_lw_dn, ℐ_sw_up_solved, ℐ_sw_dn_solved, T₀_updated,
                                          ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, T₀, solves_response, grid)
    i, j = @index(Global, NTuple)
    c = column_index(i, j, grid.Nx)
    Nz = size(grid, 3)

    @inbounds begin
        T = T₀[i, j, 1]
        T₀_updated[c] = T
        ΔT⁴ = (T + surface_temperature_increment)^4 - T^4

        for k in 1:Nz+1
            R_lw_up[i, j, k] = ifelse(solves_response, (R_lw_up[i, j, k] - ℐ_lw_up[i, j, k]) / ΔT⁴, R_lw_up[i, j, k])
            R_lw_dn[i, j, k] = ifelse(solves_response, (R_lw_dn[i, j, k] - ℐ_lw_dn[i, j, k]) / ΔT⁴, R_lw_dn[i, j, k])
            ℐ_sw_up_solved[i, j, k] = ℐ_sw_up[i, j, k]
            ℐ_sw_dn_solved[i, j, k] = ℐ_sw_dn[i, j, k]
        end
    end
end

"""
$(TYPEDSIGNATURES)

Update the fluxes of `radiation` for the current surface temperature and sun, between two solves,
and recompute its flux divergence. A no-op for a radiation that does not update between solves.
"""
update_radiation_between_solves!(radiation, model) = nothing
update_radiation_between_solves!(rtm::RadiativeTransferModel, model) =
    update_radiation_between_solves!(rtm.approximate_updates, rtm, model)
update_radiation_between_solves!(::Nothing, rtm, model) = nothing

function update_radiation_between_solves!(updates::ApproximateRadiationUpdates, rtm, model)
    grid = model.grid
    update_cos_zenith!(updates.cos_zenith, rtm.solar_position, grid, model.clock)

    launch!(architecture(grid), grid, :xy, _update_radiation_between_solves!,
            rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            updates.upwelling_longwave_response, updates.downwelling_longwave_response,
            updates.solved_upwelling_shortwave_flux, updates.solved_downwelling_shortwave_flux,
            updates.solved_cos_zenith, updates.cos_zenith, updates.surface_temperature,
            rtm.surface_radiation.surface_temperature, grid)

    compute_radiation_flux_divergence!(rtm, grid)

    return nothing
end

@kernel function _update_radiation_between_solves!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn,
                                                   R_lw_up, R_lw_dn, ℐ_sw_up_solved, ℐ_sw_dn_solved,
                                                   μ̄₀, μ₀, T₀_updated, T₀, grid)
    i, j = @index(Global, NTuple)
    c = column_index(i, j, grid.Nx)
    Nz = size(grid, 3)

    @inbounds begin
        T = T₀[i, j, 1]
        ΔT⁴ = T^4 - T₀_updated[c]^4
        T₀_updated[c] = T

        # μ̄₀ vanishes only when the whole radiation interval is night, and with it the solved fluxes
        μ₀_ratio = ifelse(μ̄₀[c] > 0, μ₀[c] / μ̄₀[c], zero(T))

        for k in 1:Nz+1
            ℐ_lw_up[i, j, k] += R_lw_up[i, j, k] * ΔT⁴
            ℐ_lw_dn[i, j, k] += R_lw_dn[i, j, k] * ΔT⁴
            ℐ_sw_up[i, j, k] = μ₀_ratio * ℐ_sw_up_solved[i, j, k]
            ℐ_sw_dn[i, j, k] = μ₀_ratio * ℐ_sw_dn_solved[i, j, k]
        end
    end
end
