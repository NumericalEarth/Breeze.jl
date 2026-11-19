#####
##### Approximate radiation updates between solves
#####
##### Between solves, the fluxes follow the surface temperature and the sun (Hogan and Bozzo, 2015).
##### Solves use the solar cosine averaged over the sunlit part of the window (Hogan and Hirahara, 2015).
#####

using Oceananigans.Utils: IterationInterval, TimeInterval, SpecifiedTimes, next_actuation_time, time_difference_in_seconds

"""
$(TYPEDEF)

State for updating a [`RadiativeTransferModel`](@ref)'s fluxes between solves
([Hogan and Bozzo, 2015](@cite HoganBozzo2015)). Each update between solves

- shifts the solved longwave fluxes by their response to `T₀⁴` times the change in `T₀⁴` since the
  solve. The response comes from a second longwave solve with the surface 1 K warmer (1 K cooler
  at the top of the solver's temperature range).
- scales the solved shortwave fluxes by `μ₀ / μ̄₀`, where `μ₀` is the current cosine of the solar
  zenith angle and `μ̄₀` is its sunlit mean over the radiation window, which the solve used for the
  slant path
  ([Hogan and Hirahara, 2015](@cite HoganHirahara2015)). The window runs from the solving step to
  the next solving step. If the interval is unknown, the solve uses the current sun.

If a step boundary finds `μ₀ > 4 μ̄₀` in any column, sunlight the solve did not average over, the
whole grid solves again. With no known interval on a grid crossed by the terminator, that happens
about every step.

# Fields
$(TYPEDFIELDS)
"""
struct ApproximateRadiationUpdates{F, C, D}
    "Response of the upwelling longwave flux to `T₀⁴` (W m⁻² K⁻⁴)"
    upwelling_longwave_response :: F
    "Response of the downwelling longwave flux to `T₀⁴` (W m⁻² K⁻⁴, signed like the flux)"
    downwelling_longwave_response :: F
    "Upwelling longwave flux of the last solve"
    solved_upwelling_longwave_flux :: F
    "Downwelling longwave flux of the last solve"
    solved_downwelling_longwave_flux :: F
    "Upwelling shortwave flux of the last solve, at `μ̄₀`"
    solved_upwelling_shortwave_flux :: F
    "Downwelling shortwave flux of the last solve, at `μ̄₀`"
    solved_downwelling_shortwave_flux :: F
    "Per-column solar cosine `μ̄₀` of the last solve"
    solved_cos_zenith :: C
    "Per-column current solar cosine `μ₀`"
    cos_zenith :: C
    "Per-column sunlit duration of the window (s), workspace for the average; `nothing` if unused"
    sunlit_duration :: D
    "Per-column surface temperature of the last solve"
    surface_temperature :: C
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The [`ApproximateRadiationUpdates`](@ref) for radiation on `grid` solved on `schedule`, or `nothing`
when nothing changes between solves: solving every iteration, a fixed sun with a constant surface
temperature, or `ReactantState`.
"""
function approximate_radiation_updates(grid::AbstractGrid, schedule, solar_position, surface_radiation)
    if solar_position isa FixedCosineZenith && surface_radiation.surface_temperature isa ConstantField
        return nothing
    end
    return approximate_radiation_updates(architecture(grid), grid, schedule, solar_position)
end

function approximate_radiation_updates(arch, grid, schedule, solar_position)
    schedule isa IterationInterval && schedule.interval == 1 && return nothing

    Nx, Ny, _ = size(grid)
    column_array() = on_architecture(arch, zeros(eltype(grid), Nx * Ny))
    cos_zenith = column_array()
    initialize_cos_zenith!(cos_zenith, solar_position)

    duration = sunlit_duration_workspace(solar_position, column_array)
    return ApproximateRadiationUpdates(ZFaceField(grid), ZFaceField(grid),
                                       ZFaceField(grid), ZFaceField(grid),
                                       ZFaceField(grid), ZFaceField(grid),
                                       column_array(), cos_zenith, duration, column_array())
end

#####
##### The cosine of the solar zenith angle for a solve
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Seconds by which the radiation window of a solve at `clock` starts before `clock`: the first stage's
offset at the first-stage update (`clock.stage == 2`), so the window starts with the step, and
zero otherwise.
"""
function radiation_window_lead(clock)
    lead = clock.last_stage_Δt
    return ifelse((clock.stage == 2) & isfinite(lead), max(0, lead), zero(lead))
end

# The first two stage abscissae of `timestepper`, or `nothing`; extended in `TimeSteppers`
stage_abscissae(timestepper) = nothing

radiation_step_duration(timestepper, clock) = stage_step_duration(stage_abscissae(timestepper), clock)
stage_step_duration(::Nothing, clock) = clock.last_Δt

function stage_step_duration((first_stage, second_stage), clock)
    clock.stage == 1 && return clock.last_Δt
    fraction = clock.stage == 2 ? first_stage : second_stage - first_stage
    duration = clock.last_stage_Δt / fraction
    return isfinite(duration) && duration > 0 ? duration : clock.last_Δt
end

# Offsets from the step start to its first stage and to `clock`, or `nothing` if the timing is unknown
function radiation_step_offsets(timestepper, clock, lead)
    Δt = radiation_step_duration(timestepper, clock)
    isfinite(Δt) && Δt > 0 || return nothing
    offsets = stage_offsets(stage_abscissae(timestepper), clock, lead)
    (isnothing(offsets) || !isfinite(last(offsets))) && return nothing
    return offsets
end

stage_offsets(::Nothing, clock, lead) = clock.stage == 2 ? (lead, lead) : nothing

function stage_offsets((first_stage, second_stage), clock, lead)
    current_offset = ifelse(clock.stage == 1, zero(lead),
                           ifelse(clock.stage == 2, clock.last_stage_Δt,
                                  second_stage * (clock.last_stage_Δt / (second_stage - first_stage))))
    return first_stage * stage_step_duration((first_stage, second_stage), clock), current_offset
end

# Start the schedule at the model's clock, as `run!` does for callbacks; a schedule belongs to one model
initialize_radiation_schedule!(radiation, clock) = nothing

function initialize_radiation_schedule!(radiation::RadiativeTransferModel, clock)
    initialize_radiation_schedule!(radiation.schedule, radiation, clock)
    return nothing
end

initialize_radiation_schedule!(schedule, radiation, clock) = nothing
initialize_radiation_schedule!(schedule::Union{TimeInterval, SpecifiedTimes}, radiation, clock) =
    Oceananigans.initialize!(schedule, (; clock))

# Between steps, the next step's first stage can still solve
awaiting_first_stage(clock) = clock.stage == 1 && clock.iteration > 0

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Length in seconds of the radiation window of a solve at `clock`, from `lead` seconds before `clock`
to the start of the next solving step, or zero when the next solve is unknown.

`TimeInterval` and `SpecifiedTimes` solve in the first step whose first stage reaches the next time
`t★`, located with the last time step and the stage abscissae of `timestepper` (or up to `t★`
when the step timing is unknown). `IterationInterval` solves at its next actuating iteration.
Does not advance the schedule.
"""
function radiation_interval(schedule::Union{TimeInterval, SpecifiedTimes}, clock, lead, timestepper=nothing)
    t★ = next_actuation_time(schedule)
    (t★ === Inf || t★ <= clock.time) && return zero(lead) # already due: actuates at the next step
    remaining = time_difference_in_seconds(t★, clock.time)
    offsets = radiation_step_offsets(timestepper, clock, lead)
    isnothing(offsets) && return finite_radiation_interval(remaining + lead)
    first_stage_offset, current_offset = offsets

    # The solving step's first stage must reach t★; a first stage already past cannot solve again
    Δt = radiation_step_duration(timestepper, clock)
    earliest_step = ifelse(awaiting_first_stage(clock), 0, 1)
    steps = max(earliest_step, ceil((remaining + current_offset - first_stage_offset) / Δt))
    return finite_radiation_interval(steps * Δt - current_offset + lead)
end

# TODO: an Oceananigans function for the iterations until a schedule next actuates would replace this
# arithmetic (it has only `actuates_next_iteration` and `next_actuation_time(::IterationInterval) = Inf`).
function radiation_interval(schedule::IterationInterval, clock, lead, timestepper=nothing)
    steps = mod(schedule.offset - clock.iteration, schedule.interval)
    steps = ifelse(steps == 0 && !awaiting_first_stage(clock), schedule.interval, steps)
    interval = steps * radiation_step_duration(timestepper, clock)
    offsets = radiation_step_offsets(timestepper, clock, lead)
    if !isnothing(offsets)
        interval += lead - last(offsets)
    end
    return finite_radiation_interval(interval)
end

radiation_interval(schedule, clock, lead, timestepper=nothing) = zero(lead)

finite_radiation_interval(interval) = ifelse(isfinite(interval), max(0, interval), zero(interval))

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Fill `cos_zenith` with the solar cosine for a solve of `rtm` now: its sunlit mean over the radiation
window when `rtm` updates between solves, else its current value.
"""
update_solve_cos_zenith!(cos_zenith, rtm, model) =
    update_solve_cos_zenith!(cos_zenith, rtm.approximate_updates, rtm, model)

update_solve_cos_zenith!(cos_zenith, ::Nothing, rtm, model) =
    update_cos_zenith!(cos_zenith, rtm.solar_position, model.grid, model.clock)

function update_solve_cos_zenith!(μ̄₀, updates::ApproximateRadiationUpdates, rtm, model)
    grid = model.grid
    clock = model.clock
    μ₀ = updates.cos_zenith
    lead = radiation_window_lead(clock)
    interval = radiation_interval(rtm.schedule, clock, lead, model.timestepper)
    update_cos_zenith!(μ₀, rtm.solar_position, grid, clock)
    copyto!(μ̄₀, μ₀)
    if interval > 0
        average_solve_cos_zenith!(μ̄₀, updates.sunlit_duration, rtm.solar_position, grid, clock, interval, lead)
    end
    copyto!(updates.solved_cos_zenith, μ̄₀)

    return nothing
end

#####
##### Updates
#####

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Whether a solve of `rtm` also computes the longwave response to the surface temperature: only when
`rtm` updates between solves and its surface temperature can change.
"""
solves_longwave_response(rtm) = !isnothing(rtm.approximate_updates) &&
                                !(rtm.surface_radiation.surface_temperature isa ConstantField)

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

The `(Tmin, Tmax)` to which the solver of `rtm` clamps the surface temperature, or `nothing`.
"""
surface_temperature_bounds(rtm) = nothing

# The surface temperature `T₀` as a solve sees it
@inline solved_surface_temperature(T₀, ::Nothing) = T₀
@inline solved_surface_temperature(T₀, bounds) = clamp(T₀, bounds...)

# Surface temperature change (K) of the second longwave solve
const surface_temperature_increment = 1

# Surface temperature of the second longwave solve: 1 K warmer, or 1 K cooler at the top of the range
@inline perturbed_surface_temperature(T₀, ::Nothing) = T₀ + surface_temperature_increment

@inline function perturbed_surface_temperature(T₀, bounds)
    T = solved_surface_temperature(T₀, bounds)
    warmer = T + surface_temperature_increment
    return ifelse(warmer > last(bounds), T - surface_temperature_increment, warmer)
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Finish a solve of `rtm` and compute its flux divergence. With updates between solves, also record
the solved fluxes and surface temperature, turn the second longwave solve into the response to
`T₀⁴`, and scale the shortwave to the current sun.
"""
function complete_radiation_solve!(rtm, model)
    complete_radiation_solve!(rtm.approximate_updates, rtm, model)
    return nothing
end

complete_radiation_solve!(::Nothing, rtm, model) = compute_radiation_flux_divergence!(rtm, model.grid)

function complete_radiation_solve!(updates::ApproximateRadiationUpdates, rtm, model)
    grid = model.grid
    T₀ = rtm.surface_radiation.surface_temperature
    launch!(architecture(grid), grid, :xy, _record_radiation_solve!,
            updates.upwelling_longwave_response, updates.downwelling_longwave_response,
            updates.solved_upwelling_longwave_flux, updates.solved_downwelling_longwave_flux,
            updates.solved_upwelling_shortwave_flux, updates.solved_downwelling_shortwave_flux,
            updates.surface_temperature,
            rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            T₀, surface_temperature_bounds(rtm), solves_longwave_response(rtm), grid)

    apply_radiation_updates!(rtm, model)

    return nothing
end

@kernel function _record_radiation_solve!(R_lw_up, R_lw_dn, ℐ_lw_up_solved, ℐ_lw_dn_solved,
                                          ℐ_sw_up_solved, ℐ_sw_dn_solved, T₀_solved,
                                          ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn, T₀, T₀_bounds, solves_response, grid)
    i, j = @index(Global, NTuple)
    c = column_index(i, j, grid.Nx)
    Nz = size(grid, 3)

    @inbounds begin
        T = solved_surface_temperature(T₀[i, j, 1], T₀_bounds)
        T₀_solved[c] = T
        ΔT⁴ = perturbed_surface_temperature(T, T₀_bounds)^4 - T^4

        for k in 1:Nz+1
            R_lw_up[i, j, k] = ifelse(solves_response, (R_lw_up[i, j, k] - ℐ_lw_up[i, j, k]) / ΔT⁴, R_lw_up[i, j, k])
            R_lw_dn[i, j, k] = ifelse(solves_response, (R_lw_dn[i, j, k] - ℐ_lw_dn[i, j, k]) / ΔT⁴, R_lw_dn[i, j, k])
            ℐ_lw_up_solved[i, j, k] = ℐ_lw_up[i, j, k]
            ℐ_lw_dn_solved[i, j, k] = ℐ_lw_dn[i, j, k]
            ℐ_sw_up_solved[i, j, k] = ℐ_sw_up[i, j, k]
            ℐ_sw_dn_solved[i, j, k] = ℐ_sw_dn[i, j, k]
        end
    end
end

"""
$(TYPEDSIGNATURES_WITH_DEFAULTS)

Update the fluxes of `radiation` for the current surface temperature and sun and recompute its flux
divergence, or, at a step boundary, solve again if sunlight arrived that the last solve did not
average over. A no-op without approximate updates.
"""
update_radiation_between_solves!(radiation, model) = nothing

update_radiation_between_solves!(rtm::RadiativeTransferModel, model) =
    update_radiation_between_solves!(rtm.approximate_updates, rtm, model)

update_radiation_between_solves!(::Nothing, rtm, model) = nothing

# Within its window the sun stays below 4 μ̄₀; a larger μ₀ is sunlight the solve did not average over,
# for which μ̄₀ would give far too long a slant path
const maximum_solar_cosine_ratio = 4

@inline needs_daylight_solve(μ₀, μ̄₀) = μ₀ > maximum_solar_cosine_ratio * μ̄₀

function update_radiation_between_solves!(updates::ApproximateRadiationUpdates, rtm, model)
    update_cos_zenith!(updates.cos_zenith, rtm.solar_position, model.grid, model.clock)

    # One fused reduction, synchronized to the host
    daylight_solves = Broadcast.instantiate(Broadcast.broadcasted(needs_daylight_solve,
                                                                  updates.cos_zenith, updates.solved_cos_zenith))

    if model.clock.stage == 1 && mapreduce(identity, |, daylight_solves; init=false)
        _update_radiation!(rtm, model)
    else
        apply_radiation_updates!(rtm, model)
    end

    return nothing
end

function apply_radiation_updates!(rtm, model)
    grid = model.grid
    updates = rtm.approximate_updates
    launch!(architecture(grid), grid, :xy, _update_radiation_between_solves!,
            rtm.upwelling_longwave_flux, rtm.downwelling_longwave_flux,
            rtm.upwelling_shortwave_flux, rtm.downwelling_shortwave_flux,
            updates.upwelling_longwave_response, updates.downwelling_longwave_response,
            updates.solved_upwelling_longwave_flux, updates.solved_downwelling_longwave_flux,
            updates.solved_upwelling_shortwave_flux, updates.solved_downwelling_shortwave_flux,
            updates.solved_cos_zenith, updates.cos_zenith, updates.surface_temperature,
            rtm.surface_radiation.surface_temperature, surface_temperature_bounds(rtm), grid)

    compute_radiation_flux_divergence!(rtm, grid)

    return nothing
end

@kernel function _update_radiation_between_solves!(ℐ_lw_up, ℐ_lw_dn, ℐ_sw_up, ℐ_sw_dn,
                                                   R_lw_up, R_lw_dn, ℐ_lw_up_solved, ℐ_lw_dn_solved,
                                                   ℐ_sw_up_solved, ℐ_sw_dn_solved,
                                                   μ̄₀, μ₀, T₀_solved, T₀, T₀_bounds, grid)
    i, j = @index(Global, NTuple)
    c = column_index(i, j, grid.Nx)
    Nz = size(grid, 3)

    @inbounds begin
        # Offset from the solve, not the last update, so small Float32 changes are not rounded away
        T = solved_surface_temperature(T₀[i, j, 1], T₀_bounds)
        ΔT⁴ = T^4 - T₀_solved[c]^4

        # No shortwave after a night-time solve; sunrise triggers a new solve
        μ₀_ratio = ifelse(μ̄₀[c] > 0, μ₀[c] / μ̄₀[c], zero(μ₀[c]))

        for k in 1:Nz+1
            ℐ_lw_up[i, j, k] = ℐ_lw_up_solved[i, j, k] + R_lw_up[i, j, k] * ΔT⁴
            ℐ_lw_dn[i, j, k] = ℐ_lw_dn_solved[i, j, k] + R_lw_dn[i, j, k] * ΔT⁴
            ℐ_sw_up[i, j, k] = μ₀_ratio * ℐ_sw_up_solved[i, j, k]
            ℐ_sw_dn[i, j, k] = μ₀_ratio * ℐ_sw_dn_solved[i, j, k]
        end
    end
end
