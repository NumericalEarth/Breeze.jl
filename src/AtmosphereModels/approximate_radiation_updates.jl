#####
##### Approximate radiation updates between solves
#####
##### Between two solves of the radiative transfer equations the surface temperature changes and the
##### sun moves. Following Hogan and Bozzo (2015), each state update between solves updates the fluxes
##### for both, so that the surface and the atmosphere exchange the same radiative fluxes at every
##### step. The solve uses the cosine of the solar zenith angle averaged over the sunlit part of the
##### radiation interval (Hogan and Hirahara, 2015), so that a sun rising between solves gets a
##### shortwave transmittance.
#####

using Oceananigans.Utils: IterationInterval, TimeInterval, SpecifiedTimes, next_actuation_time, time_difference_seconds

"""
$(TYPEDEF)

The state of the approximate updates of a [`RadiativeTransferModel`](@ref)'s fluxes between solves
([Hogan and Bozzo, 2015](@cite HoganBozzo2015)). Each state update between solves

- shifts the solved longwave fluxes by the change in the fourth power of the surface temperature `T₀`
  since that solve, times their response to `T₀⁴`, computed by repeating the longwave solve with the
  surface 1 K warmer (1 K cooler at the top of a solver's surface temperature range);
- scales the shortwave fluxes of the last solve by `μ₀ / μ̄₀`, where `μ₀` is the current cosine of
  the solar zenith angle and `μ̄₀` its mean over the sunlit part of the radiation window, with
  which the solve computed the slant path of the direct beam
  ([Hogan and Hirahara, 2015](@cite HoganHirahara2015)). The window runs from the start of the
  time step of the solve to the start of the time step that solves next. When the radiation
  interval is unknown, the solve uses the current solar angle and the fluxes still follow the
  moving sun.

Over its window the sun stays within four times `μ̄₀`. A state update that finds `μ₀` beyond that,
sunlight the last solve did not average over, solves again instead: after a solve at night, or with
an unknown or underestimated radiation interval.

# Fields
$(TYPEDFIELDS)
"""
struct ApproximateRadiationUpdates{F, C}
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
    "Per-column cosine of the solar zenith angle used by the last solve"
    solved_cos_zenith :: C
    "Per-column cosine of the solar zenith angle"
    cos_zenith :: C
    "Per-column sunlit duration of the radiation window (s), a workspace of the solve"
    sunlit_duration :: C
    "Per-column surface temperature of the last solve"
    surface_temperature :: C
end

"""
$(TYPEDSIGNATURES)

The [`ApproximateRadiationUpdates`](@ref) of a radiative transfer model on `grid` solved on
`schedule`, or `nothing` when the model solves every iteration, and so has no steps between solves.
`BreezeReactantExt` returns `nothing` on `ReactantState`, whose traced time steps do not update
the fluxes between solves.
"""
approximate_radiation_updates(grid, schedule, solar_position) =
    approximate_radiation_updates(architecture(grid), grid, schedule, solar_position)

function approximate_radiation_updates(arch, grid, schedule, solar_position)
    schedule isa IterationInterval && schedule.interval == 1 && return nothing

    Nx, Ny, _ = size(grid)
    column_array() = on_architecture(arch, zeros(eltype(grid), Nx * Ny))
    cos_zenith = column_array()
    initialize_cos_zenith!(cos_zenith, solar_position)

    return ApproximateRadiationUpdates(ZFaceField(grid), ZFaceField(grid),
                                       ZFaceField(grid), ZFaceField(grid),
                                       ZFaceField(grid), ZFaceField(grid),
                                       column_array(), cos_zenith, column_array(), column_array())
end

#####
##### The cosine of the solar zenith angle for a solve
#####

"""
$(TYPEDSIGNATURES)

The time in seconds by which the radiation window of a solve at `clock` begins before `clock`: the
first stage of the time step at the first state update of a step (`clock.stage == 2`), so that the
window begins with the step, and zero at other state updates. Later stages can return to earlier
times than the first: SSP RK3 evaluates its third stage at `tⁿ + Δt / 2`, after its second at
`tⁿ + Δt`.
"""
function radiation_window_lead(clock)
    lead = clock.last_stage_Δt
    return ifelse((clock.stage == 2) & isfinite(lead), max(0, lead), zero(lead))
end

# The first two stage abscissae of `timestepper`, supplied by the time steppers that know their
# stage timing. Declared here because `TimeSteppers` loads after `AtmosphereModels`.
stage_abscissae(timestepper) = nothing

# The offsets from the start of the time step of `clock` to its first stage and to `clock`, or
# `nothing` when the step timing is unknown: before a positive time step, and for time steppers
# without stage abscissae at any but the first-stage clock, which still gives its own offset.
function radiation_step_offsets(timestepper, clock, lead)
    Δt = clock.last_Δt
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
    return first_stage * clock.last_Δt, current_offset
end

# Between the steps of a run, the first stage of the step that begins at `clock` can still solve
awaiting_first_stage(clock) = clock.stage == 1 && clock.iteration > 0

"""
$(TYPEDSIGNATURES)

The length in seconds of the radiation window of a solve at `clock` on `schedule`, which begins
`lead` seconds before `clock` and ends at the start of the time step that solves next. Zero when
the next solve is not known: past the last explicit time, for an infinite interval, before the
first time step of an `IterationInterval`, and for other schedules.

A `TimeInterval` or `SpecifiedTimes` schedule solves in the first time step whose first stage
reaches its next time `t★`. Estimate that step's start using the last time step and the stage
abscissae of `timestepper`, or the first-stage clock when those abscissae are unknown. Without
known step timing, average up to `t★`. An `IterationInterval` solves in the time step of its
next actuating iteration. Neither calculation advances the schedule.
"""
function radiation_interval(schedule::Union{TimeInterval, SpecifiedTimes}, clock, lead, timestepper=nothing)
    t★ = next_actuation_time(schedule)
    (t★ === Inf || t★ <= clock.time) && return zero(lead) # a schedule counting from before the clock actuates at the next step
    remaining = time_difference_seconds(t★, clock.time)
    offsets = radiation_step_offsets(timestepper, clock, lead)
    isnothing(offsets) && return finite_radiation_interval(remaining + lead)
    first_stage_offset, current_offset = offsets

    # The next solve's first stage must reach t★, but its window begins at the step's start.
    # A first stage already visited cannot solve again; iteration zero never actuates a schedule.
    Δt = clock.last_Δt
    earliest_step = ifelse(awaiting_first_stage(clock), 0, 1)
    steps = max(earliest_step, ceil((remaining + current_offset - first_stage_offset) / Δt))
    return finite_radiation_interval(steps * Δt - current_offset + lead)
end

# TODO: Oceananigans could provide the number of iterations until a schedule next actuates
# (Oceananigans.Utils offers only `actuates_next_iteration(::IterationInterval)`, and
# `next_actuation_time(::IterationInterval) = Inf`), which would replace the modular arithmetic here.
function radiation_interval(schedule::IterationInterval, clock, lead, timestepper=nothing)
    steps = mod(schedule.offset - clock.iteration, schedule.interval)
    steps = ifelse(steps == 0 && !awaiting_first_stage(clock), schedule.interval, steps)
    interval = steps * clock.last_Δt
    offsets = radiation_step_offsets(timestepper, clock, lead)
    if !isnothing(offsets)
        interval += lead - last(offsets)
    end
    return finite_radiation_interval(interval)
end

radiation_interval(schedule, clock, lead, timestepper=nothing) = zero(lead)

finite_radiation_interval(interval) = ifelse(isfinite(interval), max(0, interval), zero(interval))

"""
$(TYPEDSIGNATURES)

Fill the per-column `cos_zenith` with the cosine of the solar zenith angle with which `rtm` solves
at the time of `model.clock`: its mean over the sunlit part of the radiation window when `rtm`
updates its fluxes between solves, and its current value otherwise.
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
$(TYPEDSIGNATURES)

Whether a solve of `rtm` also computes the response of its longwave fluxes to the surface temperature:
only when `rtm` updates its fluxes between solves and its surface temperature can change.
"""
solves_longwave_response(rtm) = !isnothing(rtm.approximate_updates) &&
                                !(rtm.surface_radiation.surface_temperature isa ConstantField)

"""
$(TYPEDSIGNATURES)

The range `(Tmin, Tmax)` into which the solver of `rtm` clamps the surface temperature, or
`nothing` when it evaluates any surface temperature.
"""
surface_temperature_bounds(rtm) = nothing

# The surface temperature `T₀` as a solve sees it
@inline solved_surface_temperature(T₀, ::Nothing) = T₀
@inline solved_surface_temperature(T₀, bounds) = clamp(T₀, bounds...)

# The change of the surface temperature in the second longwave solve, from which the response follows
const surface_temperature_increment = 1

# The surface temperature of the second longwave solve: 1 K warmer than the solve's, or 1 K cooler
# where warming would leave the solver's range, which would hold the surface emission fixed
@inline perturbed_surface_temperature(T₀, ::Nothing) = T₀ + surface_temperature_increment

@inline function perturbed_surface_temperature(T₀, bounds)
    T = solved_surface_temperature(T₀, bounds)
    warmer = T + surface_temperature_increment
    return ifelse(warmer > last(bounds), T - surface_temperature_increment, warmer)
end

"""
$(TYPEDSIGNATURES)

Complete a solve of `rtm` and compute its flux divergence. When `rtm` updates its fluxes between
solves, record the fluxes and surface temperature of the solve. Turn the fluxes of the second
longwave solve, which the solve left in the longwave responses, into the responses to `T₀⁴`, then
bring the shortwave fluxes to the current sun.
"""
function complete_radiation_solve!(rtm, model)
    complete_radiation_solve!(rtm.approximate_updates, rtm, model)
    set_radiation_cache_validity!(rtm.cache_valid, true)
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
$(TYPEDSIGNATURES)

Update the fluxes of `radiation` for the current surface temperature and sun, between two solves,
and recompute its flux divergence; solve again instead when sunlight arrives that the last solve
did not average over. A no-op for radiation without approximate updates. A missing radiation
cache is refreshed by [`update_radiation!`](@ref), which solves instead of calling this.
"""
update_radiation_between_solves!(radiation, model) = nothing

update_radiation_between_solves!(rtm::RadiativeTransferModel, model) =
    update_radiation_between_solves!(rtm.approximate_updates, rtm, model)

update_radiation_between_solves!(::Nothing, rtm, model) = nothing

# Over its radiation window, the sun stays within four times the sunlit mean `μ̄₀` of the window:
# the bound is approached only by a sun skimming the horizon on both sides of a short night. A
# larger `μ₀` is sunlight the solve did not average over (after a solve at night, or with an
# unknown or underestimated radiation interval), and `μ̄₀` would give it a slant path many times
# too long, depositing the beam high in the column.
const maximum_solar_cosine_ratio = 4

@inline needs_daylight_solve(μ₀, μ̄₀) = μ₀ > maximum_solar_cosine_ratio * μ̄₀

function update_radiation_between_solves!(updates::ApproximateRadiationUpdates, rtm, model)
    update_cos_zenith!(updates.cos_zenith, rtm.solar_position, model.grid, model.clock)

    # One fused reduction; the decision to solve is taken on the host, which waits for it
    daylight_solves = Broadcast.instantiate(Broadcast.broadcasted(needs_daylight_solve,
                                                                  updates.cos_zenith, updates.solved_cos_zenith))

    if mapreduce(identity, |, daylight_solves; init=false)
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
        # Anchor to the full solve: small Float32 increments must accumulate before rounding
        # the flux, rather than being lost independently at every state update.
        T = solved_surface_temperature(T₀[i, j, 1], T₀_bounds)
        ΔT⁴ = T^4 - T₀_solved[c]^4

        # A wholly nocturnal window has zero shortwave flux; sunlight after it triggers a fresh solve.
        μ₀_ratio = ifelse(μ̄₀[c] > 0, μ₀[c] / μ̄₀[c], zero(μ₀[c]))

        for k in 1:Nz+1
            ℐ_lw_up[i, j, k] = ℐ_lw_up_solved[i, j, k] + R_lw_up[i, j, k] * ΔT⁴
            ℐ_lw_dn[i, j, k] = ℐ_lw_dn_solved[i, j, k] + R_lw_dn[i, j, k] * ΔT⁴
            ℐ_sw_up[i, j, k] = μ₀_ratio * ℐ_sw_up_solved[i, j, k]
            ℐ_sw_dn[i, j, k] = μ₀_ratio * ℐ_sw_dn_solved[i, j, k]
        end
    end
end
