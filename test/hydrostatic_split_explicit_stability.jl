include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.HydrostaticEquations: VerletSubsteps, ForwardBackwardSubsteps, hydrostatic_state_check,
                                   accumulate_state_checks
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: day
using Printf
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

#####
##### (5) Stability smoke run: a balanced jet on the sphere with Coriolis, 30 days at Δt = 1200 s
#####
#
# The steady state of Jablonowski and Williamson (2006): a zonal jet of 35 m s⁻¹ in each
# hemisphere in thermal-wind balance with its temperature field, over the surface geopotential
# that balances the jet at the ground, under a uniform surface pressure of 1e5 Pa. It is an exact
# steady solution of the hydrostatic equations on the sphere. It is integrated here on a 2°
# latitude-longitude grid between walls at 70°S and 70°N, along which the flow is zonal, on ten
# sigma levels with the model top at 50 hPa, by each split-explicit configuration, with the
# default rule for the number of substeps. There is no dissipation of any kind other than the
# divergence damping of the forward-backward substeps.
#
# The state is balanced analytically, not discretely: the model's hydrostatic integral and
# gradients leave a residual acceleration, which is measured and logged at the start, and which
# sets off gravity waves that nothing removes. The run therefore tests whether the split schemes
# stay bounded about the jet for 30 days at this time step, and records what grows. It is not a
# validation of any configuration: with no dissipation, with walls in place of the poles, and
# with a zonally symmetric state, it cannot be. The state stays zonal to rounding (the departure
# of u from its zonal mean is logged), so nothing seeds the baroclinic instability of the jet
# and the run exercises the zonally symmetric subspace only. The perturbed jet, in which the
# instability grows, is the subject of `hydrostatic_baroclinic_wave.jl`.
#
# What is recorded, every 5 days and at the end: the largest departure of u from the initial jet,
# the largest v, the largest departure of the surface pressure, the largest departure of u
# from its zonal mean, and the extrema of every prognostic field, of the pressure thickness of
# the layers and of the temperatures. A run that stops with an error, as it does when a layer
# loses its mass, is recorded with the day and the error.
#
# Criteria, written before the first run.
#
#   * Initial balance. The residual acceleration of v is below 10 % of the largest Coriolis
#     acceleration f u of the jet: the second-order horizontal differences on a 2° grid leave
#     ~ (k Δφ)² / 6 ≈ 1 % for a jet 20° wide, and the hydrostatic integral over ten layers of a
#     temperature sampled at the layer pressure a few per cent of the geopotential differences.
#   * Boundedness. The run completes 30 days, and at every checkpoint (every 5 days), not only
#     at the last one: every value of every prognostic field is a number, every layer has a
#     positive pressure thickness, the temperature lies in (150, 350) K and the potential
#     temperature within the bounds that those temperatures give at the layer pressures
#     (`hydrostatic_state_check`), and u has moved from the initial jet by less than the jet's
#     own 35 m s⁻¹. The extrema over the checkpoints are logged.
#
# A configuration that does not meet the boundedness criterion is a finding to report, not a
# tolerance to adjust.

# The jet, its temperature and its surface geopotential are those of `balanced_jet_model` in
# `hydrostatic_test_utils.jl`.

# The state at a checkpoint: its departure from the initial jet, and the check of every
# prognostic field, of the pressure thickness of the layers and of the temperatures.
function record(model, initial)
    u = Array(interior(model.velocities.u))
    v = Array(interior(model.velocities.v))
    pˢ = Array(interior(model.dynamics.surface_pressure))
    return (day = model.clock.time / day,
            Δu = maximum(abs, u .- initial.u),
            v = maximum(abs, v),
            Δpˢ = maximum(abs, pˢ .- initial.pˢ),
            zonal_departure = zonal_departure(u),
            check = hydrostatic_state_check(model))
end

format(r) = @sprintf("day %5.1f: max |u - u(0)| = %.3e m s⁻¹, max |v| = %.3e m s⁻¹, max |pˢ - pˢ(0)| = %.3e Pa, max |u - zonal mean| = %.3e m s⁻¹, min Δp = %.3f Pa, θ in (%.2f, %.2f) K, T in (%.2f, %.2f) K, finite: %s, valid: %s",
                     r.day, r.Δu, r.v, r.Δpˢ, r.zonal_departure, r.check.extrema.Δp[1], r.check.extrema.θ...,
                     r.check.extrema.T..., r.check.finite, r.check.valid)

# Integrate for `days`, recording every `interval` days. Returns the records and the error that
# stopped the run, if any. A checkpoint that is not valid ends the run.
function smoke_run(model, Δt; days=30, interval=5)
    initial = (u = Array(interior(model.velocities.u)), pˢ = Array(interior(model.dynamics.surface_pressure)))
    records = [record(model, initial)]
    steps_per_record = round(Int, interval * day / Δt)
    failure = nothing

    try
        for _ in 1:round(Int, days / interval)
            for _ in 1:steps_per_record
                time_step!(model, Δt)
            end
            push!(records, record(model, initial))
            last(records).check.valid || break
        end
    catch err
        failure = err
        push!(records, record(model, initial))
    end

    return records, failure
end

@testset "Initial balance of the jet" begin
    model = balanced_jet_model()
    update_state!(model)

    Gv = Array(interior(model.timestepper.Gⁿ.v))[:, 2:end, :] # the first face is a wall
    u = Array(interior(model.velocities.u))
    φ = Array(φnodes(model.grid, Center()))
    coriolis_acceleration = maximum(abs, [2 * rotation_rate * sind(φ[j]) * u[i, j, k] for i in axes(u, 1), j in axes(u, 2), k in axes(u, 3)])
    residual = maximum(abs, Gv)

    @info @sprintf("Balanced jet: largest u %.2f m s⁻¹, largest Coriolis acceleration %.3e m s⁻², largest residual acceleration of v %.3e m s⁻² (%.2f %%), largest tendency of u %.3e m s⁻²",
                   maximum(abs, u), coriolis_acceleration, residual, 100 * residual / coriolis_acceleration,
                   maximum(abs, Array(interior(model.timestepper.Gⁿ.u))))

    @test maximum(abs, u) ≈ jet_speed rtol=0.05
    @test residual < 0.1 * coriolis_acceleration
end

const stability_configurations = [(tableau, inner_solver) for tableau in (:MIS3C, :MIS4a, :WickerSkamarock3)
                                  for inner_solver in (VerletSubsteps(), ForwardBackwardSubsteps(divergence_damping=0.1))]

@testset "Balanced jet, 30 days at Δt = 1200 s [$tableau + $(summary(inner_solver))]" for
        (tableau, inner_solver) in stability_configurations

    model = balanced_jet_model(timestepper = :GravityWaveMultirateInfinitesimalStep,
                               timestepper_kwargs = (; tableau, inner_solver))

    elapsed = @elapsed records, failure = smoke_run(model, 1200)

    name = string(tableau, " + ", summary(inner_solver))
    for r in records
        @info "Balanced jet, $name, " * format(r)
    end

    final = last(records)
    outcome = isnothing(failure) ? "completed" : "STOPPED by $(sprint(showerror, failure)[1:min(end, 160)])"
    @info @sprintf("Balanced jet, %s: %s at day %.2f after %.0f s of wall time", name, outcome, final.day, elapsed)

    # The extrema over all the checkpoints, not those of the last one
    overall = accumulate_state_checks([r.check for r in records])
    largest_Δu = maximum(r.Δu for r in records)
    @info @sprintf("Balanced jet, %s, over %d checkpoints: max |u - u(0)| = %.3e m s⁻¹, extrema %s", name,
                   length(records), largest_Δu, overall.extrema)

    @test isnothing(failure)
    @test final.day ≈ 30
    @test overall.finite
    @test overall.positive_layers
    @test overall.physical_temperatures
    @test largest_Δu < jet_speed
end
