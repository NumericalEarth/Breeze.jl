# # An inverse experiment for the July 18 ENA case
#
# This separate development script prepares a sounding loss and three independent
# control blocks for the same anelastic P3 model used by ena_forcing_comparison.jl:
# initial θ, qᵛ, u, v; large-scale temperature/moisture tendencies, geostrophic
# winds and subsidence; and surface sensible heat, latent heat and stress.
# Corrections use smooth height modes and two time modes, with dimensionless
# coefficients, a quadratic prior, a ±3 bound and an Armijo line search.
#
# The sounding operator bins observations vertically and samples the forecast at
# each bin's actual ascent time. Its default scales (0.5 K, 0.0005 kg/kg, 1 m/s)
# represent tunable model/observation mismatch scales, not instrument errors.
# A single sounding cannot identify every forcing, flux and initial correction.
# Start with a small basis and inspect sensitivity/regularization dependence.
#
# Run `julia --project=examples examples/ena_optimization.jl` to inspect targets.
# Download the extra observations first with `download_ena_observations(turbulence=true)`
# from download_ena_observations.jl, using the existing ARM credential environment.
# `--check-fd` runs a 16³, two-second synthetic control/optimizer test on CPU,
# using explicitly labeled central differences. It is NOT an AD validation or
# a fit to the 11:38 observations. It exercises all three control groups cheaply.
# `--check-ad` is the experimental Reactant/Enzyme reverse-mode counterpart.
# Reactant runs on CPU on this laptop; forward production LES remains on Metal.
# The full P3 adjoint is not yet validated: Reactant 0.2.293 exhausted the
# compiler task stack during model initialization in the latest bounded check.
# The profile interpolation and WENO AD tests pass independently. Finite-difference
# optimizer results below must not be presented as successful full-model AD.
#
# These SHORT tests prescribe a nonzero synthetic radiative heating profile in
# W/m³, independent of the evolving cloud state. The CPU mean-column RRTMGP
# callback is not differentiated. Long-window fitting requires
# a declared prescribed heating history or a differentiated radiation coupling.
# ena_radiation_differentiability.jl separately checks cloudy NumericalRadiation
# reverse derivatives and native Metal fluxes. It currently uses the numerical-
# radiation Breeze branch; passing that column check does not validate this
# coupled P3/Reactant inverse problem.
# New forward runs save radiative_heating.csv and turbulence_profiles.csv for
# this next step. No full-duration fit is launched by this development script.
#
# Planned combined example: run the forcing ensemble and SAM/observation comparison;
# optimize a selected baseline against the later sounding; rerun the optimized
# controls with the original radiation/Metal configuration; compare before/after
# and reserve LWP and quality-screened lidar variance as independent checks.
# The Covert idealized initial profile uses the later sounding, so fitting that
# sounding is calibration, not independent validation. The downloaded 05:31
# sounding provides an alternative earlier initial-state prior for a later study.
#
# Turbulence data: ARM DLPROF-WSTATS, https://doi.org/10.5439/1178583,
# enadlprofwstats4newsC1.c1.20170718.000000.nc. Its 30-minute statistics overlap
# at 10-minute spacing; they are not independent observations. The file includes
# w variance/skewness/kurtosis and cloud-base updraft fraction, plus island ECOR
# TKE, w variance and heat/moisture covariances. There are no QC bit fields in this
# version. Subcloud/SNR screening alone does not remove drizzle or island effects.
# See https://doi.org/10.1175/JAMC-D-21-0087.1 for those ENA-specific issues.
# Cloud base and MWRRET LWP also constrain mixing and entrainment indirectly.

using Breeze
using Oceananigans
using DelimitedFiles
using LinearAlgebra: dot, norm

include("setups/ena_simulation.jl")
include("setups/ena_controls.jl")

function ena_inverse_smoke_test(; backend=:finite_difference, iterations=1,
                                 directory=joinpath(@__DIR__, "ena_output", "optimization", "p3_prescribed"))
    backend in (:finite_difference, :reactant) || throw(ArgumentError("Unknown derivative backend."))
    Oceananigans.defaults.FloatType = Float64
    arch = if backend == :reactant
        isdefined(@__MODULE__, :ena_compile_inverse) ||
            error("Include setups/ena_differentiability.jl before selecting the Reactant backend.")
        Reactant.set_default_backend("cpu")
        Oceananigans.Architectures.ReactantState()
    else
        CPU()
    end
    radiation = TimeHeightProfile([0.0, 750, 1500, 3000], [0.0, 2.0],
                                  repeat([0.0, -0.05, -0.02, 0.0], 1, 2))
    construct() = ena_simulation(arch; size=(16, 16, 16), stop_time=2, radiation)
    simulation = backend == :reactant ? ena_with_stack(construct) : construct()
    set!(simulation, covert_ena_case())
    problem = ena_control_problem(simulation; centers=(750,))
    parameters = zeros(length(problem.controls))
    current = Array(ena_mean_profiles(simulation.model))
    target = (values=current .+ [0.5 0.0002 0.5 -0.5],
              times=fill(2.0, size(current)),
              weights=repeat([1/0.5^2 1/0.0005^2 1.0 1.0], size(current, 1)) ./ (4size(current, 1)))
    sampling = ena_sampling_weights(target, 1.0, 2)
    if backend == :finite_difference
        value = p -> ena_fit_loss(ena_prediction_native!(problem.workspace, p, problem.baseline, sampling, 1.0, 2),
                                 target, p, 0.01)
        value_gradient = p -> (value(p), ena_central_difference(value, p))
    else
        inverse = ena_compile_inverse(problem, target; Δt=1, steps=2)
        value, value_gradient = inverse.value, inverse.value_gradient
    end
    original_loss, gradient = value_gradient(parameters)
    original_loss == value(parameters) || error("Forecast reset is not reproducible.")
    mkpath(directory)
    tag = backend == :reactant ? "ad" : "finite_difference"
    open(joinpath(directory, "$(tag)_gradient_checks.csv"), "w") do io
        println(io, "control_group,directional_derivative,central_difference,relative_error")
        for group in (:initial, :forcing, :surface)
            direction = zeros(length(parameters))
            range = problem.ranges[group]
            direction[range] .= sin.(collect(range))
            direction ./= norm(direction)
            increment = 0.002
            reference = (value(parameters .+ increment .* direction) - value(parameters .- increment .* direction)) / (2increment)
            derivative = dot(gradient, direction)
            relative_error = abs(derivative - reference) / max(abs(reference), 1e-10)
            println(io, join((group, derivative, reference, relative_error), ','))
            isfinite(relative_error) && relative_error < 0.02 || error("Gradient check failed for $group.")
        end
    end
    result = ena_optimize(value_gradient, value, parameters; iterations)
    result.history[end].loss < original_loss || error("Synthetic optimization did not improve the objective.")
    open(joinpath(directory, "$(tag)_controls.csv"), "w") do io
        println(io, "parameter,initial_gradient,optimized_coefficient")
        for i in eachindex(parameters)
            println(io, join((problem.labels[i], gradient[i], result.controls[i]), ','))
        end
    end
    open(joinpath(directory, "$(tag)_history.csv"), "w") do io
        println(io, "iteration,loss,gradient_norm")
        for row in result.history
            println(io, join(values(row), ','))
        end
    end
    write(joinpath(directory, "$(tag)_configuration.txt"),
          "synthetic_target = true\nderivative_backend = $backend\ngrid = (16,16,16)\n" *
          "forecast_seconds = 2\nmicrophysics = P3\nradiation = prescribed synthetic heating in W/m3\n" *
          "observational_fit = false\n")
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    if "--check-ad" in ARGS
        include("setups/ena_differentiability.jl")
        ena_inverse_smoke_test(backend=:reactant)
    elseif "--check-fd" in ARGS
        ena_inverse_smoke_test()
    elseif isempty(ARGS)
        include("ena_observation_targets.jl")
        plot_ena_observation_targets()
        target = ena_sounding_target(collect(0:40:3000))
        directory = joinpath(@__DIR__, "ena_output", "optimization")
        open(joinpath(directory, "sounding_targets.csv"), "w") do io
            println(io, "height_m,T_K,qv_kg_kg,u_m_s,v_m_s,T_time_s,qv_time_s,u_time_s,v_time_s,T_weight,qv_weight,u_weight,v_weight")
            writedlm(io, hcat(target.height, target.values, target.times, target.weights), ',')
        end
        @info "Prepared ENA observation targets; no forecasts launched" last_sample_seconds=maximum(target.times)
    else
        error("Use no argument, --check-fd, or --check-ad.")
    end
end
