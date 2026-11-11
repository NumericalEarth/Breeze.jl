# # [Differentiating a LASSO-ENA forecast](@id ena_differentiability_example)
#
# This experiment uses the ARM LASSO default ERA5 / SBM warm-rain aer2 input member
# for 18 July 2017, with Breeze's original prognostic-aerosol P3 configuration.
# It does not use the Covert case or fixed-droplet-number sensitivity experiment.
# Stage the member's `snd`, `lsf`, `sfc`, `prm`, and `grd` files in
# `ENV["ENA_LASSO_BUNDLE"]`; the adapter checks their SHA-256 hashes before running.
# Obtain the inputs from [ARM LASSO](https://doi.org/10.5439/2572661).
#
# The clock starts at 00 UTC on 18 July. The supplied hourly forcing covers
# 17 July 18 UTC through 19 July 06 UTC. Height-dependent advection, subsidence,
# geostrophic winds, and mean-wind targets are interpolated in time. Surface fluxes
# use the SAM ocean exchange law and prescribed SST; radiation uses ecCKD.
# There is no additional temperature/humidity nudging or fitted P3 coefficient.
# The prescribed upper-boundary relaxation is retained.
#
# The default 32 × 32 horizontal grid is a reduced-domain differentiation probe,
# not the 256 × 256 baseline: both use 100 m spacing and all 260 LASSO vertical cells.
# Gradients of this P3/ecCKD configuration must pass the checks below before being
# used for calibration. This script does not imply that the ARM scores are reproduced.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.TimeSteppers: reset!, update_state!
using Oceananigans: initialize!
using Breeze.TimeSteppers: prepare_first_time_step!
using Reactant
using Reactant: @trace
using Enzyme
using CUDA
using CairoMakie
using Printf
using Random
using Statistics: mean

include(joinpath(pkgdir(Breeze), "examples", "lasso_ena", "LassoENA.jl"))
Random.seed!(718)
Oceananigans.defaults.FloatType = Float64
architecture = CUDA.functional() ? GPU() : CPU()
Reactant.set_default_backend(CUDA.functional() ? "gpu" : "cpu")
bundle = ENV["ENA_LASSO_BUNDLE"]
Nx = parse(Int, get(ENV, "ENA_NX", "32"))
Ny = parse(Int, get(ENV, "ENA_NY", "32"))
spinup_seconds = parse(Float64, get(ENV, "ENA_SPINUP_SECONDS", "3600"))
steps = parse(Int, get(ENV, "ENA_FORECAST_STEPS", "240"))
0 ≤ spinup_seconds && steps > 0 && spinup_seconds + steps ≤ 86400 ||
    throw(ArgumentError("The spin-up and forecast must fit within the July 18 UTC day"))
case = LassoENA.build_case(bundle, architecture; Nx, Ny)
model = case.model
Nz = length(case.z)
Lz = last(case.faces)
z_centers = case.z
z_inversion = 1500.0 # Center of the perturbation used in the directional check.

# The liquid water path, in kg/m², is the column-integrated cloud and rain water.
# We compute it from the prognostic densities so that the same function can be traced.

function liquid_water_path(model)
    ρqˡ = interior(model.microphysical_fields.ρqᶜˡ) .+ interior(model.microphysical_fields.ρqʳ)
    return sum(ρqˡ .* reshape(on_architecture(Oceananigans.architecture(model.grid), diff(case.faces)), 1, 1, :)) / (Nx * Ny)
end

@info @sprintf("Initial liquid water path: %.3f kg/m²", liquid_water_path(model))

# ## Spin up
#
# Before differentiating, we spin up turbulence for one hour.

simulation = Simulation(model; Δt=1, stop_time=spinup_seconds)
# A fixed 1 s step makes IterationInterval(30) exactly a 30 s radiation schedule,
# including under Reactant (which requires a stateless schedule).
add_callback!(simulation, sim -> LassoENA.refresh_inputs!(sim.model, case.sst), IterationInterval(1))

function progress(simulation)
    model = simulation.model
    @info @sprintf("Spin-up: t = %s, Δt = %s, max|w| = %.2f m/s, LWP = %.3f kg/m²",
                   prettytime(simulation), prettytime(simulation.Δt),
                   maximum(abs, model.velocities.w), liquid_water_path(model))
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10minutes))
run!(simulation)

# ## The forecast and its derivative
#
# Starting from the spun-up state, we forecast for 4 minutes with a fixed 1 s step.
# The controls are horizontally uniform corrections ``δθ(z)`` and ``δqᵛ(z)`` to the
# initial potential temperature and water vapor. Every other prognostic variable starts
# from the spun-up state, which we copy once onto the device that will run the forecast.

Δt = 1.0

function forecast_state(model, arch, sst)
    θ = liquid_ice_potential_temperature(model)
    compute!(θ)
    copy_to(field) = on_architecture(arch, Array(interior(field)))
    fixed = (; (name => copy_to(field) for (name, field) in pairs(Oceananigans.prognostic_fields(model))
                if name ∉ (:ρθ, :ρqᵛ))...)
    return (; θ=copy_to(θ), qᵛ=copy_to(model.microphysical_fields.qᵛ),
              temperature=copy_to(model.temperature), fixed, time=model.clock.time, sst)
end

# Each forecast restores every prognostic variable and the clock, so the same model
# (and the same compiled derivative) can be reused for any number of forecasts.

function forecast_liquid_water_path(model, δθ, δqᵛ, state, Δt, steps)
    reset!(model.clock)
    model.clock.time = state.time
    LassoENA.refresh_inputs!(model, state.sst)
    set!(model.temperature, state.temperature)
    set!(model; θ = state.θ .+ reshape(δθ, 1, 1, :),
                qᵛ = state.qᵛ .+ reshape(δqᵛ, 1, 1, :),
                state.fixed...)
    ## Prepare the first step outside the traced loop and take every step inside it,
    ## so the compiled program contains a single time step.
    initialize!(model)
    update_state!(model)
    LassoENA.refresh_inputs!(model, state.sst)
    prepare_first_time_step!(model, Δt)
    @trace mincut=true checkpointing=true track_numbers=false for step in 1:steps
        LassoENA.refresh_inputs!(model, state.sst)
        time_step!(model, Δt)
    end
    return liquid_water_path(model)
end

# Reverse-mode differentiation returns the gradient with respect to both profiles at
# once. `set_strong_zero` keeps zero sensitivities exactly zero through inactive
# microphysical branches, whose fractional powers have singular derivatives at zero.

function liquid_water_path_gradient!(model, dmodel, δθ, dθ, δqᵛ, dqᵛ, state, Δt, steps)
    Enzyme.remake_zero!(dmodel)
    dθ .= 0
    dqᵛ .= 0
    _, path = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                              forecast_liquid_water_path, Enzyme.Active,
                              Enzyme.Duplicated(model, dmodel),
                              Enzyme.Duplicated(δθ, dθ), Enzyme.Duplicated(δqᵛ, dqᵛ),
                              Enzyme.Const(state), Enzyme.Const(Δt), Enzyme.Const(steps))
    return path, dθ, dqᵛ
end

# Now we build the model on `ReactantState`, copy in the spun-up state, and compile.
# Compilation runs inside a task with a large stack, which Reactant's compiler needs
# for a model of this size.

function with_large_stack(f)
    task = Task(f, 256 * 1024^2)
    schedule(task)
    return fetch(task)
end

reactant_case = with_large_stack(() -> LassoENA.build_case(bundle, ReactantState(); Nx, Ny))
reactant_model = reactant_case.model
reactant_state = forecast_state(model, ReactantState(), reactant_case.sst)
dmodel = Enzyme.make_zero(reactant_model)
δθ = Reactant.to_rarray(zeros(Nz))
δqᵛ = Reactant.to_rarray(zeros(Nz))
dθ = Reactant.to_rarray(zeros(Nz))
dqᵛ = Reactant.to_rarray(zeros(Nz))

# We exclude one of Reactant's optimization patterns, `concat_insert_dim_elementwise`,
# which batches concatenated elementwise operations. On the differentiated time-step loop
# it more than triples the time spent optimizing the derivative, and excluding it leaves
# the derivative unchanged.

compile_options = Reactant.CompileOptions(raise=true, raise_first=true, sync=true,
                                          excluded_passes=["concat_insert_dim_elementwise"])

compile_time = @elapsed compiled_gradient! = with_large_stack() do
    Reactant.@compile compile_options=compile_options liquid_water_path_gradient!(
        reactant_model, dmodel, δθ, dθ, δqᵛ, dqᵛ, reactant_state, Δt, steps)
end

path, ∂θ, ∂qᵛ = compiled_gradient!(reactant_model, dmodel, δθ, dθ, δqᵛ, dqᵛ, reactant_state, Δt, steps)
path = Float64(path)
∂θ, ∂qᵛ = Array(∂θ), Array(∂qᵛ)
@info @sprintf("Compiled the derivative in %.0f s. Final liquid water path: %.4f kg/m²", compile_time, path)

# ## Checking the derivative
#
# We check the reverse-mode derivative against central differences of the forward
# model, along two directions: a 0.1 K warming of the cloud layer, and a
# 0.1 g/kg moistening of the subcloud layer.

state = forecast_state(model, architecture, case.sst)
zero_profile = zeros(Nz)
forecast(δθ, δqᵛ) = forecast_liquid_water_path(model, on_architecture(architecture, δθ),
                                               on_architecture(architecture, δqᵛ), state, Δt, steps)

reference_path = forecast(zero_profile, zero_profile)

cloud_layer = @. 0.1 * exp(-((z_centers - z_inversion + 200) / 150)^2)
subcloud_layer = @. 1e-4 * (z_centers < 500)

checks = map((("warming the cloud layer", cloud_layer, zero_profile, ∂θ),
              ("moistening the subcloud layer", zero_profile, subcloud_layer, ∂qᵛ))) do (label, θ_direction, q_direction, gradient)
    ε = 1e-2
    plus = forecast(ε .* θ_direction, ε .* q_direction)
    minus = forecast(-ε .* θ_direction, -ε .* q_direction)
    finite_difference = (plus - minus) / 2ε
    automatic = sum(gradient .* (θ_direction .+ q_direction))
    @assert isapprox(automatic, finite_difference; rtol=0.05, atol=1e-8) "Directional derivative failed: $label"
    @info @sprintf("%s: automatic %.6e, finite difference %.6e kg/m²", label, automatic, finite_difference)
    (; label, automatic, finite_difference)
end

@assert isapprox(path, reference_path; rtol=1e-5, atol=1e-8) "Native and compiled forecasts disagree"
@info @sprintf("Forward and Reactant forecasts agree to %.1e", abs(path - reference_path) / reference_path)

# ## Sensitivity profiles
#
# Each sensitivity panel shows how the final liquid water path responds to a change of
# the initial profile *in one model layer*, beside the mean profiles at the start of the forecast.

θ̄ = vec(mean(Array(state.θ); dims=(1, 2)))
q̄ᵛ = vec(mean(Array(state.qᵛ); dims=(1, 2)))
ρ̄ = vec(Array(interior(model.dynamics.reference_state.density)))
q̄ᶜˡ = vec(mean(Array(state.fixed.ρqᶜˡ); dims=(1, 2))) ./ ρ̄

fig = Figure(size=(1200, 500), fontsize=14)
axθ = Axis(fig[1, 1]; xlabel="θ (K)", ylabel="z (m)", title="Initial state")
axq = Axis(fig[1, 2]; xlabel="qᵛ (kg/kg)", yticklabelsvisible=false)
axc = Axis(fig[1, 3]; xlabel="qᶜˡ (kg/kg)", yticklabelsvisible=false)
axdθ = Axis(fig[1, 4]; xlabel="∂LWP/∂θ (kg m⁻² K⁻¹)", yticklabelsvisible=false, title="Sensitivity")
axdq = Axis(fig[1, 5]; xlabel="∂LWP/∂qᵛ (kg m⁻²)", yticklabelsvisible=false)

lines!(axθ, θ̄, z_centers; color=:magenta)
lines!(axq, q̄ᵛ, z_centers; color=:dodgerblue)
lines!(axc, q̄ᶜˡ, z_centers; color=:lime)
lines!(axdθ, ∂θ, z_centers; color=:magenta)
lines!(axdq, ∂qᵛ, z_centers; color=:dodgerblue)

for ax in (axθ, axq, axc, axdθ, axdq)
    hlines!(ax, z_inversion; color=:gray, linestyle=:dash)
    ylims!(ax, 0, Lz)
end

for ax in (axdθ, axdq)
    vlines!(ax, 0; color=:gray)
end

Label(fig[0, :], @sprintf("ENA, 18 July 2017: sensitivity of the liquid water path after %s", prettytime(steps * Δt)),
      fontsize=16, tellwidth=false)

save("ena_differentiability.png", fig) #src
fig

# These comparisons test the derivative of the discrete forecast, including its
# active microphysical and advection branches. It is not the sensitivity of
# long-time turbulent statistics: over longer windows, chaotic growth makes
# trajectory derivatives large and noisy.
