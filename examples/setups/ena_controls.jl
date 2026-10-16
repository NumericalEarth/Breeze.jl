# Smooth, dimensionless controls for the common P3 ENA model. No forecast is run
# by including this file. The coefficient vector has independent initial-state,
# atmospheric-forcing, and surface-flux blocks.
using Oceananigans.Architectures: on_architecture, architecture
using Oceananigans.Grids: znodes
using Oceananigans.TimeSteppers: reset!, first_time_step!
using Statistics: mean

function ena_vertical_basis(heights; centers=(300, 1000, 2200), width=600)
    basis = exp.(-((heights .- permutedims(collect(centers))) ./ width).^2)
    return basis ./ sum(basis; dims=2)
end

function ena_control_problem(simulation; centers=(300, 1000, 2200), width=600)
    model = simulation.model
    arch = architecture(model)
    FT = eltype(model.grid)
    device(array) = on_architecture(arch, FT.(array))
    drivers = simulation.callbacks[:ena_drivers].func
    drivers.initialized[] || throw(ArgumentError("Load the forcing case first."))
    height = vec(Array(znodes(model.grid, Center(), Center(), Center(); with_halos=false)))
    basis = device(ena_vertical_basis(height; centers, width))
    θ = liquid_ice_potential_temperature(model)
    compute!(θ)
    initial = (θ=device(Array(interior(θ))), qᵛ=device(Array(interior(model.microphysical_fields.qᵛ))),
               u=device(Array(interior(model.velocities.u))), v=device(Array(interior(model.velocities.v))))
    fixed = (; (name => device(Array(interior(field)))
               for (name, field) in pairs(Oceananigans.prognostic_fields(model))
               if name ∉ (:ρθ, :ρqᵛ, :ρu, :ρv))...)
    initial_scales = (θ=FT(1), qᵛ=FT(0.1), u=FT(2), v=FT(2))
    forcing_scales = (temperature_tendency=FT(1/86400), moisture_tendency=FT(0.001/86400),
                      u=FT(2), v=FT(2), subsidence=FT(0.003))
    surface_scales = (sensible_heat_flux=FT(10), latent_heat_flux=FT(0.3), kinematic_stress=FT(0.3))
    labels = String[]
    ranges = Dict{Symbol, UnitRange{Int}}()
    next = 1
    initial_ranges = map(keys(initial)) do name
        range = next:next+length(centers)-1
        next += length(centers)
        append!(labels, ["initial.$name.$i" for i in eachindex(centers)])
        range
    end
    ranges[:initial] = 1:next-1
    function table_block(tables, scales, group)
        start = next
        blocks = map(keys(scales)) do name
            table = tables[name]
            z = Array(table.heights)
            times = Array(table.times)
            vertical = group == :surface ? ones(length(z), 1) : ena_vertical_basis(z; centers, width)
            fraction = times ./ simulation.stop_time
            temporal = hcat(1 .- fraction, fraction)
            # vec(table) is height-fast; each coefficient is a height × time mode.
            matrix = reduce(hcat, [vec(vertical[:, k] * temporal[:, n]')
                                   for n in 1:2 for k in axes(vertical, 2)])
            range = next:next+size(matrix, 2)-1
            next += size(matrix, 2)
            append!(labels, ["$group.$name.$i" for i in axes(matrix, 2)])
            (; baseline=device(Array(table.values)), basis=device(matrix), range, scale=scales[name])
        end
        ranges[group] = start:next-1
        return NamedTuple{keys(scales)}(blocks)
    end
    forcing = table_block(drivers.profiles, forcing_scales, :forcing)
    surface = table_block(drivers.surface, surface_scales, :surface)
    baseline = (; initial, fixed, basis, initial_ranges, initial_scales, forcing, surface,
                  temperature_guess=device(Array(interior(model.temperature))),
                  radiative_heating=device(Array(interior(drivers.radiative_heating))))
    # A single workspace preserves aliases between the model's forcing and BC
    # parameters and the tables changed below.
    workspace = (; model, profiles=drivers.profiles, surface=drivers.surface,
                   fields=drivers.fields, radiative_heating=drivers.radiative_heating,
                   prescribed_radiation=drivers.prescribed_radiation)
    return (; workspace, baseline, labels, ranges, controls=device(zeros(next-1)))
end

function ena_set_controls!(workspace, baseline, controls)
    model = workspace.model
    reset!(model.clock)
    for name in keys(baseline.forcing)
        block = baseline.forcing[name]
        table = workspace.profiles[name].values
        table .= block.baseline .+ block.scale .* reshape(block.basis * controls[block.range], size(table))
    end
    for name in keys(baseline.surface)
        block = baseline.surface[name]
        table = workspace.surface[name].values
        correction = block.scale .* reshape(block.basis * controls[block.range], size(table))
        if name == :sensible_heat_flux
            table .= block.baseline .+ correction
        else
            # Keep latent flux's sign and keep stress positive.
            table .= block.baseline .* exp.(correction)
        end
    end
    latent_heat = model.thermodynamic_constants.liquid.reference_latent_heat
    workspace.surface.vapor_flux.values .= workspace.surface.latent_heat_flux.values ./ latent_heat
    initial = map(keys(baseline.initial), baseline.initial_ranges) do name, range
        background = baseline.initial[name]
        correction = baseline.initial_scales[name] .* reshape(baseline.basis * controls[range], 1, 1, :)
        name == :qᵛ ? background .* exp.(correction) : background .+ correction
    end
    set!(model.temperature, baseline.temperature_guess)
    set!(workspace.radiative_heating, baseline.radiative_heating)
    set!(model; NamedTuple{keys(baseline.initial)}(initial)..., baseline.fixed...)
    return nothing
end

function ena_update_drivers!(workspace)
    for name in keys(workspace.fields)
        set!(workspace.fields[name], workspace.profiles[name], workspace.model.clock.time)
    end
    ena_update_prescribed_radiation!(workspace.radiative_heating, workspace.prescribed_radiation,
                                   workspace.model.clock.time)
    return nothing
end

function ena_mean_profiles(model)
    variables = (model.temperature, model.microphysical_fields.qᵛ, model.velocities.u, model.velocities.v)
    return reduce(hcat, map(field -> vec(mean(interior(field); dims=(1, 2))), variables))
end

# Piecewise-linear temporal observation operator. Its weights are fixed before AD.
# A sonde ascends: each cell/variable can therefore have a different sample time.
function ena_sampling_weights(target, Δt, steps)
    Δt > 0 && steps ≥ 1 || throw(ArgumentError("Use a positive time step and at least one step."))
    all((target.times[target.weights .> 0] .≥ 0) .&
        (target.times[target.weights .> 0] .≤ steps * Δt)) ||
        throw(ArgumentError("Forecast must cover every valid sounding sample time."))
    times = reshape(target.times, size(target.times)..., 1)
    indices = reshape(collect(0:steps), 1, 1, :)
    return max.(0, 1 .- abs.(times ./ Δt .- indices))
end

function ena_fit_loss(prediction, target, controls, regularization)
    return sum(target.weights .* (prediction .- target.values).^2) / 2 +
           regularization * sum(abs2, controls) / (2length(controls))
end

function ena_forecast_begin!(workspace, controls, baseline, sampling, Δt)
    ena_set_controls!(workspace, baseline, controls)
    prediction = ena_mean_profiles(workspace.model) .* sampling[:, :, 1]
    ena_update_drivers!(workspace)
    first_time_step!(workspace.model, Δt)
    prediction .+= ena_mean_profiles(workspace.model) .* sampling[:, :, 2]
    return prediction
end

function ena_forecast_step!(prediction, workspace, sampling, Δt, step)
    ena_update_drivers!(workspace)
    time_step!(workspace.model, Δt)
    prediction .+= ena_mean_profiles(workspace.model) .* sampling[:, :, step+1]
    return nothing
end

function ena_prediction_native!(workspace, controls, baseline, sampling, Δt, steps)
    prediction = ena_forecast_begin!(workspace, controls, baseline, sampling, Δt)
    for step in 2:steps
        ena_forecast_step!(prediction, workspace, sampling, Δt, step)
    end
    return prediction
end

# Independent numerical reference for validating AD, not an adjoint substitute.
function ena_central_difference(value, parameters; increment=0.01)
    gradient = similar(parameters)
    positive, negative = copy(parameters), copy(parameters)
    for i in eachindex(parameters)
        positive[i] += increment
        negative[i] -= increment
        gradient[i] = (value(positive) - value(negative)) / (2increment)
        positive[i] = negative[i] = parameters[i]
    end
    return gradient
end

# Projected gradient descent with an Armijo search. The derivative provider can
# be tested separately; no objective evaluation is silently used as a gradient.
function ena_optimize(value_gradient, value, initial; iterations=5, bound=3, max_trials=8)
    controls = copy(initial)
    history = NamedTuple[]
    for iteration in 0:iterations
        loss, gradient = value_gradient(controls)
        isfinite(loss) && all(isfinite, gradient) || error("Nonfinite loss/gradient; optimization stopped.")
        push!(history, (; iteration, loss, gradient_norm=sqrt(sum(abs2, gradient))))
        @info "ENA optimization" history[end]...
        iteration == iterations && break
        step = min(1, 1 / max(maximum(abs, gradient), eps(eltype(gradient))))
        accepted = false
        for trial in 1:max_trials
            candidate = clamp.(controls .- step .* gradient, -bound, bound)
            slope = sum(gradient .* (candidate .- controls))
            slope < 0 || break
            candidate_loss = value(candidate)
            if isfinite(candidate_loss) && candidate_loss ≤ loss + 1e-4slope
                controls .= candidate
                accepted = true
                break
            end
            step /= 2
        end
        accepted || break
    end
    return (; controls, history)
end
