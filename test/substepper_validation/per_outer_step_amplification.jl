#####
##### Direct measurement of the per-outer-step linearized amplification
##### at rest. Add a tiny perturbation to one cell, take ONE outer step,
##### measure how much each prognostic field changed. Sweep over (Δt, ω).
#####
##### This measures the ACTUAL operator (substep loop + freeze +
##### recovery) on a known input, without any abstraction errors.
#####

using CUDA
using Oceananigans
using Oceananigans.Grids: znode, Center, Face
using Breeze
using BreezyBaroclinicInstability
using JLD2
using Printf
using LinearAlgebra: norm

include("sweep_runner.jl")

const arch = CUDA.functional() ? GPU() : CPU()

"""Take one outer step from a known state, return change in (ρ, ρθ, ρu, ρv, ρw)."""
function one_step_response(model, Δt; perturb_field = :ρw, perturb_value = 1e-8,
                                       perturb_loc = (4, 4, 16))
    # Extract the model's current state
    ρ_field = BreezyBaroclinicInstability.dynamics_density(model.dynamics)
    ρθ_field = model.formulation.potential_temperature_density
    ρu = model.momentum.ρu
    ρv = model.momentum.ρv
    ρw = model.momentum.ρw

    # Snapshot pre-step
    ρ_pre = Array(interior(ρ_field))
    ρθ_pre = Array(interior(ρθ_field))
    ρu_pre = Array(interior(ρu))
    ρv_pre = Array(interior(ρv))
    ρw_pre = Array(interior(ρw))

    # Apply perturbation
    i, j, k = perturb_loc
    if perturb_field === :ρw
        f = ρw
    elseif perturb_field === :ρu
        f = ρu
    elseif perturb_field === :ρθ
        f = ρθ_field
    elseif perturb_field === :ρ
        f = ρ_field
    end
    arr = Array(interior(f))
    arr[i, j, k] += perturb_value
    Oceananigans.set!(f, arr)
    Oceananigans.TimeSteppers.update_state!(model, compute_tendencies = false)

    # Take one outer step
    Oceananigans.TimeSteppers.time_step!(model, Δt)

    # Measure response
    ρ_post = Array(interior(ρ_field))
    ρθ_post = Array(interior(ρθ_field))
    ρu_post = Array(interior(ρu))
    ρv_post = Array(interior(ρv))
    ρw_post = Array(interior(ρw))

    # Subtract perturbation from input position to compare deltas
    Δρ = ρ_post .- ρ_pre
    Δρθ = ρθ_post .- ρθ_pre
    Δρu = ρu_post .- ρu_pre
    Δρv = ρv_post .- ρv_pre
    Δρw = ρw_post .- ρw_pre

    # Subtract the linear part of the perturbation propagation:
    # the perturbation vector entered as 1e-8 in (i,j,k) of ρw_pre. We've added it.
    # The DELTA we measure = (post − pre) where pre had the perturbation already.
    # So Δ here is the response of the perturbed-IC integration. To get the
    # linearized response, we'd compare to an unperturbed integration too.

    # Compute amplification: max|response| / perturb_value
    # using the response in the field we perturbed.
    response_max = if perturb_field === :ρw
        maximum(abs, Δρw)
    elseif perturb_field === :ρu
        maximum(abs, Δρu)
    elseif perturb_field === :ρθ
        maximum(abs, Δρθ)
    elseif perturb_field === :ρ
        maximum(abs, Δρ)
    end

    return (response_max = response_max,
            amplification = response_max / abs(perturb_value),
            max_Δρ = maximum(abs, Δρ),
            max_Δρθ = maximum(abs, Δρθ),
            max_Δρu = maximum(abs, Δρu),
            max_Δρv = maximum(abs, Δρv),
            max_Δρw = maximum(abs, Δρw))
end

"""
Build TWO models at rest with the same parameters, perturb one and not the
other, take one outer step on both, and return the difference.
"""
function linearized_response(; Δt, ω = 0.55, Lz = 30e3, Nx = 8, Ny = 8, Nz = 64,
                                T₀ = 250.0, perturb_value = 1e-10,
                                perturb_field = :ρw,
                                perturb_loc = (4, 4, 16))

    model_unperturbed, _ = build_rest_model(; Lz, Nx, Ny, Nz, T₀,
                                              td_kwargs = (forward_weight = ω,))
    Oceananigans.TimeSteppers.update_state!(model_unperturbed)
    Oceananigans.TimeSteppers.time_step!(model_unperturbed, Δt)

    model_perturbed, _ = build_rest_model(; Lz, Nx, Ny, Nz, T₀,
                                            td_kwargs = (forward_weight = ω,))
    Oceananigans.TimeSteppers.update_state!(model_perturbed)
    # Apply perturbation
    f = if perturb_field === :ρw
        model_perturbed.momentum.ρw
    elseif perturb_field === :ρu
        model_perturbed.momentum.ρu
    elseif perturb_field === :ρθ
        model_perturbed.formulation.potential_temperature_density
    end
    arr = Array(interior(f))
    arr[perturb_loc...] += perturb_value
    Oceananigans.set!(f, arr)
    Oceananigans.TimeSteppers.time_step!(model_perturbed, Δt)

    # Differences (subtract out unperturbed evolution to isolate the linearized response)
    fields_p = (
        BreezyBaroclinicInstability.dynamics_density(model_perturbed.dynamics),
        model_perturbed.formulation.potential_temperature_density,
        model_perturbed.momentum.ρu, model_perturbed.momentum.ρv,
        model_perturbed.momentum.ρw)
    fields_u = (
        BreezyBaroclinicInstability.dynamics_density(model_unperturbed.dynamics),
        model_unperturbed.formulation.potential_temperature_density,
        model_unperturbed.momentum.ρu, model_unperturbed.momentum.ρv,
        model_unperturbed.momentum.ρw)
    names = (:ρ, :ρθ, :ρu, :ρv, :ρw)

    deltas = NamedTuple()
    response_max = 0.0
    for (n, fp, fu) in zip(names, fields_p, fields_u)
        δ = Array(interior(fp)) .- Array(interior(fu))
        m = maximum(abs, δ)
        deltas = merge(deltas, NamedTuple{(Symbol(:max_, n),)}((m,)))
        # The response in the perturbed field
        if n == perturb_field
            response_max = m
        end
    end

    amp = response_max / abs(perturb_value)
    return (Δt = Δt, ω = ω, perturb_field = perturb_field,
            perturb_value = perturb_value,
            response_max = response_max, amplification = amp,
            deltas...)
end

#####
##### Sweep
#####

results = NamedTuple[]

@info "Per-outer-step linearized amplification at rest, ρw perturbation"
for ω in (0.50, 0.55, 0.60, 0.70, 0.80, 0.99), Δt in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
    @info "ω=$ω Δt=$(Δt)s"
    r = linearized_response(; Δt, ω, perturb_field = :ρw)
    push!(results, r)
end

println("\n=== Per-outer-step amplification of ρw perturbation (at rest) ===")
println(rpad("Δt", 10), join([rpad("ω=$ω", 12) for ω in (0.50, 0.55, 0.60, 0.70, 0.80, 0.99)]))
for Δt in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
    print(rpad("Δt=$(Δt)s", 10))
    for ω in (0.50, 0.55, 0.60, 0.70, 0.80, 0.99)
        r = filter(x -> x.ω == ω && x.Δt == Δt, results)
        if isempty(r); print("    -       "); continue; end
        print(rpad(@sprintf("%.4f", r[1].amplification), 12))
    end
    println()
end

# Also probe σ and η responses
@info "\nNow ρθ-perturbation"
results2 = NamedTuple[]
for ω in (0.55, 0.70), Δt in (5.0, 20.0, 40.0)
    @info "ω=$ω Δt=$(Δt)s"
    r = linearized_response(; Δt, ω, perturb_field = :ρθ, perturb_value = 1e-6)
    push!(results2, r)
end

for r in results2
    @info "ρθ perturbation: ω=$(r.ω) Δt=$(r.Δt)  amplification=$(r.amplification)  max_Δρw=$(r.max_ρw)"
end

jldsave(joinpath(@__DIR__, "amp_results.jld2"); results = results, results_θ = results2)
@info "Saved amp_results.jld2"
