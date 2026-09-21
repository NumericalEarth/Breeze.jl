#####
##### Time stepping without Simulation overhead
#####

"""
    many_time_steps!(model, Δt, N=100)

Execute `N` time steps of `model` with time step `Δt`.
This directly calls `time_step!` without any Simulation overhead.
"""
function many_time_steps!(model, Δt, N=100)
    for _ in 1:N
        time_step!(model, Δt)
    end
    return nothing
end

"""
    step_loop!(model, Δt, Nsteps)

Drive `Nsteps` time steps of `model` inside a `Reactant.@trace` loop so that
`Reactant.@compile` lowers the entire stepping loop into a single XLA program.
On non-Reactant backends `@trace` is a no-op decorator and this is equivalent
to `many_time_steps!`.
"""
function step_loop!(model, Δt, Nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return nothing
end

#####
##### Forward+backward loop for AD benchmarks (Reactant + Enzyme reverse mode).
#####
##### Mirrors the test/reactant_*_compilation.jl pattern: `loss` runs the
##### checkpointed step loop and reduces to a scalar; `grad_loss!` calls
##### Enzyme reverse-mode AD over `loss`. Both are compiled together via
##### `Reactant.@compile raise=true`.
#####
##### `checkpointing` selects the loop checkpointing strategy Enzyme uses in
##### the reverse pass and is one of
#####   - `false`: no checkpointing, the whole forward tape is kept;
#####   - `true`: Reactant's default, `Periodic(isqrt(Nsteps))`;
#####   - `Reactant.Periodic(n)`: `n` evenly spaced checkpoints;
#####   - `Reactant.Binomial(budget)`: revolve schedule with at most `budget`
#####     checkpoints live at once.
##### It is a plain Julia value (not traced), so each strategy is a separate
##### compile. `benchmark_time_stepping` resolves `true` to an explicit
##### `Periodic(isqrt(Nsteps))` before compiling so the result records it.
#####

function loss(model, θ_init, Δt, Nsteps, checkpointing)
    set!(model; θ=θ_init, ρ=1.0)
    # NOTE: forward step_loop! uses mincut=true, but the min-cut planner
    # blows host RAM during AD compile (graph algorithm over the dataflow
    # graph of one body iteration). Run AD without mincut.
    @trace checkpointing=checkpointing track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss!(model, dmodel, θ_init, dθ_init, Δt, Nsteps, checkpointing)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(Nsteps),
        Enzyme.Const(checkpointing))
    return loss_value
end

"""
    checkpointing_label(checkpointing)

Short string identifying a loop checkpointing strategy, used in benchmark
names and recorded in `BenchmarkResult.checkpointing`.
"""
checkpointing_label(checkpointing::Bool) = checkpointing ? "auto" : "none"
checkpointing_label(checkpointing::Reactant.Periodic) = "periodic$(checkpointing.n)"
checkpointing_label(checkpointing::Reactant.Binomial) = "binomial$(checkpointing.budget)"
