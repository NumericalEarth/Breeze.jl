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
