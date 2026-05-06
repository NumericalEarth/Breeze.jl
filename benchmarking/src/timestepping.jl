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
    many_compiled_time_steps!(compiled_step!, model, Δt, N=100)

Execute `N` time steps of `model` by calling a precompiled time-stepping
function (e.g. produced by `Reactant.@compile sync=true time_step!(model, Δt)`).
"""
function many_compiled_time_steps!(compiled_step!, model, Δt, N=100)
    for _ in 1:N
        compiled_step!(model, Δt)
    end
    return nothing
end
