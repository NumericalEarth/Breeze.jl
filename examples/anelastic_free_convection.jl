using Breeze
using Oceananigans.Units
using Printf

arch = CPU()
Nx = Nz = 128
Lz = 10e3
grid = RectilinearGrid(arch, size=(Nx, Nz), x=(0, 2Lz), z=(0, Lz), topology=(Periodic, Flat, Bounded))

ρe_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(1000))
advection = WENO()
model = AtmosphereModel(grid; advection, boundary_conditions=(; ρe=ρe_bcs))

Lz = grid.Lz
Δθ = 2 # K
Tₛ = model.formulation.constants.reference_potential_temperature # K
θᵢ(x, z) = Tₛ + Δθ * z / Lz + 1e-2 * Δθ * randn()
Ξ(x, z) = 1e-2 * randn()
set!(model, θ=θᵢ, ρu=Ξ, ρv=Ξ, ρw=Ξ)

simulation = Simulation(model, Δt=5, stop_iteration=1000) #0stop_time=4hours)

# TODO make this work
# conjure_time_step_wizard!(simulation, cfl=0.7)

ρu, ρv, ρw = model.momentum
δ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))

function progress(sim)
    compute!(δ)
    u, v, w = sim.model.velocities
    T = sim.model.temperature
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)
    δmax = maximum(abs, δ)
    Tmin = minimum(T)
    Tmax = maximum(T)

    msg = @sprintf("Iter: %d, t: %s, max|u|: (%.2e, %.2e, %.2e), max|δ|: %.2e, extrema(T): (%.2e, %.2e)",
                    iteration(sim), prettytime(sim), umax, vmax, wmax, δmax, Tmin, Tmax)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

ow = JLD2Writer(model, prognostic_fields(model),
                filename = "free_convection.jld2",
                schedule = TimeInterval(1minutes),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

ρwt = FieldTimeSeries("free_convection.jld2", "ρw")
ρet = FieldTimeSeries("free_convection.jld2", "ρe")
times = ρet.times
Nt = length(ρet)

using GLMakie, Printf

fig = Figure(size=(1200, 600), fontsize=12)
axw = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)")
axe = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)")
slider = Slider(fig[2, 1:2], range=1:Nt, startvalue=1)

n = slider.value
ρwn = @lift interior(ρwt[$n], :, 1, :)
ρen = @lift interior(ρet[$n], :, 1, :)
title = @lift "t = $(prettytime(times[$n]))"

fig[0, :] = Label(fig, title, fontsize=22, tellwidth=false)

Tmin = minimum(ρet)
Tmax = maximum(ρet)
wlim = maximum(abs, ρwt) / 2

hmw = heatmap!(axw, ρwn, colorrange=(-wlim, wlim), colormap=:balance)
hme = heatmap!(axe, ρen, colorrange=(Tmin, Tmax), colormap=:balance)

# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)
# Label(fig[0, 1], "θ", tellwidth=false)
# Label(fig[0, 2], "q", tellwidth=false)

Colorbar(fig[1, 0], hmw, label = "w", vertical=true, flipaxis=true)
Colorbar(fig[1, 3], hme, label = "ρe", vertical=true)

fig

record(fig, "free_convection.mp4", 1:Nt, framerate=12) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
