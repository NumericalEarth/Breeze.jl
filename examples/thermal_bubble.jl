# # Thermal bubble
#
# Simulates a moist static energy bubble rising through a stably stratified
# atmosphere, following Ahmad & Lindeman (2007) and Sridhar et al. (2022). This
# script doubles as a Literate.jl example rendered into the documentation.
#
# Run it directly with `julia examples/thermal_bubble.jl`. Documentation builds
# and CI automatically pick a lighter configuration via the
# `BREEZE_FAST_EXAMPLES=true` flag to keep runtimes short.

using Breeze
using CairoMakie
using Oceananigans.Units
using Printf

# ## Helper toggles
#
# `fast_mode` trims the grid and duration when `BREEZE_FAST_EXAMPLES=true` (the
# default for documentation builds) or on CI. Set
# `BREEZE_THERMAL_VISUALIZE=true` to produce Makie figures and animations.

const TRUE_STRINGS = ("1", "true", "yes")

is_enabled(var::AbstractString, default = "false") =
    lowercase(get(ENV, var, default)) in TRUE_STRINGS

fast_mode = is_enabled("BREEZE_FAST_EXAMPLES", get(ENV, "CI", "false"))
visualize_example = is_enabled("BREEZE_THERMAL_VISUALIZE", "false")

# ## Architecture and grid

arch = CPU() # switch to GPU() (after `using CUDA`) for accelerated runs

Lx = fast_mode ? 10e3 : 20e3  # horizontal extent [m]
Lz = fast_mode ? 5e3  : 10e3  # vertical extent   [m]
Nx = fast_mode ? 64   : 128
Nz = fast_mode ? 64   : 128

grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, Lz),
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))

advection = WENO(order = 5)
model = AtmosphereModel(grid; advection)

# ## Moist static energy perturbation
#
# We add a localized potential temperature perturbation that translates into a
# moist static energy anomaly.

x₀ = Lx / 2
z₀ = fast_mode ? 2e3 : 4e3
r₀ = fast_mode ? 1.5e3 : 2e3
Δθ = 10 # K

N² = 1e-6
θ₀ = model.formulation.reference_state.potential_temperature
dθdz = N² * θ₀ / 9.81

function θᵢ(x, z)
    θ̄ = θ₀ + dθdz * z
    r = sqrt((x - x₀)^2 + (z - z₀)^2)
    θ′ = Δθ * max(0, 1 - r / r₀)
    return θ̄ + θ′
end

set!(model, θ = θᵢ)

# ## Simulation controls

stop_time = fast_mode ? 10minutes : 30minutes
simulation = Simulation(model; Δt = 1.0, stop_time)

conjure_time_step_wizard!(simulation, cfl = 0.7)

function progress(sim)
    ρe = sim.model.energy
    u, _, w = sim.model.velocities

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, extrema(ρe): (%.2f, %.2f) J/kg, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   minimum(ρe), maximum(ρe),
                   maximum(abs, u), maximum(abs, w))

    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# ## Diagnostics and output
#
# Save velocity components, energy, temperature, and vertical vorticity to JLD2.

output_dir = joinpath(@__DIR__, "outputs")
mkpath(output_dir)

u, v, w = model.velocities
ζ = ∂x(w) - ∂z(u)

ρE = Field{Nothing, Nothing, Center}(grid)
set!(ρE, Field(Average(model.energy, dims = (1, 2))))
ρe′ = model.energy - ρE
ρe = model.energy
T = model.temperature

outputs = merge(model.velocities, model.tracers, (; ζ, ρe′, ρe, T))

filename = joinpath(output_dir, "thermal_bubble_$(Nx)x$(Nz).jld2")
writer = JLD2Writer(model, outputs;
                    filename,
                    schedule = TimeInterval(30seconds),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

@info "Running thermal bubble simulation on grid: $grid"
@info "Bubble parameters: center=($x₀, $z₀), radius=$r₀, Δθ=$Δθ K"
@info "Domain: $(Lx / 1000) km × $(Lz / 1000) km, resolution: $Nx × $Nz"

run!(simulation)

# ## Visualization (optional)
#
# Use Makie to explore the bubble evolution when visualization is enabled.

if visualize_example
    @info "Creating visualization..."

    ρet  = FieldTimeSeries(filename, "ρe")
    Tt   = FieldTimeSeries(filename, "T")
    ut   = FieldTimeSeries(filename, "u")
    wt   = FieldTimeSeries(filename, "w")
    ζt   = FieldTimeSeries(filename, "ζ")
    ρe′t = FieldTimeSeries(filename, "ρe′")

    times = ρet.times
    Nt = length(ρet)

    fig = Figure(size = (1500, 1000), fontsize = 12)
    axe = Axis(fig[1, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Energy ρe (J / kg)")
    axe′ = Axis(fig[1, 2], xlabel = "x (km)", ylabel = "z (km)", title = "Energy Perturbation ρe′ (J / kg)")
    axT = Axis(fig[2, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Temperature T (ᵒK)")
    axζ = Axis(fig[2, 2], xlabel = "x (km)", ylabel = "z (km)", title = "Vorticity ζ (s⁻¹)")
    axu = Axis(fig[3, 1], xlabel = "x (km)", ylabel = "z (km)", title = "Horizontal Velocity u (m/s)")
    axw = Axis(fig[3, 2], xlabel = "x (km)", ylabel = "z (km)", title = "Vertical Velocity w (m/s)")

    slider = Slider(fig[3, 1:2], range = 1:Nt, startvalue = 1)
    n = slider.value

    ρen  = @lift interior(ρet[$n], :, 1, :)
    ρe′n = @lift interior(ρe′t[$n], :, 1, :)
    Tn   = @lift interior(Tt[$n], :, 1, :)
    ζn   = @lift interior(ζt[$n], :, 1, :)
    un   = @lift interior(ut[$n], :, 1, :)
    wn   = @lift interior(wt[$n], :, 1, :)

    x = xnodes(ρet) ./ 1000
    z = znodes(ρet) ./ 1000

    title = @lift "Thermal Bubble Evolution - t = $(prettytime(times[$n]))"
    fig[0, :] = Label(fig, title, fontsize = 16, tellwidth = false)

    ρe_range  = (minimum(ρet), maximum(ρet))
    ρe′_range = (minimum(ρe′t), maximum(ρe′t))
    T_range   = (minimum(Tt), maximum(Tt))
    ζ_range   = maximum(abs, ζt)
    w_range   = maximum(abs, wt)

    hme  = heatmap!(axe,  x, z, ρen,  colorrange = ρe_range,   colormap = :thermal)
    hme′ = heatmap!(axe′, x, z, ρe′n, colorrange = ρe′_range,  colormap = :balance)
    hmT  = heatmap!(axT,  x, z, Tn,   colorrange = T_range,    colormap = :thermal)
    hmζ  = heatmap!(axζ,  x, z, ζn,   colorrange = (-ζ_range, ζ_range), colormap = :balance)
    hmu  = heatmap!(axu,  x, z, un,   colorrange = (-w_range, w_range), colormap = :balance)
    hmw  = heatmap!(axw,  x, z, wn,   colorrange = (-w_range, w_range), colormap = :balance)

    Colorbar(fig[1, 3], hme,  label = "ρe (J/kg)",    vertical = true)
    Colorbar(fig[1, 4], hme′, label = "ρe′ (J/kg)",   vertical = true)
    Colorbar(fig[2, 3], hmT,  label = "T (ᵒK)",       vertical = true)
    Colorbar(fig[2, 4], hmζ,  label = "ζ (s⁻¹)",      vertical = true)
    Colorbar(fig[3, 3], hmu,  label = "u (m/s)",      vertical = true)
    Colorbar(fig[3, 4], hmw,  label = "w (m/s)",      vertical = true)

    for ax in (axe, axe′, axT, axζ, axu, axw)
        xlims!(ax, 0, Lx / 1000)
        ylims!(ax, 0, Lz / 1000)
    end

    save(joinpath(output_dir, "thermal_bubble_evolution.png"), fig)
    @info "Saved visualization to $(joinpath(output_dir, "thermal_bubble_evolution.png"))"

    CairoMakie.record(fig, joinpath(output_dir, "thermal_bubble.mp4"), 1:Nt, framerate = 10) do nn
        @info "Drawing frame $nn of $Nt..."
        n[] = nn
    end

    @info "Saved animation to $(joinpath(output_dir, "thermal_bubble.mp4"))"
else
    @info "Skipping visualization. Set BREEZE_THERMAL_VISUALIZE=true to build it."
end

@info "Thermal bubble simulation completed!"
