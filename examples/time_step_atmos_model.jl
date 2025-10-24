using Breeze.AtmosphereModels: AtmosphereModel
using Oceananigans
using Oceananigans.Units
using Printf

Lx = 1e3
Ly = 1e3
Lz = 1e3

grid = RectilinearGrid(size=(64, 1, 64), x=(0, Lx), y=(0, Ly), z=(0, Lz))

Q₀ = 1000 # heat flux in W / m²
e_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(Q₀))
model = AtmosphereModel(grid, advection=WENO(), boundary_conditions=(; e=e_bcs))

Lz = grid.Lz
Δθ = 5 # K
Tₛ = model.formulation.thermo.potential_temperature
# θᵢ(x, y, z) = Tₛ + Δθ * z / Lz
qᵢ(x, y, z) = 0
Ξᵢ(x, y, z) = 1e-2 * randn()

# Thermal bubble parameters
N² = 1e-6        # Brunt-Väisälä frequency squared (s⁻²)
x₀ = Lx / 2      # Center of bubble in x
z₀ = Lz / 3      # Center of bubble in z
r₀ = Lz / 6      # Initial radius of bubble
dθdz = N² * Tₛ / 9.81  # Background potential temperature gradient

# Initial conditions
function θᵢ(x, y, z)
    θ̄ = Tₛ + dθdz * z # background stratification
    r = sqrt((x - x₀)^2 + (z - z₀)^2) # distance from bubble center
    θ′ = Δθ * max(0, 1 - r / r₀) # bubble
    return θ̄ + θ′
end

set!(model, θ=θᵢ, q=qᵢ, u=Ξᵢ, v=Ξᵢ)

ρu, ρv, ρw = model.momentum
δ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))
compute!(δ)

stop_time = 5minutes
simulation = Simulation(model, Δt=0.1, stop_iteration=10)
# conjure_time_step_wizard!(simulation, cfl=0.7)

using Printf

ρu, ρv, ρw = model.momentum
δ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))

function progress(sim)
    T = sim.model.temperature
    u, v, w = sim.model.velocities
    T_max = maximum(T)
    T_min = minimum(T)
    u_max = maximum(abs, u)
    v_max = maximum(abs, v)
    w_max = maximum(abs, w)
    compute!(δ)
    δ_max = maximum(abs, δ)
    msg = @sprintf("Iter: %d, time: %s, extrema(T): (%.2f, %.2f) K, max|δ|: %.2e",
                   iteration(sim), prettytime(sim), T_min, T_max, δ_max)
    msg *= @sprintf(", max|u|: (%.2e, %.2e, %.2e) m s⁻¹", u_max, v_max, w_max)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

# Calculate vorticity using abstract operations (2D: only z-component)
u, v, w = model.velocities
ζ = ∂x(w) - ∂z(u)

# Temperature perturbation from background state
θ_bg_field = Field{Nothing, Nothing, Center}(grid)
set!(θ_bg_field, z -> Tₛ + dθdz * z)

T = model.temperature
ρʳ = model.formulation.reference_density
cᵖᵈ = model.thermodynamics.dry_air.heat_capacity
ρe = model.energy
θ = ρe / (ρʳ * cᵖᵈ)
θ′ = θ - θ_bg_field

outputs = merge(model.velocities, (; ζ, θ′, T, ρe))

Nx, Ny, Nz = size(grid)
filename = "thermal_bubble_$(Nx)x$(Nz).jld2"
writer = JLD2Writer(model, outputs; filename,
                    schedule = IterationInterval(10),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

run!(simulation)

using GLMakie

# Read the output data
et = FieldTimeSeries(filename, "ρe")
wt = FieldTimeSeries(filename, "w")
ζt = FieldTimeSeries(filename, "ζ")
θ′t = FieldTimeSeries(filename, "θ′")
Tt = FieldTimeSeries(filename, "T")

times = wt.times
Nt = length(wt)

fig = Figure(size=(1500, 1000), fontsize=12)

axθ′ = Axis(fig[1, 1], title="Perturbation potentital temperature θ′ (K)")
axT = Axis(fig[1, 2], title="Temperature T (K)")
#axζ = Axis(fig[1, 2], title="Vorticity ζ (s⁻¹)")
axw = Axis(fig[2, 1], title="Vertical Velocity w (m/s)")
axe = Axis(fig[2, 2], title="Internal Energy ρe (J/kg)")

# Time slider
slider = Slider(fig[3, 1:3], range=1:Nt, startvalue=1)
n = slider.value

# Observable fields
θ′n = @lift interior(θ′t[$n], :, 1, :)
Tn = @lift interior(Tt[$n], :, 1, :)
# ζn = @lift interior(ζt[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)
en = @lift interior(et[$n], :, 1, :)

# Title with time
title = @lift "Thermal Bubble - t = $(prettytime(times[$n]))"
fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

# Create heatmaps
θ′_range = (minimum(θ′t), maximum(θ′t))
T_range = (minimum(Tt), maximum(Tt))
# ζ_range = maximum(abs, ζt)
w_range = maximum(abs, wt)
e_range = maximum(abs, et)

hmθ′ = heatmap!(axθ′, θ′n, colorrange=θ′_range, colormap=:balance)
hmT = heatmap!(axT, Tn, colorrange=T_range, colormap=:thermal)
# hmζ = heatmap!(axζ, ζn, colorrange=(-ζ_range, ζ_range), colormap=:balance)
hmw = heatmap!(axw, wn, colorrange=(-w_range, w_range), colormap=:balance)
hme = heatmap!(axe, en, colorrange=(-e_range, e_range), colormap=:balance)

# Add colorbars
Colorbar(fig[1, 0], hmθ′, label="θ′ (K)", vertical=true)
Colorbar(fig[1, 3], hmT, label="T (K)", vertical=true)
# Colorbar(fig[1, 3], hmζ, label="ζ (s⁻¹)", vertical=true)
Colorbar(fig[2, 0], hmw, label="w (m/s)", vertical=true)
Colorbar(fig[2, 3], hme, label="ρe (J/kg)", vertical=true)

display(fig)

#=
# Create animation
@info "Creating animation..."
GLMakie.record(fig, "thermal_bubble.mp4", 1:Nt, framerate=10) do nn
    @info "Drawing frame $nn of $Nt..."
    n[] = nn
end
@info "Saved animation to thermal_bubble.mp4"
=#
