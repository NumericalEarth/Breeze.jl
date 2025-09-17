# Thermal Bubble Simulation following Ahmad & Lindeman (2007) and Sridhar et al (2022)
# This simulates a circular potential temperature perturbation in 2D

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie

# Architecture and grid setup
arch = CPU() # Change to GPU() for GPU acceleration

# Domain size - following typical thermal bubble studies
Lx = 20e3  # 20 km horizontal domain
Lz = 10e3  # 10 km vertical domain

# Grid resolution - higher resolution for better bubble dynamics
Nx = 128
Nz = 128

# Create 2D grid (periodic in x, bounded in z)
grid = RectilinearGrid(arch,
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, Lz),
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))

# Thermodynamic setup
p₀ = 101325  # Pa - standard atmospheric pressure
θ₀ = 300.0   # K - reference potential temperature
reference_constants = Breeze.Thermodynamics.ReferenceConstants(base_pressure=p₀, potential_temperature=θ₀)
buoyancy = Breeze.MoistAirBuoyancy(; reference_constants)

# Advection scheme - WENO for high-order accuracy
advection = WENO(order=5)

# Create the model
#model = NonhydrostaticModel(; grid, advection, buoyancy, tracers = (:θ, :q))
model = AtmosphereModel(grid; advection) #, buoyancy, tracers = (:θ, :q))

# Thermal bubble parameters
x₀ = Lx / 2      # Center of bubble in x
z₀ = 4e3         # Center of bubble in z (2 km height)
r₀ = 2e3         # Initial radius of bubble (2 km)
Δθ = 2           # Potential temperature perturbation (K)

# Background stratification
N² = 1e-6        # Brunt-Väisälä frequency squared (s⁻²)
dθdz = N² * θ₀ / 9.81  # Background potential temperature gradient

# Initial conditions
function θᵢ(x, z)
    θ̄ = θ₀ + dθdz * z # background stratification
    r = sqrt((x - x₀)^2 + (z - z₀)^2) # distance from bubble center
    θ′ = Δθ * max(0, 1 - r / r₀) # bubble
    return θ̄ + θ′
end

set!(model, θ=θᵢ)

# Simulation parameters
stop_time = 30minutes
simulation = Simulation(model; Δt=1.0, stop_time)

# Adaptive time stepping
conjure_time_step_wizard!(simulation, cfl=0.7)

# Progress monitoring
function progress(sim)
    θ = sim.model.tracers.θ
    u, w = sim.model.velocities
    
    θ_max = maximum(θ)
    θ_min = minimum(θ)
    u_max = maximum(abs, u)
    w_max = maximum(abs, w)
    
    msg = @sprintf("Iter: %d, t: %s, Δt: %s, extrema(θ): (%.2f, %.2f) K, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), θ_min, θ_max, u_max, w_max)
    
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# Output setup
# T = Breeze.TemperatureField(model)

# Calculate vorticity using abstract operations (2D: only z-component)
u, v, w = model.velocities
ζ = ∂x(w) - ∂z(u)

# Temperature perturbation from background state
θ_bg_field = Field{Nothing, Nothing, Center}(grid)
set!(θ_bg_field, z -> θ₀ + dθdz * z)
θ′ = model.tracers.θ - θ_bg_field

#outputs = merge(model.velocities, model.tracers, (; T, ζ, θ′))
outputs = merge(model.velocities, model.tracers, (; ζ, θ′))

filename = "thermal_bubble_$(Nx)x$(Nz).jld2"
writer = JLD2Writer(model, outputs; filename,
                    schedule = TimeInterval(30seconds),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

@info "Running thermal bubble simulation on grid: $grid"
@info "Bubble parameters: center=($x₀, $z₀), radius=$r₀, Δθ=$Δθ K"
@info "Domain: $(Lx/1000) km × $(Lz/1000) km, resolution: $Nx × $Nz"

run!(simulation)

# Post-processing and visualization
if get(ENV, "CI", "false") == "false"
    @info "Creating visualization..."
    
    # Read the output data
    θt = FieldTimeSeries(filename, "θ")
    Tt = FieldTimeSeries(filename, "T")
    ut = FieldTimeSeries(filename, "u")
    wt = FieldTimeSeries(filename, "w")
    ζt = FieldTimeSeries(filename, "ζ")
    θ′t = FieldTimeSeries(filename, "θ′")
    
    times = θt.times
    Nt = length(θt)
    
    # Create visualization - expanded to 6 panels
    fig = Figure(size=(1500, 1000), fontsize=12)
    
    # Subplot layout - 3 rows, 2 columns
    axθ = Axis(fig[1, 1], xlabel="x (km)", ylabel="z (km)", title="Potential Temperature θ (K)")
    axθ′ = Axis(fig[1, 2], xlabel="x (km)", ylabel="z (km)", title="Temperature Perturbation θ′ (K)")
    axT = Axis(fig[2, 1], xlabel="x (km)", ylabel="z (km)", title="Temperature T (K)")
    axζ = Axis(fig[2, 2], xlabel="x (km)", ylabel="z (km)", title="Vorticity ζ (s⁻¹)")
    axu = Axis(fig[3, 1], xlabel="x (km)", ylabel="z (km)", title="Horizontal Velocity u (m/s)")
    axw = Axis(fig[3, 2], xlabel="x (km)", ylabel="z (km)", title="Vertical Velocity w (m/s)")
    
    # Time slider
    slider = Slider(fig[4, 1:2], range=1:Nt, startvalue=1)
    n = slider.value
    
    # Observable fields
    θn = @lift interior(θt[$n], :, 1, :)
    θ′n = @lift interior(θ′t[$n], :, 1, :)
    Tn = @lift interior(Tt[$n], :, 1, :)
    ζn = @lift interior(ζt[$n], :, 1, :)
    un = @lift interior(ut[$n], :, 1, :)
    wn = @lift interior(wt[$n], :, 1, :)
    
    # Grid coordinates
    x = xnodes(θt) ./ 1000  # Convert to km
    z = znodes(θt) ./ 1000  # Convert to km
    
    # Title with time
    title = @lift "Thermal Bubble Evolution - t = $(prettytime(times[$n]))"
    fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)
    
    # Create heatmaps
    θ_range = (minimum(θt), maximum(θt))
    θ′_range = (minimum(θ′t), maximum(θ′t))
    T_range = (minimum(Tt), maximum(Tt))
    ζ_range = maximum(abs, ζt)
    u_range = maximum(abs, ut)
    w_range = maximum(abs, wt)
    
    hmθ = heatmap!(axθ, x, z, θn, colorrange=θ_range, colormap=:thermal)
    hmθ′ = heatmap!(axθ′, x, z, θ′n, colorrange=θ′_range, colormap=:balance)
    hmT = heatmap!(axT, x, z, Tn, colorrange=T_range, colormap=:thermal)
    hmζ = heatmap!(axζ, x, z, ζn, colorrange=(-ζ_range, ζ_range), colormap=:balance)
    hmu = heatmap!(axu, x, z, un, colorrange=(-u_range, u_range), colormap=:balance)
    hmw = heatmap!(axw, x, z, wn, colorrange=(-w_range, w_range), colormap=:balance)
    
    # Add colorbars
    Colorbar(fig[1, 3], hmθ, label="θ (K)", vertical=true)
    Colorbar(fig[1, 4], hmθ′, label="θ′ (K)", vertical=true)
    Colorbar(fig[2, 3], hmT, label="T (K)", vertical=true)
    Colorbar(fig[2, 4], hmζ, label="ζ (s⁻¹)", vertical=true)
    Colorbar(fig[3, 3], hmu, label="u (m/s)", vertical=true)
    Colorbar(fig[3, 4], hmw, label="w (m/s)", vertical=true)
    
    # Set axis limits
    for ax in [axθ, axθ′, axT, axζ, axu, axw]
        xlims!(ax, 0, Lx/1000)
        ylims!(ax, 0, Lz/1000)
    end
    
    # Save the figure
    save("thermal_bubble_evolution.png", fig)
    @info "Saved visualization to thermal_bubble_evolution.png"
    
    # Create animation
    @info "Creating animation..."
    CairoMakie.record(fig, "thermal_bubble.mp4", 1:Nt, framerate=10) do nn
        @info "Drawing frame $nn of $Nt..."
        n[] = nn
    end
    @info "Saved animation to thermal_bubble.mp4"
end

@info "Thermal bubble simulation completed!"
