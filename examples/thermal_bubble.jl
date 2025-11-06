# Thermal Bubble Simulation following Ahmad & Lindeman (2007) and Sridhar et al (2022)
# This simulates a circular moist static energy perturbation in 2D

using Breeze
using Oceananigans.Units
using Printf
using CairoMakie

# Architecture and grid setup
arch = CPU() # if changing to GPU() add `using CUDA` above

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
reference_constants = ReferenceStateConstants(base_pressure=p₀, potential_temperature=θ₀)
buoyancy = MoistAirBuoyancy(; reference_constants)

# Advection scheme - WENO for high-order accuracy
advection = WENO(order=5)

# Create the model
model = AtmosphereModel(grid; advection)

# Thermal bubble parameters (moist static energy)
x₀ = Lx / 2      # Center of bubble in x
z₀ = 4e3         # Center of bubble in z (2 km height)
r₀ = 2e3         # Initial radius of bubble (2 km)
Δe = 2 * 1004.0  # Moist static energy perturbation scale (J/m³) ≈ 2 K × ρᵣ × cᵖᵈ at low levels

# Background stratification (used to construct a gently increasing MSE with height via θ)
N² = 1e-6        # Brunt-Väisälä frequency squared (s⁻²)
dθdz = N² * θ₀ / 9.81  # Background potential temperature gradient used to build MSE

# Initial conditions (set moist static energy directly)
# We form MSE ρe ≈ ρᵣ cᵖᵈ θ with a localized perturbation.
ρʳ = model.formulation.reference_density
cᵖᵈ = model.thermodynamics.dry_air.heat_capacity

function eᵢ(x, z)
    θ̄ = θ₀ + dθdz * z # background potential temperature used to build MSE
    r = sqrt((x - x₀)^2 + (z - z₀)^2) # distance from bubble center
    w = max(0, 1 - r / r₀)             # bubble weight (cone)
    ρ = @inbounds ρʳ[1, 1, max(1, min(size(ρʳ, 3), Int(clamp(round(z / (Lz / Nz)) + 1, 1, Nz))))]
    # Background + perturbation in ρe
    ē = ρ * cᵖᵈ * θ̄
    e′ = Δe * w
    return ē + e′
end

set!(model, ρe=eᵢ)

# Simulation parameters
stop_time = 30minutes
simulation = Simulation(model; Δt=1.0, stop_time)

# Adaptive time stepping
conjure_time_step_wizard!(simulation, cfl=0.7)

# Progress monitoring
function progress(sim)
    e = sim.model.energy
    u, w = sim.model.velocities

    e_max = maximum(e)
    e_min = minimum(e)
    u_max = maximum(abs, u)
    w_max = maximum(abs, w)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, extrema(ρe): (%.2f, %.2f) J/m³, max|u|: %.2f m/s, max|w|: %.2f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), e_min, e_max, u_max, w_max)

    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# Output setup
# T = Breeze.TemperatureField(model)

# Calculate vorticity using abstract operations (2D: only z-component)
u, v, w = model.velocities
ζ = ∂x(w) - ∂z(u)

# Moist static energy perturbation from background state
E = Field{Nothing, Nothing, Center}(grid)
set!(E, Field(Average(model.energy, dims=(1, 2))))
e′ = model.energy - E

outputs = merge(model.velocities, (; ζ, ρe=model.energy, e′))

filename = "thermal_bubble_$(Nx)x$(Nz).jld2"
writer = JLD2Writer(model, outputs; filename,
                    schedule = TimeInterval(30seconds),
                    overwrite_existing = true)

simulation.output_writers[:jld2] = writer

@info "Running thermal bubble simulation on grid: $grid"
@info "Bubble parameters: center=($x₀, $z₀), radius=$r₀, Δe=$Δe J/m³"
@info "Domain: $(Lx/1000) km × $(Lz/1000) km, resolution: $Nx × $Nz"

run!(simulation)

# Post-processing and visualization
if get(ENV, "CI", "false") == "false"
    @info "Creating visualization..."

    # Read the output data
    et = FieldTimeSeries(filename, "ρe")
    ut = FieldTimeSeries(filename, "u")
    wt = FieldTimeSeries(filename, "w")
    ζt = FieldTimeSeries(filename, "ζ")
    e′t = FieldTimeSeries(filename, "e′")

    times = et.times
    Nt = length(et)

    # Create visualization - 4 panels
    fig = Figure(size=(1500, 1000), fontsize=12)

    # Subplot layout - 2 rows, 2 columns
    axe = Axis(fig[1, 1], xlabel="x (km)", ylabel="z (km)", title="Moist Static Energy ρe (J/m³)")
    axe′ = Axis(fig[1, 2], xlabel="x (km)", ylabel="z (km)", title="MSE Perturbation e′ (J/m³)")
    axζ = Axis(fig[2, 1], xlabel="x (km)", ylabel="z (km)", title="Vorticity ζ (s⁻¹)")
    axw = Axis(fig[2, 2], xlabel="x (km)", ylabel="z (km)", title="Vertical Velocity w (m/s)")

    # Time slider
    slider = Slider(fig[3, 1:2], range=1:Nt, startvalue=1)
    n = slider.value

    # Observable fields
    en = @lift interior(et[$n], :, 1, :)
    e′n = @lift interior(e′t[$n], :, 1, :)
    ζn = @lift interior(ζt[$n], :, 1, :)
    wn = @lift interior(wt[$n], :, 1, :)

    # Grid coordinates
    x = xnodes(et) ./ 1000  # Convert to km
    z = znodes(et) ./ 1000  # Convert to km

    # Title with time
    title = @lift "Thermal Bubble Evolution - t = $(prettytime(times[$n]))"
    fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

    # Create heatmaps
    e_range = (minimum(et), maximum(et))
    e′_range = maximum(abs, e′t)
    ζ_range = maximum(abs, ζt)
    w_range = maximum(abs, wt)

    hme = heatmap!(axe, x, z, en, colorrange=e_range, colormap=:thermal)
    hme′ = heatmap!(axe′, x, z, e′n, colorrange=(-e′_range, e′_range), colormap=:balance)
    hmζ = heatmap!(axζ, x, z, ζn, colorrange=(-ζ_range, ζ_range), colormap=:balance)
    hmw = heatmap!(axw, x, z, wn, colorrange=(-w_range, w_range), colormap=:balance)

    # Add colorbars
    Colorbar(fig[1, 3], hme, label="ρe (J/m³)", vertical=true)
    Colorbar(fig[1, 4], hme′, label="e′ (J/m³)", vertical=true)
    Colorbar(fig[2, 3], hmζ, label="ζ (s⁻¹)", vertical=true)
    Colorbar(fig[2, 4], hmw, label="w (m/s)", vertical=true)

    # Set axis limits
    for ax in [axe, axe′, axζ, axw]
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
