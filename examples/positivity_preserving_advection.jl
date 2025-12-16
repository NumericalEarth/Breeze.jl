# # Positivity-preserving advection of a Gaussian tracer
#
# This example demonstrates positivity-preserving advection using directionally-split
# integration, following the algorithm described by the MITgcm documentation:
# https://mitgcm.org/sealion/online_documents/node80.html
#
# We advect a Gaussian tracer distribution diagonally across a periodic domain
# and compare standard advection (which can produce spurious negative values)
# with positivity-preserving advection (which maintains tracer bounds).
#
# This is a classic test case for advection schemes, used in many papers and
# model validation studies.

using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf
using Statistics

# ## Domain and grid setup
#
# We use a square periodic domain with uniform resolution. The resolution is
# relatively coarse to make the differences between schemes visible.

Nx = Ny = 30
Lx = Ly = 1.0 # domain length
Lz = 100 # meters, thin vertical extent

grid = RectilinearGrid(CPU();
                       size = (Nx, Ny, 1),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

# ## Advection velocity
#
# We prescribe a uniform diagonal velocity. The Courant number determines
# how challenging the advection problem is for maintaining positivity.

courant_number = 0.27  # moderate Courant number (also try 0.01 for low, 0.47 for high)
Δx = Lx / Nx
velocity_magnitude = courant_number * Δx / 1.0  # assuming Δt = 1 second

# Diagonal advection: equal velocity in x and y
u_velocity = velocity_magnitude / sqrt(2)
v_velocity = velocity_magnitude / sqrt(2)

@info "Courant number: $courant_number, velocity magnitude: $velocity_magnitude m/s"

# ## Gaussian initial condition
#
# We initialize a Gaussian tracer distribution centered in the domain.
# The key feature to observe is whether negative values appear during advection.

function gaussian_tracer(x, y, z;
                         x₀ = 0.5,
                         y₀ = 0.5,
                         σ = 0.1)
    r² = (x - x₀)^2 + (y - y₀)^2
    return exp(-r² / (2 * σ^2))
end

# ## Standard advection model (may produce negative values)
#
# First, we run with standard advection to see if spurious negative values appear.

advection_standard = WENO(order=5)

model_standard = AtmosphereModel(grid;
                                 tracers = :c,
                                 advection = advection_standard,
                                 microphysics = nothing,
                                 closure = nothing)

# Set initial tracer distribution
set!(model_standard, c = gaussian_tracer)

# The tracer is advected by the background flow. For pure advection test,
# we modify the velocities to be uniform.
u, v, w = model_standard.velocities
set!(model_standard, u = u_velocity, v = v_velocity)

# ## Bounds-preserving WENO advection (should maintain bounds)
#
# Now we set up the same problem but with bounds-preserving WENO.
# The bounds are set to (0, 1) since the Gaussian is normalized.

advection_bounded = WENO(order=5, bounds=(0, 1))

model_bounded = AtmosphereModel(grid;
                                tracers = :c,
                                advection = advection_bounded,
                                microphysics = nothing,
                                closure = nothing)

set!(model_bounded, c = gaussian_tracer)
set!(model_bounded, u = u_velocity, v = v_velocity)

# ## Run simulations
#
# We run both simulations for the same time period (one complete traversal
# of the domain, T = 0.5 which corresponds to diagonal transport to the
# opposite corner).

Δt = 1.0  # seconds, chosen so velocity * Δt / Δx = courant_number
stop_time = Lx / velocity_magnitude  # time for one domain traversal

simulation_standard = Simulation(model_standard; Δt, stop_time)
simulation_bounded = Simulation(model_bounded; Δt, stop_time)

# Track minimum values during the simulation
min_values_standard = Float64[]
min_values_bounded = Float64[]

function track_minimum_standard!(sim)
    c = sim.model.tracers.c
    push!(min_values_standard, minimum(c))
    return nothing
end

function track_minimum_bounded!(sim)
    c = sim.model.tracers.c
    push!(min_values_bounded, minimum(c))
    return nothing
end

add_callback!(simulation_standard, track_minimum_standard!, IterationInterval(1))
add_callback!(simulation_bounded, track_minimum_bounded!, IterationInterval(1))

@info "Running standard advection..."
run!(simulation_standard)

@info "Running bounds-preserving advection..."
run!(simulation_bounded)

# ## Results
#
# Check for negative values

@info "Standard advection: min(c) = $(minimum(model_standard.tracers.c)), max(c) = $(maximum(model_standard.tracers.c))"
@info "Bounded advection: min(c) = $(minimum(model_bounded.tracers.c)), max(c) = $(maximum(model_bounded.tracers.c))"

# ## Visualization
#
# Create a comparison plot similar to the MITgcm documentation figures.

fig = Figure(size = (1000, 500), fontsize = 14)

# Extract tracer data for plotting
c_standard = interior(model_standard.tracers.c, :, :, 1)
c_bounded = interior(model_bounded.tracers.c, :, :, 1)

x = xnodes(grid, Center())
y = ynodes(grid, Center())

# Common colormap limits (include potential negative values for visualization)
cmin = min(minimum(c_standard), minimum(c_bounded), 0)
cmax = max(maximum(c_standard), maximum(c_bounded))

ax1 = Axis(fig[1, 1],
           aspect = 1,
           xlabel = "x",
           ylabel = "y",
           title = "Standard WENO(order=5)\nCourant = $courant_number")

ax2 = Axis(fig[1, 2],
           aspect = 1,
           xlabel = "x",
           ylabel = "y",
           title = "Bounds-preserving WENO(order=5)\nCourant = $courant_number")

hm1 = heatmap!(ax1, x, y, c_standard, colorrange = (cmin, cmax), colormap = :viridis)
hm2 = heatmap!(ax2, x, y, c_bounded, colorrange = (cmin, cmax), colormap = :viridis)

# Add contour at c = 0 to highlight negative values (white line)
contour!(ax1, x, y, c_standard, levels = [0], color = :white, linewidth = 2)
contour!(ax2, x, y, c_bounded, levels = [0], color = :white, linewidth = 2)

Colorbar(fig[1, 3], hm1, label = "Tracer concentration")

# Add text annotations showing min/max values
min_std = @sprintf("%.2e", minimum(c_standard))
min_bnd = @sprintf("%.2e", minimum(c_bounded))
text!(ax1, 0.02, 0.98, text = "min = $min_std", align = (:left, :top), 
      color = :white, fontsize = 12, space = :relative)
text!(ax2, 0.02, 0.98, text = "min = $min_bnd", align = (:left, :top),
      color = :white, fontsize = 12, space = :relative)

save("positivity_preserving_comparison.png", fig)

fig

# ## Minimum value evolution
#
# Plot how the minimum tracer value evolves during the simulation.

fig2 = Figure(size = (600, 400))
ax = Axis(fig2[1, 1],
          xlabel = "Iteration",
          ylabel = "Minimum tracer value",
          title = "Evolution of minimum tracer value during diagonal advection")

lines!(ax, 1:length(min_values_standard), min_values_standard, 
       label = "Standard WENO", linewidth = 2)
lines!(ax, 1:length(min_values_bounded), min_values_bounded,
       label = "Bounds-preserving WENO", linewidth = 2, linestyle = :dash)
hlines!(ax, [0], color = :red, linestyle = :dot, linewidth = 1, label = "Zero line")

axislegend(ax, position = :rb)

save("minimum_value_evolution.png", fig2)

fig2

# ## Summary
#
# The white contour lines in the first figure show where the tracer concentration
# crosses zero. In standard advection, you may see these contours indicating
# spurious negative values. The bounds-preserving scheme should maintain
# positivity (no white contours inside the tracer distribution).
#
# Note: The full positivity guarantee requires the `PositivityPreservingRK3TimeStepper`
# which applies advection dimension-by-dimension. The `WENO(bounds=...)` scheme alone
# limits flux interpolants but doesn't guarantee positivity in multi-dimensional
# advection with combined tendencies. See:
# https://mitgcm.org/sealion/online_documents/node80.html
# https://github.com/CliMA/Oceananigans.jl/pull/3434

