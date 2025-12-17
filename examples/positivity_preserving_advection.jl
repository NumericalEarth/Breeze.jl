# # Positivity-preserving advection of a Gaussian tracer
#
# This example demonstrates positivity-preserving advection using directionally-split
# integration, following the algorithm described by the MITgcm documentation:
# https://mitgcm.org/sealion/online_documents/node80.html
#
# We advect a Gaussian tracer distribution diagonally across a periodic domain
# and compare standard advection (which can produce spurious negative values)
# with positivity-preserving advection using `PositivityPreservingRK3TimeStepper`.
#
# This is a classic test case for advection schemes, used in many papers and
# model validation studies.

using Breeze
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: time_step!
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
                       size = (Nx, Ny, 5),
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
# First, we run with standard RK3 time stepping and WENO advection
# to see if spurious negative values appear.

advection = WENO(order=5, bounds=(0, 1))

model_standard = AtmosphereModel(grid;
                                 tracers = :c,
                                 advection,
                                 microphysics = nothing,
                                 closure = nothing)

# Set initial tracer distribution and uniform velocity
set!(model_standard, c = gaussian_tracer, u = u_velocity, v = v_velocity)

# ## Positivity-preserving advection with directional splitting
#
# Now we set up the same problem but with `PositivityPreservingRK3TimeStepper`
# which applies advection dimension-by-dimension to maintain positivity.
#
# **Key**: The advection scheme is passed to the TIME STEPPER via `split_advection`,
# NOT to the model. The model uses `scalar_advection = (; c = nothing)` for the tracer
# to avoid double-counting the advection tendency.

# First create a temporary model to get the prognostic fields layout
temp_model = AtmosphereModel(grid; tracers=:c, microphysics=nothing, closure=nothing)
pf = prognostic_fields(temp_model)

# The advection scheme is stored in the time stepper via `split_advection`
pp_timestepper = PositivityPreservingRK3TimeStepper(grid, pf;
                                                    split_advection = (; c = advection))

# The model uses scalar_advection = (; c = nothing) so normal tendency computation
# skips advection for tracer c. Split advection is handled by the time stepper.
model_pp = AtmosphereModel(grid;
                           tracers = :c,
                           momentum_advection = advection,
                           scalar_advection = (; c = nothing),  # No normal advection for c
                           microphysics = nothing,
                           closure = nothing,
                           timestepper = pp_timestepper)

set!(model_pp, c = gaussian_tracer, u = u_velocity, v = v_velocity)

# ## Diagnostic tests
#
# Before running the full simulation, let's verify the setup is correct.

# Check that advection is configured correctly
@info "Checking advection configuration..."
@info "  model_standard.advection.c = $(model_standard.advection.c)"
@info "  model_pp.advection.c = $(model_pp.advection.c)"  # Should be `nothing`
@info "  model_pp.timestepper.split_advection.c = $(model_pp.timestepper.split_advection.c)"

# Check initial state
@info "Initial state:"
@info "  Standard model: min(c) = $(minimum(model_standard.tracers.c)), max(c) = $(maximum(model_standard.tracers.c))"
@info "  PP model: min(c) = $(minimum(model_pp.tracers.c)), max(c) = $(maximum(model_pp.tracers.c))"

# Run a single time step test to compare behavior
Δt = 1.0  # seconds

# Store initial values
c_standard_init = copy(interior(model_standard.tracers.c))
c_pp_init = copy(interior(model_pp.tracers.c))

@info "\nRunning single time step test..."
time_step!(model_standard, Δt)
time_step!(model_pp, Δt)

@info "After 1 time step (Δt = $Δt):"
@info "  Standard: min(c) = $(minimum(model_standard.tracers.c)), max(c) = $(maximum(model_standard.tracers.c))"
@info "  PP model: min(c) = $(minimum(model_pp.tracers.c)), max(c) = $(maximum(model_pp.tracers.c))"

# Compare the change
Δc_standard = interior(model_standard.tracers.c) .- c_standard_init
Δc_pp = interior(model_pp.tracers.c) .- c_pp_init

@info "Change in tracer field:"
@info "  Standard: Δc range = [$(minimum(Δc_standard)), $(maximum(Δc_standard))]"
@info "  PP model: Δc range = [$(minimum(Δc_pp)), $(maximum(Δc_pp))]"
@info "  Ratio of max changes: $(maximum(abs.(Δc_pp)) / maximum(abs.(Δc_standard)))"

# Reset models for full simulation
set!(model_standard, c = gaussian_tracer, u = u_velocity, v = v_velocity)
set!(model_pp, c = gaussian_tracer, u = u_velocity, v = v_velocity)

# ## Run simulations
#
# We run both simulations for the same time period (one complete traversal
# of the domain, T = 0.5 which corresponds to diagonal transport to the
# opposite corner).

stop_time = Lx / velocity_magnitude  # time for one domain traversal

simulation_standard = Simulation(model_standard; Δt, stop_time)
simulation_pp = Simulation(model_pp; Δt, stop_time)

# Track minimum and maximum values during the simulation
min_values_standard = Float64[]
min_values_pp = Float64[]
max_values_standard = Float64[]
max_values_pp = Float64[]

function track_extrema_standard!(sim)
    c = sim.model.tracers.c
    push!(min_values_standard, minimum(c))
    push!(max_values_standard, maximum(c))
    return nothing
end

function track_extrema_pp!(sim)
    c = sim.model.tracers.c
    push!(min_values_pp, minimum(c))
    push!(max_values_pp, maximum(c))
    return nothing
end

add_callback!(simulation_standard, track_extrema_standard!, IterationInterval(1))
add_callback!(simulation_pp, track_extrema_pp!, IterationInterval(1))

@info "Running standard RK3 advection..."
run!(simulation_standard)

@info "Running positivity-preserving RK3 advection..."
run!(simulation_pp)

# ## Results
#
# Check for negative values

@info "Standard RK3: min(c) = $(minimum(model_standard.tracers.c)), max(c) = $(maximum(model_standard.tracers.c))"
@info "Positivity-preserving RK3: min(c) = $(minimum(model_pp.tracers.c)), max(c) = $(maximum(model_pp.tracers.c))"

# ## Visualization
#
# Create a comparison plot similar to the MITgcm documentation figures.

fig = Figure(size = (1000, 500), fontsize = 14)

# Extract tracer data for plotting
c_standard = interior(model_standard.tracers.c, :, :, 1)
c_pp = interior(model_pp.tracers.c, :, :, 1)

x = xnodes(grid, Center())
y = ynodes(grid, Center())

# Common colormap limits (include potential negative values for visualization)
cmin = min(minimum(c_standard), minimum(c_pp), 0)
cmax = max(maximum(c_standard), maximum(c_pp))

ax1 = Axis(fig[1, 1],
           aspect = 1,
           xlabel = "x",
           ylabel = "y",
           title = "Standard RK3 + WENO(bounds)\nCourant = $courant_number")

ax2 = Axis(fig[1, 2],
           aspect = 1,
           xlabel = "x",
           ylabel = "y",
           title = "Positivity-preserving RK3\nCourant = $courant_number")

hm1 = heatmap!(ax1, x, y, c_standard, colorrange = (cmin, cmax), colormap = :viridis)
hm2 = heatmap!(ax2, x, y, c_pp, colorrange = (cmin, cmax), colormap = :viridis)

# Add contour at c = 0 to highlight negative values (white line)
contour!(ax1, x, y, c_standard, levels = [0], color = :white, linewidth = 2)
contour!(ax2, x, y, c_pp, levels = [0], color = :white, linewidth = 2)

Colorbar(fig[1, 3], hm1, label = "Tracer concentration")

# Add text annotations showing min/max values
min_std = @sprintf("%.2e", minimum(c_standard))
min_pp_val = @sprintf("%.2e", minimum(c_pp))
text!(ax1, 0.02, 0.98, text = "min = $min_std", align = (:left, :top), 
      color = :white, fontsize = 12, space = :relative)
text!(ax2, 0.02, 0.98, text = "min = $min_pp_val", align = (:left, :top),
      color = :white, fontsize = 12, space = :relative)

save("positivity_preserving_comparison.png", fig)

fig

# ## Extrema evolution
#
# Plot how the minimum and maximum tracer values evolve during the simulation.

fig2 = Figure(size = (900, 400))

ax_min = Axis(fig2[1, 1],
              xlabel = "Iteration",
              ylabel = "Minimum tracer value",
              title = "Minimum tracer value")

ax_max = Axis(fig2[1, 2],
              xlabel = "Iteration",
              ylabel = "Maximum tracer value",
              title = "Maximum tracer value")

lines!(ax_min, 1:length(min_values_standard), min_values_standard, 
       label = "Standard RK3", linewidth = 2)
lines!(ax_min, 1:length(min_values_pp), min_values_pp,
       label = "PP RK3", linewidth = 2, linestyle = :dash)
hlines!(ax_min, [0], color = :red, linestyle = :dot, linewidth = 1, label = "Zero")

lines!(ax_max, 1:length(max_values_standard), max_values_standard, 
       label = "Standard RK3", linewidth = 2)
lines!(ax_max, 1:length(max_values_pp), max_values_pp,
       label = "PP RK3", linewidth = 2, linestyle = :dash)

axislegend(ax_min, position = :rb)
axislegend(ax_max, position = :rb)

@info "Final maximum values:"
@info "  Standard: $(max_values_standard[end])"
@info "  PP: $(max_values_pp[end])"

save("minimum_value_evolution.png", fig2)

fig2

# ## Summary
#
# The white contour lines in the first figure show where the tracer concentration
# crosses zero. In standard advection, you may see these contours indicating
# spurious negative values. The positivity-preserving scheme with directional
# splitting should maintain positivity (no white contours inside the tracer
# distribution).
#
# The `PositivityPreservingRK3TimeStepper` applies advection dimension-by-dimension
# following the MITgcm algorithm:
# https://mitgcm.org/sealion/online_documents/node80.html
#
# See also: https://github.com/CliMA/Oceananigans.jl/pull/3434
