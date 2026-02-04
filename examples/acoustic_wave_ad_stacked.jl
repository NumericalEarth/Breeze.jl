# # Automatic Differentiation through Acoustic Wave Propagation
#
# This example demonstrates computing gradients through a compressible acoustic wave
# simulation using Reactant and Enzyme. We show how the sensitivity ∂L/∂ρ_init 
# evolves as we increase the number of simulation timesteps.
#
# The figure shows:
# - Row 1: Initial density perturbation
# - Rows 2-4: Sensitivity fields at increasing timestep counts, with parameter gradients

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.Grids: xnodes, ynodes
using Oceananigans.TimeSteppers: time_step!
using Breeze
using Reactant
using Reactant: @allowscalar
using Enzyme
using Statistics: mean
using CairoMakie
using CUDA  # Required for ReactantCUDA extension (even for CPU-only)

@info "Package versions" Breeze=pkgversion(Breeze) Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)

Reactant.set_default_backend("cpu")
Reactant.allowscalar(true)

# ============================================================================
# Grid and model setup
# ============================================================================

Nx, Ny = 256, 128
Lx, Ly = 1000.0, 200.0  # meters

@time "Constructing grid" grid = RectilinearGrid(ReactantState();
    size = (Nx, Ny),
    extent = (Lx, Ly),
    halo = (3, 3),
    topology = (Periodic, Periodic, Flat)
)

@time "Constructing model" model = AtmosphereModel(grid; dynamics = CompressibleDynamics())
@time "Creating shadow model" dmodel = Enzyme.make_zero(model)

# ============================================================================
# Physical constants
# ============================================================================

constants = model.thermodynamic_constants
θ₀ = 300.0      # Reference potential temperature (K)
p₀ = 101325.0   # Surface pressure (Pa)

Rᵈ = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ = cᵖᵈ / (cᵖᵈ - Rᵈ)
ρ_ref = p₀ / (Rᵈ * θ₀)
𝕌ˢⁱ = sqrt(γ * Rᵈ * θ₀)
U₀ = 20.0

# Time stepping
Δx, Δy = Lx / Nx, Ly / Ny
𝕌ˢ = 𝕌ˢⁱ + U₀ * 1.5
Δt = 0.5 * min(Δx, Δy) / 𝕌ˢ

# Step counts to compare (proof of concept: 1², 2², 3²)
# Later change to: [12^2, 18^2, 24^2] = [144, 324, 576]
step_bases = [12, 18, 24]
nsteps_list = [n^2 for n in step_bases]

# Coordinate arrays
xc = Array(xnodes(grid, Center()))
yc = Array(ynodes(grid, Center()))
xc_r = Reactant.to_rarray(xc)
yc_r = Reactant.to_rarray(yc)

# Observation point for loss function (middle of top-right quadrant)
i_obs = 3Nx ÷ 4
j_obs = 3Ny ÷ 4

# Initial Gaussian parameters
δρ_val = 0.001      # density perturbation amplitude (kg/m³)
σ_val = 50.0        # width (m)
x₀_val = Lx / 2     # x-position at domain center (m)
y₀_val = Ly / 2     # y-position at domain center (m)

println("=" ^ 70)
println("Acoustic Wave AD: Multi-Timestep Sensitivity Comparison")
println("=" ^ 70)
println()
println("Grid: $Nx × $Ny, Domain: $Lx m × $Ly m")
println("Time step: Δt = $(round(Δt, sigdigits=3)) s")
println("Step counts: $(nsteps_list) (bases: $(step_bases)²)")
println("Observation point: ($i_obs, $j_obs) at x=$(xc[i_obs])m, y=$(yc[j_obs])m")
println("Density perturbation: δρ=$δρ_val kg/m³, σ=$σ_val m")
println()

# ============================================================================
# Create initial density field
# ============================================================================

@time "Creating initial density field" begin
    ρ_init = CenterField(grid)
    set!(ρ_init, (x, y) -> ρ_ref + δρ_val * exp(-((x - x₀_val)^2 + (y - y₀_val)^2) / (2 * σ_val^2)))
end

@time "Creating shadow density field" begin
    dρ_init = CenterField(grid)
    set!(dρ_init, 0.0)
end

# Extract initial perturbation for plotting
ρ_init_array = Array(interior(ρ_init))[:, :, 1]
ρ_perturbation = ρ_init_array .- ρ_ref

# ============================================================================
# Define loss and gradient functions
# ============================================================================

function loss_field(model, ρ_init, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    ρ = model.dynamics.density
    ρθ = model.formulation.potential_temperature_density
    u = model.velocities.u
    
    interior(ρ) .= interior(ρ_init)
    interior(ρθ) .= interior(ρ_init) .* θ₀
    parent(u) .= U₀
    
    @trace track_numbers=false mincut=true checkpointing=true for i in 1:nsteps
        time_step!(model, Δt)
    end
    
    ρ_final = model.dynamics.density
    ρ_obs = @allowscalar ρ_final[i_obs, j_obs, 1]
    return ρ_obs^2
end

function grad_loss_field(model, dmodel, ρ_init, dρ_init, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    parent(dρ_init) .= 0
    
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_field,
        Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(ρ_init, dρ_init),
        Enzyme.Const(θ₀),
        Enzyme.Const(U₀),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps),
        Enzyme.Const(i_obs),
        Enzyme.Const(j_obs)
    )
    
    return dρ_init, loss_value
end

function loss_params(model, params, xc, yc, ρ_ref, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    δρ = @allowscalar params[1]
    σ = @allowscalar params[2]
    x₀ = @allowscalar params[3]
    y₀ = @allowscalar params[4]
    
    ρ = model.dynamics.density
    ρθ = model.formulation.potential_temperature_density
    u = model.velocities.u
    
    X = reshape(xc, :, 1)
    Y = reshape(yc, 1, :)
    r² = (X .- x₀).^2 .+ (Y .- y₀).^2
    gaussian = exp.(-r² ./ (2 * σ^2))
    ρ_vals = ρ_ref .+ δρ .* gaussian
    ρθ_vals = ρ_vals .* θ₀
    
    interior(ρ) .= reshape(ρ_vals, size(interior(ρ)))
    interior(ρθ) .= reshape(ρθ_vals, size(interior(ρθ)))
    parent(u) .= U₀
    
    @trace track_numbers=false mincut=true checkpointing=true for i in 1:nsteps
        time_step!(model, Δt)
    end
    
    ρ_final = model.dynamics.density
    ρ_obs = @allowscalar ρ_final[i_obs, j_obs, 1]
    return ρ_obs^2
end

function grad_loss_params(model, dmodel, params, dparams, xc, yc, ρ_ref, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    dparams .= 0
    
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_params,
        Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(params, dparams),
        Enzyme.Const(xc),
        Enzyme.Const(yc),
        Enzyme.Const(ρ_ref),
        Enzyme.Const(θ₀),
        Enzyme.Const(U₀),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps),
        Enzyme.Const(i_obs),
        Enzyme.Const(j_obs)
    )
    
    return dparams, loss_value
end

# ============================================================================
# Compile for the maximum step count (will work for smaller counts too)
# ============================================================================

max_nsteps = maximum(nsteps_list)

println("Compiling gradient functions for nsteps up to $max_nsteps...")
println()

@time "Compiling grad_loss_field" compiled_field = Reactant.@compile raise_first=true raise=true sync=true grad_loss_field(
    model, dmodel, ρ_init, dρ_init, θ₀, U₀, Δt, max_nsteps, i_obs, j_obs)

params_r = Reactant.to_rarray([δρ_val, σ_val, x₀_val, y₀_val])
dparams_r = Reactant.to_rarray(zeros(4))

@time "Compiling grad_loss_params" compiled_params = Reactant.@compile raise_first=true raise=true sync=true grad_loss_params(
    model, dmodel, params_r, dparams_r, xc_r, yc_r, ρ_ref, θ₀, U₀, Δt, max_nsteps, i_obs, j_obs)

# ============================================================================
# Compute gradients for each step count
# ============================================================================

# Storage for results
results = Dict{Int, NamedTuple}()

for nsteps in nsteps_list
    println("=" ^ 50)
    println("Computing gradients for nsteps = $nsteps")
    println("=" ^ 50)
    
    # Field gradient
    @time "Running grad_loss_field (nsteps=$nsteps)" dρ_result, loss_val = compiled_field(
        model, dmodel, ρ_init, dρ_init, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    
    dρ_array = Array(interior(dρ_result))[:, :, 1]
    
    # Parameter gradients
    @time "Running grad_loss_params (nsteps=$nsteps)" grads_result, _ = compiled_params(
        model, dmodel, params_r, dparams_r, xc_r, yc_r, ρ_ref, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    
    grads_array = Array(grads_result)
    
    results[nsteps] = (
        dρ_array = copy(dρ_array),
        loss = loss_val,
        ∂L_∂δρ = grads_array[1],
        ∂L_∂σ = grads_array[2],
        ∂L_∂x₀ = grads_array[3],
        ∂L_∂y₀ = grads_array[4],
        sim_time = nsteps * Δt
    )
    
    println("  Loss: $(loss_val)")
    println("  ∂L/∂δρ = $(grads_array[1])")
    println("  ∂L/∂σ  = $(grads_array[2])")
    println("  ∂L/∂x₀ = $(grads_array[3])")
    println("  ∂L/∂y₀ = $(grads_array[4])")
    println()
end

# ============================================================================
# Create figure with 4 stacked rows
# ============================================================================

println("Creating visualization...")
println()

aspect_ratio = Lx / Ly
n_sensitivity_plots = length(nsteps_list)

# Figure: taller to accommodate 4 rows + text
fig = Figure(size = (900, 250 + 220 * n_sensitivity_plots), fontsize = 11)

# Supertitle
step_str = join(["$(n)²" for n in step_bases], ", ")
fig[0, :] = Label(fig, "Acoustic Wave AD: Sensitivity Evolution (nsteps = $step_str)", 
                  fontsize = 16, tellwidth = false, font = :bold)

# Row 1: Initial density perturbation
ax_init = Axis(fig[1, 1]; 
    aspect = aspect_ratio,
    ylabel = "y (m)",
    title = "Initial Density Perturbation ρ′(x,y)",
    titlesize = 12)
hidexdecorations!(ax_init)

ρ_lim = δρ_val * 1.1
hm_init = heatmap!(ax_init, xc, yc, ρ_perturbation; 
    colormap = :balance,
    colorrange = (-ρ_lim, ρ_lim))
Colorbar(fig[1, 2], hm_init; label = "ρ′ (kg/m³)", height = Relative(0.85), labelsize = 10)
scatter!(ax_init, [xc[i_obs]], [yc[j_obs]]; color = :red, markersize = 8, marker = :star5)

# Rows 2-4: Sensitivity plots for each step count
for (idx, nsteps) in enumerate(nsteps_list)
    row = idx + 1
    res = results[nsteps]
    
    # Determine if this is the last row (show x-axis label)
    is_last = (idx == n_sensitivity_plots)
    
    # Create axis
    ax = Axis(fig[row, 1]; 
        aspect = aspect_ratio,
        ylabel = "y (m)",
        xlabel = is_last ? "x (m)" : "",
        title = "Sensitivity ∂L/∂ρ₀  (n=$(step_bases[idx])² = $nsteps steps, t=$(round(res.sim_time, digits=3))s)",
        titlesize = 12)
    
    if !is_last
        hidexdecorations!(ax)
    end
    
    # Symmetric colorrange
    grad_max = max(abs(minimum(res.dρ_array)), abs(maximum(res.dρ_array)))
    if grad_max == 0
        grad_max = 1e-10  # Avoid zero range
    end
    
    hm = heatmap!(ax, xc, yc, res.dρ_array; 
        colormap = :balance, 
        colorrange = (-grad_max, grad_max))
    Colorbar(fig[row, 2], hm; label = "∂L/∂ρ", height = Relative(0.85), labelsize = 10)
    
    # Mark observation point
    scatter!(ax, [xc[i_obs]], [yc[j_obs]]; color = :red, markersize = 8, marker = :star5)
    
    # Parameter gradients text below this row
    grad_text = "∂L/∂δρ=$(round(res.∂L_∂δρ, sigdigits=3))  " *
                "∂L/∂σ=$(round(res.∂L_∂σ, sigdigits=3))  " *
                "∂L/∂x₀=$(round(res.∂L_∂x₀, sigdigits=3))  " *
                "∂L/∂y₀=$(round(res.∂L_∂y₀, sigdigits=3))"
    
    # Create a sub-grid for the text below the heatmap
    Label(fig[row, 1, Bottom()], grad_text; 
          fontsize = 9, 
          halign = :center,
          padding = (0, 0, 5, 0))
end

# Adjust spacing
rowgap!(fig.layout, 5)
colgap!(fig.layout, 10)

# Save figure
step_str_filename = join(step_bases, "_")
output_filename = "acoustic_wave_sensitivity_nsteps_$(step_str_filename)sq.png"
save(output_filename, fig; px_per_unit = 2)
println("Figure saved to: $output_filename")
println()

# ============================================================================
# Summary
# ============================================================================

println("=" ^ 70)
println("Summary")
println("=" ^ 70)
println()
println("This demonstration computed ∂L/∂ρ_init at multiple timestep counts:")
println()
for (idx, nsteps) in enumerate(nsteps_list)
    res = results[nsteps]
    println("  n=$(step_bases[idx])² = $nsteps steps (t=$(round(res.sim_time, digits=3))s):")
    println("    Loss = $(res.loss)")
    println("    ∂L/∂δρ = $(res.∂L_∂δρ)")
    println()
end
println("Loss function: L = ρ²(i_obs, j_obs) at observation point ($i_obs, $j_obs)")
println()
