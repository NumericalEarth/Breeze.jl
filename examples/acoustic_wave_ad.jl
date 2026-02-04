# # Automatic Differentiation through Acoustic Wave Propagation
#
# This example demonstrates computing gradients through a compressible acoustic wave
# simulation using Reactant and Enzyme in two pedagogical steps:
#
# **Part 1**: Compute ∂L/∂ρ_init - the gradient of the loss w.r.t. the initial density field
#             This shows how the final density depends on each point in the initial condition.
#
# **Part 2**: Compute ∂L/∂params - the gradient w.r.t. the Gaussian parameters (δρ, σ, x₀, y₀)
#             This shows how the final density depends on the shape/position of the perturbation.

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
# Grid and model setup (small grid for AD demonstration)
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
nsteps = 24*24

# Coordinate arrays
xc = Array(xnodes(grid, Center()))
yc = Array(ynodes(grid, Center()))
xc_r = Reactant.to_rarray(xc)
yc_r = Reactant.to_rarray(yc)

# Observation point for loss function (middle of top-right quadrant)
# Top-right quadrant: x ∈ [Lx/2, Lx], y ∈ [Ly/2, Ly]
# Middle of that quadrant: x = 3Lx/4, y = 3Ly/4
i_obs = 3Nx ÷ 4     # 3/4 of the way in x (middle of right half)
j_obs = 3Ny ÷ 4     # 3/4 of the way in y (middle of top half)

# Initial Gaussian parameters - center at middle of domain
δρ_val = 0.001      # density perturbation amplitude (kg/m³) - small for linear acoustics
σ_val = 50.0        # width (m)
x₀_val = Lx / 2     # x-position at domain center (m)
y₀_val = Ly / 2     # y-position at domain center (m)

println("=" ^ 70)
println("Acoustic Wave AD Demonstration (Density Perturbation)")
println("=" ^ 70)
println()
println("Grid: $Nx × $Ny, Domain: $Lx m × $Ly m")
println("Time step: $(round(Δt, sigdigits=3)) s, Steps: $nsteps")
println("Observation point: ($i_obs, $j_obs) at x=$(xc[i_obs])m, y=$(yc[j_obs])m")
println("Density perturbation: δρ=$δρ_val kg/m³, σ=$σ_val m, x₀=$x₀_val m, y₀=$y₀_val m")
println()

# ============================================================================
# PART 1: Gradient w.r.t. Initial Density Field
# ============================================================================

println("=" ^ 70)
println("PART 1: Gradient w.r.t. Initial Density Field (∂L/∂ρ_init)")
println("=" ^ 70)
println()

# Create initial density field with Gaussian perturbation
@time "Creating initial density field" begin
    ρ_init = CenterField(grid)
    set!(ρ_init, (x, y) -> ρ_ref + δρ_val * exp(-((x - x₀_val)^2 + (y - y₀_val)^2) / (2 * σ_val^2)))
end

@time "Creating shadow density field" begin
    dρ_init = CenterField(grid)
    set!(dρ_init, 0.0)
end

# Loss function for Part 1: takes the initial density field directly
# Returns squared density at observation point (i_obs, j_obs)
function loss_field(model, ρ_init, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    ρ = model.dynamics.density
    ρθ = model.formulation.potential_temperature_density
    u = model.velocities.u
    
    # Copy initial density
    interior(ρ) .= interior(ρ_init)
    interior(ρθ) .= interior(ρ_init) .* θ₀
    
    # No velocity perturbation - just background
    parent(u) .= U₀
    
    # Time-stepping
    @trace track_numbers=false mincut=true checkpointing=true for i in 1:nsteps
        time_step!(model, Δt)
    end
    
    # Loss: squared density at observation point
    ρ_final = model.dynamics.density
    ρ_obs = @allowscalar ρ_final[i_obs, j_obs, 1]
    return ρ_obs^2
end

# Gradient function for Part 1
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

# Compile and run Part 1
@time "Compiling grad_loss_field" compiled_field = Reactant.@compile raise_first=true raise=true sync=true grad_loss_field(
    model, dmodel, ρ_init, dρ_init, θ₀, U₀, Δt, nsteps, i_obs, j_obs)

@time "Running grad_loss_field" dρ_result, loss_val_1 = compiled_field(
    model, dmodel, ρ_init, dρ_init, θ₀, U₀, Δt, nsteps, i_obs, j_obs)

# Extract results
dρ_array = Array(interior(dρ_result))[:, :, 1]
ρ_init_array = Array(interior(ρ_init))[:, :, 1]
ρ_perturbation = ρ_init_array .- ρ_ref  # Perturbation from background

println()
println("Loss value: $loss_val_1")
println("Gradient ∂L/∂ρ_init statistics:")
println("  Max: $(maximum(dρ_array))")
println("  Min: $(minimum(dρ_array))")
println("  Norm: $(sqrt(sum(dρ_array.^2)))")
println()

# ============================================================================
# Visualize Part 1: Initial condition and its gradient
# (Figure will be completed and saved after Part 2 with parameter gradients)
# ============================================================================

# Vertical stack layout like acoustic_wave.jl (density on top, sensitivity below)
aspect_ratio = Lx / Ly
fig = Figure(size = (800, 500), fontsize = 12)

# Supertitle (include number of timesteps)
fig[0, :] = Label(fig, "Acoustic Wave AD: Density Perturbation and Sensitivity (nsteps=$nsteps)", 
                  fontsize = 16, tellwidth = false)

# Top panel: Initial density perturbation (ρ - ρ_ref)
ax1 = Axis(fig[1, 1]; 
    aspect = aspect_ratio,
    ylabel = "y (m)",
    title = "Initial Density Perturbation  ρ′(x,y)")

hidexdecorations!(ax1)

# Use symmetric colorrange centered at zero
ρ_lim = δρ_val / 2
hm1 = heatmap!(ax1, xc, yc, ρ_perturbation; 
    colormap = :balance,
    colorrange = (-ρ_lim, ρ_lim))
Colorbar(fig[1, 2], hm1; label = "ρ′ (kg/m³)", height = Relative(0.8))

# Mark observation point on top panel
scatter!(ax1, [xc[i_obs]], [yc[j_obs]]; color = :red, markersize = 10, marker = :star5)

# Bottom panel: Gradient of loss w.r.t. initial density
ax2 = Axis(fig[2, 1]; 
    aspect = aspect_ratio,
    xlabel = "x (m)", 
    ylabel = "y (m)",
    title = "Sensitivity  ∂L/∂ρ₀")

# Colorrange defined by actual min and max of gradient (symmetric around zero)
grad_max_abs = max(abs(minimum(dρ_array)), abs(maximum(dρ_array)))
hm2 = heatmap!(ax2, xc, yc, dρ_array; 
    colormap = :balance, 
    colorrange = (-grad_max_abs, grad_max_abs))
Colorbar(fig[2, 2], hm2; label = "∂L/∂ρ", height = Relative(0.8))

# Mark observation point on bottom panel
scatter!(ax2, [xc[i_obs]], [yc[j_obs]]; color = :red, markersize = 10, marker = :star5)

# ============================================================================
# PART 2: Gradient w.r.t. Gaussian Parameters
# ============================================================================

println("=" ^ 70)
println("PART 2: Gradient w.r.t. Parameters (∂L/∂δρ, ∂L/∂σ, ∂L/∂x₀, ∂L/∂y₀)")
println("=" ^ 70)
println()

# Pack parameters into array for Duplicated
params_r = Reactant.to_rarray([δρ_val, σ_val, x₀_val, y₀_val])
dparams_r = Reactant.to_rarray(zeros(4))

# Loss function for Part 2: constructs initial condition from parameters
# Returns squared density at observation point (i_obs, j_obs)
function loss_params(model, params, xc, yc, ρ_ref, θ₀, U₀, Δt, nsteps, i_obs, j_obs)
    # Unpack parameters
    δρ = @allowscalar params[1]
    σ = @allowscalar params[2]
    x₀ = @allowscalar params[3]
    y₀ = @allowscalar params[4]
    
    ρ = model.dynamics.density
    ρθ = model.formulation.potential_temperature_density
    u = model.velocities.u
    
    # Construct Gaussian density perturbation using broadcasting
    X = reshape(xc, :, 1)
    Y = reshape(yc, 1, :)
    r² = (X .- x₀).^2 .+ (Y .- y₀).^2
    gaussian = exp.(-r² ./ (2 * σ^2))
    ρ_vals = ρ_ref .+ δρ .* gaussian
    ρθ_vals = ρ_vals .* θ₀
    
    interior(ρ) .= reshape(ρ_vals, size(interior(ρ)))
    interior(ρθ) .= reshape(ρθ_vals, size(interior(ρθ)))
    
    # No velocity perturbation - just background
    parent(u) .= U₀
    
    # Time-stepping
    @trace track_numbers=false mincut=true checkpointing=true for i in 1:nsteps
        time_step!(model, Δt)
    end
    
    # Loss: squared density at observation point
    ρ_final = model.dynamics.density
    ρ_obs = @allowscalar ρ_final[i_obs, j_obs, 1]
    return ρ_obs^2
end

# Gradient function for Part 2
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

# Compile and run Part 2
@time "Compiling grad_loss_params" compiled_params = Reactant.@compile raise_first=true raise=true sync=true grad_loss_params(
    model, dmodel, params_r, dparams_r, xc_r, yc_r, ρ_ref, θ₀, U₀, Δt, nsteps, i_obs, j_obs)

@time "Running grad_loss_params" grads_result, loss_val_2 = compiled_params(
    model, dmodel, params_r, dparams_r, xc_r, yc_r, ρ_ref, θ₀, U₀, Δt, nsteps, i_obs, j_obs)

# Extract gradients
grads_array = Array(grads_result)
∂L_∂δρ = grads_array[1]
∂L_∂σ = grads_array[2]
∂L_∂x₀ = grads_array[3]
∂L_∂y₀ = grads_array[4]

println()
println("Loss value: $loss_val_2")
println()
println("-" ^ 50)
println("PARAMETER GRADIENTS")
println("-" ^ 50)
println()
println("  ∂L/∂δρ (amplitude)  = $∂L_∂δρ")
println("  ∂L/∂σ  (width)      = $∂L_∂σ")
println("  ∂L/∂x₀ (x-position) = $∂L_∂x₀")
println("  ∂L/∂y₀ (y-position) = $∂L_∂y₀")
println()
println("Position gradient: ∇_{(x₀,y₀)} L = ($∂L_∂x₀, $∂L_∂y₀)")
println()

# Add parameter gradients to the figure (row 3, below the two heatmaps)
grad_text = "Parameter Gradients:  ∂L/∂δρ = $(round(∂L_∂δρ, sigdigits=4)),  ∂L/∂σ = $(round(∂L_∂σ, sigdigits=4)),  ∂L/∂x₀ = $(round(∂L_∂x₀, sigdigits=4)),  ∂L/∂y₀ = $(round(∂L_∂y₀, sigdigits=4))"

Label(fig[3, 1:2], grad_text; fontsize = 11, tellwidth = false)

# Save the completed figure (include nsteps in filename)
output_filename = "acoustic_wave_gradient_field_nsteps$(nsteps).png"
save(output_filename, fig; px_per_unit = 2)
println("Figure saved to: $output_filename")
println()

# ============================================================================
# PART 3: Chain Rule Verification (Pedagogical)
# ============================================================================

println("=" ^ 70)
println("PART 3: Chain Rule Verification")
println("=" ^ 70)
println()
println("The chain rule connects Parts 1 and 2:")
println()
println("  ∂L/∂param = Σᵢⱼ (∂L/∂ρ_init[i,j]) × (∂ρ_init[i,j]/∂param)")
println()
println("For a Gaussian: ρ_init(x,y) = ρ_ref + δρ × exp(-r²/(2σ²))")
println("where r² = (x-x₀)² + (y-y₀)²")
println()

# Compute ∂ρ_init/∂params analytically
X = reshape(xc, :, 1)
Y = reshape(yc, 1, :)
r² = (X .- x₀_val).^2 .+ (Y .- y₀_val).^2
gaussian = exp.(-r² ./ (2 * σ_val^2))

# Analytical derivatives of ρ_init w.r.t. parameters
∂ρ_∂δρ = gaussian                                           # = exp(-r²/(2σ²))
∂ρ_∂σ = δρ_val .* gaussian .* r² ./ σ_val^3                # = δρ × g × r²/σ³
∂ρ_∂x₀ = δρ_val .* gaussian .* (X .- x₀_val) ./ σ_val^2    # = δρ × g × (x-x₀)/σ²
∂ρ_∂y₀ = δρ_val .* gaussian .* (Y .- y₀_val) ./ σ_val^2    # = δρ × g × (y-y₀)/σ²

# Chain rule: ∂L/∂param = sum(∂L/∂ρ × ∂ρ/∂param)
∂L_∂δρ_chain = sum(dρ_array .* ∂ρ_∂δρ)
∂L_∂σ_chain = sum(dρ_array .* ∂ρ_∂σ)
∂L_∂x₀_chain = sum(dρ_array .* ∂ρ_∂x₀)
∂L_∂y₀_chain = sum(dρ_array .* ∂ρ_∂y₀)

println("Chain rule verification (AD vs analytical chain rule):")
println()
println("  ∂L/∂δρ: AD = $∂L_∂δρ, Chain = $∂L_∂δρ_chain")
println("  ∂L/∂σ:  AD = $∂L_∂σ,  Chain = $∂L_∂σ_chain")
println("  ∂L/∂x₀: AD = $∂L_∂x₀, Chain = $∂L_∂x₀_chain")
println("  ∂L/∂y₀: AD = $∂L_∂y₀, Chain = $∂L_∂y₀_chain")
println()

# ============================================================================
# Summary
# ============================================================================

println("=" ^ 70)
println("Summary")
println("=" ^ 70)
println()
println("Loss function: L = ρ²(i_obs, j_obs) = squared density at observation point")
println("Observation point: ($i_obs, $j_obs)")
println()
println("This demonstration showed:")
println()
println("1. PART 1: Computing ∂L/∂ρ_init - the sensitivity of the loss")
println("   to each grid point in the initial density field.")
println("   → Visualized in: $output_filename")
println()
println("2. PART 2: Computing ∂L/∂params - the sensitivity to the")
println("   Gaussian parameters (amplitude, width, position).")
println()
println("3. PART 3: The chain rule connects them:")
println("   ∂L/∂param = Σ (∂L/∂ρ_init) × (∂ρ_init/∂param)")
println()
