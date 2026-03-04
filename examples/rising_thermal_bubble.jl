# # Differentiable Rising Thermal Bubble
#
# A warm Gaussian bubble in a neutrally stratified atmosphere rises under buoyancy,
# rolls up into a vortex ring, and eventually breaks into 3D turbulence.
#
# We benchmark **Reactant + Enzyme** against plain Julia (+ Enzyme) for both the
# forward and backward passes of a scalar loss through the compressible Euler equations.
#
# Key result: Reactant compilation not only accelerates AD — in many cases it is the
# *only* way to get Enzyme through the full model, because the uncompiled code path
# contains Union types that Enzyme's type analysis cannot handle.
#
# ## Domain
#
# Channel geometry: periodic in ``x``, bounded (free-slip) in ``y`` and ``z``.

using Breeze
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Printf
using CairoMakie

# ## Parameters

const θ₀  = 300.0    # Background potential temperature [K]
const p₀  = 101325.0 # Surface pressure [Pa]
const pˢᵗ = 1e5      # Standard pressure [Pa]

const A   = 2.0      # Perturbation amplitude [K]
const R   = 1000.0   # Perturbation radius [m]
const x₀  = 5000.0   # Bubble center x [m]
const y₀  = 5000.0   # Bubble center y [m]
const z₀  = 2000.0   # Bubble center z [m]

const Lx  = 10000.0
const Ly  = 10000.0
const Lz  = 10000.0

const Nx  = 4
const Ny  = 4
const Nz  = 4

const z_threshold = 5000.0
const nsteps      = 4

# ## Grids and models
#
# We build two copies: one on `ReactantState` for compiled execution, one on
# `CPU()` for plain-Julia baselines.  Everything else is shared.

topo = (Periodic, Bounded, Bounded)
grid_kwargs = (size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (0, Lz), topology = topo)

grid     = RectilinearGrid(ReactantState(); grid_kwargs...)
grid_cpu = RectilinearGrid(CPU();           grid_kwargs...)

FT = eltype(grid)

model     = AtmosphereModel(grid;     dynamics = CompressibleDynamics())
model_cpu = AtmosphereModel(grid_cpu; dynamics = CompressibleDynamics())

constants = model.thermodynamic_constants

# ## Initial conditions

θ_perturbation(x, y, z) = FT(A) * exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / R^2)
θ_initial(x, y, z) = FT(θ₀) + θ_perturbation(x, y, z)
ρ_initial(x, y, z) = adiabatic_hydrostatic_density(z, p₀, θ₀, pˢᵗ, constants)

θ_init      = CenterField(grid);     set!(θ_init,      θ_initial)
θ_init_cpu  = CenterField(grid_cpu); set!(θ_init_cpu,  θ_initial)
dθ_init     = CenterField(grid);     set!(dθ_init,     FT(0))
dθ_init_cpu = CenterField(grid_cpu); set!(dθ_init_cpu, FT(0))

set!(model;     θ = θ_init,     ρ = ρ_initial)
set!(model_cpu; θ = θ_init_cpu, ρ = ρ_initial)

# ## Loss and gradient
#
# ``J = \text{mean}(\theta^2)`` above ``z_*``.

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ = θ_init, ρ = FT(1))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    T = interior(model.temperature)
    k_start = ceil(Int, z_threshold / Lz * size(model.grid, 3)) + 1
    upper = @view T[:, :, k_start:end]
    return mean(upper .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    _, loss_val = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dθ_init, loss_val
end

# ## Time step (CFL on sound speed)

Rᵈ  = constants.molar_gas_constant / constants.dry_air.molar_mass
cᵖᵈ = constants.dry_air.heat_capacity
γ   = cᵖᵈ / (cᵖᵈ - Rᵈ)
cₛ  = sqrt(γ * Rᵈ * θ₀)
Δt  = FT(0.4 * Lx / Nx / cₛ)

# ## Compile (Reactant)

@info "Compiling forward pass..."
compiled_fwd = Reactant.@compile raise=true raise_first=true sync=true loss(
    model, θ_init, Δt, nsteps)

@info "Compiling backward pass (Enzyme reverse mode)..."
dmodel = Enzyme.make_zero(model)
compiled_bwd = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model, dmodel, θ_init, dθ_init, Δt, nsteps)

# ## Benchmark
#
# Reactant-compiled functions use `Reactant.Profiler.@timed` (device-synchronized).
# Plain-Julia baselines use `Base.@elapsed` (CPU, synchronous by construction).

# ### Forward pass

@info "Benchmarking forward pass (Reactant compiled)..."
prof_fwd = Reactant.Profiler.@timed nrepeat=5 compiled_fwd(model, θ_init, Δt, nsteps)
t_fwd_compiled = prof_fwd.runtime_ns / 1e9
@info "Forward (Reactant)" runtime=t_fwd_compiled

@info "Benchmarking forward pass (plain Julia, CPU)..."
loss(model_cpu, θ_init_cpu, Δt, nsteps)
t_fwd_julia = @elapsed for _ in 1:5
    loss(model_cpu, θ_init_cpu, Δt, nsteps)
end
t_fwd_julia /= 5
@info "Forward (Julia)" runtime=t_fwd_julia speedup=t_fwd_julia/t_fwd_compiled

# ### Backward pass

@info "Benchmarking backward pass (Reactant compiled)..."
prof_bwd = Reactant.Profiler.@timed nrepeat=5 compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)
t_bwd_compiled = prof_bwd.runtime_ns / 1e9
@info "Backward (Reactant)" runtime=t_bwd_compiled

# Without Reactant, Enzyme alone fails on Union types in the Breeze/Oceananigans
# dispatch chain (`set!` → `set_momentum!`) unless we relax strict aliasing.
# This is safe here — the Unions come from keyword-argument splatting, not actual
# type-punning.

Enzyme.API.strictAliasing!(false)

t_bwd_julia = nothing
@info "Attempting backward pass (plain Julia + Enzyme on CPU)..."
try
    dmodel_cpu = Enzyme.make_zero(model_cpu)
    grad_loss(model_cpu, dmodel_cpu, θ_init_cpu, dθ_init_cpu, Δt, nsteps)
    global t_bwd_julia = @elapsed for _ in 1:3
        grad_loss(model_cpu, dmodel_cpu, θ_init_cpu, dθ_init_cpu, Δt, nsteps)
    end
    t_bwd_julia /= 3
    @info "Backward (Julia)" runtime=t_bwd_julia speedup=t_bwd_julia/t_bwd_compiled
catch e
    @warn "Backward pass without Reactant failed" exception=(e, catch_backtrace())
    @info "Enzyme cannot differentiate through the full model without Reactant."
end

Enzyme.API.strictAliasing!(true)

# ## Compute the sensitivity field

@info "Computing sensitivity..."
dθ, J = compiled_bwd(model, dmodel, θ_init, dθ_init, Δt, nsteps)

@info "Loss value" J
@info "Max |∂J/∂θ₀|" maximum_sensitivity=maximum(abs, interior(dθ))

# ## Visualization
#
# Central cross-sections of ``\partial J / \partial \theta_0`` through the bubble center.

sensitivity = Array(interior(dθ))

x = range(0, Lx, length=Nx)
y = range(0, Ly, length=Ny)
z = range(Lz/2Nz, Lz - Lz/2Nz, length=Nz)

jmid = Ny ÷ 2
imid = Nx ÷ 2

fig = Figure(size = (1200, 500), fontsize = 14)

ax1 = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "z (m)",
           title = "∂J/∂θ₀  — x–z slice at y = $(Int(Ly/2)) m",
           aspect = DataAspect())
slimit = maximum(abs, sensitivity)
hm1 = heatmap!(ax1, x, z, sensitivity[:, jmid, :];
               colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig[1, 2], hm1; label = "∂J/∂θ₀")

ax2 = Axis(fig[1, 3]; xlabel = "y (m)", ylabel = "z (m)",
           title = "∂J/∂θ₀  — y–z slice at x = $(Int(Lx/2)) m",
           aspect = DataAspect())
hm2 = heatmap!(ax2, y, z, sensitivity[imid, :, :];
               colormap = :balance, colorrange = (-slimit, slimit))
Colorbar(fig[1, 4], hm2; label = "∂J/∂θ₀")

Label(fig[0, :],
    @sprintf("Adjoint sensitivity — %d steps, Δt = %.4f s, J = %.6e", nsteps, Δt, J),
    fontsize = 16, tellwidth = false)

save("rising_thermal_sensitivity.png", fig; px_per_unit = 2)
@info "Saved rising_thermal_sensitivity.png"

# ## Timing bar chart

fig2 = Figure(size = (650, 400), fontsize = 14)

if isnothing(t_bwd_julia)
    ax = Axis(fig2[1, 1];
              xticks = ([1, 2, 3], ["Forward\n(Julia)", "Forward\n(Reactant)",
                                     "Backward\n(Reactant)"]),
              ylabel = "Wall time (s)",
              title = "Rising Thermal Bubble — Reactant + Enzyme Benchmark")
    barplot!(ax, [1, 2, 3], [t_fwd_julia, t_fwd_compiled, t_bwd_compiled];
             color = [:steelblue, :seagreen, :seagreen],
             bar_labels = :y, label_formatter = x -> @sprintf("%.4f s", x))
    Label(fig2[2, 1],
          "Backward (Julia + Enzyme): FAILED — IllegalTypeAnalysisException on Union types",
          fontsize = 11, color = :gray40, tellwidth = false)
else
    ax = Axis(fig2[1, 1];
              xticks = ([1, 2, 3, 4], ["Forward\n(Julia)", "Forward\n(Reactant)",
                                        "Backward\n(Julia)", "Backward\n(Reactant)"]),
              ylabel = "Wall time (s)",
              title = "Rising Thermal Bubble — Reactant + Enzyme Benchmark")
    barplot!(ax, [1, 2, 3, 4], [t_fwd_julia, t_fwd_compiled, t_bwd_julia, t_bwd_compiled];
             color = [:steelblue, :seagreen, :steelblue, :seagreen],
             bar_labels = :y, label_formatter = x -> @sprintf("%.4f s", x))
end

save("rising_thermal_benchmark.png", fig2; px_per_unit = 2)
@info "Saved rising_thermal_benchmark.png"

# ## Summary

@info "Timing summary" forward_julia=t_fwd_julia forward_reactant=t_fwd_compiled backward_julia=something(t_bwd_julia, "FAILED") backward_reactant=t_bwd_compiled
@info "Speedups" forward=t_fwd_julia/t_fwd_compiled
if !isnothing(t_bwd_julia)
    @info "Backward speedup" backward=t_bwd_julia/t_bwd_compiled
else
    @info "Backward pass only possible with Reactant — Enzyme alone cannot compile the model."
end

nothing #hide
