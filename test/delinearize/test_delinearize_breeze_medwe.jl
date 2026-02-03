#=
Investigation: DelinearizingIndexPassing Segmentation Fault (B.6.3)
Status: ACTIVE INVESTIGATION
Purpose: MedWE to reproduce DelinearizingIndexPassing segfault with Breeze model
Related: cursor-toolchain/rules/domains/differentiability/investigations/delinearizing-segfault.md
Synchronized with: Manteia.jl/test/delinearize/ (downstream - awaiting PR merge)

⚠️ PR DEPENDENCY: This is the primary investigation location. Manteia.jl tests
   depend on BreezeReactantExt, so compilation must work here first.

This test uses:
- Breeze AtmosphereModel with CompressibleDynamics (FFT-free)
- Periodic, Periodic, Flat topology (the topology that triggers B.6.3)
- Small 4x4 grid to minimize compilation time

The segfault is associated with:
- ParallelTestRunner
- --check-bounds=yes
- fill_halos! operations on periodic boundaries

Run with: julia --project=test test/delinearize/test_delinearize_breeze_medwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_breeze_medwe.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Breeze
using Reactant
using Enzyme
using Statistics: mean
using CUDA  # Required for ReactantCUDA extension

# Log package versions for debugging
@info "Package versions" Breeze=pkgversion(Breeze) Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
@info "Julia options" check_bounds=Base.JLOptions().check_bounds

Reactant.set_default_backend("cpu")

# Grid: Periodic topology in x,y (this is what triggers B.6.3)
@time "Constructing grid (Periodic, Periodic, Flat)" grid = RectilinearGrid(ReactantState();
    size = (4, 4),
    extent = (1000.0, 1000.0),
    halo = (3, 3),
    topology = (Periodic, Periodic, Flat)
)

# Model: CompressibleDynamics to avoid FFT issues (B.6.1)
@time "Constructing AtmosphereModel (CompressibleDynamics)" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
@time "Creating shadow model (Enzyme.make_zero)" dmodel = Enzyme.make_zero(model)

# Initial condition field
@time "Constructing initial condition field (θ_init)" begin
    θ_init = CenterField(grid)
    set!(θ_init, (x, y) -> 300.0 + 0.01 * x + 0.01 * y)  # Small perturbation
end

# Shadow field for gradient accumulation
@time "Constructing shadow field (dθ_init)" begin
    dθ_init = CenterField(grid)
    set!(dθ_init, 0.0)
end

# Loss function: simple L2 norm of temperature field after time stepping
function loss(model, θ_init, Δt, nsteps)
    set!(model, θ=θ_init, ρ=1.0)
    
    # The @trace loop triggers fill_halos! calls which may cause the segfault
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:nsteps
        time_step!(model, Δt)
    end
    
    return mean(interior(model.temperature).^2)
end

# Gradient function using Enzyme reverse-mode AD
function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    parent(dθ_init) .= 0
    
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps)
    )
    
    return dθ_init, loss_value
end

# Parameters
Δt = 0.01
nsteps = 4  # Multiple steps to exercise fill_halos! repeatedly

# Attempt compilation and execution
try
    @time "Compiling grad_loss (Reactant.@compile)" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
        model, dmodel, θ_init, dθ_init, Δt, nsteps)
    
    @time "Running compiled gradient function" dθ, loss_val = compiled(model, dmodel, θ_init, dθ_init, Δt, nsteps)
    
    @info "Results" loss_value=loss_val gradient_max=maximum(abs, interior(dθ)) any_nan=any(isnan, interior(dθ))
    
    # Basic sanity checks
    if !isnan(loss_val) && loss_val > 0
        @info "✅ SUCCESS: Forward pass produced valid loss"
    else
        @warn "⚠️ ISSUE: Loss is NaN or non-positive"
    end
    
    if maximum(abs, interior(dθ)) > 0 && !any(isnan, interior(dθ))
        @info "✅ SUCCESS: Gradient is non-zero and finite"
    else
        @warn "⚠️ ISSUE: Gradient is zero or contains NaN"
    end
    
catch e
    @error "❌ FAILED" exception=(e, catch_backtrace())
    
    # Check if this is the DelinearizingIndexPassing segfault
    if e isa InterruptException || e isa ProcessExited
        @error "Likely segfault - this may be B.6.3 DelinearizingIndexPassing issue"
    end
    
    rethrow(e)
end
