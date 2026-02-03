#=
Investigation: DelinearizingIndexPassing Segmentation Fault (B.6.3)
Status: ACTIVE INVESTIGATION
Purpose: MedWE to reproduce DelinearizingIndexPassing segfault with pure Oceananigans model
Related: cursor-toolchain/rules/domains/differentiability/investigations/delinearizing-segfault.md

This test uses pure Oceananigans (no Breeze) to help isolate whether the issue is:
- Breeze-specific (only Breeze test fails)
- Oceananigans-level (both tests fail)
- Reactant/Enzyme-level (need further MWE simplification)

This test uses:
- Oceananigans HydrostaticFreeSurfaceModel with ExplicitFreeSurface (FFT-free)
- Periodic, Periodic, Bounded topology (Periodic in x,y like Breeze test)
- Small 4x4x2 grid to minimize compilation time

The segfault is associated with:
- ParallelTestRunner
- --check-bounds=yes
- fill_halos! operations on periodic boundaries

Run with: julia --project=test test/delinearize/test_delinearize_oceananigans_medwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_oceananigans_medwe.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface
using Reactant
using Enzyme
using Statistics: mean
using CUDA  # Required for ReactantCUDA extension

# Log package versions for debugging
@info "Package versions" Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
@info "Julia options" check_bounds=Base.JLOptions().check_bounds

Reactant.set_default_backend("cpu")

# Grid: Periodic topology in x,y, Bounded in z (standard ocean setup)
# The periodic boundaries are what trigger B.6.3
@time "Constructing grid (Periodic, Periodic, Bounded)" grid = RectilinearGrid(ReactantState();
    size = (4, 4, 2),
    extent = (1000.0, 1000.0, 100.0),
    halo = (3, 3, 3),
    topology = (Periodic, Periodic, Bounded)
)

# Model: HydrostaticFreeSurfaceModel with ExplicitFreeSurface to avoid FFT issues (B.6.1)
@time "Constructing HydrostaticFreeSurfaceModel (ExplicitFreeSurface)" model = HydrostaticFreeSurfaceModel(
    grid;
    free_surface = ExplicitFreeSurface(),
    buoyancy = nothing,
    tracers = :T,  # Single tracer for simplicity
    closure = nothing
)
@time "Creating shadow model (Enzyme.make_zero)" dmodel = Enzyme.make_zero(model)

# Initial condition field for tracer T
@time "Constructing initial condition field (T_init)" begin
    T_init = CenterField(grid)
    set!(T_init, (x, y, z) -> 20.0 + 0.01 * x + 0.01 * y)  # Small perturbation
end

# Shadow field for gradient accumulation
@time "Constructing shadow field (dT_init)" begin
    dT_init = CenterField(grid)
    set!(dT_init, 0.0)
end

# Loss function: simple L2 norm of tracer field after time stepping
function loss(model, T_init, Δt, nsteps)
    set!(model, T=T_init)
    
    # The @trace loop triggers fill_halos! calls which may cause the segfault
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:nsteps
        time_step!(model, Δt)
    end
    
    return mean(interior(model.tracers.T).^2)
end

# Gradient function using Enzyme reverse-mode AD
function grad_loss(model, dmodel, T_init, dT_init, Δt, nsteps)
    parent(dT_init) .= 0
    
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T_init, dT_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps)
    )
    
    return dT_init, loss_value
end

# Parameters
Δt = 60.0   # 1 minute timestep (typical for ocean models)
nsteps = 4  # Multiple steps to exercise fill_halos! repeatedly

# Attempt compilation and execution
try
    @time "Compiling grad_loss (Reactant.@compile)" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
        model, dmodel, T_init, dT_init, Δt, nsteps)
    
    @time "Running compiled gradient function" dT, loss_val = compiled(model, dmodel, T_init, dT_init, Δt, nsteps)
    
    @info "Results" loss_value=loss_val gradient_max=maximum(abs, interior(dT)) any_nan=any(isnan, interior(dT))
    
    # Basic sanity checks
    if !isnan(loss_val) && loss_val > 0
        @info "✅ SUCCESS: Forward pass produced valid loss"
    else
        @warn "⚠️ ISSUE: Loss is NaN or non-positive"
    end
    
    if maximum(abs, interior(dT)) > 0 && !any(isnan, interior(dT))
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
