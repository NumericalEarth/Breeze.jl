#=
Investigation: DelinearizingIndexPassing Segmentation Fault (B.6.3)
Status: ACTIVE INVESTIGATION
Purpose: Isolate which component of time_step! triggers the segfault

Since fill_halo_regions! PASSES, this test breaks down time_step! into components:
1. set!(model, ...) - field initialization
2. compute_tendencies! - stencil operations
3. update_state! - state management
4. Individual RK substeps

Run with: julia --project=test test/delinearize/test_delinearize_timestep_components_medwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_timestep_components_medwe.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface
using Reactant
using Enzyme
using Statistics: mean
using CUDA

# Log package versions and options
@info "Package versions" Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
@info "Julia options" check_bounds=Base.JLOptions().check_bounds

Reactant.set_default_backend("cpu")

#####
##### Setup: Create model once
#####

@info "=" ^ 60
@info "Setting up model..."
@info "=" ^ 60

@time "Constructing grid" grid = RectilinearGrid(ReactantState();
    size = (4, 4, 2),
    extent = (1.0, 1.0, 0.1),
    halo = (3, 3, 3),
    topology = (Periodic, Periodic, Bounded)
)

@time "Constructing HydrostaticFreeSurfaceModel" model = HydrostaticFreeSurfaceModel(
    grid;
    free_surface = ExplicitFreeSurface(),
    buoyancy = nothing,
    tracers = :T,
    closure = nothing
)

@time "Creating shadow model" dmodel = Enzyme.make_zero(model)

#####
##### Test 1: Just set!(model, ...) 
#####

@info "=" ^ 60
@info "Test 1: set!(model, T=...) only"
@info "=" ^ 60

@time "Constructing T_init field" begin
    T_init = CenterField(grid)
    set!(T_init, (x, y, z) -> 20.0 + x + y)
end

@time "Constructing dT_init field" begin
    dT_init = CenterField(grid)
    set!(dT_init, 0.0)
end

function loss_set_only(model, T_init)
    set!(model, T=T_init)
    return mean(interior(model.tracers.T).^2)
end

function grad_set_only(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_set_only, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T_init, dT_init)
    )
    return dT_init, loss_value
end

@info "Testing autodiff through set!..."
try
    @time "Compiling grad_set_only" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_only(
        model, dmodel, T_init, dT_init)
    @time "Running grad_set_only" dT, loss_val = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 1 PASSED: set! works" loss_value=loss_val
catch e
    @error "❌ Test 1 FAILED: set!" exception=(e, catch_backtrace())
end

#####
##### Test 2: set! + fill_halo_regions!
#####

@info "=" ^ 60
@info "Test 2: set! + fill_halo_regions!"
@info "=" ^ 60

function loss_set_and_halos(model, T_init)
    set!(model, T=T_init)
    fill_halo_regions!(model.tracers.T)
    return mean(interior(model.tracers.T).^2)
end

function grad_set_and_halos(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_set_and_halos, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T_init, dT_init)
    )
    return dT_init, loss_value
end

@info "Testing autodiff through set! + fill_halos!..."
try
    @time "Compiling grad_set_and_halos" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_and_halos(
        model, dmodel, T_init, dT_init)
    @time "Running grad_set_and_halos" dT, loss_val = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 2 PASSED: set! + fill_halos! works" loss_value=loss_val
catch e
    @error "❌ Test 2 FAILED: set! + fill_halos!" exception=(e, catch_backtrace())
end

#####
##### Test 3: set! + update_state!
#####

@info "=" ^ 60
@info "Test 3: set! + update_state!"
@info "=" ^ 60

function loss_set_and_update(model, T_init)
    set!(model, T=T_init)
    update_state!(model)
    return mean(interior(model.tracers.T).^2)
end

function grad_set_and_update(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_set_and_update, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T_init, dT_init)
    )
    return dT_init, loss_value
end

@info "Testing autodiff through set! + update_state!..."
try
    @time "Compiling grad_set_and_update" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_and_update(
        model, dmodel, T_init, dT_init)
    @time "Running grad_set_and_update" dT, loss_val = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 3 PASSED: set! + update_state! works" loss_value=loss_val
catch e
    @error "❌ Test 3 FAILED: set! + update_state!" exception=(e, catch_backtrace())
end

#####
##### Test 4: Single time_step! (nsteps=1)
#####

@info "=" ^ 60
@info "Test 4: Single time_step! (nsteps=1)"
@info "=" ^ 60

Δt = 0.001

function loss_single_timestep(model, T_init, Δt)
    set!(model, T=T_init)
    time_step!(model, Δt)
    return mean(interior(model.tracers.T).^2)
end

function grad_single_timestep(model, dmodel, T_init, dT_init, Δt)
    parent(dT_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_single_timestep, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T_init, dT_init),
        Enzyme.Const(Δt)
    )
    return dT_init, loss_value
end

@info "Testing autodiff through single time_step!..."
try
    @time "Compiling grad_single_timestep" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_single_timestep(
        model, dmodel, T_init, dT_init, Δt)
    @time "Running grad_single_timestep" dT, loss_val = compiled(model, dmodel, T_init, dT_init, Δt)
    @info "✅ Test 4 PASSED: Single time_step! works" loss_value=loss_val
catch e
    @error "❌ Test 4 FAILED: Single time_step!" exception=(e, catch_backtrace())
end

#####
##### Test 5: Multiple time_step! in @trace loop
#####

@info "=" ^ 60
@info "Test 5: Multiple time_step! in @trace loop (nsteps=2)"
@info "=" ^ 60

nsteps = 2

function loss_traced_timesteps(model, T_init, Δt, nsteps)
    set!(model, T=T_init)
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.tracers.T).^2)
end

function grad_traced_timesteps(model, dmodel, T_init, dT_init, Δt, nsteps)
    parent(dT_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_traced_timesteps, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(T_init, dT_init),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps)
    )
    return dT_init, loss_value
end

@info "Testing autodiff through traced time_step! loop..."
try
    @time "Compiling grad_traced_timesteps" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_traced_timesteps(
        model, dmodel, T_init, dT_init, Δt, nsteps)
    @time "Running grad_traced_timesteps" dT, loss_val = compiled(model, dmodel, T_init, dT_init, Δt, nsteps)
    @info "✅ Test 5 PASSED: Traced time_step! loop works" loss_value=loss_val
catch e
    @error "❌ Test 5 FAILED: Traced time_step! loop" exception=(e, catch_backtrace())
end

@info "=" ^ 60
@info "All tests completed"
@info "=" ^ 60
