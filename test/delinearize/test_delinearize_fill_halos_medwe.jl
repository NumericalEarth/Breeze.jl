#=
Investigation: DelinearizingIndexPassing Segmentation Fault (B.6.3)
Status: ACTIVE INVESTIGATION
Purpose: Minimal MedWE focusing on fill_halo_regions! to isolate the segfault

This test removes model complexity entirely and focuses on:
- Grid construction with ReactantState
- Field construction
- fill_halo_regions! call (the suspected culprit)
- Autodiff through fill_halo_regions!

If this fails with --check-bounds=yes, we've isolated the issue to fill_halo_regions!
If this passes, the issue is elsewhere in the model time-stepping.

Run with: julia --project=test test/delinearize/test_delinearize_fill_halos_medwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_fill_halos_medwe.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Enzyme
using Statistics: mean
using CUDA

# Log package versions and options
@info "Package versions" Oceananigans=pkgversion(Oceananigans) Reactant=pkgversion(Reactant) Enzyme=pkgversion(Enzyme)
@info "Julia options" check_bounds=Base.JLOptions().check_bounds

Reactant.set_default_backend("cpu")

#####
##### Test 1: fill_halo_regions! in forward pass only
#####

@info "=" ^ 60
@info "Test 1: Forward pass with fill_halo_regions!"
@info "=" ^ 60

@time "Constructing grid (Periodic, Periodic, Flat)" grid = RectilinearGrid(ReactantState();
    size = (4, 4),
    extent = (1.0, 1.0),
    halo = (1, 1),
    topology = (Periodic, Periodic, Flat)
)

@time "Constructing field" begin
    c = CenterField(grid)
    set!(c, (x, y) -> x + y)
end

# Simple function that uses fill_halo_regions!
function forward_with_halos(c)
    fill_halo_regions!(c)
    return mean(interior(c).^2)
end

@info "Testing forward pass compilation..."
try
    @time "Compiling forward_with_halos" compiled_forward = Reactant.@compile raise_first=true raise=true sync=true forward_with_halos(c)
    @time "Running forward_with_halos" result = compiled_forward(c)
    @info "✅ Forward pass SUCCESS" result=result
catch e
    @error "❌ Forward pass FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 2: fill_halo_regions! with Enzyme autodiff
#####

@info "=" ^ 60
@info "Test 2: Autodiff through fill_halo_regions!"
@info "=" ^ 60

@time "Constructing field for autodiff" begin
    c2 = CenterField(grid)
    set!(c2, (x, y) -> x + y)
end

@time "Constructing shadow field" begin
    dc2 = CenterField(grid)
    set!(dc2, 0.0)
end

# Loss that goes through fill_halo_regions!
function loss_with_halos(c)
    fill_halo_regions!(c)
    return mean(interior(c).^2)
end

# Gradient function
function grad_loss_halos(c, dc)
    parent(dc) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_with_halos, Enzyme.Active,
        Enzyme.Duplicated(c, dc)
    )
    return dc, loss_value
end

@info "Testing autodiff through fill_halo_regions!..."
try
    @time "Compiling grad_loss_halos" compiled_grad = Reactant.@compile raise_first=true raise=true sync=true grad_loss_halos(c2, dc2)
    @time "Running grad_loss_halos" dc_result, loss_val = compiled_grad(c2, dc2)
    @info "✅ Autodiff SUCCESS" loss_value=loss_val gradient_max=maximum(abs, interior(dc_result))
catch e
    @error "❌ Autodiff FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 3: Multiple fill_halo_regions! calls (simulating time stepping)
#####

@info "=" ^ 60
@info "Test 3: Multiple fill_halo_regions! calls in @trace loop"
@info "=" ^ 60

@time "Constructing field for loop test" begin
    c3 = CenterField(grid)
    set!(c3, (x, y) -> x + y)
end

@time "Constructing shadow field for loop test" begin
    dc3 = CenterField(grid)
    set!(dc3, 0.0)
end

# Loss with multiple fill_halo_regions! calls in a loop
function loss_with_halo_loop(c, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for i in 1:nsteps
        # Simulate what happens in time_step!: modify field then fill halos
        parent(c) .= parent(c) .* 0.99  # Simple "tendency"
        fill_halo_regions!(c)
    end
    return mean(interior(c).^2)
end

function grad_loss_halo_loop(c, dc, nsteps)
    parent(dc) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_with_halo_loop, Enzyme.Active,
        Enzyme.Duplicated(c, dc),
        Enzyme.Const(nsteps)
    )
    return dc, loss_value
end

nsteps = 4

@info "Testing autodiff through fill_halo_regions! loop (nsteps=$nsteps)..."
try
    @time "Compiling grad_loss_halo_loop" compiled_loop = Reactant.@compile raise_first=true raise=true sync=true grad_loss_halo_loop(c3, dc3, nsteps)
    @time "Running grad_loss_halo_loop" dc_result, loss_val = compiled_loop(c3, dc3, nsteps)
    @info "✅ Loop autodiff SUCCESS" loss_value=loss_val gradient_max=maximum(abs, interior(dc_result))
catch e
    @error "❌ Loop autodiff FAILED" exception=(e, catch_backtrace())
end

@info "=" ^ 60
@info "All tests completed"
@info "=" ^ 60
