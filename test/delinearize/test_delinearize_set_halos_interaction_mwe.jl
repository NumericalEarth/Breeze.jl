#=
MWE: The interaction between set! and fill_halo_regions! triggers DelinearizeIndexingPass segfault

Key findings:
- set!(field, field) alone: PASSES
- fill_halo_regions! alone: PASSES
- COMBINATION in same function: SEGFAULTS

This test explores the exact interaction.

Run with: julia --project=test test/delinearize/test_delinearize_set_halos_interaction_mwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_set_halos_interaction_mwe.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Enzyme
using Statistics: mean
using CUDA

@info "Julia options" check_bounds=Base.JLOptions().check_bounds
Reactant.set_default_backend("cpu")

# Setup
@time "Constructing grid" grid = RectilinearGrid(ReactantState();
    size=(4,4,2), extent=(1,1,0.1), halo=(3,3,3), topology=(Periodic,Periodic,Bounded))

# Use standalone fields, not model fields
@time "Constructing field_a" field_a = CenterField(grid)
set!(field_a, 0.0)
dfield_a = CenterField(grid)
set!(dfield_a, 0.0)

@time "Constructing field_b" field_b = CenterField(grid)
set!(field_b, (x,y,z) -> x + y)
dfield_b = CenterField(grid)
set!(dfield_b, 0.0)

#####
##### Test 1: fill_halo_regions! on fresh field (baseline)
#####

@info "=" ^ 50
@info "Test 1: fill_halo_regions! only (baseline)"

function loss_halos_only(field_a)
    fill_halo_regions!(field_a)
    return mean(interior(field_a).^2)
end

function grad_halos_only(field_a, dfield_a)
    parent(dfield_a) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_halos_only, Enzyme.Active,
        Enzyme.Duplicated(field_a, dfield_a))
    return dfield_a, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_halos_only(field_a, dfield_a)
    @time "Running" result = compiled(field_a, dfield_a)
    @info "✅ Test 1 PASSED: fill_halo_regions! alone works"
catch e
    @error "❌ Test 1 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 2: set! only (baseline)
#####

@info "=" ^ 50
@info "Test 2: set!(field, field) only (baseline)"

function loss_set_only(field_a, field_b)
    set!(field_a, field_b)
    return mean(interior(field_a).^2)
end

function grad_set_only(field_a, dfield_a, field_b, dfield_b)
    parent(dfield_a) .= 0
    parent(dfield_b) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_set_only, Enzyme.Active,
        Enzyme.Duplicated(field_a, dfield_a), Enzyme.Duplicated(field_b, dfield_b))
    return dfield_a, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_only(field_a, dfield_a, field_b, dfield_b)
    @time "Running" result = compiled(field_a, dfield_a, field_b, dfield_b)
    @info "✅ Test 2 PASSED: set! alone works"
catch e
    @error "❌ Test 2 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 3: set! then fill_halo_regions! on SAME field (the failing case)
#####

@info "=" ^ 50
@info "Test 3: set! then fill_halo_regions! on SAME field"

function loss_set_then_halos(field_a, field_b)
    set!(field_a, field_b)
    fill_halo_regions!(field_a)  # fill halos on the SAME field we just set!
    return mean(interior(field_a).^2)
end

function grad_set_then_halos(field_a, dfield_a, field_b, dfield_b)
    parent(dfield_a) .= 0
    parent(dfield_b) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_set_then_halos, Enzyme.Active,
        Enzyme.Duplicated(field_a, dfield_a), Enzyme.Duplicated(field_b, dfield_b))
    return dfield_a, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_then_halos(field_a, dfield_a, field_b, dfield_b)
    @time "Running" result = compiled(field_a, dfield_a, field_b, dfield_b)
    @info "✅ Test 3 PASSED"
catch e
    @error "❌ Test 3 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 4: set! then fill_halo_regions! on DIFFERENT field
#####

@info "=" ^ 50
@info "Test 4: set! then fill_halo_regions! on DIFFERENT field"

function loss_set_then_halos_different(field_a, field_b)
    set!(field_a, field_b)
    fill_halo_regions!(field_b)  # fill halos on DIFFERENT field
    return mean(interior(field_a).^2)
end

function grad_set_then_halos_different(field_a, dfield_a, field_b, dfield_b)
    parent(dfield_a) .= 0
    parent(dfield_b) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_set_then_halos_different, Enzyme.Active,
        Enzyme.Duplicated(field_a, dfield_a), Enzyme.Duplicated(field_b, dfield_b))
    return dfield_a, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_then_halos_different(field_a, dfield_a, field_b, dfield_b)
    @time "Running" result = compiled(field_a, dfield_a, field_b, dfield_b)
    @info "✅ Test 4 PASSED"
catch e
    @error "❌ Test 4 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 5: fill_halo_regions! then set! (reversed order)
#####

@info "=" ^ 50
@info "Test 5: fill_halo_regions! then set! (reversed order)"

function loss_halos_then_set(field_a, field_b)
    fill_halo_regions!(field_a)  # fill halos first
    set!(field_a, field_b)       # then set
    return mean(interior(field_a).^2)
end

function grad_halos_then_set(field_a, dfield_a, field_b, dfield_b)
    parent(dfield_a) .= 0
    parent(dfield_b) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_halos_then_set, Enzyme.Active,
        Enzyme.Duplicated(field_a, dfield_a), Enzyme.Duplicated(field_b, dfield_b))
    return dfield_a, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_halos_then_set(field_a, dfield_a, field_b, dfield_b)
    @time "Running" result = compiled(field_a, dfield_a, field_b, dfield_b)
    @info "✅ Test 5 PASSED"
catch e
    @error "❌ Test 5 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 6: parent .= parent then fill_halo_regions! (bypass set!)
#####

@info "=" ^ 50
@info "Test 6: parent() .= parent() then fill_halo_regions!"

function loss_parent_then_halos(field_a, field_b)
    parent(field_a) .= parent(field_b)  # bypass set!
    fill_halo_regions!(field_a)
    return mean(interior(field_a).^2)
end

function grad_parent_then_halos(field_a, dfield_a, field_b, dfield_b)
    parent(dfield_a) .= 0
    parent(dfield_b) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_parent_then_halos, Enzyme.Active,
        Enzyme.Duplicated(field_a, dfield_a), Enzyme.Duplicated(field_b, dfield_b))
    return dfield_a, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_parent_then_halos(field_a, dfield_a, field_b, dfield_b)
    @time "Running" result = compiled(field_a, dfield_a, field_b, dfield_b)
    @info "✅ Test 6 PASSED"
catch e
    @error "❌ Test 6 FAILED" exception=(e, catch_backtrace())
end

@info "=" ^ 50
@info "All tests completed"
