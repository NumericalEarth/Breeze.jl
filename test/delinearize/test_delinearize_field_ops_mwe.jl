#=
MWE: Isolate which Oceananigans Field operation triggers the --check-bounds=yes issue

The set!(model, T=field) call does:
1. set!(model.tracers.T, T_init) → set_to_field! → parent(u) .= parent(v)
2. initialization_update_state!(model) → update_state! → lots of stuff

This test isolates each step to find the exact culprit.

Run with: julia --project=test test/delinearize/test_delinearize_field_ops_mwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_field_ops_mwe.jl
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

@info "Julia options" check_bounds=Base.JLOptions().check_bounds
Reactant.set_default_backend("cpu")

# Setup
@time "Constructing grid" grid = RectilinearGrid(ReactantState();
    size=(4,4,2), extent=(1,1,0.1), halo=(3,3,3), topology=(Periodic,Periodic,Bounded))

@time "Constructing model" model = HydrostaticFreeSurfaceModel(
    grid; free_surface=ExplicitFreeSurface(), buoyancy=nothing, tracers=:T, closure=nothing)
dmodel = Enzyme.make_zero(model)

@time "Constructing T_init" T_init = CenterField(grid)
set!(T_init, (x,y,z) -> x + y)
dT_init = CenterField(grid)
set!(dT_init, 0.0)

#####
##### Test 1: Just parent(field) .= parent(other_field)  
#####

@info "=" ^ 50
@info "Test 1: parent(field) .= parent(field) only"

function loss_parent_assign(model, T_init)
    parent(model.tracers.T) .= parent(T_init)
    return mean(interior(model.tracers.T).^2)
end

function grad_parent_assign(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_parent_assign, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(T_init, dT_init))
    return dT_init, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_parent_assign(model, dmodel, T_init, dT_init)
    @time "Running" result = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 1 PASSED"
catch e
    @error "❌ Test 1 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 2: Just set!(field, field) - no model
#####

@info "=" ^ 50
@info "Test 2: set!(field, field) - direct field copy"

function loss_field_set(model, T_init)
    set!(model.tracers.T, T_init)
    return mean(interior(model.tracers.T).^2)
end

function grad_field_set(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_field_set, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(T_init, dT_init))
    return dT_init, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_field_set(model, dmodel, T_init, dT_init)
    @time "Running" result = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 2 PASSED"
catch e
    @error "❌ Test 2 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 3: set!(field, field) + fill_halo_regions!
#####

@info "=" ^ 50
@info "Test 3: set!(field, field) + fill_halo_regions!"

function loss_set_and_halos(model, T_init)
    set!(model.tracers.T, T_init)
    fill_halo_regions!(model.tracers.T)
    return mean(interior(model.tracers.T).^2)
end

function grad_set_and_halos(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_set_and_halos, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(T_init, dT_init))
    return dT_init, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_and_halos(model, dmodel, T_init, dT_init)
    @time "Running" result = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 3 PASSED"
catch e
    @error "❌ Test 3 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 4: set!(field, field) + update_state! (no initialize!)
#####

@info "=" ^ 50
@info "Test 4: set!(field, field) + update_state!"

function loss_set_and_update(model, T_init)
    set!(model.tracers.T, T_init)
    update_state!(model)
    return mean(interior(model.tracers.T).^2)
end

function grad_set_and_update(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_set_and_update, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(T_init, dT_init))
    return dT_init, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_set_and_update(model, dmodel, T_init, dT_init)
    @time "Running" result = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 4 PASSED"
catch e
    @error "❌ Test 4 FAILED" exception=(e, catch_backtrace())
end

#####
##### Test 5: Full set!(model, T=...) - the original failing case
#####

@info "=" ^ 50
@info "Test 5: Full set!(model, T=field)"

function loss_full_set(model, T_init)
    set!(model, T=T_init)
    return mean(interior(model.tracers.T).^2)
end

function grad_full_set(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss_full_set, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(T_init, dT_init))
    return dT_init, lv
end

try
    @time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_full_set(model, dmodel, T_init, dT_init)
    @time "Running" result = compiled(model, dmodel, T_init, dT_init)
    @info "✅ Test 5 PASSED"
catch e
    @error "❌ Test 5 FAILED" exception=(e, catch_backtrace())
end

@info "=" ^ 50
@info "All tests completed"
