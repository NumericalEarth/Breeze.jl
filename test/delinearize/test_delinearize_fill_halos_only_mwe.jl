#=
MINIMAL MWE: Just fill_halo_regions! on an Oceananigans Field
No model, no set!, just the bare minimum.

Testing both 2D (Flat) and 3D (Bounded) topologies.

Run with: julia --project=test test/delinearize/test_delinearize_fill_halos_only_mwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_fill_halos_only_mwe.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Reactant
using Enzyme
using Statistics: mean
using CUDA

@info "Julia options" check_bounds=Base.JLOptions().check_bounds
Reactant.set_default_backend("cpu")

#####
##### Test 1: 2D grid (Periodic, Periodic, Flat) - this worked before
#####

@info "=" ^ 50
@info "Test 1: 2D grid (Periodic, Periodic, Flat)"

@time "Constructing 2D grid" grid_2d = RectilinearGrid(ReactantState();
    size=(4,4), extent=(1,1), halo=(1,1), topology=(Periodic,Periodic,Flat))

@time "Constructing 2D field" field_2d = CenterField(grid_2d)
@time "Constructing 2D shadow" dfield_2d = CenterField(grid_2d)
parent(field_2d) .= 1.0
parent(dfield_2d) .= 0.0

function loss_2d(f)
    fill_halo_regions!(f)
    return mean(interior(f).^2)
end

function grad_2d(f, df)
    parent(df) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_2d, Enzyme.Active, Enzyme.Duplicated(f, df))
    return df, lv
end

try
    @time "Compiling 2D" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_2d(field_2d, dfield_2d)
    @time "Running 2D" result = compiled(field_2d, dfield_2d)
    @info "✅ Test 1 PASSED (2D Flat)"
catch e
    @error "❌ Test 1 FAILED (2D Flat)" exception=(e, catch_backtrace())
end

#####
##### Test 2: 3D grid (Periodic, Periodic, Bounded) - this might fail
#####

@info "=" ^ 50
@info "Test 2: 3D grid (Periodic, Periodic, Bounded)"

@time "Constructing 3D grid" grid_3d = RectilinearGrid(ReactantState();
    size=(4,4,2), extent=(1,1,1), halo=(1,1,1), topology=(Periodic,Periodic,Bounded))

@time "Constructing 3D field" field_3d = CenterField(grid_3d)
@time "Constructing 3D shadow" dfield_3d = CenterField(grid_3d)
parent(field_3d) .= 1.0
parent(dfield_3d) .= 0.0

function loss_3d(f)
    fill_halo_regions!(f)
    return mean(interior(f).^2)
end

function grad_3d(f, df)
    parent(df) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_3d, Enzyme.Active, Enzyme.Duplicated(f, df))
    return df, lv
end

try
    @time "Compiling 3D" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_3d(field_3d, dfield_3d)
    @time "Running 3D" result = compiled(field_3d, dfield_3d)
    @info "✅ Test 2 PASSED (3D Bounded)"
catch e
    @error "❌ Test 2 FAILED (3D Bounded)" exception=(e, catch_backtrace())
end

#####
##### Test 3: 3D grid (Periodic, Periodic, Periodic) - fully periodic
#####

@info "=" ^ 50
@info "Test 3: 3D grid (Periodic, Periodic, Periodic)"

@time "Constructing 3D periodic grid" grid_3d_periodic = RectilinearGrid(ReactantState();
    size=(4,4,2), extent=(1,1,1), halo=(1,1,1), topology=(Periodic,Periodic,Periodic))

@time "Constructing field" field_3d_p = CenterField(grid_3d_periodic)
@time "Constructing shadow" dfield_3d_p = CenterField(grid_3d_periodic)
parent(field_3d_p) .= 1.0
parent(dfield_3d_p) .= 0.0

function loss_3d_periodic(f)
    fill_halo_regions!(f)
    return mean(interior(f).^2)
end

function grad_3d_periodic(f, df)
    parent(df) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss_3d_periodic, Enzyme.Active, Enzyme.Duplicated(f, df))
    return df, lv
end

try
    @time "Compiling 3D periodic" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_3d_periodic(field_3d_p, dfield_3d_p)
    @time "Running 3D periodic" result = compiled(field_3d_p, dfield_3d_p)
    @info "✅ Test 3 PASSED (3D Periodic)"
catch e
    @error "❌ Test 3 FAILED (3D Periodic)" exception=(e, catch_backtrace())
end

@info "=" ^ 50
@info "All tests completed"
