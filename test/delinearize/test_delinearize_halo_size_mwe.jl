#=
MINIMAL MWE: Test if halo size affects the DelinearizeIndexingPass segfault

Hypothesis: halo=(3,3,3) fails but halo=(1,1,1) passes

Run with: julia --project=test test/delinearize/test_delinearize_halo_size_mwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_halo_size_mwe.jl
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

# Test different halo sizes
for halo_size in [1, 2, 3]
    @info "=" ^ 50
    @info "Testing halo size = $halo_size"
    
    grid = RectilinearGrid(ReactantState();
        size=(6,6,2), extent=(1,1,1), halo=(halo_size,halo_size,halo_size), 
        topology=(Periodic,Periodic,Bounded))
    
    field = CenterField(grid)
    dfield = CenterField(grid)
    parent(field) .= 1.0
    parent(dfield) .= 0.0
    
    # Define functions inside loop to avoid name conflicts
    loss_fn = f -> begin
        fill_halo_regions!(f)
        mean(interior(f).^2)
    end
    
    grad_fn = (f, df) -> begin
        parent(df) .= 0
        _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss_fn, Enzyme.Active, Enzyme.Duplicated(f, df))
        return df, lv
    end
    
    try
        @time "Compiling halo=$halo_size" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_fn(field, dfield)
        @time "Running halo=$halo_size" result = compiled(field, dfield)
        @info "✅ halo=$halo_size PASSED"
    catch e
        @error "❌ halo=$halo_size FAILED" exception=(e, catch_backtrace())
        break  # Stop on first failure
    end
end

@info "=" ^ 50
@info "Done"
