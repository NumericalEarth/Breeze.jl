#=
MINIMAL MWE: Exact reproduction of failing Test 1 from test_delinearize_set_halos_interaction_mwe.jl

Run with: julia --project=test test/delinearize/test_delinearize_minimal_mwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_minimal_mwe.jl
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

# Exact same setup as the failing test
halo = (2, 2, 2)
grid = RectilinearGrid(ReactantState();
    size=(3,3,2), extent=(1,1,0.1), halo=halo, topology=(Periodic,Periodic,Bounded))

field = CenterField(grid)
set!(field, 0.0)  # Uses Oceananigans set! which fills interior only

dfield = CenterField(grid)
set!(dfield, 0.0)

function loss(f)
    fill_halo_regions!(f)
    return mean(interior(f).^2)
end

function grad(f, df)
    parent(df) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active, Enzyme.Duplicated(f, df))
    return df, lv
end

@info "Compiling..."
@time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad(field, dfield)