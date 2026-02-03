#=
MWE: set!(model, T=...) with Reactant + Enzyme triggers DelinearizingIndexPassing segfault
Run with: julia --project=test test/delinearize/test_delinearize_set_mwe.jl
Run with bounds checking: julia --project=test --check-bounds=yes test/delinearize/test_delinearize_set_mwe.jl
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface
using Reactant
using Enzyme
using Statistics: mean
using CUDA

@info "Julia options" check_bounds=Base.JLOptions().check_bounds
Reactant.set_default_backend("cpu")

# Grid
grid = RectilinearGrid(ReactantState(); size=(4,4,2), extent=(1,1,0.1), halo=(3,3,3), topology=(Periodic,Periodic,Bounded))

# Model
model = HydrostaticFreeSurfaceModel(grid; free_surface=ExplicitFreeSurface(), buoyancy=nothing, tracers=:T, closure=nothing)
dmodel = Enzyme.make_zero(model)

# Field
T_init = CenterField(grid); set!(T_init, (x,y,z) -> x + y)
dT_init = CenterField(grid); set!(dT_init, 0.0)

# Loss: just set! and return mean
function loss(model, T_init)
    set!(model, T=T_init)
    return mean(interior(model.tracers.T).^2)
end

# Gradient
function grad_loss(model, dmodel, T_init, dT_init)
    parent(dT_init) .= 0
    _, lv = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(T_init, dT_init))
    return dT_init, lv
end

# Compile and run
@time "Compiling" compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(model, dmodel, T_init, dT_init)
@time "Running" dT, lv = compiled(model, dmodel, T_init, dT_init)
@info "Result" loss=lv grad_max=maximum(abs, interior(dT))
