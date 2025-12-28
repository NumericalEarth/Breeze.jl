#=
MWE: Oceananigans reductions + ReactantState hit scalar-indexing errors

This MWE reproduces the *current* Oceananigans↔Reactant integration problem:
Oceananigans reductions (`minimum/maximum/sum/...`) over `AbstractField` /
`AbstractOperation` can scalar-index the underlying Reactant arrays, which
Reactant disallows.

Two key triggers:
1) `Oceananigans.Advection.cell_advection_timescale(grid, velocities)` uses
   `minimum(KernelFunctionOperation(...))` internally.
2) `maximum(field)` uses Oceananigans' `AbstractField` reduction machinery.

How to run:
  julia --project=test test/mwe_reactant_scalar_indexing_oceananigans_reductions.jl

Notes:
- This file is written to be useful both for upstream reporting (OceananigansReactantExt)
  and for Breeze-local validation.
- If you uncomment `using Breeze`, Breeze will load `BreezeReactantExt`, which currently
  contains a workaround that *avoids scalar iteration* by redirecting reductions to
  `interior(field)` (for `Field`) or by materializing `AbstractOperation` to a `Field`
  and reducing that.
=#

# Uncomment to enable the Breeze-local workaround:
# using Breeze

using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Advection: cell_advection_timescale
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: interior
using CUDA

Reactant.set_default_backend("cpu")

println("="^80)
println("MWE: Oceananigans reductions + ReactantState")
println("="^80)
println("Reactant:      v", pkgversion(Reactant))
println("Oceananigans:  v", pkgversion(Oceananigans))
println("CUDA:          v", pkgversion(CUDA))
println()

# ----------------------------------------------------------------------
# Trigger A: cell_advection_timescale -> minimum(KernelFunctionOperation)
# ----------------------------------------------------------------------
grid = RectilinearGrid(ReactantState(); size=(4, 4), extent=(1, 1), topology=(Periodic, Flat, Bounded))
u = XFaceField(grid); parent(u) .= 1
v = YFaceField(grid); parent(v) .= 0
w = ZFaceField(grid); parent(w) .= 0

println("-"^80)
println("Trigger A: cell_advection_timescale(grid, (u, v, w))")
println("-"^80)
try
    τ = cell_advection_timescale(grid, (u, v, w))
    println("SUCCESS: τ = ", τ)
catch e
    println("FAILED:")
    println(sprint(showerror, e, catch_backtrace()))
end
println()

# ----------------------------------------------------------------------
# Trigger B: Field reductions (maximum/minimum/sum)
# ----------------------------------------------------------------------
f = CenterField(grid)
parent(f) .= 1

println("-"^80)
println("Trigger B: maximum(f)")
println("-"^80)
try
    println("SUCCESS: maximum(f) = ", maximum(f))
catch e
    println("FAILED:")
    println(sprint(showerror, e, catch_backtrace()))
end
println()

println("-"^80)
println("Control: reductions on plain Reactant arrays DO work")
println("-"^80)
A = Reactant.to_rarray(rand(4,4,4))
println("maximum(A) = ", maximum(A))
println("minimum(A) = ", minimum(A))
println("sum(A)     = ", sum(A))
println()

