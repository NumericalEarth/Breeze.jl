#=
MWE: `FourierTridiagonalPoissonSolver` compilation on `ReactantState` (issue #223 context)

This MWE focuses on the exact solver used by Breeze's anelastic pressure solve:
`Oceananigans.Solvers.FourierTridiagonalPoissonSolver`.

## Key Issues Addressed (Option B workaround)

1) **Construction**: Oceananigans does not define FFT planning methods for Reactant arrays.
   - Without a workaround, solver construction fails with a `MethodError` for
     `plan_forward_transform(::ConcretePJRTArray, ...)`.
   - **Fix**: BreezeReactantExt returns `nothing` for FFT plans on Reactant arrays.

2) **Tridiagonal Solve Compilation**: The KA kernel `solve_batched_tridiagonal_system_kernel!`
   fails during `Reactant.@compile raise=true` due to unsupported affine loop patterns.
   - **Fix**: BreezeReactantExt provides a pure-Julia Thomas algorithm using broadcasts
     that Reactant can compile successfully.

3) **Complex Number Handling**: The original solver operates on complex arrays post-FFT.
   Reactant fails to lower `real(complex)` inside KA kernels.
   - **Fix**: BreezeReactantExt solves the tridiagonal system directly on complex arrays
     using pure-Julia broadcasts. The tridiagonal coefficients are real (Laplacian eigenvalues),
     while only the RHS and solution are complex. This works because pure Julia broadcasts
     (not KA kernels) handle complex numbers correctly in Reactant.

## Current Status

With BreezeReactantExt loaded, `FourierTridiagonalPoissonSolver.solve!` compiles successfully
with `raise=true` for Periodic/Flat transform dimensions (no DCT support yet).

## How to Run

```bash
# Baseline (no Breeze workaround loaded) — expected to fail at construction:
julia --project=test test/mwe_reactant_fourier_tridiagonal_solver_compile.jl

# With Breeze workaround loaded — should succeed:
julia --project=test -e 'using Breeze; include("test/mwe_reactant_fourier_tridiagonal_solver_compile.jl")'
```
=#

using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState, on_architecture
using Oceananigans.Grids: Periodic, Flat, Bounded
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver, solve!
using CUDA

Reactant.set_default_backend("cpu")

println("="^80)
println("MWE: FourierTridiagonalPoissonSolver compilation on ReactantState")
println("="^80)
println("Reactant:     v", pkgversion(Reactant))
println("Oceananigans: v", pkgversion(Oceananigans))
println("CUDA:         v", pkgversion(CUDA))
println("Breeze loaded: ", isdefined(Main, :Breeze))
println()

arch = ReactantState()

# Typical Breeze-like topology: x periodic, y flat (2D), z bounded.
# Note: For Flat topology, size is 2D (Nx, Nz) not 3D.
grid = RectilinearGrid(arch;
    size = (4, 4),
    x = (0, 1),
    z = (0, 1),
    topology = (Periodic, Flat, Bounded))

println("Grid: ", typeof(grid))

println("-"^80)
println("Step 1: Construct FourierTridiagonalPoissonSolver")
println("-"^80)
local solver = nothing
try
    global solver = FourierTridiagonalPoissonSolver(grid)
    println("SUCCESS: solver = ", typeof(solver))
catch e
    println("FAILED (expected without Breeze workaround):")
    println(first(sprint(showerror, e, catch_backtrace()), 800))
end
println()

if solver === nothing
    println("Stopping: solver construction failed.")
else
    println("-"^80)
    println("Step 2: Compile solve!(x, solver, b) with Reactant.@compile")
    println("-"^80)

    x = on_architecture(arch, zeros(Float64, size(grid)...))
    b = on_architecture(arch, rand(Float64, size(grid)...))

    function do_solve!(x, solver, b)
        solve!(x, solver, b)
        return x
    end

    try
        compiled = Reactant.@compile sync=true raise=true do_solve!(x, solver, b)
        compiled(x, solver, b)
        println("SUCCESS: compiled solve! executed")
    catch e
        println("FAILED:")
        println(first(sprint(showerror, e, catch_backtrace()), 1200))
    end
end


