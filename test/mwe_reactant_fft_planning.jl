#=
MWE: FFT Planning for Reactant arrays

This MWE demonstrates the issue with Oceananigans' FFT planning functions
not being defined for Reactant array types.

The error: MethodError: no method matching plan_forward_transform(
    ::ConcretePJRTArray{ComplexF64, 3, 1}, ::Periodic, ...)

This occurs when constructing FourierTridiagonalPoissonSolver with ReactantState.

How to run:
  julia --project=test test/mwe_reactant_fft_planning.jl

Status: RESOLVED in BreezeReactantExt
  The fix adds no-op methods for plan_forward_transform and plan_backward_transform
  because XLA handles FFT planning internally.

Fix pattern:
  function Oceananigans.Solvers.plan_forward_transform(A::ReactantArray, ::Periodic, dims, planner_flag=nothing)
      return nothing  # XLA handles planning
  end
=#

# Uncomment to enable the Breeze fix:
# using Breeze

using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Solvers: plan_forward_transform, plan_backward_transform
using CUDA

Reactant.set_default_backend("cpu")

println("="^80)
println("MWE: FFT Planning for Reactant arrays")
println("="^80)
println("Reactant:      v", pkgversion(Reactant))
println("Oceananigans:  v", pkgversion(Oceananigans))
println()

# ----------------------------------------------------------------------
# Trigger: Create complex array and try to plan FFT
# ----------------------------------------------------------------------
println("-"^80)
println("Trigger: plan_forward_transform on Reactant array")
println("-"^80)

# Create a ConcretePJRTArray (Reactant's array type)
complex_array = Reactant.to_rarray(rand(ComplexF64, 8, 1, 8))

println("Array type: ", typeof(complex_array))
println()

println("Attempting plan_forward_transform with Periodic topology...")
try
    plan = plan_forward_transform(complex_array, Periodic(), (1,))
    println("SUCCESS: plan = ", plan)
catch e
    error_str = sprint(showerror, e)
    if occursin("MethodError", error_str) && occursin("plan_forward_transform", error_str)
        println("FAILED (expected): No method for Reactant array types")
    else
        println("FAILED with different error:")
    end
    println()
    println("Error (first 400 chars):")
    println(first(error_str, 400))
end
println()

println("Attempting plan_forward_transform with Bounded topology...")
try
    plan = plan_forward_transform(complex_array, Bounded(), (3,))
    println("SUCCESS: plan = ", plan)
catch e
    error_str = sprint(showerror, e)
    if occursin("MethodError", error_str)
        println("FAILED (expected): No method for Reactant array types")
    else
        println("FAILED with different error:")
    end
    println()
    println("Error (first 400 chars):")
    println(first(error_str, 400))
end
println()

# ----------------------------------------------------------------------
# Impact: FourierTridiagonalPoissonSolver construction fails
# ----------------------------------------------------------------------
println("-"^80)
println("Impact: FourierTridiagonalPoissonSolver construction")
println("-"^80)

println("Attempting to create grid with ReactantState...")
grid = RectilinearGrid(ReactantState();
    size = (8, 8),
    halo = (5, 5),
    x = (-10e3, 10e3),
    z = (0, 10e3),
    topology = (Periodic, Flat, Bounded))
println("Grid created: ", typeof(grid))
println()

println("Attempting to create FourierTridiagonalPoissonSolver...")
try
    using Oceananigans.Solvers: FourierTridiagonalPoissonSolver
    solver = FourierTridiagonalPoissonSolver(grid)
    println("SUCCESS: solver created")
catch e
    error_str = sprint(showerror, e)
    if occursin("plan_forward_transform", error_str) || occursin("plan_backward_transform", error_str)
        println("FAILED (expected): FFT planning not defined for Reactant arrays")
    else
        println("FAILED with different error:")
    end
    println()
    println("Error (first 500 chars):")
    println(first(error_str, 500))
end
println()

# ----------------------------------------------------------------------
# Fix demonstration (manual)
# ----------------------------------------------------------------------
println("-"^80)
println("Fix: Add no-op methods for Reactant arrays")
println("-"^80)

println("""
# The fix in BreezeReactantExt adds these methods:

const ReactantArray = Union{
    Reactant.ConcretePJRTArray,
    Reactant.ConcreteIFRTArray,
    Reactant.TracedRArray
}

function Oceananigans.Solvers.plan_forward_transform(A::ReactantArray, ::Periodic, dims, planner_flag=nothing)
    return nothing  # XLA handles FFT planning internally
end

function Oceananigans.Solvers.plan_forward_transform(A::ReactantArray, ::Bounded, dims, planner_flag=nothing)
    return nothing
end

# Similar for plan_backward_transform
""")
println()

println("="^80)
println("Summary:")
println("  - Oceananigans FFT planning functions are not defined for Reactant arrays")
println("  - XLA handles FFT planning internally, so we return nothing (no-op)")
println("  - BreezeReactantExt provides these methods automatically")
println("="^80)
