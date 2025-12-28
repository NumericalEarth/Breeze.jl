#=
MWE: TracedRNumber{Bool} in boolean context

This MWE demonstrates the issue with using TracedRNumber{Bool} in Julia's
native conditional operators (&& and ||).

The issue occurs in Oceananigans' time_step! function:
  Δt == 0 && @warn "..."
  model.clock.iteration == 0 && update_state!(...)

When clock fields are ConcretePJRTNumber (Reactant traced numbers), comparisons
return TracedRNumber{Bool}, which cannot be used in short-circuit && operators.

Error: TypeError(:if, "", Bool, TracedRNumber{Bool}(()))

How to run:
  julia --project=test test/mwe_reactant_boolean_context.jl

Status: RESOLVED in BreezeReactantExt
  The fix provides a custom time_step! method that removes these conditionals.
  
Fix pattern:
  - Remove runtime conditionals that depend on traced values
  - Use `ifelse(cond, a, b)` instead of `cond ? a : b` when needed
  - Ensure model is initialized before compilation
=#

using Reactant
using CUDA

Reactant.set_default_backend("cpu")

println("="^80)
println("MWE: TracedRNumber{Bool} in boolean context")
println("="^80)
println("Reactant: v", pkgversion(Reactant))
println()

# ----------------------------------------------------------------------
# Trigger 1: Direct use of TracedRNumber{Bool} in && operator
# ----------------------------------------------------------------------
println("-"^80)
println("Trigger 1: TracedRNumber{Bool} in && short-circuit")
println("-"^80)

function problematic_conditional(x)
    # This comparison produces TracedRNumber{Bool}
    is_zero = x == 0
    # This && operator expects native Bool, not TracedRNumber{Bool}
    is_zero && println("x is zero!")  # Will fail during tracing
    return x + 1
end

x_traced = Reactant.ConcreteRNumber(5.0)

println("Attempting to compile function with && conditional...")
try
    compiled = Reactant.@compile problematic_conditional(x_traced)
    result = compiled(x_traced)
    println("SUCCESS (unexpected): result = ", result)
catch e
    error_str = sprint(showerror, e)
    if occursin("TracedRNumber{Bool}", error_str) || occursin("TypeError", error_str)
        println("FAILED (expected): TracedRNumber{Bool} cannot be used in && context")
    else
        println("FAILED with different error:")
    end
    println()
    println("Error (first 300 chars):")
    println(first(error_str, 300))
end
println()

# ----------------------------------------------------------------------
# Trigger 2: Ternary operator with traced condition
# ----------------------------------------------------------------------
println("-"^80)
println("Trigger 2: TracedRNumber{Bool} in ternary operator")
println("-"^80)

function problematic_ternary(x)
    # Ternary also requires native Bool
    return x > 0 ? x : -x
end

println("Attempting to compile function with ternary operator...")
try
    compiled = Reactant.@compile problematic_ternary(x_traced)
    result = compiled(x_traced)
    println("SUCCESS (unexpected): result = ", result)
catch e
    error_str = sprint(showerror, e)
    if occursin("TracedRNumber{Bool}", error_str) || occursin("TypeError", error_str)
        println("FAILED (expected): TracedRNumber{Bool} cannot be used in ternary")
    else
        println("FAILED with different error:")
    end
    println()
    println("Error (first 300 chars):")
    println(first(error_str, 300))
end
println()

# ----------------------------------------------------------------------
# Fix: Use ifelse() instead of conditionals
# ----------------------------------------------------------------------
println("-"^80)
println("Fix: Use ifelse() for Reactant-compatible conditionals")
println("-"^80)

function fixed_conditional(x)
    # ifelse() works with TracedRNumber{Bool}
    return ifelse(x > 0, x, -x)
end

println("Compiling function with ifelse()...")
try
    compiled = Reactant.@compile fixed_conditional(x_traced)
    result = compiled(x_traced)
    println("SUCCESS: result = ", result)
catch e
    println("FAILED (unexpected):")
    println(sprint(showerror, e, catch_backtrace()))
end
println()

# ----------------------------------------------------------------------
# Fix: Remove conditionals for invariants
# ----------------------------------------------------------------------
println("-"^80)
println("Fix: Remove conditionals for invariants (e.g., iteration == 0)")
println("-"^80)

# For checks like `iteration == 0 && update_state!(...)`, the fix is:
# 1. Assume the user calls update_state! before compilation
# 2. Remove the runtime check from the traced code

mutable struct Clock
    iteration::Any  # Can be ConcreteRNumber
    time::Any
end

function original_step!(clock)
    # This pattern causes issues:
    clock.iteration == 0 && return "need init"
    clock.iteration += 1
    clock.time += 1.0
    return "stepped"
end

function fixed_step!(clock)
    # Remove the iteration check - assume already initialized
    clock.iteration += 1
    clock.time += 1.0
    return nothing
end

clock = Clock(Reactant.ConcreteRNumber(1), Reactant.ConcreteRNumber(0.0))

println("Compiling fixed step function...")
try
    compiled_step! = Reactant.@compile fixed_step!(clock)
    compiled_step!(clock)
    println("SUCCESS: iteration = ", clock.iteration, ", time = ", clock.time)
catch e
    println("FAILED:")
    println(sprint(showerror, e, catch_backtrace()))
end
println()

println("="^80)
println("Summary:")
println("  - TracedRNumber{Bool} cannot be used in && or ?: operators")
println("  - Use ifelse(cond, a, b) for Reactant-compatible conditionals")
println("  - For invariant checks, remove them and document requirements")
println("  - BreezeReactantExt provides a custom time_step! that removes these")
println("="^80)
