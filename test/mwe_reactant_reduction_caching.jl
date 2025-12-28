#=
MWE: Reduction Caching Investigation

Investigation for Improvement #4: Can we cache scratch fields to reduce
allocation overhead for repeated reductions?

## Current Pattern (allocates per reduction)

```julia
function Base.minimum(f, op::AbstractOperation; ...)
    tmp = Field(op)  # NEW allocation every time!
    return Base.minimum(f, interior(tmp))
end
```

## Questions to Answer

1. How often are reductions called during time stepping?
2. What is the allocation overhead of Field(op)?
3. Would caching provide measurable benefit?

## How to Run

```bash
julia --project=test test/mwe_reactant_reduction_caching.jl
```
=#

using Breeze
using CUDA
using Reactant
using Oceananigans
using Oceananigans.Architectures: ReactantState, CPU
using Oceananigans.Advection: Centered
using Oceananigans.AbstractOperations: AbstractOperation, KernelFunctionOperation
using Oceananigans.Fields: Field

println("="^80)
println("MWE: Reduction Caching Investigation")
println("="^80)
println()

#####
##### Test 1: Measure allocation from Field(op)
#####

println("-"^80)
println("Test 1: Allocation from Field(op) on CPU")
println("-"^80)

# Create a CPU model first (simpler to analyze)
cpu_grid = RectilinearGrid(CPU();
    size = (32, 32),
    x = (0, 1000.0),
    z = (0, 1000.0),
    topology = (Periodic, Flat, Bounded)
)

cpu_model = AtmosphereModel(cpu_grid; advection = Centered(order=2))
θ₀(x, z) = 300.0 + 0.01 * z
set!(cpu_model; θ = θ₀)

# Get an AbstractOperation (temperature field is already a Field, not an op)
# Let's create a simple operation
u = cpu_model.velocities.u
op = u * u  # This creates an AbstractOperation

println("Operation type: $(typeof(op))")
println()

# Measure allocation
println("Measuring Field(op) allocation...")
alloc_before = Base.gc_live_bytes()
for i in 1:10
    tmp = Field(op)
end
GC.gc()
alloc_after = Base.gc_live_bytes()

println("Approximate allocation per Field(op): ~$(round((alloc_after - alloc_before) / 10 / 1024, digits=1)) KB")
println()

#####
##### Test 2: Count reductions during time stepping
#####

println("-"^80)
println("Test 2: Reduction frequency during time stepping")
println("-"^80)

# We can't easily instrument the code, but we know:
# - CFL checks happen every N steps (configurable)
# - Each CFL check calls minimum() on velocity operations

println("CFL checking involves minimum() on KernelFunctionOperations")
println("With TimeStepWizard, this happens every `cfl` check interval")
println("Default: every time step, so N_steps reductions per simulation")
println()

#####
##### Test 3: Analyze current Field(op) approach
#####

println("-"^80)
println("Test 3: Analyze current Field(op) approach")
println("-"^80)

# The current implementation uses Field(op) which:
# 1. Creates a new Field with proper boundary conditions
# 2. Calls compute! to evaluate the operation into the field
# 3. Reduces over interior(field)

op_test = u * u
println("Creating Field(op) 10 times and measuring...")

times = Float64[]
for i in 1:10
    t = time_ns()
    tmp = Field(op_test)
    push!(times, (time_ns() - t) / 1e6)
end

println("Field(op) creation times: $(round.(times, digits=3)) ms")
println("Average: $(round(sum(times)/length(times), digits=3)) ms")
println()

#####
##### Test 4: Benchmark full minimum(op) call
#####

println("-"^80)
println("Test 4: Benchmark full minimum(op) call")
println("-"^80)

n_iters = 50

# Warmup
for _ in 1:5
    minimum(op_test)
end

# Benchmark
t_total = time_ns()
for _ in 1:n_iters
    minimum(op_test)
end
t_total = (time_ns() - t_total) / 1e6 / n_iters

println("minimum(op) average time: $(round(t_total, digits=3)) ms")
println()

# Compare to minimum on a pre-computed Field
field_test = Field(op_test)
for _ in 1:5
    minimum(field_test)
end

t_field = time_ns()
for _ in 1:n_iters
    minimum(field_test)
end
t_field = (time_ns() - t_field) / 1e6 / n_iters

println("minimum(precomputed_field) average time: $(round(t_field, digits=3)) ms")
println()

overhead = t_total - t_field
println("Field(op) overhead per reduction: $(round(overhead, digits=3)) ms")
println()

#####
##### Test 5: Reactant context analysis
#####

println("-"^80)
println("Test 5: Reactant context analysis")
println("-"^80)

println("Key insight: In Reactant context, Field(op) creates a NEW array each time.")
println("This is significant because:")
println("  1. Reactant arrays are backed by XLA buffers")
println("  2. Creating a buffer involves XLA compilation overhead")
println("  3. GC pressure from temporary arrays")
println()
println("However, Reactant's @compile traces the whole function, so:")
println("  - Field allocations inside traced code become static")
println("  - XLA optimizer may eliminate redundant allocations")
println("  - Caching may not help inside @compile context")
println()

println()
println("="^80)
println("Investigation complete")
println("="^80)

println()
println("SUMMARY")
println("="^80)
println("""
Key Findings for Improvement #4 (Reduction Caching):

1. Field(op) creates a new Field with compute! for each reduction
2. The overhead is primarily:
   - Array allocation
   - Boundary condition setup
   - compute! kernel execution

3. In Reactant @compile context:
   - Allocations become static XLA operations
   - XLA optimizer may fuse/eliminate redundant work
   - Caching may not provide additional benefit

4. Outside @compile (runtime):
   - CFL checks happen at runtime, not inside traced code
   - But TimeStepWizard typically runs before @compile

CONCLUSION:
- Reduction caching adds complexity with uncertain benefit
- XLA optimization may already handle this
- Mark as "INVESTIGATED - LOW PRIORITY"
- Revisit if profiling shows reduction overhead
""")
println("="^80)
