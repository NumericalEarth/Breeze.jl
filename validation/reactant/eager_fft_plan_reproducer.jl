# Minimal reproducer: applying a Reactant FFT plan to a *concrete* Reactant array
# recurses forever, allocating a new array on each call, until the process runs out of memory.
#
#   julia --project reproducer.jl            # run under a memory limit (e.g. ulimit -v)!
using Reactant
using AbstractFFTs

const FFTExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

x = Reactant.to_rarray(ones(ComplexF64, 2, 2, 2))
plan = FFTExt.ReactantFFTInPlacePlan{ComplexF64}([1, 2, 3])  # the plan Oceananigans builds for ReactantState

# Inside a compiled function the plan works:
apply(plan, x) = (plan * x; nothing)
Reactant.@jit apply(plan, x)
println("compiled: ", Array(x)[1])   # 8.0 + 0.0im

# Eagerly it dispatches to AbstractFFTs' fallback `*(p::Plan{T}, x::AbstractArray) = p * copy1(T, x)`,
# and `copy1` returns another ConcretePJRTArray, so the same method is called again, forever:
println(which(*, Tuple{typeof(plan), typeof(x)}))
plan * x   # never returns; resident memory grows by GBs per minute
