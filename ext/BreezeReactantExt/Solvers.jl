using Oceananigans.Solvers: Solvers, FourierTridiagonalPoissonSolver

const EagerReactantFourierTridiagonalPoissonSolver =
    FourierTridiagonalPoissonSolver{<:Any, <:Any, <:Any, <:Any, <:Any, <:ConcreteReactantArray}

# TODO Oceananigans: move this method into OceananigansReactantExt, and delete it once
# Reactant applies its FFT plans to concrete arrays (https://github.com/EnzymeAD/Reactant.jl/issues/3259).
# `AtmosphereModel` with `AnelasticDynamics` projects its initial momentum with the FFT pressure
# solver outside of any trace (`set!` → `enforce_mass_conservation!`), where two operations fail
# on concrete Reactant arrays:
#
# - Reactant's FFT plans act only on traced arrays. Applied to a concrete array, AbstractFFTs'
#   fallback `p * copy1(T, x)` calls itself forever, allocating a copy of `x` on every call.
# - Removing the mean, `ϕ .- mean(ϕ)`, reads a scalar, which device buffers forbid.
#
# Compile the whole eager solve with `@jit` instead. Inside a trace the solver storage is
# a traced array, so this method does not apply. `@jit` compiles on every call, which is
# acceptable for the few eager solves that model construction and `set!` perform, but makes
# eager (untraced) time stepping of a ReactantState anelastic model slow.
function Solvers.solve!(x, solver::EagerReactantFourierTridiagonalPoissonSolver, b=nothing)
    Reactant.@jit Solvers.solve!(x, solver, b)
    return nothing
end
