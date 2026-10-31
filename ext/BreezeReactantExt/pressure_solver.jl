using Oceananigans: ReactantState
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Solvers: FourierTridiagonalPoissonSolver
using Breeze.AnelasticEquations: AnelasticEquations, solve_anelastic_poisson_equation!
using Reactant: Reactant

const ReactantGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:ReactantState}
const ReactantFourierTridiagonalPoissonSolver = FourierTridiagonalPoissonSolver{<:ReactantGrid}

# Outside a compiled program (for example `set!` → `enforce_mass_conservation!` during model
# construction) the pressure solve runs on concrete arrays. Reactant's FFT plans only implement
# `plan * x` for traced arrays; on a concrete array, `plan * x` falls back to AbstractFFTs'
# `p * copy1(T, x)`, which calls itself forever, allocating a copy of `x` on every call until the
# process runs out of memory. Compiling the whole solve as one program also avoids compiling
# each kernel separately.
#
# TODO: remove once Reactant applies its FFT plans to concrete arrays, or move to
# OceananigansReactantExt as an eager `solve!` method: Oceananigans' `set!(::NonhydrostaticModel)`
# on `ReactantState` runs out of memory in the same way.
function AnelasticEquations.solve_for_anelastic_pressure!(pₙ, solver::ReactantFourierTridiagonalPoissonSolver, ρŨ, Δt)
    if Reactant.within_compile()
        solve_anelastic_poisson_equation!(pₙ, solver, ρŨ, Δt)
    else
        Reactant.@jit solve_anelastic_poisson_equation!(pₙ, solver, ρŨ, Δt)
    end
    return pₙ
end
