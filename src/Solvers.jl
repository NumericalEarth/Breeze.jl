"""
    Solvers

Iterative solvers for the small nonlinear scalar problems that arise in Breeze's
thermodynamics: equation-of-state temperature inversions, saturation adjustment,
and dewpoint computation.

A "solver" is a lightweight, isbits description of an iteration's stopping rule that
algorithms dispatch on:

* [`NewtonSolver`](@ref) and [`SecantSolver`](@ref) iterate until a tolerance-based
  convergence criterion is met (or `maxiter` is reached).
* [`FixedIterations`](@ref) performs an exact number of iterations with no convergence
  test at all. Because the trip count is fixed, the loop unrolls to straight-line code,
  which is required for Reactant tracing and cheap reverse-mode differentiation
  (a tolerance-based `while` loop traces to an XLA `while` op whose adjoint is
  pathological — see NumericalEarth/Breeze.jl#767).
* `nothing` means "do not iterate": the algorithm returns its initial guess
  (typically a closed-form approximation).

The drivers [`newton_solve`](@ref) and [`secant_solve`](@ref) implement the iterations
once, so every algorithm shares the same loop logic and the same solver vocabulary.
"""
module Solvers

export NewtonSolver, SecantSolver, FixedIterations

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF
using Oceananigans: Oceananigans
using Oceananigans.Utils: prettysummary

#####
##### Solver types
#####

"""
$(TYPEDEF)

A Newton iteration that terminates when the step size `Δx` satisfies
`|Δx| ≤ max(abstol, reltol * |x|)`, or after `maxiter` iterations.
"""
struct NewtonSolver{FT}
    reltol :: FT
    abstol :: FT
    maxiter :: Int
end

"""
$(TYPEDSIGNATURES)

Return a [`NewtonSolver`](@ref) with relative tolerance `reltol`, absolute tolerance
`abstol`, and iteration cap `maxiter`.

```jldoctest
using Breeze

NewtonSolver(reltol=1e-6, maxiter=4)

# output
NewtonSolver(reltol=1.0e-6, abstol=0.0, maxiter=4)
```
"""
NewtonSolver(FT::DataType=Oceananigans.defaults.FloatType; reltol=1e-8, abstol=0, maxiter=8) =
    NewtonSolver(convert(FT, reltol), convert(FT, abstol), Int(maxiter))

"""
$(TYPEDEF)

A secant iteration that terminates when the residual `r` satisfies
`|r| ≤ max(abstol, reltol * |scale|)` — where `scale` is an algorithm-supplied
magnitude for the residual — or after `maxiter` iterations.
"""
struct SecantSolver{FT}
    reltol :: FT
    abstol :: FT
    maxiter :: Int
end

"""
$(TYPEDSIGNATURES)

Return a [`SecantSolver`](@ref) with relative tolerance `reltol`, absolute tolerance
`abstol`, and iteration cap `maxiter`.

```jldoctest
using Breeze

SecantSolver(abstol=1e-4, maxiter=20)

# output
SecantSolver(reltol=0.0, abstol=0.0001, maxiter=20)
```
"""
SecantSolver(FT::DataType=Oceananigans.defaults.FloatType; reltol=0, abstol=1e-3, maxiter=100) =
    SecantSolver(convert(FT, reltol), convert(FT, abstol), Int(maxiter))

"""
$(TYPEDEF)

A solver that performs exactly `iterations` iterations with no convergence test.

The fixed trip count means the iteration unrolls to straight-line, branch-free code,
making it the right choice for Reactant tracing and reverse-mode differentiation,
where data-dependent `while` loops are pathological (NumericalEarth/Breeze.jl#767).

```jldoctest
using Breeze

FixedIterations(2)

# output
FixedIterations(2)
```
"""
struct FixedIterations
    iterations :: Int
end

#####
##### Float type materialization
#####

"""
$(TYPEDSIGNATURES)

Return `solver` with its tolerances converted to float type `FT`, so that
solver parameters stored on `Float32` models do not promote kernel arithmetic.
"""
materialize_solver(solver::NewtonSolver, FT) =
    NewtonSolver(convert(FT, solver.reltol), convert(FT, solver.abstol), solver.maxiter)

materialize_solver(solver::SecantSolver, FT) =
    SecantSolver(convert(FT, solver.reltol), convert(FT, solver.abstol), solver.maxiter)

materialize_solver(solver::FixedIterations, FT) = solver
materialize_solver(::Nothing, FT) = nothing

#####
##### Newton iteration driver
#####

# The two iteration forms below are NOT collapsible into one: the `while` form has a
# data-dependent trip count (fewest iterations on vanilla CPU/GPU), whereas the
# `FixedIterations` form has a compile-time trip count that unrolls to straight-line
# code (required for Reactant tracing and cheap reverse-mode differentiation).

"""
$(TYPEDSIGNATURES)

Solve `r(x) = 0` by Newton iteration from initial guess `x`, where
`residual_and_derivative(x)` returns the tuple `(r(x), r′(x))`.

The iteration is controlled by `solver`:

* `NewtonSolver`: iterate until `|Δx| ≤ max(abstol, reltol * |x|)` or `maxiter` is reached
* `FixedIterations`: perform exactly `iterations` Newton steps (no convergence test)
* `nothing`: return the initial guess `x` unmodified

```jldoctest
using Breeze
using Breeze.Solvers: newton_solve

solver = NewtonSolver(reltol=1e-12, maxiter=20)
x = newton_solve(x -> (x^2 - 2, 2x), solver, 1.0)
round(x, digits=10)

# output
1.4142135624
```
"""
@inline function newton_solve(residual_and_derivative, solver::NewtonSolver, x)
    Δx = x # guarantees the convergence criterion fails before the first step
    iter = 0
    while abs(Δx) > max(solver.abstol, solver.reltol * abs(x)) && iter < solver.maxiter
        r, r′ = residual_and_derivative(x)
        Δx = -r / r′
        x += Δx
        iter += 1
    end
    return x
end

@inline function newton_solve(residual_and_derivative, solver::FixedIterations, x)
    for _ in 1:solver.iterations
        r, r′ = residual_and_derivative(x)
        x -= r / r′
    end
    return x
end

@inline newton_solve(residual_and_derivative, ::Nothing, x) = x

#####
##### Secant iteration driver
#####

"""
$(TYPEDSIGNATURES)

Solve `r(x) = 0` by secant iteration from the initial guesses `x₁` and `x₂`, where
`residual(x)` returns `r(x)`. The convergence criterion compares the residual against
`scale`: iteration stops when `|r| ≤ max(abstol, reltol * |scale|)`.

The iteration is controlled by `solver`:

* `SecantSolver`: iterate until the residual converges or `maxiter` is reached
* `FixedIterations`: perform exactly `iterations` secant steps (no convergence test)

A degenerate step (`r₂ = r₁`, slope undefined) terminates a `SecantSolver` iteration
at the current iterate and leaves a `FixedIterations` iterate unchanged.

```jldoctest
using Breeze
using Breeze.Solvers: secant_solve

solver = SecantSolver(abstol=1e-12, maxiter=20)
x = secant_solve(x -> x^2 - 2, solver, 1.0, 2.0, 1.0)
round(x, digits=10)

# output
1.4142135624
```
"""
@inline function secant_solve(residual, solver::SecantSolver, x₁, x₂, scale)
    r₁ = residual(x₁)
    r₂ = residual(x₂)
    iter = 0
    while abs(r₂) > max(solver.abstol, solver.reltol * abs(scale)) && iter < solver.maxiter
        # Compute slope; guard against stagnation (r₂ = r₁ → division by zero).
        ΔxΔr = (x₂ - x₁) / (r₂ - r₁)
        valid_step = isfinite(ΔxΔr)
        ΔxΔr = ifelse(valid_step, ΔxΔr, zero(ΔxΔr))

        x₁, r₁ = x₂, r₂
        x₂ -= r₂ * ΔxΔr
        r₂ = residual(x₂)

        # Ensures the loop terminates naturally on the next header check instead of a 'break'
        r₂ = ifelse(valid_step, r₂, zero(r₂))
        iter += 1
    end
    return x₂
end

@inline function secant_solve(residual, solver::FixedIterations, x₁, x₂, scale)
    r₁ = residual(x₁)
    r₂ = residual(x₂)
    for _ in 1:solver.iterations
        ΔxΔr = (x₂ - x₁) / (r₂ - r₁)
        valid_step = isfinite(ΔxΔr)
        ΔxΔr = ifelse(valid_step, ΔxΔr, zero(ΔxΔr))

        x₁, r₁ = x₂, r₂
        x₂ -= r₂ * ΔxΔr
        r₂ = residual(x₂)
    end
    return x₂
end

#####
##### Show methods
#####

Base.summary(solver::NewtonSolver) =
    string("NewtonSolver(reltol=", prettysummary(solver.reltol),
           ", abstol=", prettysummary(solver.abstol),
           ", maxiter=", solver.maxiter, ")")

Base.summary(solver::SecantSolver) =
    string("SecantSolver(reltol=", prettysummary(solver.reltol),
           ", abstol=", prettysummary(solver.abstol),
           ", maxiter=", solver.maxiter, ")")

Base.summary(solver::FixedIterations) = string("FixedIterations(", solver.iterations, ")")

Base.show(io::IO, solver::Union{NewtonSolver, SecantSolver, FixedIterations}) =
    print(io, summary(solver))

end # module
