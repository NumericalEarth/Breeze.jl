# Applying a Reactant FFT plan to a concrete array recurses forever (OOM)

**Reactant 0.2.303** (also seen with 0.2.292), AbstractFFTs 1.5, Julia 1.12.6, CPU and GPU backends.

## Summary

`ReactantAbstractFFTsExt` defines `Base.:*(p::ReactantFFTInPlacePlan{T}, x::AnyTracedRArray{T})`
(and the same for the other plan types) only for *traced* arrays. When the plan is applied
outside a compiled function to a `ConcretePJRTArray`, dispatch falls through to AbstractFFTs'
generic fallback

```julia
*(p::Plan{T}, x::AbstractArray) where {T} = p * copy1(T, x)
```

`copy1` returns another `ConcretePJRTArray`, so the same method is called again, forever. Each
call allocates (and compiles a copy for) a new array; resident memory grows to tens of GB within
a minute, independent of the array size (2×2×2 behaves like 8×8×8), and the process is
eventually OOM-killed rather than hitting a `StackOverflowError`.

Even if the fallback returned, it would be wrong for in-place plans: `plan * x` would transform a
copy and leave `x` unchanged.

## Reproducer

```julia
using Reactant
using AbstractFFTs

const FFTExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

x = Reactant.to_rarray(ones(ComplexF64, 2, 2, 2))
plan = FFTExt.ReactantFFTInPlacePlan{ComplexF64}([1, 2, 3])

apply(plan, x) = (plan * x; nothing)
Reactant.@jit apply(plan, x)   # works: x[1] == 8 + 0im

which(*, Tuple{typeof(plan), typeof(x)})
# *(p::AbstractFFTs.Plan{T}, x::AbstractArray) where T @ AbstractFFTs .../definitions.jl:224

plan * x   # never returns; memory grows without bound (run under `ulimit -v`!)
```

## Where this bites

Oceananigans' `OceananigansReactantExt` builds these plans for `FFTBasedPoissonSolver` and
`FourierTridiagonalPoissonSolver` on `ReactantState` grids. Inside `@compile` everything works,
but any *eager* pressure solve hangs and exhausts memory, for example

```julia
using Oceananigans, Reactant
grid = RectilinearGrid(ReactantState(); size=(4, 4, 4), extent=(1, 1, 1))
model = NonhydrostaticModel(grid)
set!(model, u=(x, y, z) -> sin(2π * x))   # pressure projection → plan * storage → OOM
```

and building any anelastic `Breeze.AtmosphereModel` on `ReactantState` (the constructor calls
`set!`, which projects the velocity field). We first mis-attributed this to the MLIR pass pipeline
of the kernel launched just before the transform, because the stack sampled during the blow-up
is inside the compilation of one of the recursive `copy1` copies.

## Suggested fix

Give the Reactant plans methods for concrete arrays that compile the traced method, e.g.

```julia
for P in (ReactantFFTPlan, ReactantIFFTPlan, ReactantRFFTPlan, ReactantIRFFTPlan)
    @eval Base.:*(p::$P, x::Reactant.AnyConcreteRArray) = Reactant.@jit p * x
end
for P in (ReactantFFTInPlacePlan, ReactantIFFTInPlacePlan)
    @eval function Base.:*(p::$P, x::Reactant.AnyConcreteRArray)
        Reactant.@jit p * x   # mutates x
        return x
    end
end
# ...and LinearAlgebra.mul!(y, p, x) for concrete y, x
```

or, at least, throw an informative error instead of recursing.

<!-- Breeze workaround: NumericalEarth/Breeze.jl branch glw/reactant-anelastic,
     ext/BreezeReactantExt/pressure_solver.jl (jit the eager anelastic pressure solve). -->
