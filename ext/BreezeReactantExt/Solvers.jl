using Reactant: Reactant
using Oceananigans.Solvers: Solvers, FourierTridiagonalPoissonSolver

const EagerReactantFourierTridiagonalPoissonSolver =
    FourierTridiagonalPoissonSolver{<:Any, <:Any, <:Any, <:Any, <:Any, <:ConcreteReactantArray}

# TODO Oceananigans: move this method into OceananigansReactantExt. `AtmosphereModel` with
# `AnelasticDynamics` projects its initial momentum with the FFT pressure solver outside of
# any trace, where two operations fail on concrete Reactant arrays:
#
# - Reactant's FFT plans act only on traced arrays. Applied to a concrete array,
#   AbstractFFTs' fallback copies it into an `Array` and recurses until the stack overflows.
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

# Select a complex value component by component. A complex `ifelse` lowers to an
# `llvm.select` on an aggregate, which Reactant cannot raise to StableHLO
# (https://github.com/EnzymeAD/Reactant.jl/issues/3475).
@inline select_components(condition, a::Complex, b::Complex) =
    complex(ifelse(condition, real(a), real(b)), ifelse(condition, imag(a), imag(b)))

# TODO Oceananigans: replace the complex `ifelse` in `solve_batched_tridiagonal_system_z!`
# with a componentwise selection and delete this copy. The Fourier-tridiagonal anelastic
# pressure solver solves complex vertical systems, so its kernel cannot be raised otherwise.
# Apart from that selection, this method is identical to Oceananigans'.
@inline function Solvers.solve_batched_tridiagonal_system_z!(i, j, Nz, ϕ::AbstractArray{<:Complex}, a, b, c, f, t,
                                                              grid, p, args, tridiagonal_direction)
    @inbounds begin
        β  = Solvers.get_coefficient(i, j, 1, grid, b, p, tridiagonal_direction, args...)
        f₁ = Solvers.get_coefficient(i, j, 1, grid, f, p, tridiagonal_direction, args...)
        ϕ[i, j, 1] = f₁ / β

        for k = 2:Nz
            cᵏ⁻¹ = Solvers.get_coefficient(i, j, k-1, grid, c, p, tridiagonal_direction, args...)
            bᵏ   = Solvers.get_coefficient(i, j, k,   grid, b, p, tridiagonal_direction, args...)
            aᵏ⁻¹ = Solvers.get_coefficient(i, j, k-1, grid, a, p, tridiagonal_direction, args...)

            t[i, j, k] = cᵏ⁻¹ / β
            β = bᵏ - Solvers.masked_multiply(aᵏ⁻¹, t[i, j, k])
            fᵏ = Solvers.get_coefficient(i, j, k, grid, f, p, tridiagonal_direction, args...)

            # Elide the forward update where `β ≈ 0` and the system is not diagonally dominant.
            definitely_diagonally_dominant = abs(β) > 10 * eps(Solvers.float_eltype(ϕ))
            ϕ★ = (fᵏ - Solvers.masked_multiply(aᵏ⁻¹, ϕ[i, j, k-1])) / β
            ϕ[i, j, k] = select_components(definitely_diagonally_dominant, ϕ★, ϕ[i, j, k])
        end

        for k = Nz-1:-1:1
            ϕ[i, j, k] -= Solvers.masked_multiply(t[i, j, k+1], ϕ[i, j, k+1])
        end
    end
end
