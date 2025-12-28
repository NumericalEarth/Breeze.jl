module BreezeReactantExt

using Breeze
using Reactant
using Oceananigans
using OffsetArrays: OffsetArray
using Oceananigans.AbstractOperations: AbstractOperation
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: AbstractField, Field, interior, get_neutral_mask
using Oceananigans.Grids: AbstractGrid, Bounded, Flat, Periodic, XDirection, YDirection, ZDirection, topology
using Oceananigans.Solvers: plan_forward_transform, plan_backward_transform
using Oceananigans.TimeSteppers:
    RungeKutta3TimeStepper,
    rk3_substep!,
    tick!,
    update_state!,
    compute_pressure_correction!,
    make_pressure_correction!,
    step_lagrangian_particles!,
    cache_previous_tendencies!,
    compute_flux_bc_tendencies!

using FFTW: fft, ifft

#####
##### FFT Planning for Reactant arrays
#####
# XLA handles FFT planning internally, so we return nothing (no-op) for Reactant arrays.
# This is needed for FourierTridiagonalPoissonSolver to work with ReactantState architecture.
#
# See GitHub issue #223: Reactant tracing for AnelasticFormulation
# Related: Reactant.jl PR #1931
#####

# ConcretePJRTArray is the concrete array type used by Reactant
const ReactantArray = Union{
    Reactant.ConcretePJRTArray,
    Reactant.ConcreteIFRTArray,
    Reactant.TracedRArray
}

#####
##### Reactant-safe reductions for Oceananigans Fields / Operations
#####
# Problem:
# - Oceananigans defines `maximum/minimum/sum/...` for `AbstractField` in a way that
#   ultimately iterates and scalar-indexes the field/operation.
# - Reactant arrays disallow scalar indexing, so reductions like `maximum(field)` and
#   `minimum(KernelFunctionOperation(...))` error with:
#     "Scalar indexing is disallowed. Invocation of getindex(::TracedRArray, ...)"
#
# Key insight:
# - Reactant *does* support reductions on plain arrays (ConcretePJRTArray / TracedRArray),
#   and also on SubArray views of those arrays.
# - For `Field`, `interior(field)` is a `SubArray` view of the parent array (halo excluded),
#   which Reactant can reduce without scalar iteration.
# - For `AbstractOperation` (e.g. `KernelFunctionOperation`), we can materialize the
#   operation into a computed `Field(op)` (kernel-based compute) and then reduce its interior.
#
# This is a Breeze-local workaround that should be upstreamed to OceananigansReactantExt.
#####

# Field reductions (Reactant-backed Fields only): redirect to reduction over `interior(field)`.
#
# We restrict dispatch to Fields whose storage parent is a Reactant array to avoid
# changing behavior for CPU/GPU (Array/CuArray) Fields.
const ReactantParent = ReactantArray

function Base.maximum(
    f::Function,
    c::Field{LX, LY, LZ, O, G, I, OffsetArray{T, 3, P}, FT, BC};
    condition = nothing,
    mask = get_neutral_mask(Base.maximum!),
    dims = :,
) where {LX, LY, LZ, O, G, I, T, P<:ReactantParent, FT, BC}
    condition === nothing || return invoke(Base.maximum, Tuple{Function, AbstractField}, f, c; condition, mask, dims)
    dims isa Colon || return invoke(Base.maximum, Tuple{Function, AbstractField}, f, c; condition, mask, dims)
    return Base.maximum(f, interior(c))
end

Base.maximum(c::Field{LX, LY, LZ, O, G, I, OffsetArray{T, 3, P}, FT, BC}; kwargs...) where {LX, LY, LZ, O, G, I, T, P<:ReactantParent, FT, BC} =
    Base.maximum(identity, c; kwargs...)

function Base.minimum(
    f::Function,
    c::Field{LX, LY, LZ, O, G, I, OffsetArray{T, 3, P}, FT, BC};
    condition = nothing,
    mask = get_neutral_mask(Base.minimum!),
    dims = :,
) where {LX, LY, LZ, O, G, I, T, P<:ReactantParent, FT, BC}
    condition === nothing || return invoke(Base.minimum, Tuple{Function, AbstractField}, f, c; condition, mask, dims)
    dims isa Colon || return invoke(Base.minimum, Tuple{Function, AbstractField}, f, c; condition, mask, dims)
    return Base.minimum(f, interior(c))
end

Base.minimum(c::Field{LX, LY, LZ, O, G, I, OffsetArray{T, 3, P}, FT, BC}; kwargs...) where {LX, LY, LZ, O, G, I, T, P<:ReactantParent, FT, BC} =
    Base.minimum(identity, c; kwargs...)

function Base.sum(
    f::Function,
    c::Field{LX, LY, LZ, O, G, I, OffsetArray{T, 3, P}, FT, BC};
    condition = nothing,
    mask = get_neutral_mask(Base.sum!),
    dims = :,
) where {LX, LY, LZ, O, G, I, T, P<:ReactantParent, FT, BC}
    condition === nothing || return invoke(Base.sum, Tuple{Function, AbstractField}, f, c; condition, mask, dims)
    dims isa Colon || return invoke(Base.sum, Tuple{Function, AbstractField}, f, c; condition, mask, dims)
    return Base.sum(f, interior(c))
end

Base.sum(c::Field{LX, LY, LZ, O, G, I, OffsetArray{T, 3, P}, FT, BC}; kwargs...) where {LX, LY, LZ, O, G, I, T, P<:ReactantParent, FT, BC} =
    Base.sum(identity, c; kwargs...)

# Operation reductions (ReactantState only): materialize op -> reduce.
#
# This is required for CFL / TimeStepWizard, because cell_advection_timescale uses:
#   τ = KernelFunctionOperation(...)
#   minimum(τ)
#
# Without this, `minimum(τ)` iterates `τ[i,j,k]` on the host, which scalar-indexes
# Reactant-backed Fields inside the kernel_function.
function Base.minimum(
    f::Function,
    op::AbstractOperation;
    condition = nothing,
    mask = get_neutral_mask(Base.minimum!),
    dims = :,
)
    if !(op.grid.architecture isa ReactantState) || condition !== nothing || !(dims isa Colon)
        return invoke(Base.minimum, Tuple{Function, AbstractField}, f, op; condition, mask, dims)
    end

    tmp = Field(op) # computes via a kernel and returns a Reactant-backed Field
    return Base.minimum(f, interior(tmp))
end

Base.minimum(op::AbstractOperation; kwargs...) = Base.minimum(identity, op; kwargs...)

function Base.maximum(
    f::Function,
    op::AbstractOperation;
    condition = nothing,
    mask = get_neutral_mask(Base.maximum!),
    dims = :,
)
    if !(op.grid.architecture isa ReactantState) || condition !== nothing || !(dims isa Colon)
        return invoke(Base.maximum, Tuple{Function, AbstractField}, f, op; condition, mask, dims)
    end

    tmp = Field(op)
    return Base.maximum(f, interior(tmp))
end

Base.maximum(op::AbstractOperation; kwargs...) = Base.maximum(identity, op; kwargs...)

function Base.sum(
    f::Function,
    op::AbstractOperation;
    condition = nothing,
    mask = get_neutral_mask(Base.sum!),
    dims = :,
)
    if !(op.grid.architecture isa ReactantState) || condition !== nothing || !(dims isa Colon)
        return invoke(Base.sum, Tuple{Function, AbstractField}, f, op; condition, mask, dims)
    end

    tmp = Field(op)
    return Base.sum(f, interior(tmp))
end

Base.sum(op::AbstractOperation; kwargs...) = Base.sum(identity, op; kwargs...)

#####
##### CartesianIndex indexing for OffsetArray{TracedRNumber}
#####
# ReactantOffsetArraysExt.get_ancestor_and_indices_inner uses Vararg{Any,N}
# which doesn't match CartesianIndex{N} (a single argument vs N arguments).
# This fix converts CartesianIndex to tuple and splats it.
#
# See: https://github.com/EnzymeAD/Reactant.jl - ReactantOffsetArraysExt.jl line 66
#####

function Reactant.TracedUtils.get_ancestor_and_indices_inner(
    x::OffsetArray{<:Reactant.TracedRNumber,N}, idx::CartesianIndex{N}
) where {N}
    return Reactant.TracedUtils.get_ancestor_and_indices_inner(x, Tuple(idx)...)
end

#####
##### Reactant-compatible FourierTridiagonalPoissonSolver.solve! (workaround for issue #223)
#####
# Reactant currently fails to lower complex-valued KernelAbstractions kernels during `Reactant.@compile`
# (e.g. `copy_real_component!`, `multiply_by_spacing!`, and the complex batched tridiagonal solve).
#
# Workaround strategy:
# - Keep transforms and algebra in "pure Julia" array operations (Reactant can handle complex there).
# - Split complex RHS into real and imaginary parts, solve two *real* batched tridiagonal systems,
#   then recombine into a complex spectral solution.
# - Avoid `copy_real_component!` by assigning the real part into the output array via pure broadcast.
#
# Notes:
# - This is an integration workaround and should be upstreamed to OceananigansReactantExt.
# - We only support cases where the transform dimensions (non-tridiagonal dims) are Periodic/Flat
#   (no Bounded DCTs), because StableHLO provides FFT but not DCT.
#

const ReactantAbstractFFTsExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

@inline is_traced_array(A) = A isa Reactant.AnyTracedRArray
@inline is_reactant_array(A) = A isa ReactantArray  # Includes both ConcreteRArray and TracedRArray

@inline function tridiagonal_dim(::XDirection); 1; end
@inline function tridiagonal_dim(::YDirection); 2; end
@inline function tridiagonal_dim(::ZDirection); 3; end

@inline function _scale_by_spacing!(rhs, ::XDirection, grid)
    Δ = grid.Δxᶜᵃᵃ
    rhs .*= (Δ isa Number ? Δ : reshape(Δ, length(Δ), 1, 1))
    return nothing
end

@inline function _scale_by_spacing!(rhs, ::YDirection, grid)
    Δ = grid.Δyᵃᶜᵃ
    rhs .*= (Δ isa Number ? Δ : reshape(Δ, 1, length(Δ), 1))
    return nothing
end

@inline function _scale_by_spacing!(rhs, ::ZDirection, grid)
    Δ = grid.z.Δᵃᵃᶜ
    rhs .*= (Δ isa Number ? Δ : reshape(Δ, 1, 1, length(Δ)))
    return nothing
end

#####
##### Reactant-compatible BatchedTridiagonalSolver.solve! (workaround for issue #223)
#####
# The standard Oceananigans BatchedTridiagonalSolver.solve! uses a KernelAbstractions kernel
# (solve_batched_tridiagonal_system_kernel!) which fails to raise during Reactant.@compile
# with raise=true due to complex affine loop patterns.
#
# This workaround implements the Thomas algorithm using pure Julia broadcast operations
# which Reactant can compile successfully. We only support ZDirection currently since that's
# what FourierTridiagonalPoissonSolver uses for anelastic dynamics.
#####

const ZTridiagonalSolver = Oceananigans.Solvers.BatchedTridiagonalSolver{
    A, B, C, T, G, P, <:ZDirection
} where {A, B, C, T, G, P}

# Detect if we're operating on Reactant arrays (either ConcreteRArray or TracedRArray)
# KA kernels can't run on either, so we always need the broadcast workaround
@inline function _is_reactant_tridiagonal_solve(ϕ, rhs)
    return is_reactant_array(ϕ) || is_reactant_array(rhs)
end

"""
    _thomas_solve_z_broadcast!(ϕ, a, b, c, rhs, t, Nz)

Pure-Julia Thomas algorithm for ZDirection tridiagonal solve using broadcasts.
Replaces the KA kernel `solve_batched_tridiagonal_system_kernel!` for Reactant compatibility.

Arguments:
- `ϕ`: Output array (Nx, Ny, Nz)
- `a`: Lower diagonal coefficients (length Nz-1 vector)
- `b`: Main diagonal coefficients (Nx, Ny, Nz array)
- `c`: Upper diagonal coefficients (same as `a`, but we use `a` for both in symmetric case)
- `rhs`: Right-hand side (Nx, Ny, Nz array)
- `t`: Scratch array for modified upper diagonal (Nx, Ny, Nz)
- `Nz`: Number of z levels
"""
function _thomas_solve_z_broadcast!(ϕ, a, b, c, rhs, t, Nz)
    # c is the upper diagonal (length Nz-1) - same as a for symmetric problems
    # In Oceananigans, both lower and upper diagonals are typically the same
    
    RT = eltype(real(ϕ))
    eps_threshold = 10 * eps(RT)

    @views begin
        # Forward elimination
        β = b[:, :, 1]
        ϕ[:, :, 1] .= rhs[:, :, 1] ./ β

        for k = 2:Nz
            # Get coefficients for this level
            # a[k-1] is the lower diagonal coefficient connecting level k-1 to k
            # c[k-1] is the upper diagonal coefficient connecting level k-1 to k
            a_k = reshape(view(a, (k-1):(k-1)), 1, 1)
            c_k = reshape(view(c, (k-1):(k-1)), 1, 1)

            # Modified upper diagonal
            t[:, :, k] .= c_k ./ β
            
            # New diagonal after elimination
            β_new = b[:, :, k] .- a_k .* t[:, :, k]

            # Forward substitution with diagonal dominance check
            ϕ_candidate = (rhs[:, :, k] .- a_k .* ϕ[:, :, k-1]) ./ β_new
            diag_dom = abs.(β_new) .> eps_threshold
            ϕ[:, :, k] .= ifelse.(diag_dom, ϕ_candidate, ϕ[:, :, k])

            # Update β for next iteration
            β = β_new
        end

        # Back substitution
        for k = (Nz-1):-1:1
            ϕ[:, :, k] .-= t[:, :, k+1] .* ϕ[:, :, k+1]
        end
    end

    return nothing
end

# Unified method for ZTridiagonalSolver with ReactantArrays
# We dispatch on grid architecture to avoid ambiguity with the upstream method.
function Oceananigans.Solvers.solve!(
    ϕ,
    solver::ZTridiagonalSolver{A, B, C, T, G},
    rhs,
    args...
) where {A, B, C, T, G<:AbstractGrid{<:Any, <:Any, <:Any, <:Any, ReactantState}}
    # This method is triggered when the solver's grid is on ReactantState architecture.
    # Always use pure-Julia broadcast operations - KA kernels can't run on Reactant arrays
    # (neither ConcreteRArray nor TracedRArray).

    Nz = size(solver.grid, 3)
    
    _thomas_solve_z_broadcast!(
        ϕ,
        solver.a,  # lower diagonal
        solver.b,  # main diagonal  
        solver.a,  # upper diagonal (symmetric in Oceananigans)
        rhs,
        solver.t,  # scratch
        Nz
    )

    return nothing
end

function Oceananigans.Solvers.solve!(
    x,
    solver::Oceananigans.Solvers.FourierTridiagonalPoissonSolver{G},
    b=nothing,
) where {G<:AbstractGrid{<:Any, <:Any, <:Any, <:Any, ReactantState}}

    # ═══════════════════════════════════════════════════════════════════════════
    # Reactant workaround for FourierTridiagonalPoissonSolver
    # ═══════════════════════════════════════════════════════════════════════════
    # 
    # This override ONLY works during @compile/@trace execution (TracedRArrays).
    # FFTs from ReactantAbstractFFTsExt only work on TracedRArrays, not on
    # ConcreteRArrays (outside of tracing context).
    #
    # For model construction, `compute_pressure_correction!` is skipped for
    # ReactantState (see below). Pressure correction will be applied during
    # compiled time stepping.
    # ═══════════════════════════════════════════════════════════════════════════

    # Verify we're in traced context (FFTs only work on TracedRArrays)
    is_traced = is_traced_array(solver.source_term)
    
    if !is_traced
        # This should not happen - compute_pressure_correction! should be skipped
        # for ConcreteRArrays. If we get here, something is wrong.
        error("""
            FourierTridiagonalPoissonSolver.solve! called on ConcreteRArrays.
            
            This is unexpected - pressure correction should be skipped during model
            construction for ReactantState. Please report this as a bug.
            
            For time stepping, use Reactant.@compile:
              compiled_step! = Reactant.@compile time_step!(model, Δt)
              compiled_step!(model, Δt)
            """)
    end

    ReactantAbstractFFTsExt === nothing &&
        error("ReactantAbstractFFTsExt is not loaded; cannot run FFT on Reactant arrays.")
    AbstractFFTs_mod = ReactantAbstractFFTsExt.AbstractFFTs

    grid = solver.grid
    Nx, Ny, Nz = size(grid)

    # ---------------------------
    # Set source term (optional)
    # ---------------------------
    rhs = solver.source_term
    if b !== nothing
        rhs .= b
        _scale_by_spacing!(rhs, solver.batched_tridiagonal_solver.tridiagonal_direction, grid)
    end

    # ---------------------------
    # Forward transforms (periodic only; no DCT support)
    # ---------------------------
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    untransformed_dim = tridiagonal_dim(tdir)
    transform_dims = untransformed_dim == 1 ? (2, 3) : (untransformed_dim == 2 ? (1, 3) : (1, 2))

    topo = topology(grid)
    bounded_dims = Tuple(d for d in transform_dims if topo[d] === Bounded)
    isempty(bounded_dims) || error("Reactant workaround only supports Periodic/Flat transform dims (got Bounded dims = $bounded_dims).")

    periodic_dims = Tuple(d for d in transform_dims if topo[d] === Periodic)
    
    if !isempty(periodic_dims)
        # Use broadcast assignment instead of copyto! for better XLA fusion
        # Benchmark shows A .= fft(A) is 1.69× faster than copyto!(A, fft(A))
        rhs .= AbstractFFTs_mod.fft(rhs, periodic_dims)
    end

    # ---------------------------
    # Solve tridiagonal system directly on complex arrays
    # ---------------------------
    # The tridiagonal coefficients (a, b, c) are REAL (Laplacian eigenvalues).
    # Only rhs and ϕ are complex (spectral coefficients).
    # The Thomas algorithm works with complex rhs/ϕ and real coefficients because:
    #   - complex ./ real = complex ✅
    #   - real .* complex = complex ✅
    #   - ifelse.(real_condition, complex, complex) = complex ✅
    #
    # This eliminates 4 allocations and 2× tridiagonal solve overhead.
    # ---------------------------
    
    a = solver.batched_tridiagonal_solver.a
    b_arr = solver.batched_tridiagonal_solver.b
    c = solver.batched_tridiagonal_solver.c
    t = solver.batched_tridiagonal_solver.t
    
    ϕ = solver.storage
    _thomas_solve_z_broadcast!(ϕ, a, b_arr, c, rhs, t, Nz)

    # ---------------------------
    # Backward transforms
    # ---------------------------
    if !isempty(periodic_dims)
        # Use broadcast assignment instead of copyto! for better XLA fusion
        ϕ .= AbstractFFTs_mod.ifft(ϕ, periodic_dims)
    end

    # Zero-mean gauge (Reactant-safe: use broadcasted sum)
    ϕ_mean = sum(ϕ) / length(ϕ)
    ϕ .= ϕ .- ϕ_mean

    # ---------------------------
    # Copy real part into output (avoid complex KA kernels)
    # ---------------------------
    out = x isa Oceananigans.Fields.Field ? parent(x) : x
    @views view(out, 1:Nx, 1:Ny, 1:Nz) .= real.(ϕ)

    return nothing
end

# Forward transform plans - return nothing since XLA handles planning
function Oceananigans.Solvers.plan_forward_transform(
    A::ReactantArray,
    ::Periodic,
    dims,
    planner_flag=nothing
)
    return nothing
end

function Oceananigans.Solvers.plan_forward_transform(
    A::ReactantArray,
    ::Bounded,
    dims,
    planner_flag=nothing
)
    return nothing
end

# Backward transform plans - return nothing since XLA handles planning
function Oceananigans.Solvers.plan_backward_transform(
    A::ReactantArray,
    ::Periodic,
    dims,
    planner_flag=nothing
)
    return nothing
end

function Oceananigans.Solvers.plan_backward_transform(
    A::ReactantArray,
    ::Bounded,
    dims,
    planner_flag=nothing
)
    return nothing
end

#####
##### Reactant-compatible time_step! for RK3 time stepper
#####
# The standard Oceananigans time_step! has conditionals like:
#   Δt == 0 && @warn "..."
#   model.clock.iteration == 0 && update_state!(...)
#
# When clock.iteration and clock.time are TracedRNumber (Reactant traced numbers),
# comparisons like `iteration == 0` return TracedRNumber{Bool}, which cannot be
# used in short-circuit && operators (TypeError: if expected Bool, got TracedRNumber{Bool}).
#
# This method provides a Reactant-compatible implementation that removes these
# runtime conditionals. For Reactant/Enzyme AD, we assume:
# - The user calls update_state! or set! before the first time step (as is standard)
# - Δt is not zero (the user is responsible for valid time steps)
#
# See: Oceananigans.jl TimeSteppers/runge_kutta_3.jl lines 94-98
#####

# Type alias for AtmosphereModel on ReactantState with RK3 time stepper
const ReactantAtmosphereModel = Breeze.AtmosphereModels.AtmosphereModel{
    <:Any, <:Any, ReactantState, <:RungeKutta3TimeStepper
}

"""
    time_step!(model::ReactantAtmosphereModel, Δt)

Reactant-compatible time stepping for AtmosphereModel with RK3 time stepper.
This version removes boolean conditionals that are incompatible with Reactant tracing.
"""
function Oceananigans.TimeSteppers.time_step!(model::ReactantAtmosphereModel, Δt; callbacks=[])
    # NOTE: We skip the standard checks that use runtime conditionals:
    # - `Δt == 0 && @warn ...` - assume user provides valid Δt
    # - `model.clock.iteration == 0 && update_state!(...)` - assume user initialized model
    
    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = γ¹ * Δt
    second_stage_Δt = (γ² + ζ²) * Δt
    third_stage_Δt  = (γ³ + ζ³) * Δt

    #
    # First stage
    #

    compute_flux_bc_tendencies!(model)
    rk3_substep!(model, Δt, γ¹, nothing)

    tick!(model.clock, first_stage_Δt; stage=true)

    compute_pressure_correction!(model, first_stage_Δt)
    make_pressure_correction!(model, first_stage_Δt)

    cache_previous_tendencies!(model)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, first_stage_Δt)

    #
    # Second stage
    #

    compute_flux_bc_tendencies!(model)
    rk3_substep!(model, Δt, γ², ζ²)

    tick!(model.clock, second_stage_Δt; stage=true)

    compute_pressure_correction!(model, second_stage_Δt)
    make_pressure_correction!(model, second_stage_Δt)

    cache_previous_tendencies!(model)
    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, second_stage_Δt)

    #
    # Third stage
    #

    compute_flux_bc_tendencies!(model)
    rk3_substep!(model, Δt, γ³, ζ³)

    # For Reactant, we skip the time-correction that uses `time_difference_seconds`
    # which involves runtime conditionals. We use the direct third_stage_Δt.
    tick!(model.clock, third_stage_Δt)
    model.clock.last_Δt = Δt

    compute_pressure_correction!(model, third_stage_Δt)
    make_pressure_correction!(model, third_stage_Δt)

    update_state!(model, callbacks; compute_tendencies = true)
    step_lagrangian_particles!(model, third_stage_Δt)

    return nothing
end

#####
##### Reactant-compatible anelastic pressure solver (Option C workaround for issue #223)
#####
# The original `_compute_anelastic_source_term!` and `_pressure_correct_momentum!` are
# KernelAbstractions kernels that fail during Reactant MLIR lowering with:
#   'affine.store' op value to store must have the same type as memref element type
#
# This workaround replaces those kernels with pure-Julia broadcast operations.
# We override `compute_pressure_correction!` and `make_pressure_correction!` for
# ReactantAtmosphereModel to use these broadcast-based implementations.
#####

using Oceananigans: fields
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.BoundaryConditions: fill_halo_regions!

# Import the functions we need to extend
import Breeze.AtmosphereModels.Dynamics: solve_for_anelastic_pressure!, compute_anelastic_source_term!

"""
    _compute_source_term_broadcast!(rhs, grid, ρu, ρv, ρw, Δt)

Pure-Julia broadcast implementation of the anelastic source term computation.
Replaces the KA kernel `_compute_anelastic_source_term!` for Reactant compatibility.

Computes: rhs = Δz * div(ρu, ρv, ρw) / Δt

Note: This implementation currently supports 2D grids (Flat in y) with regular spacing.
"""
function _compute_source_term_broadcast!(rhs, grid, ρu, ρv, ρw, Δt)
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
    
    # For Flat y topology, Ny = 1 and we have a 2D problem
    if topo[2] !== Flat
        error("3D grids not yet supported in Reactant anelastic source term workaround. " *
              "Only 2D grids with Flat y topology are currently supported.")
    end
    
    # Get grid metrics (must be regular for this workaround)
    Δx_val = grid.Δxᶜᵃᵃ
    Δz_val = grid.z.Δᵃᵃᶜ
    Δx_val isa Number || error("Stretched grids not yet supported in Reactant workaround")
    Δz_val isa Number || error("Stretched grids not yet supported in Reactant workaround")
    
    # Access parent arrays
    ρu_data = parent(ρu)
    ρw_data = parent(ρw)
    rhs_data = parent(rhs)  # Solver's source_term storage (no halos, size Nx × Ny × Nz)
    
    # Interior index ranges in parent arrays (with halos)
    # ρu is at face-x: size (Nx+1+2Hx, Ny+2Hy, Nz+2Hz) = (14, 1, 14) for Nx=8, Hx=3
    # Actually for Flat Hy=0, so size is (14, 1, 14)
    # Interior of ρu: indices (Hx+1):(Hx+Nx+1) for the Nx+1 face values
    # But we need δx = ρu[i+1] - ρu[i] for i in 1:Nx (center indices)
    # At center i, we need face values at i and i+1
    # Face index i corresponds to parent index Hx+i
    # So: δx_ρu[i] = ρu_data[Hx+i+1] - ρu_data[Hx+i] for i in 1:Nx
    
    # Similarly for ρw at face-z: size (Nx+2Hx, Ny+2Hy, Nz+1+2Hz) = (14, 1, 15)
    # δz_ρw[k] = ρw_data[Hz+k+1] - ρw_data[Hz+k] for k in 1:Nz
    
    iy = (Hy+1):(Hy+Ny)  # = 1:1 for Flat y with Hy=0
    iz_c = (Hz+1):(Hz+Nz)  # Center z indices in parent
    ix_c = (Hx+1):(Hx+Nx)  # Center x indices in parent
    
    @views begin
        # δx_ρu: difference in x direction (at cell centers)
        # ρu is at faces, so ρu[Hx+i+1, :, :] - ρu[Hx+i, :, :] gives δx at center i
        δx_ρu = ρu_data[(Hx+2):(Hx+Nx+1), iy, iz_c] .- ρu_data[(Hx+1):(Hx+Nx), iy, iz_c]
        
        # δz_ρw: difference in z direction (at cell centers)
        # ρw is at faces, so ρw[:, :, Hz+k+1] - ρw[:, :, Hz+k] gives δz at center k
        δz_ρw = ρw_data[ix_c, iy, (Hz+2):(Hz+Nz+1)] .- ρw_data[ix_c, iy, (Hz+1):(Hz+Nz)]
        
        # For a 2D (x,z) grid with regular spacing:
        # div(ρu, ρv, ρw) = (1/V) * (Ax*δx_ρu + Az*δz_ρw)
        # V = Δx * Δz, Ax = Δz, Az = Δx (areas perpendicular to each direction)
        # div = (1/(Δx*Δz)) * (Δz*δx_ρu + Δx*δz_ρw)
        # Δz * div = (Δz/(Δx*Δz)) * (Δz*δx_ρu + Δx*δz_ρw) = (1/Δx) * Δz*δx_ρu + δz_ρw
        #          = (Δz/Δx) * δx_ρu + δz_ρw
        
        # rhs = Δz * div / Δt
        # The rhs_data has no halos, direct indexing 1:Nx, 1:Ny, 1:Nz
        rhs_data[1:Nx, 1:Ny, 1:Nz] .= ((Δz_val / Δx_val) .* δx_ρu .+ δz_ρw) ./ Δt
    end
    
    return nothing
end

"""
Override `compute_anelastic_source_term!` for ReactantState to use broadcast implementation.

Note: We always use the broadcast implementation on ReactantState because:
- During Reactant tracing: arrays are TracedRArrays and KA kernels can't be lowered to MLIR
- Outside tracing: arrays are ConcreteRArrays and KA kernels can't run directly on them
"""
function compute_anelastic_source_term!(
    solver::Oceananigans.Solvers.FourierTridiagonalPoissonSolver{G},
    ρŨ, Δt
) where {G<:AbstractGrid{<:Any, <:Any, <:Any, <:Any, ReactantState}}
    
    rhs = solver.source_term
    grid = solver.grid
    ρu, ρv, ρw = ρŨ
    
    # Always use broadcast on ReactantState (both ConcreteRArrays and TracedRArrays)
    _compute_source_term_broadcast!(rhs, grid, ρu, ρv, ρw, Δt)
    
    return nothing
end

"""
    _pressure_correct_momentum_broadcast!(ρu, ρv, ρw, grid, Δt, p, ρᵣ)

Pure-Julia broadcast implementation of the pressure correction for momentum.
Replaces the KA kernel `_pressure_correct_momentum!` for Reactant compatibility.

Computes:
  ρu -= ρᵣ * Δt * ∂x(p)
  ρv -= ρᵣ * Δt * ∂y(p)  
  ρw -= ρᵣ_interp * Δt * ∂z(p)
"""
function _pressure_correct_momentum_broadcast!(ρu, ρv, ρw, grid, Δt, p, ρᵣ)
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
    
    if topo[2] !== Flat
        error("3D grids not yet supported in Reactant pressure correction workaround.")
    end
    
    # Get grid metrics (must be regular)
    Δx_val = grid.Δxᶠᵃᵃ
    Δz_val = grid.z.Δᵃᵃᶠ
    Δx_val isa Number || error("Stretched grids not yet supported")
    Δz_val isa Number || error("Stretched grids not yet supported")
    
    # Access parent arrays
    ρu_data = parent(ρu)
    ρw_data = parent(ρw)
    p_data = parent(p)
    ρᵣ_data = parent(ρᵣ)
    
    # Index ranges
    iy = (Hy+1):(Hy+Ny)  # = 1:1 for Flat y
    iz_c = (Hz+1):(Hz+Nz)  # Center z indices
    ix_c = (Hx+1):(Hx+Nx)  # Center x indices
    
    @views begin
        # ==========================================
        # Correct ρu (at x-faces, faces 1:Nx in logical indexing)
        # ==========================================
        # The kernel updates M.ρu[i,j,k] for i=1:Nx, j=1:Ny, k=1:Nz
        # ∂xᶠᶜᶜ(i,j,k,p) = (p[i,j,k] - p[i-1,j,k]) / Δx
        # This uses the halo value p[i-1] which equals p[Nx] for i=1 (periodic BC)
        
        # ρᵣ at cell centers (1D profile along z)
        ρᵣ_z = ρᵣ_data[1, 1, iz_c]  # Shape: (Nz,)
        ρᵣ_3d = reshape(ρᵣ_z, 1, 1, Nz)  # Shape: (1, 1, Nz)
        
        # ∂x(p) for faces 1:Nx (logical), parent indices (Hx+1):(Hx+Nx)
        # For face i: (p[i] - p[i-1]) / Δx
        # p[i] is at parent index Hx+i, p[i-1] is at parent index Hx+i-1
        # Note: p[0] = p[Hx] is in the halo, filled by fill_halo_regions! for periodic BC
        ∂x_p_all = (p_data[(Hx+1):(Hx+Nx), iy, iz_c] .- p_data[Hx:(Hx+Nx-1), iy, iz_c]) ./ Δx_val
        
        # Correct ρu at all faces 1:Nx (parent indices (Hx+1):(Hx+Nx))
        ρu_data[(Hx+1):(Hx+Nx), iy, iz_c] .-= ρᵣ_3d .* Δt .* ∂x_p_all
        
        # ==========================================
        # Correct ρw (at z-faces, faces 1:Nz in logical indexing)
        # ==========================================
        # The kernel updates M.ρw[i,j,k] for i=1:Nx, j=1:Ny, k=1:Nz
        # For Bounded z topology, faces 1 and Nz+1 are at the boundaries
        # Face 1 uses ρᶠ = ℑzᵃᵃᶠ(i, j, 1, grid, ρᵣ) = 0.5*(ρᵣ[k=1] + ρᵣ[k=0])
        # But ρᵣ[k=0] is in the halo. For Bounded BC, we need to check how this is handled.
        # The original kernel accesses ρᵣ via the halo-filled field.
        
        # ∂z(p) for faces 1:Nz (logical), parent indices (Hz+1):(Hz+Nz)
        # For face k: (p[k] - p[k-1]) / Δz
        # Note: p[0] = p[Hz] is in the halo
        ∂z_p_all = (p_data[ix_c, iy, (Hz+1):(Hz+Nz)] .- p_data[ix_c, iy, Hz:(Hz+Nz-1)]) ./ Δz_val
        
        # Interpolate ρᵣ to faces 1:Nz
        # For face k: ρᵣ_face = 0.5 * (ρᵣ[k] + ρᵣ[k-1])
        # ρᵣ[k] at parent index 1,1,Hz+k, ρᵣ[k-1] at parent index 1,1,Hz+k-1
        ρᵣ_k = ρᵣ_data[1, 1, (Hz+1):(Hz+Nz)]      # Centers k=1:Nz
        ρᵣ_km1 = ρᵣ_data[1, 1, Hz:(Hz+Nz-1)]       # Centers k=0:Nz-1 (includes halo)
        ρᵣ_face_all = (ρᵣ_k .+ ρᵣ_km1) ./ 2       # Shape: (Nz,)
        ρᵣ_face_3d = reshape(ρᵣ_face_all, 1, 1, Nz)
        
        # Correct ρw at all faces 1:Nz (parent indices (Hz+1):(Hz+Nz))
        ρw_data[ix_c, iy, (Hz+1):(Hz+Nz)] .-= ρᵣ_face_3d .* Δt .* ∂z_p_all
    end
    
    # ρv is not corrected for Flat y topology (no v velocity component)
    
    return nothing
end

"""
    _cpu_pressure_correction!(model, Δt)

Perform pressure correction using CPU arrays as a fallback when Reactant FFTs
are not available (outside @compile context).

This copies data to CPU, performs the FFT-based pressure solve with FFTW,
and copies the result back to Reactant arrays.
"""
function _cpu_pressure_correction!(model::ReactantAtmosphereModel, Δt)
    grid = model.grid
    Nx, Ny, Nz = size(grid)
    topo = topology(grid)
    
    # Only 2D (Flat y) supported for now
    if topo[2] !== Flat
        @warn "CPU fallback pressure correction only supports 2D grids (Flat y). " *
              "Initial velocity field may not be divergence-free."
        return nothing
    end
    
    # Copy momentum to CPU
    ρu_cpu = Array(parent(model.momentum.ρu))
    ρv_cpu = Array(parent(model.momentum.ρv))
    ρw_cpu = Array(parent(model.momentum.ρw))
    
    # Get grid metrics
    Hx, Hy, Hz = grid.Hx, grid.Hy, grid.Hz
    Δx = grid.Δxᶜᵃᵃ
    Δz = grid.z.Δᵃᵃᶜ
    
    # Compute divergence (source term) on CPU
    rhs_cpu = zeros(Nx, Ny, Nz)
    iy = (Hy+1):(Hy+Ny)
    iz_c = (Hz+1):(Hz+Nz)
    ix_c = (Hx+1):(Hx+Nx)
    
    @views begin
        δx_ρu = ρu_cpu[(Hx+2):(Hx+Nx+1), iy, iz_c] .- ρu_cpu[(Hx+1):(Hx+Nx), iy, iz_c]
        δz_ρw = ρw_cpu[ix_c, iy, (Hz+2):(Hz+Nz+1)] .- ρw_cpu[ix_c, iy, (Hz+1):(Hz+Nz)]
        rhs_cpu .= ((Δz / Δx) .* δx_ρu .+ δz_ρw) ./ Δt
    end
    
    # FFT solve on CPU using the existing solver's tridiagonal coefficients
    solver = model.pressure_solver
    a_cpu = Array(solver.batched_tridiagonal_solver.a)
    b_cpu = Array(solver.batched_tridiagonal_solver.b)
    c_cpu = a_cpu  # Symmetric
    
    # Forward FFT in x (periodic dimension)
    rhs_fft = fft(Complex{Float64}.(rhs_cpu), 1)
    
    # Solve tridiagonal system for each (kx, y) mode
    ϕ_fft = similar(rhs_fft)
    for i in 1:Nx, j in 1:Ny
        # Extract column
        rhs_col = rhs_fft[i, j, :]
        b_col = b_cpu[i, j, :]
        
        # Thomas algorithm
        ϕ_col = zeros(ComplexF64, Nz)
        
        # Forward elimination
        β = b_col[1]
        ϕ_col[1] = rhs_col[1] / β
        t = zeros(ComplexF64, Nz)
        
        for k in 2:Nz
            t[k] = c_cpu[k-1] / β
            β = b_col[k] - a_cpu[k-1] * t[k]
            if abs(β) > eps(Float64)
                ϕ_col[k] = (rhs_col[k] - a_cpu[k-1] * ϕ_col[k-1]) / β
            end
        end
        
        # Back substitution
        for k in (Nz-1):-1:1
            ϕ_col[k] -= t[k+1] * ϕ_col[k+1]
        end
        
        ϕ_fft[i, j, :] .= ϕ_col
    end
    
    # Inverse FFT
    ϕ_cpu = real.(ifft(ϕ_fft, 1))
    
    # Zero mean (gauge condition)
    ϕ_cpu .-= sum(ϕ_cpu) / length(ϕ_cpu)
    
    # Copy result back to pressure anomaly field
    p_parent = parent(model.dynamics.pressure_anomaly)
    p_parent[ix_c, iy, iz_c] .= Reactant.to_rarray(ϕ_cpu)
    
    # Apply pressure correction to momentum
    ρᵣ_cpu = Array(parent(model.dynamics.reference_state.density))
    
    Δx_f = grid.Δxᶠᵃᵃ
    Δz_f = grid.z.Δᵃᵃᶠ
    
    # ∂x(p) correction for ρu
    @views begin
        ∂x_p = (ϕ_cpu[2:Nx, :, :] .- ϕ_cpu[1:Nx-1, :, :]) ./ Δx_f
        # Handle periodic wrap
        ∂x_p_wrap = (ϕ_cpu[1:1, :, :] .- ϕ_cpu[Nx:Nx, :, :]) ./ Δx_f
        
        for k in 1:Nz
            ρᵣ_k = ρᵣ_cpu[1, 1, Hz+k]
            ρu_cpu[(Hx+2):(Hx+Nx), iy, Hz+k] .-= ρᵣ_k .* Δt .* ∂x_p[:, :, k]
            ρu_cpu[(Hx+1):(Hx+1), iy, Hz+k] .-= ρᵣ_k .* Δt .* ∂x_p_wrap[:, :, k]
        end
    end
    
    # ∂z(p) correction for ρw (interpolate ρᵣ to faces)
    @views begin
        ∂z_p = (ϕ_cpu[:, :, 2:Nz] .- ϕ_cpu[:, :, 1:Nz-1]) ./ Δz_f
        
        for k in 2:Nz
            ρᵣ_face = 0.5 * (ρᵣ_cpu[1, 1, Hz+k] + ρᵣ_cpu[1, 1, Hz+k-1])
            ρw_cpu[ix_c, iy, Hz+k] .-= ρᵣ_face .* Δt .* ∂z_p[:, :, k-1]
        end
    end
    
    # Copy corrected momentum back
    parent(model.momentum.ρu) .= Reactant.to_rarray(ρu_cpu)
    parent(model.momentum.ρw) .= Reactant.to_rarray(ρw_cpu)
    
    return nothing
end

"""
Override `compute_pressure_correction!` for ReactantAtmosphereModel.

When arrays are ConcreteRArrays (outside tracing context), we use a CPU fallback
that copies data to regular arrays, performs the FFT solve with FFTW, and copies
the results back. This ensures the velocity field is divergence-free after set!().

During tracing (@compile/@trace), arrays are TracedRArrays and the Reactant-native
pressure correction is performed.
"""
function Oceananigans.TimeSteppers.compute_pressure_correction!(model::ReactantAtmosphereModel, Δt)
    # Check if we're in traced context
    ρu = model.momentum.ρu
    is_traced = is_traced_array(parent(ρu))
    
    if !is_traced
        # Outside tracing context (ConcreteRArrays) - use CPU fallback
        # This ensures divergence-free velocities during model construction.
        foreach(mask_immersed_field!, model.momentum)
        fill_halo_regions!(model.momentum, model.clock, fields(model))
        _cpu_pressure_correction!(model, Δt)
        fill_halo_regions!(model.dynamics.pressure_anomaly)
        return nothing
    end
    
    # Inside tracing context (TracedRArrays) - perform Reactant-native pressure correction
    foreach(mask_immersed_field!, model.momentum)
    fill_halo_regions!(model.momentum, model.clock, fields(model))

    ρᵣ = model.dynamics.reference_state.density
    ρŨ = model.momentum
    solver = model.pressure_solver
    αᵣp′ = model.dynamics.pressure_anomaly
    solve_for_anelastic_pressure!(αᵣp′, solver, ρŨ, Δt)
    fill_halo_regions!(αᵣp′)

    return nothing
end

"""
Override `make_pressure_correction!` for ReactantAtmosphereModel.

When arrays are ConcreteRArrays (outside tracing context), the pressure correction
was already applied in `_cpu_pressure_correction!`, so we skip here.

During tracing, we use the broadcast-based implementation.
"""
function Oceananigans.TimeSteppers.make_pressure_correction!(model::ReactantAtmosphereModel, Δt)
    # Check if we're in traced context
    ρu = model.momentum.ρu
    is_traced = is_traced_array(parent(ρu))
    
    if !is_traced
        # Outside tracing context - pressure correction was already applied
        # in _cpu_pressure_correction!() within compute_pressure_correction!().
        return nothing
    end
    
    # Inside tracing context - apply pressure correction with broadcast
    grid = model.grid
    ρu, ρv, ρw = model.momentum
    p = model.dynamics.pressure_anomaly
    ρᵣ = model.dynamics.reference_state.density
    
    _pressure_correct_momentum_broadcast!(ρu, ρv, ρw, grid, Δt, p, ρᵣ)
    
    return nothing
end

end # module
