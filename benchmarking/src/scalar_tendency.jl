#####
##### Scalar tendency kernel benchmark case (no model)
#####
##### Builds the bare core of Breeze's `compute_scalar_tendency!`: a single
##### kernel that writes the advective tendency Gc = -∇·(𝐮 c) for a materialized
##### WENO scheme, with no `AtmosphereModel`. Used to profile the cost of WENO
##### advection in isolation across a range of orders.
#####

using Oceananigans
using Oceananigans.Advection: div_Uc, materialize_advection
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using KernelAbstractions: @kernel, @index

@kernel function compute_scalar_tendency!(Gc, grid, advection, U, c)
    i, j, k = @index(Global, NTuple)
    @inbounds Gc[i, j, k] = -div_Uc(i, j, k, grid, advection, U, c)
end

# Launch wrapper; Reactant lowers this into a single XLA program.
function scalar_tendency!(Gc, grid, advection, U, c)
    launch!(architecture(grid), grid, :xyz, compute_scalar_tendency!, Gc, grid, advection, U, c)
    return nothing
end

"""
    scalar_tendency_problem(arch; Nx, Ny, Nz, order,
                            float_type = Float32,
                            topology = (Periodic, Periodic, Bounded))

Build the scalar-tendency workload on `arch` for a grid of size `(Nx, Ny, Nz)`
and a `WENO` scheme of the given `order`, materialized onto the grid exactly as
`AtmosphereModel` does. The halo is sized to the WENO stencil so any order up to
the requested one fits.

Returns `(scalar_tendency!, (Gc, grid, advection, U, c), grid)`, ready to hand
to `benchmark_tendency` or `Reactant.@compile`.
"""
function scalar_tendency_problem(arch;
                                 Nx, Ny, Nz,
                                 order,
                                 float_type = Float32,
                                 topology = (Periodic, Periodic, Bounded))

    Oceananigans.defaults.FloatType = float_type

    # WENO of order `order` reads (order + 1) ÷ 2 cells on each side.
    h = cld(order + 1, 2)
    grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(1, 1, 1), halo=(h, h, h), topology)

    advection = materialize_advection(WENO(float_type; order), grid)

    U  = (u = XFaceField(grid), v = YFaceField(grid), w = ZFaceField(grid))
    c  = CenterField(grid)
    Gc = CenterField(grid)
    set!(U.u, 1)
    set!(c, (x, y, z) -> sinpi(2x) * sinpi(2y))

    return scalar_tendency!, (Gc, grid, advection, U, c), grid
end

#####
##### Reverse-mode gradient of a loss of the scalar tendency
#####
##### The loss is L = Σ Gc² / 2 over the interior, so ∂L/∂Gc = Gc and the reverse
##### pass is the transpose of the tendency kernel applied to Gc, accumulated
##### into ∂L/∂c and ∂L/∂𝐮. The loss is a pointwise kernel followed by a plain
##### `sum` of the field's parent array, because both have Enzyme rules on every
##### backend: KernelAbstractions' rules for the kernel launches and CUDA.jl's for
##### `sum` (a `sum(abs2, view)` of a `CuArray` has no rule). The same function is
##### differentiated eagerly by Enzyme (vanilla) and by Enzyme-MLIR inside one
##### Reactant-compiled program, so the two backends compute the same gradient.
#####

@kernel function compute_half_square!(ℓ, Gc)
    i, j, k = @index(Global, NTuple)
    @inbounds ℓ[i, j, k] = Gc[i, j, k]^2 / 2
end

# ℓ is written only in the interior; its halo stays zero, so `sum(parent(ℓ))`
# is the interior sum.
function scalar_tendency_loss(Gc, ℓ, grid, advection, U, c)
    scalar_tendency!(Gc, grid, advection, U, c)
    launch!(architecture(grid), grid, :xyz, compute_half_square!, ℓ, Gc)
    return sum(parent(ℓ))
end

# Shadows accumulate, so they are zeroed on every call: each call is one
# complete gradient evaluation (forward, loss, reverse).
function scalar_tendency_gradient!(Gc, dGc, ℓ, dℓ, grid, advection, U, dU, c, dc)
    parent(dGc) .= 0
    parent(dℓ) .= 0
    parent(dU.u) .= 0
    parent(dU.v) .= 0
    parent(dU.w) .= 0
    parent(dc) .= 0
    Enzyme.autodiff(Enzyme.Reverse, scalar_tendency_loss, Enzyme.Active,
                    Enzyme.Duplicated(Gc, dGc),
                    Enzyme.Duplicated(ℓ, dℓ),
                    Enzyme.Const(grid),
                    Enzyme.Const(advection),
                    Enzyme.Duplicated(U, dU),
                    Enzyme.Duplicated(c, dc))
    return nothing
end

"""
    scalar_tendency_gradient_problem(arch; Nx, Ny, Nz, order,
                                     float_type = Float32,
                                     topology = (Periodic, Periodic, Bounded))

Build the reverse-mode counterpart of `scalar_tendency_problem`: one call of the
returned function evaluates the loss L = Σ Gc² / 2 of the scalar tendency and its
gradient with respect to the tracer `c` and the velocities `U`, by Enzyme reverse
mode. The grid, fields and advection scheme are those of `scalar_tendency_problem`.

Returns `(scalar_tendency_gradient!, args, grid)`, where `args` holds each
differentiated field followed by its shadow, ready to hand to `benchmark_tendency`
or `Reactant.@compile`. After a call, the shadows hold ∂L/∂Gc, ∂L/∂ℓ (zero, being
consumed), ∂L/∂U and ∂L/∂c; see `scalar_tendency_gradient_summary`.
"""
function scalar_tendency_gradient_problem(arch; kw...)
    _, (Gc, grid, advection, U, c), grid = scalar_tendency_problem(arch; kw...)

    ℓ  = CenterField(grid)
    dGc = Enzyme.make_zero(Gc)
    dℓ  = Enzyme.make_zero(ℓ)
    dU  = Enzyme.make_zero(U)
    dc  = Enzyme.make_zero(c)

    args = (Gc, dGc, ℓ, dℓ, grid, advection, U, dU, c, dc)

    return scalar_tendency_gradient!, args, grid
end

"""
    scalar_tendency_gradient_summary(args)

Return `(; loss, gradient_c_norm, gradient_u_norm)` for the `args` of
`scalar_tendency_gradient_problem` after a gradient evaluation, computed on the host
in `Float64`: the loss Σ Gc² / 2 and the 2-norms of ∂L/∂c and ∂L/∂u over the parent arrays (halos
included, since the tendency reads them). Used to check that two backends computed
the same gradient.
"""
function scalar_tendency_gradient_summary(args)
    Gc, dGc, ℓ, dℓ, grid, advection, U, dU, c, dc = args
    norm2(f) = sqrt(sum(abs2, Float64.(Array(parent(f)))))
    # ℓ = Gc² / 2 in the interior and zero in its halo
    loss = sum(Float64.(Array(parent(ℓ))))
    return (; loss, gradient_c_norm = norm2(dc), gradient_u_norm = norm2(dU.u))
end
