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
##### Reverse-mode AD of the scalar tendency
#####
##### Mirrors `loss` / `grad_loss!` in timestepping.jl: `scalar_tendency_loss`
##### evaluates the tendency and reduces it to a scalar, and
##### `grad_scalar_tendency_loss!` differentiates it with Enzyme reverse mode with
##### respect to the tracer and the velocities. The same function runs eagerly
##### (vanilla Enzyme) or compiled with `Reactant.@compile` (Enzyme-MLIR).
#####

function scalar_tendency_loss(Gc, grid, advection, U, c)
    scalar_tendency!(Gc, grid, advection, U, c)
    return mean(interior(Gc) .^ 2)
end

function grad_scalar_tendency_loss!(Gc, dGc, grid, advection, U, dU, c, dc)
    parent(dU.u) .= 0
    parent(dU.v) .= 0
    parent(dU.w) .= 0
    parent(dc) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        scalar_tendency_loss, Enzyme.Active,
        Enzyme.Duplicated(Gc, dGc),
        Enzyme.Const(grid),
        Enzyme.Const(advection),
        Enzyme.Duplicated(U, dU),
        Enzyme.Duplicated(c, dc))
    return loss_value
end

"""
    scalar_tendency_ad_problem(arch; Nx, Ny, Nz, order,
                               float_type = Float32,
                               topology = (Periodic, Periodic, Bounded))

Build the reverse-mode counterpart of `scalar_tendency_problem` on the same grid,
fields and advection scheme: one call of the returned function evaluates
L = mean(Gc²) and accumulates ∂L/∂c and ∂L/∂𝐮 into the shadows `dc` and `dU`.

Returns `(grad_scalar_tendency_loss!, (Gc, dGc, grid, advection, U, dU, c, dc), grid)`.
"""
function scalar_tendency_ad_problem(arch; kw...)
    _, (Gc, grid, advection, U, c), grid = scalar_tendency_problem(arch; kw...)
    dGc = Enzyme.make_zero(Gc)
    dU = Enzyme.make_zero(U)
    dc = Enzyme.make_zero(c)
    return grad_scalar_tendency_loss!, (Gc, dGc, grid, advection, U, dU, c, dc), grid
end
