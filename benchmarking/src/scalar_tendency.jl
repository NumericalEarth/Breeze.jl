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

Returns `(scalar_tendency!, (Gc, grid, advection, U, c))`, ready to hand to
`benchmark_scalar_tendency` or `Reactant.@compile`.
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

    return scalar_tendency!, (Gc, grid, advection, U, c)
end
