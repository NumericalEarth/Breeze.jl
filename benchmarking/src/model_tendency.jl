#####
##### Full-model tendency benchmark case
#####
##### Builds a compressible `AtmosphereModel` and times `compute_tendencies!`,
##### which fills `model.timestepper.Gⁿ` (momentum, thermodynamic, density, …)
##### in place. This is the full per-RK-stage tendency evaluation, in contrast
##### to the bare advective kernel in `scalar_tendency.jl`.
#####

using Oceananigans
using Breeze: CompressibleDynamics
using Breeze.AtmosphereModels: compute_tendencies!

"""
    model_tendency_problem(arch; Nx, Ny, Nz, order,
                           float_type = Float32,
                           topology = (Periodic, Periodic, Bounded))

Build a compressible `AtmosphereModel` on `arch` of size `(Nx, Ny, Nz)` with a
`WENO` scheme of the given `order`, initialized to a state at rest. Returns
`(compute_tendencies!, (model,), model.grid)`, ready to hand to
`benchmark_tendency` or `Reactant.@compile`. `compute_tendencies!(model)` fills
`model.timestepper.Gⁿ` in place (its `callbacks` argument defaults to `[]`).
"""
function model_tendency_problem(arch;
                                Nx, Ny, Nz,
                                order,
                                float_type = Float32,
                                topology = (Periodic, Periodic, Bounded))

    Oceananigans.defaults.FloatType = float_type

    # WENO of order `order` reads (order + 1) ÷ 2 cells on each side.
    h = cld(order + 1, 2)
    grid = RectilinearGrid(arch; size=(Nx, Ny, Nz), extent=(1e3, 1e3, 1e3), halo=(h, h, h), topology)

    model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=WENO(float_type; order))

    # `set!` runs update_state! (compute_tendencies=false), so the diagnostic
    # state is fresh before tendencies are computed. Values are irrelevant to
    # the timing — every kernel still runs.
    set!(model; θ=300, ρ=1)

    return compute_tendencies!, (model,), grid
end
