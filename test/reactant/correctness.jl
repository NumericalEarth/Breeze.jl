include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant correctness — AtmosphereModel parity over a compiled `run!`.
#####
##### The same model is built on a vanilla architecture and on `ReactantState`: compressible
##### dynamics with split-explicit acoustic substepping on a hydrostatic reference state, WENO-5
##### advection, the TKE-based turbulence closure, one-moment cloud microphysics, and Coriolis.
##### Both get the same initial state (a warm, moist bubble in a sheared wind over a stably
##### stratified background), one runs a `Simulation` of a few SSP-RK3 steps eagerly and the
##### other as one compiled program (Oceananigans' `ReactantSimulation`), and every prognostic
##### field and tendency is compared.
#####
##### Two grids: a doubly periodic `RectilinearGrid`, and a `LatitudeLongitudeGrid` bounded in
##### every direction. Both are 32 × 32 × 8. XLA's op fusion reassociates the boundary stencils,
##### which raises the floating-point noise floor on bounded directions; the tolerances below
##### are for Float64 and leave room for that.
#####

using Breeze
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Units: kilometers
using Reactant
using Printf: @printf
using Test
using CUDA

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Comparison
#####

function compare_interior(name, ψ₁, ψ₂; rtol, atol)
    a₁ = Array(interior(ψ₁))
    a₂ = Array(interior(ψ₂))
    max_ψ₁ = maximum(abs, a₁)
    max_ψ₂ = maximum(abs, a₂)
    max_δ  = maximum(abs, a₁ .- a₂)
    rel = max_δ / max(max_ψ₁, eps(eltype(a₁)))
    ok = isapprox(a₁, a₂; rtol, atol)
    @printf("  %-12s ok=%-5s  max|ψ₁|=%.6e  max|ψ₂|=%.6e  max|δ|=%.6e  rel=%.3e\n",
            string(name), ok, max_ψ₁, max_ψ₂, max_δ, rel)
    return ok
end

function report_state(label, vmodel, rmodel; rtol, atol)
    println("─── $label ───")
    ok = true
    Ψv = Oceananigans.fields(vmodel)
    Ψr = Oceananigans.fields(rmodel)
    for name in keys(Ψv)
        ok &= compare_interior(string(name), Ψv[name], Ψr[name]; rtol, atol)
    end
    Gv = vmodel.timestepper.Gⁿ
    Gr = rmodel.timestepper.Gⁿ
    for name in propertynames(Gv)
        ok &= compare_interior("Gⁿ.$name", Gv[name], Gr[name]; rtol, atol)
    end
    return ok
end

#####
##### Grids
#####

const Nx, Ny, Nz = 32, 32, 8
const Lz = 2kilometers

# 32 km across with 1 km cells on the rectilinear grid; 2° of longitude and latitude around
# 30°N on the lat-lon grid, about 190 km across with 6 km cells.
function rectilinear_grid(arch)
    return RectilinearGrid(arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                           x=(0, 32kilometers), y=(0, 32kilometers), z=(0, Lz),
                           topology=(Periodic, Periodic, Bounded))
end

function latitude_longitude_grid(arch)
    return LatitudeLongitudeGrid(arch; size=(Nx, Ny, Nz), halo=(5, 5, 5),
                                 longitude=(-1, 1), latitude=(29, 31), z=(0, Lz),
                                 topology=(Bounded, Bounded, Bounded))
end

#####
##### Model and initial state
#####

# A stably stratified background, 3 K/km above a 500 m mixed layer.
θ̄(z) = 300 + 3e-3 * max(0, z - 500)

function build_model(grid)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=4);
                                    reference_potential_temperature=θ̄)
    microphysics = OneMomentCloudMicrophysics(;
        cloud_formation = NonEquilibriumCloudFormation(nothing, :ice))
    closure = TKEBasedTurbulenceClosure()
    coriolis = grid isa LatitudeLongitudeGrid ? SphericalCoriolis() : FPlane(f=1e-4)
    return AtmosphereModel(grid; dynamics, advection=WENO(order=5), coriolis, closure, microphysics)
end

# Horizontal coordinates normalized to [-1, 1] over the domain, so that the same bubble serves
# both grids: metres on the rectilinear grid, degrees around (0°, 30°N) on the lat-lon grid.
normalized(::RectilinearGrid, x, y) = ((x - 16kilometers) / 16kilometers, (y - 16kilometers) / 16kilometers)
normalized(::LatitudeLongitudeGrid, λ, φ) = (λ, φ - 30)

# A Gaussian bubble of radius 0.4 (normalized) and 600 m, centred 600 m up.
function bubble(grid, x, y, z)
    ξ, η = normalized(grid, x, y)
    ζ = (z - 600) / 600
    return exp(-(ξ^2 + η^2) / 0.4^2 - ζ^2)
end

# The reference density as a function of height, read from the vanilla model, so that both
# models are set from the same host values: `set!` from one field to another does not run
# eagerly on a `ReactantState` GPU field (it broadcasts between device arrays, which indexes
# them one element at a time). The reference is hydrostatic, so one column describes it, and
# the cell centres it is read at are the ones it is set at.
function reference_density_profile(model)
    ρ̄ = model.dynamics.reference_state.density
    zᶜ = Array(znodes(ρ̄))
    ρᶜ = Array(interior(ρ̄))[1, 1, :]
    function ρ(z)
        k = clamp(searchsortedlast(zᶜ, z), 1, length(zᶜ) - 1)
        w = (z - zᶜ[k]) / (zᶜ[k+1] - zᶜ[k])
        return (1 - w) * ρᶜ[k] + w * ρᶜ[k+1]
    end
    return ρ
end

# The bubble is warm and moist, with cloud liquid and a little rain in its core; the zonal wind
# is sheared over the depth with a weak meridional wave; the turbulent kinetic energy decays
# with height. Vapour sits near 80% of saturation in the mixed layer and is given as a density
# with the reference's near-surface value, 1.1 kg/m³.
function set_initial_state!(model, ρ̄)
    grid = model.grid
    b(x, y, z) = bubble(grid, x, y, z)
    set!(model;
         ρ    = (x, y, z) -> ρ̄(z),
         θ    = (x, y, z) -> θ̄(z) + 2 * b(x, y, z),
         u    = (x, y, z) -> 5 + 3 * z / Lz,
         v    = (x, y, z) -> 0.5 * sinpi(normalized(grid, x, y)[1]) * cospi(z / Lz / 2),
         w    = (x, y, z) -> 0.2 * b(x, y, z) * sinpi(z / Lz),
         ρe   = (x, y, z) -> 0.3 * exp(-z / 500) + 1e-3,
         ρqᵛ  = (x, y, z) -> 1.1 * (7e-3 * exp(-z / 1500) + 4e-3 * b(x, y, z)),
         ρqᶜˡ = (x, y, z) -> 1.1 * 3e-4 * b(x, y, z)^2,
         ρqʳ  = (x, y, z) -> 1.1 * 2e-5 * b(x, y, z)^2)
    return model
end

function build_model_pair(make_grid)
    vmodel = build_model(make_grid(default_arch))
    rmodel = build_model(make_grid(ReactantState()))
    ρ̄ = reference_density_profile(vmodel)
    set_initial_state!(vmodel, ρ̄)
    set_initial_state!(rmodel, ρ̄)
    return vmodel, rmodel
end

#####
##### Test
#####

@testset "Reactant correctness — compiled run! parity" begin
    Δt = 2.0
    stop_iteration = 5
    atol = sqrt(eps(Float64))

    cases = [
        ("RectilinearGrid (Periodic, Periodic, Bounded)",     rectilinear_grid,        1e-8),
        ("LatitudeLongitudeGrid (Bounded, Bounded, Bounded)", latitude_longitude_grid, 1e-8),
    ]

    @testset "$label" for (label, make_grid, rtol) in cases
        vmodel, rmodel = build_model_pair(make_grid)

        @test report_state("$label — before run!", vmodel, rmodel; rtol, atol)

        vsimulation = Simulation(vmodel; Δt, stop_iteration, verbose=false)
        run!(vsimulation)

        rsimulation = Simulation(rmodel; Δt, stop_iteration, verbose=false)
        r_run! = Reactant.@compile raise=true sync=true run!(rsimulation)
        r_run!(rsimulation)

        @test iteration(rsimulation) == stop_iteration
        @test report_state("$label — after  run!", vmodel, rmodel; rtol, atol)
    end
end
