#####
##### Reactant correctness — compressible AtmosphereModel parity over one
##### `time_step!`. Builds the same model on a vanilla architecture and on
##### `ReactantState`, sets identical initial conditions, takes one full
##### SSP-RK3 step on each, and checks per-field parity.
#####
##### Tolerances differ by topology: XLA op-fusion in WENO-5 boundary
##### stencils raises the FP noise floor on bounded directions. The
##### all-(periodic-in-horizontal) case reaches Float64 roundoff; the
##### fully-bounded-in-y case is pinned at ~1e-6 by the BC stencil
##### reassociation.
#####

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Periodic, Bounded
using Reactant
using Random
using Printf: @printf
using Test
using CUDA

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
    ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "..", "tmp"))
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

function compare_interior(name, ψ₁, ψ₂; rtol, atol)
    a₁ = Array(interior(ψ₁))
    a₂ = Array(interior(ψ₂))
    ok = isapprox(a₁, a₂; rtol, atol)
    if !ok
        max_δ, idx = findmax(abs, a₁ .- a₂)
        @printf("(%6s) ψ₁ ≉ ψ₂: max|ψ₁|=%.6e, max|ψ₂|=%.6e, max|δ|=%.6e at %s\n",
                name, maximum(abs, a₁), maximum(abs, a₂), max_δ, string(idx.I))
    end
    return ok
end

function compare_states(vmodel, rmodel; rtol, atol)
    Ψv = Oceananigans.fields(vmodel)
    Ψr = Oceananigans.fields(rmodel)
    all(compare_interior(string(name), Ψv[name], Ψr[name]; rtol, atol)
        for name in keys(Ψv))
end

function build_model_pair(topology)
    grid_kw = (size=(8, 8, 8), halo=(5, 5, 5), extent=(1e3, 1e3, 1e3), topology=topology)
    vgrid = RectilinearGrid(default_arch;    grid_kw...)
    rgrid = RectilinearGrid(ReactantState(); grid_kw...)

    model_kw = (; dynamics=CompressibleDynamics(),
                  advection=WENO(order=5),
                  coriolis=FPlane(f=1e-4),
                  microphysics=nothing)

    vmodel = AtmosphereModel(vgrid; model_kw...)
    rmodel = AtmosphereModel(rgrid; model_kw...)

    Random.seed!(98765)
    Nx, Ny, Nz = size(vgrid)
    u_init = 0.1 .* randn(size(vmodel.velocities.u)...)
    v_init = 0.1 .* randn(size(vmodel.velocities.v)...)
    θ_init = 300.0 .+ 0.1 .* randn(Nx, Ny, Nz)

    set!(vmodel; ρ=1.0, u=u_init, v=v_init, θ=θ_init)
    set!(rmodel; ρ=1.0, u=u_init, v=v_init, θ=θ_init)

    return vmodel, rmodel
end

@testset "Reactant correctness — time_step! parity" begin
    Δt = 0.02
    atol = sqrt(eps(Float64))

    cases = [
        ("Periodic, Periodic, Bounded", (Periodic, Periodic, Bounded), 1e-10),
        ("Periodic, Bounded,  Bounded", (Periodic, Bounded,  Bounded), 1e-5),
    ]

    @testset "topology=$label" for (label, topology, rtol) in cases
        vmodel, rmodel = build_model_pair(topology)

        time_step!(vmodel, Δt)
        r_step! = Reactant.@compile raise=true sync=true time_step!(rmodel, Δt)
        r_step!(rmodel, Δt)

        @test compare_states(vmodel, rmodel; rtol, atol)
    end
end
