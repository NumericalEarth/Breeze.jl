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
using Oceananigans.TimeSteppers: first_time_step!
using Reactant
using Printf: @printf
using Test
using CUDA

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

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

    Nx, Ny, Nz = size(vgrid)
    u_init = 0.1 .* randn(size(vmodel.velocities.u)...)
    v_init = 0.1 .* randn(size(vmodel.velocities.v)...)
    θ_init = 300.0 .+ 0.1 .* randn(Nx, Ny, Nz)

    set!(vmodel; ρ=1.0, u=u_init, v=v_init, θ=θ_init)
    set!(rmodel; ρ=1.0, u=u_init, v=v_init, θ=θ_init)

    return vmodel, rmodel
end

@testset "Reactant correctness — first_time_step! parity" begin
    Δt = 0.02
    atol = sqrt(eps(Float64))

    cases = [
        ("Periodic, Periodic, Bounded", (Periodic, Periodic, Bounded), 1e-8),
        ("Periodic, Bounded,  Bounded", (Periodic, Bounded,  Bounded), 1e-8),
    ]

    @testset "topology=$label" for (label, topology, rtol) in cases
        vmodel, rmodel = build_model_pair(topology)

        @test report_state("topology=$label — before first_time_step!", vmodel, rmodel; rtol, atol)

        time_step!(vmodel, Δt)
        r_step! = Reactant.@compile raise=true sync=true first_time_step!(rmodel, Δt)
        r_step!(rmodel, Δt)

        @test report_state("topology=$label — after  first_time_step!", vmodel, rmodel; rtol, atol)
    end
end
