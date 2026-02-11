#=
Investigation: Bounded Topology Compilation Error (B.6.4)
Status: FAILING (expected)
Purpose: Reproduce "had set op which was not a direct descendant" error
Related: investigations/bounded-sinkdus-segfault.md

Minimal reproduction: Bounded topology + nsteps=2 + diagnostic velocities (u=ρu/ρ)
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Breeze
using Reactant
using Enzyme
using Statistics: mean
using Test
using CUDA
using Pkg

println("Package versions:")
for pkg in ["Oceananigans", "Breeze", "Reactant", "Enzyme"]
    v = Pkg.dependencies()[Base.UUID(Pkg.project().dependencies[pkg])].version
    println("  $pkg: v$v")
end

mlir_dump_dir = joinpath(@__DIR__, "mlir_dump")
mkpath(mlir_dump_dir)
Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
Reactant.MLIR.IR.DUMP_MLIR_DIR[] = mlir_dump_dir

Reactant.set_default_backend("cpu")

@testset "B.6.4: Bounded topology fails with nsteps > 1" begin
    
    grid = RectilinearGrid(ReactantState();
        size = (4, 4),
        extent = (1000.0, 1000.0),
        halo = (3, 3),
        topology = (Bounded, Bounded, Flat))
    
    model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
    dmodel = Enzyme.make_zero(model)
    
    θ_init = CenterField(grid)
    set!(θ_init, (x, y) -> 300.0 + 0.01 * x)
    dθ_init = CenterField(grid)
    set!(dθ_init, 0.0)
    
    function loss(model, θ_init, Δt, nsteps)
        set!(model, θ=θ_init, ρ=1.0)
        @trace mincut=true checkpointing=false track_numbers=false for i in 1:nsteps
            time_step!(model, Δt)
        end
        return mean(interior(model.temperature).^2)
    end
    
    function grad_loss(model, dmodel, θ_init, dθ_init, Δt, nsteps)
        parent(dθ_init) .= 0
        _, loss_value = Enzyme.autodiff(
            Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
            loss, Enzyme.Active,
            Enzyme.Duplicated(model, dmodel),
            Enzyme.Duplicated(θ_init, dθ_init),
            Enzyme.Const(Δt),
            Enzyme.Const(nsteps))
        return dθ_init, loss_value
    end
    
    Δt = 0.01
    nsteps = 2
    
    @info "Attempting compilation with Bounded topology and nsteps=$nsteps"
    @info "Expected error: 'had set op which was not a direct descendant'"
    
    @test_broken begin
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
            model, dmodel, θ_init, dθ_init, Δt, nsteps)
        dθ, loss_val = compiled(model, dmodel, θ_init, dθ_init, Δt, nsteps)
        @test loss_val > 0
        @test !isnan(loss_val)
        true
    end
end
