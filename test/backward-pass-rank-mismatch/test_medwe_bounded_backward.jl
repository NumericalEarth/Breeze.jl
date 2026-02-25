#=
Investigation: Backward Pass Rank Mismatch
Status: FAILING (as of 2026-02-23)
Purpose: Reproduce DUS rank mismatch in Enzyme backward pass with Bounded topology
Related: cursor-toolchain/rules/domains/differentiability/investigations/backward-pass-rank-mismatch.md
Synchronized with: N/A (Breeze-level MedWE — the bug is upstream in Reactant/Enzyme-JAX)

The forward pass compiles and runs. The backward pass crashes with:
  LLVM ERROR: Failed to infer result type(s):
  "stablehlo.dynamic_update_slice"(...) {} :
    (tensor<9x10x10xf64>, tensor<1x9x10x10xf64>, ...) -> ( ??? )

This occurs because Face-on-Bounded fields have interior sizes (N+1) that
don't match the kernel ndrange (N). A post-Enzyme optimization pass
incorrectly collapses the checkpoint cache's iteration dimension.
=#

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA
using Test
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
@info "MLIR dumps will be written to" mlir_dump_dir

Reactant.set_default_backend("cpu")

function run_timesteps!(model, Δt, Nt)
    @trace track_numbers=false for _ in 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

function make_init_fields(grid)
    FT = eltype(grid)
    θ_init = CenterField(grid)
    set!(θ_init, (args...) -> FT(300))
    dθ_init = CenterField(grid)
    set!(dθ_init, FT(0))
    return θ_init, dθ_init
end

function loss(model, θ_init, Δt, nsteps)
    FT = eltype(model.grid)
    set!(model; θ=θ_init, ρ=one(FT))
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
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

@testset "Backward pass rank mismatch — MedWE" begin

    # # ---------------------------------------------------------------
    # # Baseline: (Periodic, Periodic, Bounded) — SHOULD WORK
    # # Face on Bounded only in z → ρw has interior (N, N, N+1)
    # # ---------------------------------------------------------------
    # @testset "Baseline: (Periodic, Periodic, Bounded) size=(4,4,4)" begin
    #     @info "  Testing baseline (Periodic, Periodic, Bounded) size=(4,4,4)..."

    #     @time "Constructing grid" grid = RectilinearGrid(ReactantState();
    #         size=(4, 4, 4), extent=(1, 1, 1),
    #         topology=(Periodic, Periodic, Bounded))

    #     @time "Constructing model" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
    #     FT = eltype(grid)

    #     @testset "Forward pass compiles" begin
    #         Δt = FT(0.001)
    #         Nt = 1
    #         @time "Compiling forward" compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
    #         @time "Running forward" compiled_run!(model, Δt, Nt)
    #         @test model.clock.iteration == Nt
    #     end

    #     @testset "Backward pass (Enzyme)" begin
    #         Δt = FT(0.001)
    #         nsteps = 4
    #         @time "Creating shadow model" dmodel = Enzyme.make_zero(model)
    #         @time "Creating init fields" θ_init, dθ_init = make_init_fields(grid)

    #         @test_broken begin
    #             @time "Compiling grad_loss" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    #                 model, dmodel, θ_init, dθ_init, Δt, nsteps)
    #             compiled_grad !== nothing
    #         end
    #     end
    # end

    # ---------------------------------------------------------------
    # Primary bug topology: (Bounded, Periodic, Bounded)
    # Face on Bounded in x (ρu) and z (ρw)
    # Uses size=(10,10,10) to match the original error trace
    # ---------------------------------------------------------------
    @testset "Bug: (Bounded, Periodic, Bounded) size=(10,10,10)" begin
        @info "  Testing bug topology (Bounded, Periodic, Bounded) size=(10,10,10)..."

        @time "Constructing grid" grid = RectilinearGrid(ReactantState();
            size=(10, 10, 10), extent=(1, 1, 1),
            topology=(Bounded, Periodic, Bounded))

        @time "Constructing model" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        FT = eltype(grid)

        @testset "Forward pass compiles" begin
            Δt = FT(0.001)
            Nt = 1
            @time "Compiling forward" compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            @time "Running forward" compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end

        @testset "Backward pass (Enzyme) — EXPECTED TO CRASH" begin
            Δt = FT(0.001)
            nsteps = 4
            @time "Creating shadow model" dmodel = Enzyme.make_zero(model)
            @time "Creating init fields" θ_init, dθ_init = make_init_fields(grid)

            @test_broken begin
                @time "Compiling grad_loss" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                    model, dmodel, θ_init, dθ_init, Δt, nsteps)
                compiled_grad !== nothing
            end
        end
    end

    # ---------------------------------------------------------------
    # Smaller variant: (Bounded, Periodic, Bounded) size=(4,4,4)
    # Same topology but smaller grid — does the bug scale?
    # ---------------------------------------------------------------
    @testset "Bug variant: (Bounded, Periodic, Bounded) size=(4,4,4)" begin
        @info "  Testing (Bounded, Periodic, Bounded) size=(4,4,4)..."

        @time "Constructing grid" grid = RectilinearGrid(ReactantState();
            size=(4, 4, 4), extent=(1, 1, 1),
            topology=(Bounded, Periodic, Bounded))

        @time "Constructing model" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        FT = eltype(grid)

        @testset "Forward pass compiles" begin
            Δt = FT(0.001)
            Nt = 1
            @time "Compiling forward" compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            @time "Running forward" compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end

        @testset "Backward pass (Enzyme)" begin
            Δt = FT(0.001)
            nsteps = 4
            @time "Creating shadow model" dmodel = Enzyme.make_zero(model)
            @time "Creating init fields" θ_init, dθ_init = make_init_fields(grid)

            @test_broken begin
                @time "Compiling grad_loss" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                    model, dmodel, θ_init, dθ_init, Δt, nsteps)
                compiled_grad !== nothing
            end
        end
    end

    # ---------------------------------------------------------------
    # Control: All-Periodic topology — should NOT exhibit the bug
    # ---------------------------------------------------------------
    @testset "Control: (Periodic, Periodic, Periodic) size=(4,4,4)" begin
        @info "  Testing control (Periodic, Periodic, Periodic) size=(4,4,4)..."

        @time "Constructing grid" grid = RectilinearGrid(ReactantState();
            size=(4, 4, 4), extent=(1, 1, 1),
            topology=(Periodic, Periodic, Periodic))

        @time "Constructing model" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
        FT = eltype(grid)

        @testset "Forward pass compiles" begin
            Δt = FT(0.001)
            Nt = 1
            @time "Compiling forward" compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt)
            @time "Running forward" compiled_run!(model, Δt, Nt)
            @test model.clock.iteration == Nt
        end

        @testset "Backward pass (Enzyme)" begin
            Δt = FT(0.001)
            nsteps = 4
            @time "Creating shadow model" dmodel = Enzyme.make_zero(model)
            @time "Creating init fields" θ_init, dθ_init = make_init_fields(grid)

            @time "Compiling grad_loss" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test compiled_grad !== nothing

            @time "Running grad_loss" dθ, loss_val = compiled_grad(
                model, dmodel, θ_init, dθ_init, Δt, nsteps)
            @test loss_val > 0
            @test isfinite(loss_val)
            @test maximum(abs, interior(dθ)) > 0
            @test !any(isnan, interior(dθ))
        end
    end
end
