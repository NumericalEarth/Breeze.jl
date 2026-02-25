#####
##### Reactant CompressibleDynamics tests
#####
# Topology matrix test for AtmosphereModel + CompressibleDynamics backward pass.
# Tests both nsteps=1 (no checkpointing) and nsteps=9 (checkpointed, perfect square).
#
# 2D: PPF, BBF
# 3D: PPP, PBB, PPB, BBB

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState, GPU
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using CUDA
using Test

@testset "Reactant CompressibleDynamics" begin
    @info "Performing Reactant CompressibleDynamics tests..."

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

    topologies_2d = [
        ("PPF", (Periodic, Periodic, Flat)),
        ("BBF", (Bounded,  Bounded,  Flat)),
    ]

    topologies_3d = [
        ("PPP", (Periodic, Periodic, Periodic)),
        ("PBB", (Periodic, Bounded,  Bounded)),
        ("PPB", (Periodic, Periodic, Bounded)),
        ("BBB", (Bounded,  Bounded,  Bounded)),
    ]

    Δt_val  = 0.001
    Nt_fwd  = 4
    nsteps_list = (1, 9)

    for (label, topo) in vcat(topologies_2d, topologies_3d)
        is_2d = topo[3] === Flat
        sz  = is_2d ? (6, 6)    : (6, 6, 6)
        ext = is_2d ? (1, 1)    : (1, 1, 1)

        @testset "$label" begin
            @info "  Testing $label with size=$sz..."
            grid = RectilinearGrid(ReactantState(); size=sz, extent=ext, topology=topo)
            @time "Constructing model ($label)" model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
            FT = eltype(grid)
            Δt = FT(Δt_val)

            @test model isa AtmosphereModel
            @test model.grid.architecture isa ReactantState
            @test model.dynamics isa CompressibleDynamics

            # Compile forward once
            @info "    [$label] Compiling forward (Nt=$Nt_fwd)..."
            @time "Compiling forward ($label)" compiled_fwd = @compile raise=true raise_first=true sync=true run_timesteps!(model, Δt, Nt_fwd)

            # Compile backward once (nsteps passed as argument)
            dmodel = Enzyme.make_zero(model)
            θ_init, dθ_init = make_init_fields(grid)
            ns_compile = 9
            @info "    [$label] Compiling backward (nsteps=$ns_compile)..."
            @time "Compiling backward ($label)" compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, ns_compile)

            # Run forward
            @testset "Forward (Nt=$Nt_fwd)" begin
                @info "    [$label] Running forward..."
                @time "Running forward ($label)" compiled_fwd(model, Δt, Nt_fwd)
                @test model.clock.iteration == Nt_fwd
            end

            # Run backward with different nsteps values
            for ns in nsteps_list
                @testset "Backward (nsteps=$ns)" begin
                    @info "    [$label] Running backward (nsteps=$ns)..."
                    @time "Running backward ($label, n=$ns)" dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, ns)
                    @test loss_val > 0
                    @test isfinite(loss_val)
                    @test maximum(abs, interior(dθ)) > 0
                    @test !any(isnan, interior(dθ))
                end
            end
        end
    end
end
