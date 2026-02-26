#####
##### Reactant compilation tests for CompressibleDynamics
#####
#
# Phase structure per topology:
#   (a)   Build model on ReactantState
#   (b+c) Compile + raise forward time_step!
#   (d)   Compile + raise backward (Enzyme reverse mode)
#
# Centered advection:  full topology matrix, phases (a)–(d)
# WENO advection:      reduced topologies,   phases (a)–(b+c)

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Test
using CUDA

#####
##### Topology configurations
#####

# Each entry: (label, topology, ndims)
centered_topologies = [
    # 2D
    ("PPF", (Periodic, Periodic, Flat),    2),
    ("BBF", (Bounded,  Bounded,  Flat),    2),
    # 3D
    ("PPP", (Periodic, Periodic, Periodic), 3),
    ("PBB", (Periodic, Bounded, Bounded),  3),
]

weno_topologies = [
    ("PPF", (Periodic, Periodic, Flat), 2),
    ("BBF", (Bounded,  Bounded,  Flat), 2),
]

weno_schemes = [
    ("WENO(order=5)",              WENO(order=5)),
    ("WENO(order=9)",              WENO(order=9)),
    ("WENO(order=5, bounds=(0,1))", WENO(order=5, bounds=(0, 1))),
]


#####
##### Helpers
#####

function make_grid(topo, nd)
    sz  = nd == 2 ? (8, 8)    : (8, 8, 8)
    ext = nd == 2 ? (1e3, 1e3) : (1e3, 1e3, 1e3)
    return RectilinearGrid(ReactantState(); size=sz, extent=ext, topology=topo)
end

function make_weno_grid(topo, nd)
    sz  = nd == 2 ? (8, 8)    : (8, 8, 8)
    ext = nd == 2 ? (1e3, 1e3) : (1e3, 1e3, 1e3)
    hl  = nd == 2 ? (5, 5)    : (5, 5, 5)
    return RectilinearGrid(ReactantState(); size=sz, extent=ext, halo=hl, topology=topo)
end

function run_time_steps!(model, Δt, nsteps)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return nothing
end

get_temperature(model) = Array(interior(model.temperature))

function make_init_fields(grid)
    FT = eltype(grid)
    θ_init  = CenterField(grid); set!(θ_init,  (args...) -> FT(300))
    dθ_init = CenterField(grid); set!(dθ_init, FT(0))
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

#####
##### Backend
#####

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Centered advection: full topology matrix, phases (a)–(d)
#####

@testset "Reactant CompressibleDynamics — Centered" begin
    Δt_val = 0.02

    for (label, topo, nd) in centered_topologies
        @testset "$label" begin
            grid = make_grid(topo, nd)
            FT = eltype(grid)
            Δt = FT(Δt_val)

            # ── (a) Build ──
            @testset "(a) Build" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
                @test model isa AtmosphereModel
                @test model.dynamics isa CompressibleDynamics

                set!(model; θ=FT(300), ρ=one(FT))
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            # Reconstruct model for compilation phases (avoids stale state)
            model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
            set!(model; θ=FT(300), ρ=one(FT))

            # ── (b+c) Raise forward ──
            @testset "(b+c) Raise forward" begin
                nsteps = 2
                compiled_run = Reactant.@compile raise=true raise_first=true sync=true run_time_steps!(model, Δt, nsteps)
                @test compiled_run !== nothing

                compiled_run(model, Δt, nsteps)
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            # ── (d) Raise backward ──
            if label ∉ backward_broken
                @testset "(d) Raise backward" begin
                    θ_init, dθ_init = make_init_fields(grid)
                    dmodel = Enzyme.make_zero(model)
                    ns = 1

                    compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                        model, dmodel, θ_init, dθ_init, Δt, ns)

                    dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, ns)
                    @test loss_val > 0
                    @test isfinite(loss_val)
                    @test maximum(abs, interior(dθ)) > 0
                    @test !any(isnan, interior(dθ))
                end
            else
                @warn "  [$label] Skipping (d) Raise backward — blocked by B.6.10"
            end
        end
    end
end

#####
##### WENO advection: reduced topologies, phases (a)–(b+c)
#####

@testset "Reactant CompressibleDynamics — WENO" begin
    Δt_val = 0.02

    for (scheme_label, scheme) in weno_schemes
        @testset "$scheme_label" begin
            for (label, topo, nd) in weno_topologies
                @testset "$label" begin
                    grid = make_weno_grid(topo, nd)
                    FT = eltype(grid)
                    Δt = FT(Δt_val)

                    # ── (a) Build ──
                    @testset "(a) Build" begin
                        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)
                        @test model isa AtmosphereModel
                        @test model.dynamics isa CompressibleDynamics

                        set!(model; θ=FT(300), ρ=one(FT))
                        T = get_temperature(model)
                        @test all(isfinite, T)
                        @test all(T .> 0)
                    end

                    # ── (b+c) Raise forward ──
                    @testset "(b+c) Raise forward" begin
                        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)
                        set!(model; θ=FT(300), ρ=one(FT))

                        nsteps = 2
                        compiled_run = Reactant.@compile raise=true raise_first=true sync=true run_time_steps!(model, Δt, nsteps)
                        @test compiled_run !== nothing

                        compiled_run(model, Δt, nsteps)
                        T = get_temperature(model)
                        @test all(isfinite, T)
                        @test all(T .> 0)
                    end
                end
            end
        end
    end
end
