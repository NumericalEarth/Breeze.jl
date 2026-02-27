#####
##### Reactant compilation tests — Centered advection
#####
#
# Phase structure per topology:
#   (a)   Build model on ReactantState
#   (b)   Compile + raise backward (Enzyme reverse mode)

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Statistics: mean
using Test
using CUDA

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Topology configurations
#####

topologies = [
    ("Periodic, Periodic, Flat",     (Periodic, Periodic, Flat),     2),
    ("Bounded, Bounded, Flat",       (Bounded,  Bounded,  Flat),     2),
    ("Periodic, Periodic, Periodic", (Periodic, Periodic, Periodic), 3),
    ("Periodic, Bounded, Bounded",   (Periodic, Bounded,  Bounded),  3),
]

#####
##### Helpers
#####

function make_grid(topo, nd)
    sz  = nd == 2 ? (8, 8)     : (8, 8, 8)
    ext = nd == 2 ? (1e3, 1e3) : (1e3, 1e3, 1e3)
    return RectilinearGrid(ReactantState(); size=sz, extent=ext, topology=topo)
end

function run_time_steps!(model, Δt, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
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
    return mean(interior(model.temperature) .^ 2)
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
##### Tests
#####

@testset "Reactant CompressibleDynamics — Centered" begin
    Δt_val = 0.02

    for (label, topo, nd) in topologies
        @testset "$label" begin
            grid = make_grid(topo, nd)
            FT = eltype(grid)
            Δt = FT(Δt_val)

            # ── Build ──
            @testset "Build" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
                @test model isa AtmosphereModel
                @test model.dynamics isa CompressibleDynamics

                set!(model; θ=FT(300), ρ=one(FT))
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            # Reconstruct for compilation phases
            model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
            set!(model; θ=FT(300), ρ=one(FT))

            # ── Raise backward ──
            @testset "Raise backward" begin
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
        end
    end
end
