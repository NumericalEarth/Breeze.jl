#####
##### Reactant compilation tests — WENO advection
#####
#
# Phase structure per topology:
#   (a)   Build model on ReactantState
#   (b)   Compile + raise forward (Enzyme forward mode)

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using Test
using CUDA

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Configurations
#####

topologies = [
    ("Periodic, Periodic, Flat", (Periodic, Periodic, Flat), 2),
    ("Periodic, Bounded, Bounded",   (Periodic, Bounded, Bounded), 3),
]

schemes = [
    ("WENO(order=5)",               WENO(order=5)),
    ("WENO(order=5, bounds=(0,1))", WENO(order=5, bounds=(0, 1))),
]

#####
##### Helpers
#####

function make_grid(topo, nd)
    sz  = nd == 2 ? (8, 8)     : (8, 8, 8)
    ext = nd == 2 ? (1e3, 1e3) : (1e3, 1e3, 1e3)
    hl  = nd == 2 ? (5, 5)     : (5, 5, 5)
    return RectilinearGrid(ReactantState(); size=sz, extent=ext, halo=hl, topology=topo)
end

function run_time_steps!(model, Δt, nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return nothing
end

get_temperature(model) = Array(interior(model.temperature))

#####
##### Tests
#####

@testset "reactant_weno_compilation" begin
    Δt_val = 0.02

    for (scheme_label, scheme) in schemes
        @testset "$scheme_label" begin
            for (label, topo, nd) in topologies
                @testset "$label" begin
                    grid = make_grid(topo, nd)
                    FT = eltype(grid)
                    Δt = FT(Δt_val)

                    # ── Build ──
                    @testset "Build" begin
                        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)
                        @test model isa AtmosphereModel
                        @test model.dynamics isa CompressibleDynamics

                        set!(model; θ=FT(300), ρ=one(FT))
                        T = get_temperature(model)
                        @test all(isfinite, T)
                        @test all(T .> 0)
                    end

                    # ── Raise forward ──
                    @testset "Raise forward" begin
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
