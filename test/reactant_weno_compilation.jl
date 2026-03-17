#####
##### Reactant compilation tests — WENO advection
#####
#
# Phase structure:
#   RectilinearGrid:      Build + compile/raise backward (Enzyme reverse mode)
#   LatitudeLongitudeGrid: Build + compile/raise forward

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

function make_init_fields(grid)
    FT = eltype(grid)
    θ_init = CenterField(grid)
    set!(θ_init, (args...) -> FT(300))
    return θ_init
end

function initial_density(model)
    FT = eltype(model.grid)
    ref = model.dynamics.reference_state
    return isnothing(ref) ? one(FT) : ref.density
end

function loss(model, θ_init, Δt, nsteps)
    set!(model; θ=θ_init, ρ=initial_density(model))
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

                    # ── Raise backward ──
                    @testset "Raise backward" begin
                        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)
                        θ_init = make_init_fields(grid)
                        dθ_init = CenterField(grid)
                        set!(dθ_init, FT(0))
                        set!(model; θ=θ_init, ρ=initial_density(model))

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
    end
end
