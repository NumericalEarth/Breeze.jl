#####
##### Reactant compilation tests — WENO advection
#####
#
# Phase structure per topology:
#   (a)   Build model on ReactantState
#   (b)   Compile + raise backward (Enzyme reverse mode)
#   (c)   FD validation of AD gradients

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using GPUArraysCore: @allowscalar
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
    ("Periodic, Periodic, Flat",    (Periodic, Periodic, Flat),    2),
    ("Periodic, Bounded, Bounded",  (Periodic, Bounded,  Bounded), 3),
]

schemes = [
    ("WENO(order=5)",               WENO(order=5)),
    ("WENO(order=5, bounds=(0,1))", WENO(order=5, bounds=(0, 1))),
]

#####
##### Helpers
#####

function make_grid(topo, nd; arch=ReactantState())
    sz  = nd == 2 ? (8, 8)     : (8, 8, 8)
    ext = nd == 2 ? (1e3, 1e3) : (1e3, 1e3, 1e3)
    hl  = nd == 2 ? (5, 5)     : (5, 5, 5)
    return RectilinearGrid(arch; size=sz, extent=ext, halo=hl, topology=topo)
end

function make_latlon_grid(Nλ, Nφ, Nz)
    return LatitudeLongitudeGrid(ReactantState();
                                 size = (Nλ, Nφ, Nz),
                                 halo = (5, 5, 5),
                                 longitude = (0, 360),
                                 latitude = (-85, 85),
                                 z = (0, 1e3))
end

function run_time_steps!(model, Δt, Nsteps)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return nothing
end

get_temperature(model) = Array(interior(model.temperature))

function make_init_fields(grid)
    θ_init  = CenterField(grid); set!(θ_init,  (args...) -> 300.0)
    dθ_init = CenterField(grid); set!(dθ_init, 0)
    return θ_init, dθ_init
end

function initial_density(model)
    FT = eltype(model.grid)
    ref = model.dynamics.reference_state
    return isnothing(ref) ? one(FT) : ref.density
end

function loss(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, ρ=1.0)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

#####
##### Tests
#####

@testset "reactant_weno_compilation" begin
    Δt = 0.02

    @testset "$scheme_label" for (scheme_label, scheme) in schemes
        @testset "$label" for (label, topo, nd) in topologies
            grid = make_grid(topo, nd)

            # ── Build ──
            @testset "Build" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)
                @test model isa AtmosphereModel
                @test model.dynamics isa CompressibleDynamics

                set!(model; θ=300.0, ρ=1.0)
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            # Reconstruct for backward + FD phases
            model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)

            θ_init, dθ_init = make_init_fields(grid)
            dmodel = Enzyme.make_zero(model)
            Ns = 1

            compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, Ns)
            dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
            ad_grad = @allowscalar Array(interior(dθ))

            # ── Raise backward ──
            @testset "Raise backward" begin
                @test loss_val > 0
                @test isfinite(loss_val)
                @test maximum(abs, ad_grad) > 0
                @test !any(isnan, ad_grad)
            end

            # ── FD validation ──
            # Verify AD gradients against one-sided finite differences:
            #   ∂J/∂θ(i,j,k) ≈ (J(θ + ε·eᵢⱼₖ) - J(θ)) / ε
            # Checked at two grid cells and two step sizes to confirm
            # convergence is not an artifact of a particular ε.
            @testset "FD validation" begin
                grid_fd = make_grid(topo, nd; arch=default_arch)
                make_fd_model() = AtmosphereModel(grid_fd; dynamics=CompressibleDynamics(), advection=scheme)

                θ₀_fd = CenterField(grid_fd); set!(θ₀_fd, (args...) -> 300.0)
                J₀ = loss(make_fd_model(), θ₀_fd, Δt, Ns)

                test_cells = nd == 2 ? [(1,1,1), (4,4,1)] : [(1,1,1), (4,4,4)]

                for ε in (1e-4, 1e-6), (ic, jc, kc) in test_cells
                    @testset let ε=ε, (ic, jc, kc)=(ic, jc, kc)
                        θ_fd = CenterField(grid_fd); set!(θ_fd, (args...) -> 300.0)
                        @allowscalar interior(θ_fd, ic, jc, kc)[] += ε
                        J₊ = loss(make_fd_model(), θ_fd, Δt, Ns)
                        fd = (J₊ - J₀) / ε
                        ad = ad_grad[ic, jc, kc]
                        @test ad ≈ fd rtol=0.001
                    end
                end
            end
        end
    end
end

####
#### LatitudeLongitudeGrid (global longitude, near-global latitude)
####

@testset "Reactant CompressibleDynamics — WENO, LatitudeLongitudeGrid" begin
    Δt_val = 0.02

    Nλ = 8
    Nφ = 8
    Nz = 8

    for (scheme_label, scheme) in schemes
        @testset "$scheme_label" begin
            grid = make_latlon_grid(Nλ, Nφ, Nz)
            FT = eltype(grid)
            Δt = FT(Δt_val)

            @testset "Build" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)
                @test model isa AtmosphereModel
                @test model.dynamics isa CompressibleDynamics

                θ_init, _ = make_init_fields(grid)
                set!(model; θ=θ_init, ρ=initial_density(model))
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            @testset "Raise forward" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), advection=scheme)
                θ_init, _ = make_init_fields(grid)
                set!(model; θ=θ_init, ρ=initial_density(model))

                Nsteps = 2
                compiled_run = Reactant.@compile raise=true raise_first=true sync=true run_time_steps!(model, Δt, Nsteps)
                @test compiled_run !== nothing

                compiled_run(model, Δt, Nsteps)
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end
        end
    end
end
