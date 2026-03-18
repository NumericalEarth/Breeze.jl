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
using Oceananigans.Grids: Periodic
using Reactant
using Reactant: @trace
using Enzyme
using GPUArraysCore: @allowscalar
using Statistics: mean
using Test
using CUDA

device_arch = default_arch isa GPU ? GPU() : CPU()

if device_arch isa GPU
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

function make_grid(topo, nd; arch=ReactantState())
    sz  = nd == 2 ? (8, 8)     : (8, 8, 8)
    ext = nd == 2 ? (1e3, 1e3) : (1e3, 1e3, 1e3)
    return RectilinearGrid(arch; size=sz, extent=ext, topology=topo)
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

@testset "Reactant CompressibleDynamics — Centered" begin
    Δt = 0.02

    for (label, topo, nd) in topologies
        @testset "$label" begin
            grid = make_grid(topo, nd)

            # ── Build ──
            @info "Building model..."
            @testset "Build" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics())
                @test model isa AtmosphereModel
                @test model.dynamics isa CompressibleDynamics

                set!(model; θ=300.0, ρ=1.0)
                T = get_temperature(model)
                @test all(isfinite, T)
                @test all(T .> 0)
            end

            # Reconstruct for compilation phases
            model = AtmosphereModel(grid; dynamics=CompressibleDynamics())

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
                grid_fd = make_grid(topo, nd; arch=device_arch)
                make_fd_model() = AtmosphereModel(grid_fd; dynamics=CompressibleDynamics())

                θ₀_fd = CenterField(grid_fd); set!(θ₀_fd, (args...) -> 300.0)
                J₀ = loss(make_fd_model(), θ₀_fd, Δt, Ns)

                test_cells = nd == 2 ? [(1,1,1), (4,4,1)] : [(1,1,1), (4,4,4)]

                for ε in [1e-4, 1e-6]
                    for (ic, jc, kc) in test_cells
                        θ_fd = CenterField(grid_fd); set!(θ_fd, (args...) -> 300.0)
                        @allowscalar interior(θ_fd, ic, jc, kc)[] += ε
                        J₊ = loss(make_fd_model(), θ_fd, Δt, Ns)
                        fd = (J₊ - J₀) / ε
                        ad = ad_grad[ic, jc, kc]
                        rel = abs(ad - fd) / (max(abs(ad), abs(fd)) + eps())
                        @test rel < 0.01
                    end
                end
            end
        end
    end
end
