#####
##### Reactant compilation tests — LatitudeLongitudeGrid (Periodic, Bounded, Bounded)
#####
#
# Phase structure per dynamics variant:
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

if get(ENV, "GITHUB_ACTIONS", "false") == "true"
    Reactant.MLIR.IR.DUMP_MLIR_ALWAYS[] = true
    ENV["TMPDIR"] = mkpath(joinpath(@__DIR__, "..", "tmp"))
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Dynamics variants
#####

dynamics_variants = [
    ("ExplicitTimeStepping",            () -> CompressibleDynamics(ExplicitTimeStepping())),
    ("SplitExplicitTimeDiscretization", () -> CompressibleDynamics(
        SplitExplicitTimeDiscretization(substeps=3))),
]

#####
##### Helpers
#####

function make_grid(; arch=ReactantState())
    return LatitudeLongitudeGrid(arch;
                                 size = (8, 6, 4),
                                 halo = (5, 5, 5),
                                 longitude = (0, 360),
                                 latitude = (-85, 85),
                                 z = (0, 10_000.0),
                                 topology = (Periodic, Bounded, Bounded))
end

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
    set!(model; θ=θ_init, ρ=initial_density(model))
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

@testset "Reactant CompressibleDynamics — LatitudeLongitudeGrid (PBB)" begin
    Δt = 0.02

    @testset "$label" for (label, make_dynamics) in dynamics_variants
        grid = make_grid()

        # ── Build ──
        @testset "Build" begin
            model = AtmosphereModel(grid; dynamics=make_dynamics(), coriolis=SphericalCoriolis())
            @test model isa AtmosphereModel
            @test model.dynamics isa CompressibleDynamics
        end

        model  = AtmosphereModel(grid; dynamics=make_dynamics(), coriolis=SphericalCoriolis())
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
        @testset "FD validation" begin
            grid_fd = make_grid(arch=default_arch)
            make_fd_model() = AtmosphereModel(grid_fd; dynamics=make_dynamics(), coriolis=SphericalCoriolis())

            θ₀_fd = CenterField(grid_fd); set!(θ₀_fd, (args...) -> 300.0)
            J₀ = loss(make_fd_model(), θ₀_fd, Δt, Ns)

            for ε in (1e-4, 1e-6), (ic, jc, kc) in [(1,1,1), (4,4,4)]
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
