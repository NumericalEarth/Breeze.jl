include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant compilation tests — microphysics schemes
#####
#
# Phase structure per grid type:
#   (a)   Build model on ReactantState with the given microphysics scheme
#   (b)   Compile + raise backward (Enzyme reverse mode)
#
# Each scheme is exercised by its own driver file (e.g. `1m_microphysics.jl`,
# `p3_microphysics.jl`) which includes this setup and calls
# `run_microphysics_tests(label, microphysics, initial_state)`, where
# `initial_state` holds the scheme-specific moisture fields passed to `set!`.

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
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

#####
##### Grid configurations
#####

grid_configs = [
    ("RectilinearGrid (PPB)",
     arch -> RectilinearGrid(arch; size=(8, 8, 8), extent=(1e3, 1e3, 1e3),
                             topology=(Periodic, Periodic, Bounded))),
    ("LatitudeLongitudeGrid (PBB)",
     arch -> LatitudeLongitudeGrid(arch; size=(8, 8, 8),
                                   longitude=(-10, 10), latitude=(-10, 10), z=(-1e3, 0),
                                   topology=(Periodic, Bounded, Bounded))),
]

#####
##### Helpers
#####

function loss(model, θ_init, initial_state, Δt, Nsteps)
    set!(model; θ=θ_init, ρ=1.0, initial_state...)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function grad_loss(model, dmodel, θ_init, dθ_init, initial_state, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(initial_state),
        Enzyme.Const(Δt),
        Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

#####
##### Tests
#####

function run_microphysics_tests(scheme_label, microphysics, initial_state)
    @testset "reactant_microphysics_compilation - $scheme_label" begin
        Δt = 0.02
        Ns = 1

        @testset "$label" for (label, make_grid) in grid_configs
            grid = make_grid(ReactantState())

            @testset "Build" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), microphysics)
                @test model isa AtmosphereModel
                @test model.dynamics isa CompressibleDynamics
            end

            @testset "Raise backward" begin
                model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), microphysics)
                θ_init  = CenterField(grid); set!(θ_init,  (args...) -> 300.0)
                dθ_init = CenterField(grid); set!(dθ_init, 0)
                dmodel  = Enzyme.make_zero(model)

                compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                    model, dmodel, θ_init, dθ_init, initial_state, Δt, Ns)
                dθ, loss_val = @with_stack_size compiled_grad(model, dmodel, θ_init, dθ_init, initial_state, Δt, Ns)
                ad_grad = @allowscalar Array(interior(dθ))

                @test loss_val > 0
                @test isfinite(loss_val)
                @test any(!iszero, ad_grad)
                @test all(isfinite, ad_grad)
            end
        end
    end
end
