#####
##### Reactant compilation tests — 1-moment non-equilibrium microphysics
#####
#
# Phase structure per grid type:s
#   (a)   Build model on ReactantState with OneMomentCloudMicrophysics (MPNE1M)
#   (b)   Compile + raise backward (Enzyme reverse mode)

using Breeze
using Breeze.Microphysics: NonEquilibriumCloudFormation
using CloudMicrophysics
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using GPUArraysCore: @allowscalar
using Statistics: mean
using Test
using CUDA

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

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

function loss(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, ρ=1.0, ρqᵛ=0.01, ρqᶜˡ=1e-4, ρqᶜⁱ=1e-5, ρqʳ=1e-5, ρqˢ=1e-6)
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

@testset "Reactant 1M MPNE — backward" begin
    Δt = 0.02
    Ns = 1

    microphysics = OneMomentCloudMicrophysics(;
        cloud_formation = NonEquilibriumCloudFormation(nothing, :ice))

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

            compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                model, dmodel, θ_init, dθ_init, Δt, Ns)
            dθ, loss_val = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
            ad_grad = @allowscalar Array(interior(dθ))

            @test loss_val > 0
            @test isfinite(loss_val)
            @test maximum(abs, ad_grad) > 0
            @test !any(isnan, ad_grad)
        end
    end
end
