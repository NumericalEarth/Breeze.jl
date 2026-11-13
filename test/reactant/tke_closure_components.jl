include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant forward and gradient compilation of a column ensemble with `TKEBasedTurbulenceClosure`,
##### its `TurbulenceAugmentedShear`, a `SubgridCloudStaticStability` whose weight (a dummy) reads the
##### stored mixing length, and a mixing length with a `HarmonicBlend`
#####

using Breeze
using Breeze.TurbulenceClosures: TurbulenceClosures
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: ColumnEnsembleSize
using Reactant
using Reactant: @trace
using Enzyme
using GPUArraysCore: @allowscalar
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

struct MixingLengthWeight{FT}
    ℓ⁰ :: FT
end

@inline function TurbulenceClosures.saturated_stability_weightᶜᶜᶠ(i, j, k, grid, model::MixingLengthWeight, buoyancy, tracers, closure_fields)
    ℓ = @inbounds closure_fields.ℓᵐ[i, j, k]
    return ℓ / (ℓ + model.ℓ⁰)
end

function tke_loss(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, qᵗ=0.012, u=5, ρe=0.1)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return sum(interior(model.tracers.ρe))
end

function grad_tke_loss(model, dmodel, θ_init, dθ_init, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), tke_loss, Enzyme.Active,
                                    Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(θ_init, dθ_init),
                                    Enzyme.Const(Δt), Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

@testset "reactant_tke_closure_components" begin
    Δt = 1.0
    Ns = 2

    grid = RectilinearGrid(ReactantState(); size=ColumnEnsembleSize(Nz=8, ensemble=(2, 1), Hz=3),
                           z=(0, 2000), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    closure = TKEBasedTurbulenceClosure(mixing_length = GradientLimitedMixingLength(ℂˢ=0.79, ℂⁱ=1.78, blending=HarmonicBlend(exponent=4)),
                                        shear = TurbulenceAugmentedShear(turbulence_shear_coefficient=0.5),
                                        static_stability = SubgridCloudStaticStability(MixingLengthWeight(100.0)))
    initial_θ() = (θ = CenterField(grid); set!(θ, z -> 290 + 3e-3 * z); θ)

    @testset "Forward" begin
        model = AtmosphereModel(grid; dynamics, microphysics, closure)
        θ_init = initial_θ()
        compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true tke_loss(model, θ_init, Δt, Ns)
        value = @with_stack_size compiled(model, θ_init, Δt, Ns)
        @test isfinite(value) && value > 0
    end

    @testset "Gradient" begin
        model = AtmosphereModel(grid; dynamics, microphysics, closure)
        θ_init = initial_θ()
        dθ_init = CenterField(grid); set!(dθ_init, 0)
        dmodel = Enzyme.make_zero(model)
        compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true grad_tke_loss(
            model, dmodel, θ_init, dθ_init, Δt, Ns)
        dθ, loss_value = @with_stack_size compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
        ad_grad = @allowscalar Array(interior(dθ))
        @test isfinite(loss_value)
        @test all(isfinite, ad_grad)
        @test any(!iszero, ad_grad)
    end
end
