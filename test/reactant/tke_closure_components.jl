include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant forward and gradient compilation of a column ensemble with `TKEBasedTurbulenceClosure`,
##### its `TurbulentRichardsonStabilityFunction`, a `SaturationWeightedStaticStability` whose weight (a dummy) reads the
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
                                        static_stability = SaturationWeightedStaticStability(MixingLengthWeight(100.0)))
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

#####
##### A calibration's configuration: the closure built with traced scalars (as a template whose coefficients are trained)
##### and the trained coefficients rebound in the loss with `with_parameters`, on a stretched column ensemble (whose
##### grid, traced, has a traced eltype in the kernels), stepped by `MultirateInfinitesimalStep`; the gradient is taken
##### with respect to the trained coefficients. The closure has a `DoubleGaussian` with a `TKEBudgetThirdMoment` and a
##### `ScalarVarianceTransport`, and a `HeatingBudgetFlux` of a `PrescribedHeating` held in a reduced field (one profile
##### for the ensemble, as a forcing of θ would hold it).
#####

function trained_loss(model, coefficients, θ_init, Δt, Nsteps)
    model.closure = with_parameters(model.closure; coefficients...)
    set!(model; θ=θ_init, qᵗ=0.012, u=5, ρe=0.1)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return sum(interior(model.tracers.ρe))
end

function grad_trained_loss(model, dmodel, coefficients, dcoefficients, θ_init, Δt, Nsteps)
    _, loss_value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), trained_loss, Enzyme.Active,
                                    Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(coefficients, dcoefficients),
                                    Enzyme.Const(θ_init), Enzyme.Const(Δt), Enzyme.Const(Nsteps))
    return dcoefficients, loss_value
end

@testset "reactant_tke_trained_coefficients" begin
    Δt = 1.0
    Ns = 2
    PT = typeof(Reactant.ConcreteRNumber(0.0))

    grid = RectilinearGrid(ReactantState(); size=ColumnEnsembleSize(Nz=8, ensemble=(2, 1), Hz=3),
                           z=[2000 * (k / 8)^1.2 for k in 0:8], topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    cooling = Field{Nothing, Nothing, Center}(grid)
    set!(cooling, z -> 1000 < z < 1200 ? -5 / 3600 : 0)
    distribution = DoubleGaussian(PT; third_moment = TKEBudgetThirdMoment(PT),
                                  variance_transport = ScalarVarianceTransport(PT))
    closure = TKEBasedTurbulenceClosure(PT; static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                        subgrid_distribution = distribution,
                                        heating_budget_flux = HeatingBudgetFlux(PT; heating = PrescribedHeating(cooling)))
    trained() = map(Reactant.ConcreteRNumber, (ℂᵛᵗ = 1.0, ℂᑫ = 0.5, ℂᑫᴸ = 8.0, ℂʷᵀ = 0.3, ℂʷᴺ = 0.1, ℂʷʳ = 0.2,
                                               ℂᵍˢ = 0.1, ℂᵍˢˢ = 0.05))
    θ_init = CenterField(grid); set!(θ_init, z -> 290 + 3e-3 * z)

    @testset "Forward" begin
        model = AtmosphereModel(grid; dynamics, microphysics, closure, timestepper=:MultirateInfinitesimalStep)
        coefficients = trained()
        compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true trained_loss(
            model, coefficients, θ_init, Δt, Ns)
        value = @with_stack_size compiled(model, coefficients, θ_init, Δt, Ns)
        @test isfinite(value) && value > 0
    end

    @testset "Gradient with respect to the trained coefficients" begin
        model = AtmosphereModel(grid; dynamics, microphysics, closure, timestepper=:MultirateInfinitesimalStep)
        dmodel = Enzyme.make_zero(model)
        coefficients = trained()
        dcoefficients = map(c -> Reactant.ConcreteRNumber(0.0), coefficients)
        compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true grad_trained_loss(
            model, dmodel, coefficients, dcoefficients, θ_init, Δt, Ns)
        derivatives, loss_value = @with_stack_size compiled_grad(model, dmodel, coefficients, dcoefficients, θ_init, Δt, Ns)
        derivatives = map(d -> Float64(Reactant.to_number(d)), derivatives)
        @info "Gradient of the TKE with respect to the trained coefficients" derivatives
        @test isfinite(loss_value)
        @test all(isfinite, derivatives)
        @test !iszero(derivatives.ℂᑫ)
    end
end

#####
##### The column ensemble with a `DoubleGaussian` whose scalar second moments solve their budget with transport
##### (`ScalarVarianceTransport`, with the third moment of the PDF components) and a TKE budget flux that overshoots
##### the turbulent layer (`ℂʷᵒ`): forward and gradient
#####

@testset "reactant_tke_scalar_variance_transport" begin
    Δt = 1.0
    Ns = 2

    grid = RectilinearGrid(ReactantState(); size=ColumnEnsembleSize(Nz=8, ensemble=(2, 1), Hz=3),
                           z=(0, 2000), topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    distribution = DoubleGaussian(; third_moment = TKEBudgetThirdMoment(; ℂʷᵀ = 0.5, ℂʷᵒ = 1),
                                  variance_transport = ScalarVarianceTransport(; ℂᵛᵗ = 1, component_third_moment = true))
    closure = TKEBasedTurbulenceClosure(; subgrid_distribution = distribution)
    initial_θ() = (θ = CenterField(grid); set!(θ, z -> 290 + 3e-3 * z); θ)

    model = AtmosphereModel(grid; dynamics, microphysics, closure)
    compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true tke_loss(model, initial_θ(), Δt, Ns)
    value = @with_stack_size compiled(model, initial_θ(), Δt, Ns)
    @test isfinite(value) && value > 0

    model = AtmosphereModel(grid; dynamics, microphysics, closure)
    θ_init = initial_θ()
    dθ_init = CenterField(grid); set!(dθ_init, 0)
    dmodel = Enzyme.make_zero(model)
    compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true grad_tke_loss(
        model, dmodel, θ_init, dθ_init, Δt, Ns)
    dθ, loss_value = @with_stack_size compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
    @test isfinite(loss_value)
    @test all(isfinite, @allowscalar Array(interior(dθ)))
end
