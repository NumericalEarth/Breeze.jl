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
                                               ℂᵍˢ = 0.0, ℂᵍˢˢ = 0.0,
                                               ℂᵍ₀ = 0.4, ℂʷᵉ = 0.12, ℂʷᵋ = 0.34))
    θ_init = CenterField(grid); set!(θ_init, z -> 290 + 3e-3 * z)

    forward_model = AtmosphereModel(grid; dynamics, microphysics, closure, timestepper=:MultirateInfinitesimalStep)
    compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true trained_loss(
        forward_model, trained(), θ_init, Δt, Ns)
    loss(coefficients) = Float64(Reactant.to_number(@with_stack_size compiled(forward_model, coefficients, θ_init, Δt, Ns)))

    @testset "Forward" begin
        value = loss(trained())
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

        # Compare with central differences. The skewness coefficients of the within-component share start at zero,
        # where a branch on zero slopes once selected a constant and zeroed their derivatives, and the argument of
        # its sigmoid at zero skewness was once stored at construction, so that rebinding ℂᵍ₀ did not move it.
        h = 1e-3
        values = map(c -> Float64(Reactant.to_number(c)), trained())
        for name in (:ℂᵍˢ, :ℂᵍˢˢ, :ℂᵍ₀, :ℂʷᵉ, :ℂʷᵋ, :ℂʷʳ, :ℂʷᴺ)
            shifted(δ) = map(Reactant.ConcreteRNumber, merge(values, NamedTuple{(name,)}((values[name] + δ,))))
            finite_difference = (loss(shifted(h)) - loss(shifted(-h))) / 2h
            name in (:ℂᵍˢ, :ℂᵍˢˢ) && @test !iszero(derivatives[name])
            @info "AD and FD" name derivatives[name] finite_difference
            @test derivatives[name] ≈ finite_difference rtol=1e-2
        end
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

#####
##### The `DoubleGaussian` with `VarianceStabilityFunctions` (composed with `ScalarVarianceTransport` and the
##### flux-correlation bound ℂʳ), its coefficients traced and rebound with `with_parameters`: forward, and the
##### gradient with respect to them against central differences
#####

@testset "reactant_tke_variance_stability_functions" begin
    Δt = 1.0
    Ns = 2
    PT = typeof(Reactant.ConcreteRNumber(0.0))

    grid = RectilinearGrid(ReactantState(); size=ColumnEnsembleSize(Nz=8, ensemble=(2, 1), Hz=3),
                           z=[2000 * (k / 8)^1.2 for k in 0:8], topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    distribution = DoubleGaussian(PT; ℂʳ = 0.5, variance_transport = ScalarVarianceTransport(PT),
                                  variance_stability_functions = VarianceStabilityFunctions(PT))
    closure = TKEBasedTurbulenceClosure(PT; static_stability = SaturationWeightedStaticStability(CondensateFluxFraction()),
                                        subgrid_distribution = distribution)
    trained() = map(Reactant.ConcreteRNumber, (ℂᵛ₁⁻ = 1.5, ℂᵛ₁⁰ = 0.7, ℂᵛ₁⁺ = 0.6, ℂᵛ₂⁰ = 0.7, ℂᵛ₂⁺ = 0.6,
                                               ℂᵛ₃⁻ = 0.9, ℂᵛ₃ᵐ = 0.8, ℂᵛ₃⁺ = 0.7))
    θ_init = CenterField(grid); set!(θ_init, z -> 290 + 3e-3 * z)

    forward_model = AtmosphereModel(grid; dynamics, microphysics, closure, timestepper=:MultirateInfinitesimalStep)
    compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true trained_loss(
        forward_model, trained(), θ_init, Δt, Ns)
    loss(coefficients) = Float64(Reactant.to_number(@with_stack_size compiled(forward_model, coefficients, θ_init, Δt, Ns)))

    @testset "Forward" begin
        value = loss(trained())
        @test isfinite(value) && value > 0
    end

    @testset "Gradient with respect to the variance stability coefficients" begin
        model = AtmosphereModel(grid; dynamics, microphysics, closure, timestepper=:MultirateInfinitesimalStep)
        dmodel = Enzyme.make_zero(model)
        coefficients = trained()
        dcoefficients = map(c -> Reactant.ConcreteRNumber(0.0), coefficients)
        compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true grad_trained_loss(
            model, dmodel, coefficients, dcoefficients, θ_init, Δt, Ns)
        derivatives, loss_value = @with_stack_size compiled_grad(model, dmodel, coefficients, dcoefficients, θ_init, Δt, Ns)
        derivatives = map(d -> Float64(Reactant.to_number(d)), derivatives)
        @info "Gradient of the TKE with respect to the variance stability coefficients" derivatives
        @test isfinite(loss_value)
        @test all(isfinite, derivatives)

        h = 1e-3
        values = map(c -> Float64(Reactant.to_number(c)), trained())
        for name in keys(values)
            shifted(δ) = map(Reactant.ConcreteRNumber, merge(values, NamedTuple{(name,)}((values[name] + δ,))))
            finite_difference = (loss(shifted(h)) - loss(shifted(-h))) / 2h
            @info "AD and FD" name derivatives[name] finite_difference
            @test derivatives[name] ≈ finite_difference rtol=1e-2 atol=1e-10 * abs(Float64(Reactant.to_number(loss_value)))
        end
    end
end
