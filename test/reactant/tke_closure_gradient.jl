include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant forward and gradient of a column ensemble with the default `TKEBasedTurbulenceClosure`: its
##### `DoubleGaussian` with `TKEBudgetThirdMoment`, its `HeatingBudgetFlux` and the latent static stability, on a
##### stretched grid
#####

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: ColumnEnsembleSize
using Reactant
using Reactant: @trace, CompileOptions
using Enzyme
using GPUArraysCore: @allowscalar
using Test
using CUDA

Reactant.set_default_backend(default_arch isa GPU ? "gpu" : "cpu")

# XLA's CPU compiler moves the small time-step loops of a short column into one function whose gradient takes hours
# and tens of GB to optimize; a threshold of zero keeps them as loops
small_while_loop_options = (; xla_backend_extra_options = Dict("xla_cpu_small_while_loop_byte_threshold" => "0"))
compile_options = CompileOptions(; raise = true, raise_first = true, sync = true,
                                 xla_debug_options = default_arch isa GPU ? (;) : small_while_loop_options)

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

@testset "Reactant gradient of the default TKEBasedTurbulenceClosure" begin
    Δt = 1.0
    Ns = 2
    grid = RectilinearGrid(ReactantState(); size=ColumnEnsembleSize(Nz=6, ensemble=(2, 1), Hz=3),
                           z=[0, 20, 50, 100, 180, 300, 460], topology=(Flat, Flat, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=101325, potential_temperature=290)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
    closure = TKEBasedTurbulenceClosure()
    initial_θ() = (θ = CenterField(grid); set!(θ, z -> 290 + 3e-3 * z); θ)

    model = AtmosphereModel(grid; dynamics, microphysics, closure)
    θ_init = initial_θ()
    compiled = @with_stack_size Reactant.@compile compile_options=compile_options tke_loss(model, θ_init, Δt, Ns)
    value = @with_stack_size compiled(model, θ_init, Δt, Ns)
    @test isfinite(value) && value > 0

    model = AtmosphereModel(grid; dynamics, microphysics, closure)
    θ_init = initial_θ()
    dθ_init = CenterField(grid); set!(dθ_init, 0)
    dmodel = Enzyme.make_zero(model)
    # The gradient does not compile: the optimized StableHLO multiplies mismatched slices of the inverse face and center
    # spacings of the stretched grid ("stablehlo.multiply" of tensor<5xf64> by tensor<4xf64>, at grid.z.Δ⁻¹ᵃᵃᶠ), a
    # failure that appeared with the sinks and budget fluxes of e65b95d2. The exchange solve alone,
    # `exchange_amplitudes!` on the same column, compiles and differentiates, so the failure needs more of the closure.
    # When the gradient compiles, @test_broken fails and the checks below become the test.
    compiled_grad = try
        @with_stack_size Reactant.@compile compile_options=compile_options grad_tke_loss(model, dmodel, θ_init, dθ_init, Δt, Ns)
    catch
        nothing
    end
    @test_broken !isnothing(compiled_grad)
    if !isnothing(compiled_grad)
        dθ, loss_value = @with_stack_size compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
        gradient = @allowscalar Array(interior(dθ))
        @test isfinite(loss_value)
        @test all(isfinite, gradient)
        @test any(!iszero, gradient)
    end
end
