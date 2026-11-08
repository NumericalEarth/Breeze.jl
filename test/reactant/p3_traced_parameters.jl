include(joinpath(dirname(@__DIR__), "setup.jl"))

#####
##### Reactant forward and gradient compilation of P3 on a column-ensemble grid with
##### traced process-rate and subgrid parameters (Gaussian subgrid condensation, TKE closure)
#####

using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: ColumnEnsembleSize
using Reactant
using Reactant: @trace
using Enzyme
using GPUArraysCore: @allowscalar
using Statistics: mean
using Test
using CUDA

using Breeze.AtmosphereModels: GaussianSubgridCorrelation
using Breeze.Microphysics.PredictedParticleProperties: ProcessRate

# TODO: move upstream to Reactant (ReactantCUDAExt). Tracked parameters reach GPU kernels as
# `CuTracedRNumber{Float64, 1}`, and P3 mixes them with plain `Float64` (e.g. `FT(p)`, `clamp`,
# `rate / scale`). Reactant's `promote_rule` methods for `CuTracedRNumber` are `@nospecializeinfer`,
# so promotion is a dynamic call inside the kernel (InvalidIRError), and `Float64(::CuTracedRNumber)`
# has no method. These concrete rules are inferable and return what arithmetic on a
# `CuTracedRNumber` already returns: its loaded value.
const ReactantCUDA = Base.get_extension(Reactant, :ReactantCUDAExt)

if !isnothing(ReactantCUDA)
    const CuTracedFloat64 = ReactantCUDA.CuTracedRNumber{Float64, 1}
    Base.promote_rule(::Type{CuTracedFloat64}, ::Type{Float64}) = Float64
    Base.promote_rule(::Type{Float64}, ::Type{CuTracedFloat64}) = Float64
    Base.promote_rule(::Type{CuTracedFloat64}, ::Type{CuTracedFloat64}) = Float64
    Base.Float64(x::CuTracedFloat64) = x[]
end

if default_arch isa GPU
    Reactant.set_default_backend("gpu")
else
    Reactant.set_default_backend("cpu")
end

# Rebuild `x` with some fields replaced, re-inferring its type parameters.
function rebuild(x; replacements...)
    T = typeof(x)
    return T.name.wrapper((get(replacements, n, getfield(x, n)) for n in fieldnames(T))...)
end

function traced_parameter_loss(model, θ_init, Δt, Nsteps)
    set!(model; θ=θ_init, qᵗ=0.012)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(interior(model.temperature) .^ 2)
end

function traced_parameter_grad_loss(model, dmodel, θ_init, dθ_init, Δt, Nsteps)
    parent(dθ_init) .= 0
    _, loss_value = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        traced_parameter_loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(θ_init, dθ_init),
        Enzyme.Const(Δt),
        Enzyme.Const(Nsteps))
    return dθ_init, loss_value
end

@testset "reactant_p3_traced_parameters" begin
    Δt = 1.0
    Ns = 1

    # Every number of the process rates and subgrid correlation is traced, including the
    # condensate formation's rate. The limiter pass count is a type parameter, so tracking
    # all `Number`s leaves it alone.
    condensation = ExponentialRateCondensateFormation(1 / 120, Δt)
    subgrid = GaussianSubgridCorrelation(Float64; condensation)
    process_rates = Reactant.to_rarray(ProcessRate(Float64); track_numbers=Number)
    subgrid = Reactant.to_rarray(subgrid; track_numbers=Number)
    @test process_rates.autoconversion_coefficient isa Reactant.ConcreteRNumber
    @test subgrid.condensation.rate isa Reactant.ConcreteRNumber
    microphysics = rebuild(P3Microphysics(); process_rates, subgrid)

    grid = RectilinearGrid(ReactantState(); size=ColumnEnsembleSize(Nz=8, ensemble=(2, 1), Hz=3),
                           z=(0, 2000), topology=(Flat, Flat, Bounded))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; base_pressure=101325, potential_temperature=290)
    dynamics = AnelasticDynamics(reference_state)
    closure = TKEBasedTurbulenceClosure()

    @testset "Forward" begin
        model = AtmosphereModel(grid; dynamics, microphysics, closure)
        θ_init = CenterField(grid); set!(θ_init, z -> 290 + z / 1e3)
        compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true traced_parameter_loss(
            model, θ_init, Δt, Ns)
        value = @with_stack_size compiled(model, θ_init, Δt, Ns)
        @test isfinite(value) && value > 0
    end

    @testset "Gradient" begin
        model = AtmosphereModel(grid; dynamics, microphysics, closure)
        θ_init  = CenterField(grid); set!(θ_init, z -> 290 + z / 1e3)
        dθ_init = CenterField(grid); set!(dθ_init, 0)
        dmodel  = Enzyme.make_zero(model)
        compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true traced_parameter_grad_loss(
            model, dmodel, θ_init, dθ_init, Δt, Ns)
        dθ, loss_value = @with_stack_size compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
        ad_grad = @allowscalar Array(interior(dθ))
        @test isfinite(loss_value)
        @test all(isfinite, ad_grad)
        @test any(!iszero, ad_grad)
    end
end
