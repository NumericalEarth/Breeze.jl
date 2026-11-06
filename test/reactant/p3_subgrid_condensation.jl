include(joinpath(@__DIR__, "microphysics_compilation_setup.jl"))

#####
##### Reactant forward and gradient compilation of P3 with Gaussian subgrid condensation,
##### its variance supplied by the TKE closure
#####

using Breeze.AtmosphereModels: GaussianSubgridCorrelation

@testset "reactant_p3_subgrid_condensation" begin
    Δt = 0.02
    Ns = 1
    condensation = ExponentialRateCondensateFormation(1 / 120, Δt)
    subgrid = GaussianSubgridCorrelation(Float64; condensation)
    microphysics = P3Microphysics(; subgrid)
    initial_state = (; qᵛ=0.01, qᶜˡ=1e-4, qʳ=1e-5, nʳ=1e6, qⁱ=0, nⁱ=0)

    grid = RectilinearGrid(ReactantState(); size=(4, 4, 8), extent=(1e3, 1e3, 1e3),
                           topology=(Periodic, Periodic, Bounded))
    closure = TKEBasedTurbulenceClosure()

    @testset "Forward" begin
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), microphysics, closure)
        θ_init = CenterField(grid); set!(θ_init, (x, y, z) -> 300 + z / 1e3)
        compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true loss(
            model, θ_init, initial_state, Δt, Ns)
        value = @with_stack_size compiled(model, θ_init, initial_state, Δt, Ns)
        @test isfinite(value) && value > 0
    end

    @testset "Gradient" begin
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), microphysics, closure)
        θ_init  = CenterField(grid); set!(θ_init, (x, y, z) -> 300 + z / 1e3)
        dθ_init = CenterField(grid); set!(dθ_init, 0)
        dmodel  = Enzyme.make_zero(model)
        compiled_grad = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true grad_loss(
            model, dmodel, θ_init, dθ_init, initial_state, Δt, Ns)
        dθ, loss_value = @with_stack_size compiled_grad(model, dmodel, θ_init, dθ_init, initial_state, Δt, Ns)
        ad_grad = @allowscalar Array(interior(dθ))
        @test isfinite(loss_value)
        @test all(isfinite, ad_grad)
        @test any(!iszero, ad_grad)
    end
end
