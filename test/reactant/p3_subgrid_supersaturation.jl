include(joinpath(@__DIR__, "microphysics_compilation_setup.jl"))

#####
##### Reactant forward and gradient compilation of P3 with a subgrid supersaturation model
##### (a dummy that reads a field on the grid) in its cloud condensation and evaporation hook
#####

using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties

struct ScaledSupersaturation{FT}
    Γ :: FT
end

PredictedParticleProperties.subgrid_supersaturation_fields(::ScaledSupersaturation, model) = (; scale = model.velocities.w)

@inline PredictedParticleProperties.subgrid_supersaturation_statistics(i, j, k, grid, ::ScaledSupersaturation, fields) =
    @inbounds 1 + fields.scale[i, j, k]

@inline PredictedParticleProperties.subgrid_supersaturation(model::ScaledSupersaturation, scale,
                                                            qᵛ, qᵛ⁺ˡ, qᶜˡ, T, P, ρ, constants) = model.Γ * scale

@testset "reactant_p3_subgrid_supersaturation" begin
    Δt = 0.02
    Ns = 1
    microphysics = P3Microphysics(; subgrid_supersaturation = ScaledSupersaturation(5e-4))
    initial_state = (; qᵛ=0.01, qᶜˡ=1e-4, qʳ=1e-5, nʳ=1e6, qⁱ=0, nⁱ=0)

    grid = RectilinearGrid(ReactantState(); size=(4, 4, 8), extent=(1e3, 1e3, 1e3),
                           topology=(Periodic, Periodic, Bounded))

    @testset "Forward" begin
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), microphysics)
        θ_init = CenterField(grid); set!(θ_init, (x, y, z) -> 300 + z / 1e3)
        compiled = @with_stack_size Reactant.@compile raise=true raise_first=true sync=true loss(
            model, θ_init, initial_state, Δt, Ns)
        value = @with_stack_size compiled(model, θ_init, initial_state, Δt, Ns)
        @test isfinite(value) && value > 0
    end

    @testset "Gradient" begin
        model = AtmosphereModel(grid; dynamics=CompressibleDynamics(), microphysics)
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
