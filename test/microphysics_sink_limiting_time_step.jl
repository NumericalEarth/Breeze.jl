include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudIce, CloudLiquid
using Oceananigans
using Test

using Breeze.AtmosphereModels: with_sink_limiting_time_step, sink_limiting_time_step, tendency_microphysics
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics
using Breeze.CompressibleEquations: SplitExplicitTimeDiscretization

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics, TwoMomentCloudMicrophysics

non_equilibrium_1M(FT) =
    OneMomentCloudMicrophysics(FT; cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT)),
                               precipitation_boundary_condition = ImpenetrableBoundaryCondition())

# A warm cell loaded with snow (one level, so no sedimentation between cells): melting
# drains snow on a timescale far shorter than a 60 s step. With a fixed 10 s sink-limiting
# timescale a 60 s step overdraws the snow (ρqˢⁿ ≈ -0.027 after one step), and the next step
# throws a DomainError.
function melting_snow_column(FT; Δt, Nt)
    grid = RectilinearGrid(default_arch, FT; size = 1, z = (0, 1000), topology = (Flat, Flat, Bounded))
    model = AtmosphereModel(grid; microphysics = non_equilibrium_1M(FT))
    set!(model; θ = 300, qᵗ = 2e-3, qˢⁿ = 1e-3, qʳ = 1e-4)
    snow_minima = FT[]
    for _ in 1:Nt
        time_step!(model, Δt)
        push!(snow_minima, minimum(interior(model.microphysical_fields.ρqˢⁿ)))
    end
    return model, snow_minima
end

@testset "Sink-limiting timescale follows the time step [$FT]" for FT in test_float_types()
    sink_timescale(μ::PredictedParticlePropertiesMicrophysics) = μ.process_rates.sink_limiting_timescale
    sink_timescale(μ) = μ.categories.τⁿᵘᵐ  # 1M and 2M

    for μ in (non_equilibrium_1M(FT), TwoMomentCloudMicrophysics(FT), PredictedParticlePropertiesMicrophysics(FT))
        τ = sink_timescale(μ)
        @test τ == 10
        @test sink_timescale(with_sink_limiting_time_step(μ, 60)) == 60
        @test sink_timescale(with_sink_limiting_time_step(μ, 60)) isa FT
        @test with_sink_limiting_time_step(μ, 5) == μ   # the floor applies: identical scheme
        @test with_sink_limiting_time_step(μ, 10) == μ
        @test with_sink_limiting_time_step(μ, Inf) == μ # before the first step
    end
end

@testset "Snow stays non-negative at Δt = 60 s [$FT]" for FT in test_float_types()
    model, snow_minima = melting_snow_column(FT; Δt = 60, Nt = 3)
    @test model.clock.last_Δt == 60
    @test sink_limiting_time_step(model.timestepper, model.clock) == 60
    @test tendency_microphysics(model).categories.τⁿᵘᵐ == 60
    @test all(≥(0), snow_minima)
    for name in (:ρqᶜˡ, :ρqᶜⁱ, :ρqʳ, :ρqˢⁿ)
        @test minimum(interior(model.microphysical_fields[name])) ≥ 0
    end
    @test minimum(interior(model.moisture_density)) ≥ 0
end

@testset "WS-RK3 limits sinks over the current step [$FT]" for FT in test_float_types()
    grid = RectilinearGrid(default_arch, FT; size = 1, z = (0, 1000), topology = (Flat, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 300)
    model = AtmosphereModel(grid; dynamics, microphysics = non_equilibrium_1M(FT))
    ρ = model.dynamics.reference_state.density
    set!(model; θ = 300, qᵗ = 2e-3, qˢⁿ = 1e-3, qʳ = 1e-4, ρ)

    time_step!(model, 60)
    # Stage-1 tendencies of the next step are built from the step just taken
    @test sink_limiting_time_step(model.timestepper, model.clock) ≈ 60
    time_step!(model, 30)
    @test sink_limiting_time_step(model.timestepper, model.clock) ≈ 30
    @test tendency_microphysics(model).categories.τⁿᵘᵐ ≈ 30
    @test minimum(interior(model.microphysical_fields.ρqˢⁿ)) ≥ 0
    @test minimum(interior(model.microphysical_fields.ρqʳ)) ≥ 0
end

@testset "Steps no longer than the floor are unchanged [$FT]" for FT in test_float_types()
    for Δt in (5, 10)
        model, _ = melting_snow_column(FT; Δt, Nt = 2)
        @test tendency_microphysics(model) == model.microphysics
    end
end
