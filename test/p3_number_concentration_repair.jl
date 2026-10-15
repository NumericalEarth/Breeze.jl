include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties,
                                                       NumberConcentrationRepair,
                                                       PredictedParticlePropertiesMicrophysics

using Oceananigans: RectilinearGrid, set!
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: update_state!

const PPP = PredictedParticleProperties

@testset "P3 number concentration repair" begin
    FT = Float64

    @testset "Construction" begin
        repair = NumberConcentrationRepair(FT)
        @test repair.minimum_number_density == FT(1e-3)
        @test repair.rain_intercept == FT(1e7)
        @test repair.ice_number_minimum_temperature == FT(233)
        @test PredictedParticlePropertiesMicrophysics(FT).number_repair isa NumberConcentrationRepair{FT}
        @test isnothing(PredictedParticlePropertiesMicrophysics(FT; number_repair = nothing).number_repair)
    end

    # Two lower cells hold a trace of rain and ice with no number; the two upper cells hold
    # rain and ice with a healthy number, which the repair must leave alone.
    trace_mass = FT(1e-12)
    Nz = 4
    is_trace(z) = z < 200

    function repaired_model(number_repair)
        grid = RectilinearGrid(default_arch, FT; size = (1, 1, Nz), x = (0, 100), y = (0, 100), z = (0, 400))
        constants = ThermodynamicConstants(FT)
        reference_state = ReferenceState(grid, constants;
                                         base_pressure = FT(101325),
                                         potential_temperature = FT(285))
        dynamics = AnelasticDynamics(reference_state)
        p3 = PredictedParticlePropertiesMicrophysics(FT; number_repair)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants, microphysics = p3)
        set!(model; θ = FT(285), qᵛ = FT(1e-3),
             qʳ = (x, y, z) -> is_trace(z) ? trace_mass : FT(1e-3),
             nʳ = (x, y, z) -> is_trace(z) ? zero(FT) : FT(1e5),
             qⁱ = (x, y, z) -> is_trace(z) ? trace_mass : FT(1e-4),
             nⁱ = (x, y, z) -> is_trace(z) ? zero(FT) : FT(1e4),
             enforce_mass_conservation = false)
        update_state!(model)
        return model
    end

    repaired = repaired_model(NumberConcentrationRepair(FT))
    unrepaired = repaired_model(nothing)
    column(model, name) = vec(Array(interior(getproperty(model.microphysical_fields, name))))
    trace = 1:2
    healthy = 3:4

    @testset "Mass without number gets a bounded number" begin
        p3 = repaired.microphysics
        parameters = p3.process_rates

        ρnʳ = column(repaired, :ρnʳ)
        ρnⁱ = column(repaired, :ρnⁱ)
        @test all(ρnʳ[trace] .>= p3.number_repair.minimum_number_density)
        @test all(ρnⁱ[trace] .> 0)

        # A trace of rain is given the smallest drops the slope bounds admit, not the largest.
        qʳ = column(repaired, :qʳ)
        nʳ = column(repaired, :nʳ)
        for k in trace
            λʳ = PPP.unbounded_rain_slope_parameter(qʳ[k], nʳ[k], parameters)
            @test λʳ ≈ parameters.maximum_rain_slope rtol = 1e-6
        end

        # The written ice number already satisfies P3's bounds: bounding it again is a no-op.
        ρ = vec(Array(interior(Breeze.AtmosphereModels.total_density(repaired.dynamics))))
        qⁱ = column(repaired, :qⁱ)
        nⁱ = column(repaired, :nⁱ)
        for k in trace
            bounded = @allowscalar PPP.p3_ice_moments(p3, ρ[k], qⁱ[k], nⁱ[k], zero(FT), zero(FT), zero(FT)).nⁱ
            @test bounded ≈ nⁱ[k] rtol = 1e-6
        end

        # Without the repair the number stays zero.
        @test all(column(unrepaired, :ρnʳ)[trace] .== 0)
        @test all(column(unrepaired, :ρnⁱ)[trace] .== 0)
    end

    @testset "Populations with a number are left alone" begin
        for name in (:ρnʳ, :ρnⁱ)
            @test column(repaired, name)[healthy] == column(unrepaired, name)[healthy]
        end
    end

    @testset "Only number changes; no water moves" begin
        for name in (:ρqᶜˡ, :ρqʳ, :ρqⁱ, :ρqʷⁱ, :ρqᶠ, :ρbᶠ)
            @test column(repaired, name) == column(unrepaired, name)
        end
        @test Array(interior(repaired.moisture_density)) == Array(interior(unrepaired.moisture_density))
    end

    @testset "Time steps longer than the sink-limiting timescale warn" begin
        p3 = repaired.microphysics
        τ = p3.process_rates.sink_limiting_timescale
        repaired.clock.last_Δt = τ / 2
        @test_logs Breeze.AtmosphereModels.microphysics_model_update!(p3, repaired)
        repaired.clock.last_Δt = 2τ
        @test_logs (:warn, r"sink_limiting_timescale") Breeze.AtmosphereModels.microphysics_model_update!(p3, repaired)
    end

    @testset "Trace mass falls slowly" begin
        # wʳ and wⁱ at the bottom face of each trace cell carry that cell's fall speed.
        for name in (:wʳ, :wⁱ)
            repaired_speed = maximum(abs, column(repaired, name)[trace])
            unrepaired_speed = maximum(abs, column(unrepaired, name)[trace])
            @test repaired_speed < FT(0.1)
            @test unrepaired_speed > FT(0.5)
        end
    end
end
