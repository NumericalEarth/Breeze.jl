include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, ProcessRate

using Oceananigans: RectilinearGrid, set!
using Oceananigans.Fields: interior

const PPP = PredictedParticleProperties

@testset "P3 number concentration repair" begin
    FT = Float64

    # Two lower cells hold a trace of rain and ice with no number; the two upper cells hold
    # rain and ice with a healthy number, which the repair must leave alone.
    trace_mass = FT(1e-12)
    is_trace(z) = z < 200

    function trace_model(process_rates)
        grid = RectilinearGrid(default_arch, FT; size = (1, 1, 4), x = (0, 100), y = (0, 100), z = (0, 400))
        constants = ThermodynamicConstants(FT)
        reference_state = ReferenceState(grid, constants;
                                         base_pressure = FT(101325),
                                         potential_temperature = FT(285))
        dynamics = AnelasticDynamics(reference_state)
        p3 = PredictedParticlePropertiesMicrophysics(FT; process_rates)
        model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants, microphysics = p3)
        # `set!` ends with `update_state!`, which applies the repair.
        set!(model; θ = 285, qᵛ = 1e-3,
             qʳ = (x, y, z) -> is_trace(z) ? trace_mass : 1e-3,
             nʳ = (x, y, z) -> is_trace(z) ? 0 : 1e5,
             qⁱ = (x, y, z) -> is_trace(z) ? trace_mass : 1e-4,
             nⁱ = (x, y, z) -> is_trace(z) ? 0 : 1e4,
             enforce_mass_conservation = false)
        return model
    end

    repaired = trace_model(ProcessRate(FT))
    # A zero threshold never fires, and keeps the scheme type (and its compiled kernels) unchanged.
    unrepaired = trace_model(ProcessRate(FT; minimum_number_density = 0))
    column(model, name) = vec(Array(interior(model.microphysical_fields[name])))
    trace = 1:2
    healthy = 3:4

    @testset "Mass without number gets a bounded number" begin
        p3 = repaired.microphysics
        parameters = p3.process_rates

        @test all(column(repaired, :ρnʳ)[trace] .>= parameters.minimum_number_density)
        @test all(column(repaired, :ρnⁱ)[trace] .> 0)

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
            @test maximum(abs, column(repaired, name)[trace]) < 0.1
            @test maximum(abs, column(unrepaired, name)[trace]) > 0.5
        end
    end
end
