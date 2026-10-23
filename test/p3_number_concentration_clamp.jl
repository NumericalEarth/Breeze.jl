include(joinpath(@__DIR__, "setup.jl"))

using Test

using Breeze
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties

using Oceananigans: RectilinearGrid, set!
using Oceananigans.Fields: interior

const PPP = PredictedParticleProperties

@testset "P3 number concentrations are clamped to the admissible range" begin
    FT = Float64

    # Four cells, bottom to top: (1) rain and ice with mass but no number, (2) mass with far
    # more number than its smallest admissible particles allow, (3) a population inside the
    # bounds, (4) ice number above the global cap.
    grid = RectilinearGrid(default_arch, FT; size = (1, 1, 4), x = (0, 100), y = (0, 100), z = (0, 400))
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants;
                                     base_pressure = FT(101325),
                                     potential_temperature = FT(285))
    dynamics = AnelasticDynamics(reference_state)
    p3 = PredictedParticlePropertiesMicrophysics(FT)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants = constants, microphysics = p3)
    parameters = p3.process_rates

    cell(z) = z < 100 ? 1 : z < 200 ? 2 : z < 300 ? 3 : 4
    pick(values) = (x, y, z) -> values[cell(z)]
    qʳ_set = (1e-6, 1e-6, 1e-3, 1e-4)
    nʳ_set = (0.0, 1e12, 1e5, 1e5)
    qⁱ_set = (1e-6, 1e-12, 1e-4, 1e-3)
    nⁱ_set = (0.0, 1e6, 1e4, 1e9)
    # `set!` ends with `update_state!`, whose negative-moisture correction applies the clamp.
    set!(model; θ = 285, qᵛ = 1e-3,
         qʳ = pick(qʳ_set), nʳ = pick(nʳ_set), qⁱ = pick(qⁱ_set), nⁱ = pick(nⁱ_set),
         enforce_mass_conservation = false)

    column(name) = vec(Array(interior(model.microphysical_fields[name])))
    ρ = vec(Array(interior(Breeze.AtmosphereModels.total_density(model.dynamics))))
    qʳ, nʳ, qⁱ, nⁱ = column(:qʳ), column(:nʳ), column(:qⁱ), column(:nⁱ)

    @testset "Rain is clamped onto the slope bounds and left alone inside them" begin
        λʳ(k) = PPP.unbounded_rain_slope_parameter(qʳ[k], nʳ[k], parameters)
        @test λʳ(1) ≈ parameters.minimum_rain_slope rtol = 1e-6   # no number: largest drops
        @test λʳ(2) ≈ parameters.maximum_rain_slope rtol = 1e-6   # too many drops: smallest drops
        @test nʳ[3] ≈ nʳ_set[3] rtol = 1e-6
        @test nʳ[4] ≈ nʳ_set[4] rtol = 1e-6
        # Clamping again is a no-op: the prognostic satisfies the bounds every read applies.
        for k in 1:4
            @test PPP.bounded_rain_number(nʳ[k], qʳ[k], parameters) ≈ nʳ[k] rtol = 1e-6
        end
    end

    @testset "Ice is clamped onto the λ limits and the global cap" begin
        @test nⁱ[1] > 0
        @test nⁱ[2] < nⁱ_set[2]
        @test nⁱ[3] ≈ nⁱ_set[3] rtol = 1e-6
        @test ρ[4] * nⁱ[4] ≈ parameters.maximum_ice_number_density rtol = 1e-6
        for k in 1:4
            bounded = @allowscalar PPP.p3_ice_moments(p3, ρ[k], qⁱ[k], nⁱ[k], zero(FT), zero(FT), zero(FT)).nⁱ
            @test bounded ≈ nⁱ[k] rtol = 1e-6
        end
    end

    @testset "The clamp runs in the negative-moisture correction and moves no water" begin
        water_names = (:ρqᶜˡ, :ρqʳ, :ρqⁱ, :ρqʷⁱ, :ρqᶠ, :ρbᶠ)
        water_before = Dict(name => column(name) for name in water_names)
        moisture_before = Array(interior(model.moisture_density))
        μ = model.microphysical_fields
        set!(μ.ρnʳ, 0)
        set!(μ.ρnⁱ, 1e12)
        Breeze.AtmosphereModels.fix_negative_moisture!(model)
        for name in water_names
            @test column(name) == water_before[name]
        end
        @test Array(interior(model.moisture_density)) == moisture_before
        # Only the prognostic densities are current here; the specific numbers are refreshed
        # later in `update_state!`. A wiped rain number lands on the lower bound for the
        # cell's mass (the largest admissible drops); an inflated ice number lands on the
        # global cap, or on the upper λ bound where that is lower (cell 2, a trace of ice).
        nʳ_clamped = column(:ρnʳ) ./ ρ
        nⁱ_clamped = column(:ρnⁱ) ./ ρ
        for k in 1:4
            @test PPP.unbounded_rain_slope_parameter(qʳ[k], nʳ_clamped[k], parameters) ≈ parameters.minimum_rain_slope rtol = 1e-6
            @test ρ[k] * nⁱ_clamped[k] <= parameters.maximum_ice_number_density * (1 + 1e-6)
        end
        for k in (1, 3, 4)
            @test ρ[k] * nⁱ_clamped[k] ≈ parameters.maximum_ice_number_density rtol = 1e-6
        end
        @test ρ[2] * nⁱ_clamped[2] < parameters.maximum_ice_number_density
    end

    @testset "Time steps longer than the sink-limiting timescale warn" begin
        τ = parameters.sink_limiting_timescale
        model.clock.last_Δt = τ / 2
        @test_logs Breeze.AtmosphereModels.microphysics_model_update!(p3, model)
        model.clock.last_Δt = 2τ
        @test_logs (:warn, r"sink_limiting_timescale") Breeze.AtmosphereModels.microphysics_model_update!(p3, model)
    end

    @testset "Fall speeds follow the clamped sizes" begin
        # wʳ at the bottom face of each cell carries that cell's fall speed: the largest
        # admissible drops fall fast, the smallest barely at all.
        wʳ = column(:wʳ)
        @test abs(wʳ[1]) > 5
        @test abs(wʳ[2]) < 0.1
    end
end
