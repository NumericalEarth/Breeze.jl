include(joinpath(@__DIR__, "setup.jl"))

#####
##### Implicit relaxation of P3 cloud condensation, rain vapor exchange and droplet activation
##### toward their targets over the coming stage (`LiquidRelaxationStage`)
#####

using Breeze
using Breeze.AtmosphereModels: StageStart, implicit_relaxation, implicit_relaxation_rate,
                               shared_excess_relaxation_rates
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticleProperties, P3MicrophysicalState,
                                                       LiquidRelaxationStage, GridLiquidRelaxationStage,
                                                       compute_p3_process_rates, liquid_equilibrium_condensate
using Breeze.Thermodynamics: ThermodynamicConstants, MoistureMassFractions, LiquidIcePotentialTemperatureState,
                             with_temperature, with_moisture, saturation_specific_humidity,
                             PlanarLiquidSurface, PlanarIceSurface, density, temperature, air_pressure
using Enzyme: Enzyme, autodiff, Forward, Duplicated, Const
using Test

const PPP = PredictedParticleProperties

function cell_state(qᵛ, qᶜˡ, constants; T = 285.0)
    FT = typeof(qᵛ)
    q = MoistureMassFractions(qᵛ, qᶜˡ, zero(FT))
    return with_temperature(LiquidIcePotentialTemperatureState(zero(FT), q, FT(1e5), FT(9e4)), FT(T), constants)
end

function cell_saturation(qᶜˡ, constants; T = 285.0)
    FT = typeof(qᶜˡ)
    𝒰 = cell_state(FT(1e-2), qᶜˡ, constants; T)
    return saturation_specific_humidity(FT(T), density(𝒰, constants), constants, PlanarLiquidSurface())
end

cell_microphysics(qᶜˡ, nᶜˡ; qʳ = zero(qᶜˡ), nʳ = zero(qᶜˡ)) =
    P3MicrophysicalState(qᶜˡ, nᶜˡ, qʳ, nʳ, ntuple(_ -> zero(qᶜˡ), 8)...)

# The rates of one cell at the state (qᵛ, qᶜˡ), for a stage of length Δt that starts from that
# state (c̃ = qᶜˡ), or without a stage (the explicit rates)
function stage_rates(p3, qᵛ, qᶜˡ, constants, Δt; qʳ = zero(qᵛ), nʳ = zero(qᵛ))
    FT = typeof(qᵛ)
    ρ = one(FT)
    𝒰 = cell_state(qᵛ, qᶜˡ, constants)
    ℳ = cell_microphysics(qᶜˡ, FT(1e8); qʳ, nʳ)
    stage = isnothing(Δt) ? nothing : LiquidRelaxationStage(FT(Δt), qᶜˡ, FT(1e8), true)
    return compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing, temperature(𝒰, constants),
                                    zero(FT), zero(FT), nothing, stage)
end

# The vapor relaxation rate εˡ of the cloud droplets at saturation
function cloud_relaxation_rate(p3, qᶜˡ, constants)
    FT = typeof(qᶜˡ)
    𝒰 = cell_state(cell_saturation(qᶜˡ, constants), qᶜˡ, constants)
    transport = PPP.air_transport_properties(temperature(𝒰, constants), air_pressure(𝒰, constants), constants)
    cloud = PPP.diagnose_cloud_dsd(p3, qᶜˡ, FT(1e8), one(FT))
    return PPP.cloud_vapor_relaxation_coefficient(p3, qᶜˡ, one(FT), transport.Dᵛ, cloud.μᶜˡ, cloud.λᶜˡ, cloud.nᶜˡ)
end

@testset "Implicit relaxation helpers [$FT]" for FT in (Float32, Float64)
    for ωΔt in FT.((0.1, 1, 10, 1e3)), (c, c_eq) in ((FT(0), FT(1)), (FT(1), FT(0)), (FT(0.3), FT(0.7)))
        c⁺ = implicit_relaxation(c, c_eq, ωΔt)
        @test c⁺ isa FT
        @test min(c, c_eq) ≤ c⁺ ≤ max(c, c_eq)
        @test c + implicit_relaxation_rate(c, c_eq, ωΔt, one(FT)) ≈ c⁺
    end
    x⁺ = FT(2e-4) - FT(10) * sum(shared_excess_relaxation_rates((FT(2e-4), FT(2e-4)), (FT(0.5), FT(1.5)), FT(10)))
    @test x⁺ ≈ FT(2e-4) / (1 + 10 * 2)

    # Without a model of condensation the equilibrium condensate is that of the grid mean
    @test liquid_equilibrium_condensate(nothing, FT(3e-4), FT(1.5)) ≈ FT(2e-4)
    @test liquid_equilibrium_condensate(nothing, FT(-3e-4), FT(1.5)) == 0
    # Subgrid condensation: the equilibrium condensate ⟨s⁺⟩ / ξˡ of the distribution, ⟨s⁺⟩ = s̄ + Γ
    @test liquid_equilibrium_condensate(FT(1.5e-4), FT(-3e-4 / 2), FT(1.5)) == 0
    @test liquid_equilibrium_condensate(FT(4.5e-4), FT(-1.5e-4), FT(1.5)) ≈ FT(2e-4)
end

# March one cell by backward-Euler stages: c ← c + Δt × rate. Water is conserved and the vapor
# excess is updated with the cloud, so the saturation excess of total water is held fixed.
function march(p3, qᵛ, qᶜˡ, constants, Δt, n)
    history = [(qᵛ, qᶜˡ)]
    for _ in 1:n
        r = stage_rates(p3, qᵛ, qᶜˡ, constants, Δt)
        δ = Δt * (r.condensation + r.ccn_activation_mass)
        qᵛ, qᶜˡ = qᵛ - δ, qᶜˡ + δ
        push!(history, (qᵛ, qᶜˡ))
    end
    return history
end

@testset "One cell at ωΔt ∈ {0.1, 1, 10, 1e3}: non-negative, monotone, no overshoot [$FT]" for FT in (Float32, Float64)
    constants = ThermodynamicConstants(FT)
    p3 = P3Microphysics(FT)
    εˡ = cloud_relaxation_rate(p3, FT(3e-4), constants)
    @test εˡ > 0
    for ωΔt in (0.1, 1, 10, 1e3), (excess, qᶜˡ) in ((2e-4, 3e-4), (-2e-4, 3e-4), (-1e-3, 3e-4))
        Δt = FT(ωΔt / εˡ)
        qᵛ = cell_saturation(FT(qᶜˡ), constants) + FT(excess)
        history = march(p3, qᵛ, FT(qᶜˡ), constants, Δt, 30)
        c = last.(history)
        @test all(≥(0), c)
        increments = diff(c)
        # monotone: the cloud moves one way only, up to the round-off of the vapor excess
        # qᵛ - qᵛ⁺ˡ, a difference of numbers of size qᵛ, near equilibrium
        tol = 10 * eps(FT) * qᵛ
        monotone = all(≥(-tol), increments) || all(≤(tol), increments)
        @test monotone
        # no overshoot: the first stage moves the cloud no further than its last value
        # (the equilibrium the march converges to)
        @test abs(c[2] - c[1]) ≤ abs(c[end] - c[1]) + tol
        @test all(isfinite, c)
    end
end

@testset "Implicit relaxation conserves water and energy [$FT]" for FT in (Float32, Float64)
    constants = ThermodynamicConstants(FT)
    p3 = P3Microphysics(FT)
    for (excess, qᶜˡ) in ((2e-4, 2e-4), (-1e-4, 3e-4), (2e-4, 0.0))
        qᵛ = cell_saturation(FT(qᶜˡ), constants) + FT(excess)
        𝒰 = cell_state(qᵛ, FT(qᶜˡ), constants)
        ρ = FT(1.1)
        ℳ = cell_microphysics(FT(qᶜˡ), FT(1e8); qʳ = FT(1e-4), nʳ = FT(1e4))
        stage = LiquidRelaxationStage(FT(30), FT(qᶜˡ), FT(1e8), true)
        rates = compute_p3_process_rates(p3, ρ, ℳ, 𝒰, constants, nothing, temperature(𝒰, constants),
                                         zero(FT), zero(FT), nothing, stage)
        properties = PPP.p3_process_properties(p3, ρ, ℳ)
        result = PPP.p3_tendencies(p3, ρ, ℳ, properties, rates)
        water = result.tendency_ρqᵛ + result.tendency_ρqᶜˡ + result.tendency_ρqʳ + result.tendency_ρqⁱ + result.tendency_ρqʷⁱ
        scale = abs(result.tendency_ρqᵛ) + eps(FT)
        @test abs(water) ≤ 10 * eps(FT) * scale
        @test PPP.p3_tendency_component(result, Val(:ρθ)) == 0
    end
end

@testset "Implicit and explicit rates agree as ωΔt → 0 [$FT]" for FT in (Float32, Float64)
    constants = ThermodynamicConstants(FT)
    p3 = P3Microphysics(FT; process_rates = PPP.ProcessRate(FT; sink_limiting_timescale = FT(1e-3)))
    qᵛ = cell_saturation(FT(3e-4), constants) + FT(1e-5)
    implicit = stage_rates(p3, qᵛ, FT(3e-4), constants, FT(1e-6)).condensation
    explicit = stage_rates(p3, qᵛ, FT(3e-4), constants, nothing).condensation
    @test implicit ≈ explicit rtol = 1e-2
end

@testset "The explicit rates are used without a stage and before the first time step" begin
    FT = Float64
    constants = ThermodynamicConstants(FT)
    p3 = P3Microphysics(FT)
    qᵛ = cell_saturation(3e-4, constants) + 2e-4
    𝒰 = cell_state(qᵛ, 3e-4, constants)
    ℳ = cell_microphysics(3e-4, 1e8; qʳ = 1e-4, nʳ = 1e4)
    arguments = (p3, 1.0, ℳ, 𝒰, constants, nothing, temperature(𝒰, constants), 0.0, 0.0)
    explicit = compute_p3_process_rates(arguments...)
    @test compute_p3_process_rates(arguments..., nothing, nothing) == explicit

    # On the grid the stage is read from the tendencies and the state the stage starts from
    fields = map(x -> fill(x, 1, 1, 1), (; ρqᵛ = qᵛ, ρqᶜˡ = 3e-4, ρqʳ = 1e-4, ρqʷⁱ = 0.0))
    G = map(x -> fill(x, 1, 1, 1), (; ρqᵛ = 0.0, ρqᶜˡ = -1e-6, ρqʳ = 0.0, ρqʷⁱ = 0.0))
    unknown = GridLiquidRelaxationStage(1, 1, 1, G, StageStart(fields, fields, false, Inf))
    @test compute_p3_process_rates(arguments..., nothing, unknown) == explicit

    known = GridLiquidRelaxationStage(1, 1, 1, G, StageStart(fields, fields, false, 20.0))
    stage = PPP.liquid_relaxation_stage(known, nothing, 1.0)
    @test stage.active
    @test stage.Δt == 20
    @test stage.qᶜˡ ≈ 3e-4 - 20 * 1e-6
    @test stage.Δqᵛ == 0
    @test stage.qʳ == 1e-4
    @test stage.qʷⁱ == 0
    implicit = compute_p3_process_rates(arguments..., nothing, known)
    @test implicit == compute_p3_process_rates(arguments..., nothing, stage)
    @test 0 < implicit.condensation < explicit.condensation
end

@testset "Ice cap sees the stage-integrated liquid exchange" begin
    constants = ThermodynamicConstants(Float64)
    T, P, τ = 263.0, 9e4, 10.0
    vapor = 2.3e-3
    function capped_deposition(cond, interval; coat_cond=1e-7)
        liquid = (; cond, ccn_activation_mass=0.0, ccn_activation_number=0.0,
                  rain_cond=0.0, rain_evap=0.0, coat_cond, coat_evap=0.0)
        return PPP.limit_ice_vapor_rates(liquid, 1e-3, 0.0, 0.0, vapor, T, P, vapor,
                                         constants, τ, 273.15, 0.0, 0.0, interval).dep
    end
    @test 0 < capped_deposition(1e-5, 2.0) < capped_deposition(0.0, 2.0)
    # Cloud and coating exchange are both projected over the stage interval
    @test capped_deposition(1e-5, 2.0) ≈ capped_deposition(2e-6, τ; coat_cond=2e-8)
    @test capped_deposition(0.0, 2.0) > capped_deposition(0.0, 2.0; coat_cond=1e-5)
    # A stage longer than the limiting timescale consumes the ice excess once, not Δt / τ times.
    @test 50 * capped_deposition(0.0, 50.0; coat_cond=0.0) ≈ τ * capped_deposition(0.0, τ; coat_cond=0.0)
end

@testset "Stage vapor excess follows the limited phase changes, not the host" begin
    constants = ThermodynamicConstants(Float64)
    p3 = P3Microphysics(Float64; vapor_excess=PrognosticVaporExcess())
    ρ, cloud = 1.0, 3e-4
    vapor = cell_saturation(cloud, constants) + 2e-4
    𝒰 = cell_state(vapor, cloud, constants)
    # The prognostic vapor excess is zero and the host's is not, so the alignment is active.
    ℳ = cell_microphysics(cloud, 1e8)
    host_excess = PPP.vapor_excess_after_moisture_update(𝒰, vapor, cloud, 0.0, constants)
    @test abs(host_excess - ℳ.δqᵛ⁺ˡ) > 1e-4
    inputs = (; ρ, 𝒰, ℳ, temperature_tendency=0.0, pressure_tendency=0.0)
    properties = PPP.p3_process_properties(p3, ρ, ℳ)

    function limited_stage(Δt, ρqᵛ)
        rates = stage_rates(p3, vapor, cloud, constants, Δt)
        fields = (; ρqᵛ, ρqᶜˡ=cloud, ρqʳ=0.0, ρnʳ=0.0, ρqⁱ=0.0, ρnⁱ=0.0, ρqʷⁱ=0.0)
        fields = map(x -> fill(x, 1, 1, 1), fields)
        G = map(zero, fields)
        stage = StageStart(fields, fields, false, Δt)
        φ = PPP.p3_sink_fractions(1, 1, 1, G, stage, p3, ρ, ℳ, properties, rates)
        limited = PPP.patankar_limited_rates(rates, φ)
        tendency = PPP.stage_vapor_excess_tendency(p3.process_rates, inputs, stage, constants,
                                                   limited, rates.condensation)
        return rates, limited, tendency
    end

    for Δt in (2.0, 20.0)
        # Nothing is limited: the tendency is the one the rates project from the aligned state,
        # less the part of it that takes the prognostic excess to the diagnosis of that state.
        rates, limited, tendency = limited_stage(Δt, vapor)
        @test limited.condensation ≈ rates.condensation
        ε = p3.process_rates.sink_limiting_timescale * rates.prognostic_vapor_excess_adjustment
        aligned_excess = PPP.vapor_excess_after_moisture_update(𝒰, vapor - ε, cloud + ε, 0.0, constants)
        @test tendency / ρ ≈ rates.prognostic_vapor_excess_tendency - (aligned_excess - ℳ.δqᵛ⁺ˡ) / Δt rtol=1e-6

        # A stage with an exhausted vapor donor applies no condensation and no alignment. The
        # prognostic vapor excess then stays where it is, whatever the stage interval, and is
        # not drawn to the host's.
        rates, limited, tendency = limited_stage(Δt, 0.0)
        @test rates.condensation > 0
        @test limited.condensation == 0
        @test abs(Δt * tendency / ρ) < 0.05 * abs(host_excess - ℳ.δqᵛ⁺ˡ)
    end
end

@testset "Transported cloud does not drive condensation or evaporation" begin
    constants = ThermodynamicConstants(Float64)
    p3 = P3Microphysics(Float64)
    cloud, Δt = 3e-4, 20.0
    ℳ = cell_microphysics(cloud, 1e8)
    function condensation(vapor, remainder)
        𝒰 = cell_state(vapor, cloud, constants)
        return compute_p3_process_rates(p3, 1.0, ℳ, 𝒰, constants, nothing, 285.0, 0.0, 0.0, nothing,
                                        LiquidRelaxationStage(Δt, remainder, 1e8, true)).condensation
    end
    # Keep the rate state fixed and advect cloud into and out of the cell.
    saturated = cell_saturation(cloud, constants)
    for remainder in (cloud / 2, 2cloud)
        @test condensation(saturated, remainder) ≈ condensation(saturated, cloud)
    end

    # In dry air the cloud evaporates no more than the stage leaves of it...
    dry = saturated - 1e-3
    @test condensation(dry, cloud) < 0
    @test 0 < -Δt * condensation(dry, 1e-5) ≤ 1e-5
    # ...and a negative remainder is refilled from vapor, not left negative.
    refill = Δt * condensation(dry, -1e-5)
    @test 0 < refill ≤ 1e-5
end

@testset "Stages that restart from the step start keep the condensate of earlier stages" begin
    constants = ThermodynamicConstants(Float64)
    p3 = P3Microphysics(Float64)
    cloud = 3e-4
    # Three stages of growing length from one step start, with no transport: each is evaluated
    # at the state the one before it left, which is nearly saturated after the first.
    for excess in (1e-4, -1e-4)
        vapor = cell_saturation(cloud, constants) + excess
        start = cell_state(vapor, cloud, constants)
        state, rate_vapor, rate_cloud = start, vapor, cloud
        increments = map((100.0, 150.0, 300.0)) do Δt
            stage = LiquidRelaxationStage(Δt, cloud, 1e8, true, vapor - rate_vapor, Inf, Inf)
            r = compute_p3_process_rates(p3, 1.0, cell_microphysics(rate_cloud, 1e8), state, constants,
                                         nothing, 285.0, 0.0, 0.0, nothing, stage)
            increment = Δt * (r.condensation + r.ccn_activation_mass)
            rate_vapor, rate_cloud = vapor - increment, cloud + increment
            state = with_moisture(start, MoistureMassFractions(rate_vapor, rate_cloud, 0.0))
            increment
        end
        @test all(sign.(increments) .== sign(excess))
        # A longer stage from the same start exchanges no less, and none overshoots saturation
        @test abs(increments[1]) ≤ abs(increments[2]) ≤ abs(increments[3])
        @test abs(increments[3]) < 1.1 * abs(increments[1])
        final = PPP.vapor_excess_after_moisture_update(start, rate_vapor, rate_cloud, 0.0, constants)
        @test 0 ≤ final / excess < 0.05
    end
end

@testset "Prognostic vapor excess does not move without phase change or forcing" begin
    constants = ThermodynamicConstants(Float64)
    p3 = P3Microphysics(Float64; vapor_excess=PrognosticVaporExcess())
    # No cloud to evaporate, so the alignment cannot bring the host to the prognostic excess
    vapor = cell_saturation(0.0, constants) - 6.6e-3
    𝒰 = cell_state(vapor, 0.0, constants)
    ℳ = P3MicrophysicalState(ntuple(_ -> 0.0, 9)..., -1e-3, 0.0, 0.0)
    host_excess = PPP.vapor_excess_after_moisture_update(𝒰, vapor, 0.0, 0.0, constants)
    @test abs(host_excess - ℳ.δqᵛ⁺ˡ) > 5e-3
    inputs = (; ρ=1.0, 𝒰, ℳ, temperature_tendency=0.0, pressure_tendency=0.0)
    fields = map(x -> fill(x, 1, 1, 1), (; ρqᵛ=vapor, ρqᶜˡ=0.0, ρqʳ=0.0, ρnʳ=0.0, ρqⁱ=0.0, ρnⁱ=0.0, ρqʷⁱ=0.0))
    for Δt in (0.2, 2.0)
        rates = compute_p3_process_rates(p3, 1.0, ℳ, 𝒰, constants, nothing, 285.0, 0.0, 0.0, nothing,
                                         LiquidRelaxationStage(Δt, 0.0, 0.0, true))
        @test rates.condensation == 0
        @test rates.deposition == 0
        stage = StageStart(fields, fields, false, Δt)
        @test PPP.stage_vapor_excess_tendency(p3.process_rates, inputs, stage, constants,
                                              rates, rates.condensation) == 0
    end
end

@testset "Ice cap sees only the liquid exchange that the stage applies" begin
    constants = ThermodynamicConstants(Float64)
    p3 = P3Microphysics(Float64)
    T, Δt = 263.0, 300.0
    𝒰 = cell_state(1e-3, 0.0, constants; T)
    liquid_saturation = saturation_specific_humidity(T, density(𝒰, constants), constants, PlanarLiquidSurface())
    ice_saturation = saturation_specific_humidity(T, density(𝒰, constants), constants, PlanarIceSurface())
    ice_excess = (liquid_saturation - ice_saturation) / 50
    𝒰 = cell_state(ice_saturation + ice_excess, 0.0, constants; T)
    # Rain at the state the rates see, in air below liquid and just above ice saturation
    ℳ = P3MicrophysicalState(0.0, 0.0, 1e-3, 1e4, 1e-3, 1e6, ntuple(_ -> 0.0, 6)...)
    rates(rain) = compute_p3_process_rates(p3, 1.0, ℳ, 𝒰, constants, nothing, T, 0.0, 0.0, nothing,
                                           LiquidRelaxationStage(Δt, 0.0, 0.0, true, 0.0, rain, Inf))
    supplied, empty = rates(Inf), rates(0.0)
    # The rate asks for the same evaporation either way; the stage applies none of it where it
    # leaves no rain, and deposition then takes no more than the ice excess holds
    @test supplied.rain_evaporation > 0
    @test empty.rain_evaporation == supplied.rain_evaporation
    @test 0 < Δt * empty.deposition ≤ ice_excess
    @test Δt * supplied.deposition > ice_excess
    @test empty.deposition < rates(1e-6).deposition < supplied.deposition

    # Coating evaporation is projected over the stage, and no further than the coating it leaves
    liquid = (; cond=0.0, ccn_activation_mass=0.0, ccn_activation_number=0.0, rain_cond=0.0,
              rain_evap=0.0, coat_cond=0.0, coat_evap=1e-7)
    qᵛ = ice_saturation + ice_excess
    P = air_pressure(𝒰, constants)
    capped(coating) = PPP.limit_ice_vapor_rates(liquid, 1e-6, 0.0, 0.0, qᵛ, T, P, qᵛ, constants, 10.0, 273.15,
                                                0.0, 0.0, Δt, 0.0, Inf, coating).dep
    @test capped(0.0) < capped(1e-6) < capped(Inf)
    @test capped(Inf) ≈ capped(Δt * 1e-7)
end

@testset "Activation and rain growth share the vapor excess" begin
    constants = ThermodynamicConstants(Float64)
    p3 = P3Microphysics(Float64)
    vapor = cell_saturation(0.0, constants)
    for _ in 1:5
        state = cell_state(vapor, 0.0, constants)
        vapor = saturation_specific_humidity(285.0, density(state, constants), constants, PlanarLiquidSurface()) + 1e-7
    end
    𝒰 = cell_state(vapor, 0.0, constants)
    saturation = saturation_specific_humidity(285.0, density(𝒰, constants), constants, PlanarLiquidSurface())
    ξˡ = PPP.p3_liquid_psychrometric_correction(saturation, 285.0, constants)
    Δt = 1e4
    r = stage_rates(p3, vapor, 0.0, constants, Δt; qʳ = 1e-3, nʳ = 1e6)
    @test r.rain_condensation > 0
    @test r.ccn_activation_mass > 0
    @test Δt * (r.ccn_activation_mass + r.rain_condensation) ≤ (vapor - saturation) / ξˡ * (1 + 5eps())
end

@testset "Implicit activation relaxes toward the activated number" begin
    FT = Float64
    constants = ThermodynamicConstants(FT)
    p3 = P3Microphysics(FT)  # prescribed droplet number: the seed mass
    qᵛ = cell_saturation(zero(FT), constants) + 2e-4
    seed = p3.cloud.number_concentration * PPP.activated_droplet_mass(p3.process_rates, FT)
    for Δt in (0.1, 1.0, 10.0, 1e3, 1e5)
        r = stage_rates(p3, qᵛ, zero(FT), constants, Δt)
        @test 0 < Δt * r.ccn_activation_mass ≤ seed * (1 + 1e-12)
    end
end

@testset "Condensation rate: automatic vs finite-difference derivative at small and large ωΔt" begin
    FT = Float64
    constants = ThermodynamicConstants(FT)
    p3 = P3Microphysics(FT)
    qᶜˡ = 3e-4
    q⁺ = cell_saturation(qᶜˡ, constants)
    for Δt in (0.01, 1e3), excess in (2e-4, -1e-4)
        f(x) = stage_rates(p3, q⁺ + x, qᶜˡ, constants, Δt).condensation
        x₀ = excess
        ad = only(autodiff(Forward, Const(f), Duplicated(x₀, 1.0)))
        h = 1e-9
        fd = (f(x₀ + h) - f(x₀ - h)) / 2h
        @test isfinite(ad)
        @test ad ≈ fd rtol = 1e-4
    end
end

@testset "Implicit aerosol activation stays within the aerosol reservoir" begin
    FT = Float64
    constants = ThermodynamicConstants(FT)
    aerosol = PPP.AerosolActivation(PPP.AerosolMode(FT); prognostic = true)
    p3 = P3Microphysics(FT; aerosol)
    qᵛ = cell_saturation(zero(FT), constants) + 1e-3
    for nᵃ in (0.0, 1e5, 1e8), Δt in (0.1, 10.0, 1e4)
        𝒰 = cell_state(qᵛ, zero(FT), constants)
        ℳ = P3MicrophysicalState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, nᵃ, 0.0)
        stage = LiquidRelaxationStage(Δt, 0.0, 0.0, true)
        r = compute_p3_process_rates(p3, 1.0, ℳ, 𝒰, constants, nothing, temperature(𝒰, constants),
                                     0.0, 0.0, nothing, stage)
        # backward Euler toward a target capped by the reservoir: never more than nᵃ in one stage
        @test 0 ≤ Δt * r.ccn_activation_number ≤ nᵃ * (1 + 1e-12)
    end
end
