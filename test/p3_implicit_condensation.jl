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
                             with_temperature, saturation_specific_humidity, PlanarLiquidSurface, density,
                             temperature, air_pressure
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
    fields = (; ρqᶜˡ = fill(3e-4, 1, 1, 1))
    G = (; ρqᶜˡ = fill(-1e-6, 1, 1, 1))
    unknown = GridLiquidRelaxationStage(1, 1, 1, G, StageStart(fields, fields, false, Inf))
    @test compute_p3_process_rates(arguments..., nothing, unknown) == explicit

    known = GridLiquidRelaxationStage(1, 1, 1, G, StageStart(fields, fields, false, 20.0))
    stage = PPP.liquid_relaxation_stage(known, nothing, 1.0)
    @test stage.active
    @test stage.Δt == 20
    @test stage.qᶜˡ ≈ 3e-4 - 20 * 1e-6
    implicit = compute_p3_process_rates(arguments..., nothing, known)
    @test implicit == compute_p3_process_rates(arguments..., nothing, stage)
    @test 0 < implicit.condensation < explicit.condensation
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
