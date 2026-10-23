using Oceananigans
using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.Thermodynamics: Thermodynamics as TD
using Breeze.ParcelModels: ParcelModels as PM
using Test

function coupling_parcel(; formulation=:LiquidIcePotentialTemperature, weight=1., cloudy=true, mixed_phase=false)
    grid = RectilinearGrid(size=101, z=(-10, 2010), topology=(Flat, Flat, Bounded))
    microphysics = P3Microphysics(; vapor_excess=PrognosticVaporExcess())
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics, formulation)
    p(z) = 1e5 - 40z
    T = mixed_phase ? 250 : 300
    set!(model; T, p, ρ=weight, qᵗ=cloudy ? 0.015 : 0.0001, z=250, w=2)
    if cloudy
        state = model.dynamics.state
        state.μ = merge(state.μ, (; ρqᶜˡ=weight*0.001, ρqʳ=weight*0.0003, ρnʳ=weight*1e5))
        if mixed_phase
            state.μ = merge(state.μ, (; ρqⁱ=weight*0.001, ρnⁱ=weight*1e4,
                                      ρqᶠ=weight*0.0003, ρbᶠ=weight*1e-6, ρqʷⁱ=weight*0.0002))
        end
        set!(model; T, p, ρ=weight, qᵗ=0.015, z=250, w=2)
    end
    state = model.dynamics.state
    constants = model.thermodynamic_constants
    saturation = TD.saturation_specific_humidity(state.𝒰, constants, TD.PlanarLiquidSurface())
    state.μ = merge(state.μ, (; ρδqᵛ⁺ˡ=state.ρ*(state.𝒰.moisture_mass_fractions.vapor-saturation)))
    return model
end

@testset "Parcel microphysics uses parcel density" begin
    first = coupling_parcel(weight=1.)
    second = coupling_parcel(weight=2.)
    PM.compute_parcel_tendencies!(first)
    PM.compute_parcel_tendencies!(second)
    # Same physical parcel and prescribed ascent: changing only the arbitrary
    # environmental density weight must not change any specific process rate.
    for name in keys(first.dynamics.state.μ)
        @test first.dynamics.timestepper.G.Gμ[name] ≈ second.dynamics.timestepper.G.Gμ[name]/2
    end
end

@testset "Parcel P3 vapor excess follows phase-change density" begin
    for formulation in (:StaticEnergy, :LiquidIcePotentialTemperature), mixed_phase in (false, true)
        model = coupling_parcel(; formulation, mixed_phase)
        set!(model.velocities.w, 0)
        state = model.dynamics.state
        constants = model.thermodynamic_constants
        initial_density = TD.density(state.𝒰, constants)
        initial_pressure = TD.air_pressure(state.𝒰, constants)
        initial_energy = state.ℰ
        initial_moisture = state.qᵗ
        τ = model.microphysics.process_rates.sink_limiting_timescale

        # A resting parcel changes only through microphysics. Over one limiter
        # interval, the predicted excess must match the actual parcel state after
        # phase-change heating or cooling at fixed pressure.
        PM.ssp_rk3_parcel_substep!(model, PM.ParcelInitialState(state), τ, 1)
        saturation = TD.saturation_specific_humidity(state.𝒰, constants, TD.PlanarLiquidSurface())
        diagnosed_excess = state.𝒰.moisture_mass_fractions.vapor - saturation

        @test TD.density(state.𝒰, constants) != initial_density
        @test TD.air_pressure(state.𝒰, constants) == initial_pressure
        @test state.ℰ == initial_energy
        @test state.qᵗ == initial_moisture
        @test state.μ.ρδqᵛ⁺ˡ / state.ρ ≈ diagnosed_excess atol=1e-12
    end
end

@testset "Parcel P3 tendencies match the scalar state evaluation: $formulation" for formulation in (:StaticEnergy, :LiquidIcePotentialTemperature)
    PPP = Breeze.Microphysics.PredictedParticleProperties
    model = coupling_parcel(; formulation, mixed_phase=true)
    PM.compute_parcel_tendencies!(model)
    state = model.dynamics.state
    constants = model.thermodynamic_constants
    G = model.dynamics.timestepper.G
    ρ = TD.density(state.𝒰, constants)
    μ = PM.rescale_parcel_microphysics(state.μ, ρ / state.ρ)
    velocities = (; u=G.Gx, v=G.Gy, w=G.Gz)
    ℳ = Breeze.AtmosphereModels.microphysical_state(model.microphysics, ρ, μ, state.𝒰, velocities)
    result = PPP.p3_state_tendencies(model.microphysics, ρ, ℳ, state.𝒰, constants)
    for name in keys(state.μ)
        expected = PPP.p3_tendency_component(result, Val(name)) * state.ρ / ρ
        @test G.Gμ[name] ≈ expected
    end
end

@testset "Kessler parcel update uses the parcel density" begin
    constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
    function kessler_parcel(weight)
        grid = RectilinearGrid(size=20, z=(0, 2000), topology=(Flat, Flat, Bounded))
        microphysics = DCMIP2016KesslerMicrophysics()
        model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics, thermodynamic_constants=constants)
        # A nearly saturated parcel carrying cloud and rain; the environmental density is
        # an arbitrary weight for the stored prognostics.
        set!(model; T=290, p=z -> 1e5 - 10z, ρ=weight, qᵗ=0.014, z=500, w=1)
        state = model.dynamics.state
        state.μ = (; ρqᶜˡ=weight * 1e-3, ρqʳ=weight * 5e-4)
        set!(model; T=290, p=z -> 1e5 - 10z, ρ=weight, qᵗ=0.014, z=500, w=1)
        time_step!(model, 10)
        return (state.qᵗ, state.μ.ρqᶜˡ / state.ρ, state.μ.ρqʳ / state.ρ, temperature(state.𝒰, constants))
    end
    first = kessler_parcel(1.)
    second = kessler_parcel(2.)
    @test first[2] > 0
    @test all(first .≈ second)
end
