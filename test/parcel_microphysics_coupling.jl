using Oceananigans
using Breeze
using Breeze.Thermodynamics: Thermodynamics as TD
using Breeze.ParcelModels: ParcelModels as PM
using Test

function coupling_parcel(; formulation=:LiquidIcePotentialTemperature, weight=1., cloudy=true, mixed_phase=false)
    grid = RectilinearGrid(size=101, z=(-10, 2010), topology=(Flat, Flat, Bounded))
    microphysics = P3Microphysics(; predict_supersaturation=true)
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
    state.μ = merge(state.μ, (; ρsᵛ⁺ˡ=state.ρ*(state.𝒰.moisture_mass_fractions.vapor-saturation)))
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

@testset "Parcel P3 tendencies match the gridded path: $formulation" for formulation in (:StaticEnergy, :LiquidIcePotentialTemperature)
    # Parcels must evolve P3 exactly as LES and regional models do: the same adiabatic
    # forcing from w and the same prognostic supersaturation tendency.
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

@testset "Vanishing cloud number transfer remains finite" begin
    PPP = Breeze.Microphysics.PredictedParticleProperties
    for FT in (Float32, Float64)
        # The old intermediate N/(ρ q) overflows; the answer remains finite.
        q = floatmin(FT) / 1000
        number = FT(1e5)
        density = FT(0.5)
        @test PPP.cloud_number_transfer_rate(number, density, q, zero(FT)) === zero(FT)
        @test PPP.cloud_number_transfer_rate(number, density, q, q) ≈ number/density
        @test PPP.cloud_number_transfer_rate(number, density, zero(FT), zero(FT)) === zero(FT)
        @test PPP.cloud_riming_number_rate(q, number, density, zero(FT)) === zero(FT)
    end
end
