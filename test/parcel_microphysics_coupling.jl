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

@testset "Parcel supersaturation chain rule: $formulation, mixed phase=$mixed_phase" for formulation in (:StaticEnergy, :LiquidIcePotentialTemperature), mixed_phase in (false, true)
    model = coupling_parcel(; formulation, mixed_phase)
    PM.compute_parcel_tendencies!(model)
    state = model.dynamics.state
    constants = model.thermodynamic_constants
    G = model.dynamics.timestepper.G
    q = state.𝒰.moisture_mass_fractions
    liquid = (G.Gμ.ρqᶜˡ + G.Gμ.ρqʳ + G.Gμ.ρqʷⁱ) / state.ρ
    ice = G.Gμ.ρqⁱ / state.ρ
    vapor = -liquid - ice
    function displaced_supersaturation(δt)
        updated = PM.reconstruct_thermodynamic_state(state.𝒰, state.ℰ, state.z + δt*G.Gz,
                                                      state.𝒰.reference_pressure - 40δt*G.Gz)
        updated = TD.with_moisture(updated, TD.MoistureMassFractions(q.vapor+δt*vapor, q.liquid+δt*liquid, q.ice+δt*ice))
        return updated.moisture_mass_fractions.vapor - TD.saturation_specific_humidity(updated, constants, TD.PlanarLiquidSurface())
    end
    δt = 1e-3
    derivative = (displaced_supersaturation(δt) - displaced_supersaturation(-δt)) / (2δt)
    @test G.Gμ.ρsᵛ⁺ˡ / state.ρ ≈ derivative rtol=1e-6 atol=1e-12
end

@testset "Supersaturation retains RK3 convergence" begin
    errors = Float64[]
    for dt in (20., 10., 5.)
        model = coupling_parcel(cloudy=false)
        state = model.dynamics.state
        initial_theta = state.ℰ
        initial_water = state.qᵗ
        run!(Simulation(model; Δt=dt, stop_time=100, verbose=false))
        diagnosed = state.𝒰.moisture_mass_fractions.vapor -
                    TD.saturation_specific_humidity(state.𝒰, model.thermodynamic_constants, TD.PlanarLiquidSurface())
        push!(errors, abs(state.μ.ρsᵛ⁺ˡ/state.ρ - diagnosed))
        @test state.ℰ ≈ initial_theta
        @test state.qᵗ ≈ initial_water
    end
    @info "RK3 supersaturation errors" errors
    @test errors[1]/errors[2] > 6
    @test errors[2]/errors[3] > 6
end

@testset "Saturation derivatives follow selected formula" begin
    for FT in (Float32, Float64), fit in (TD.ClausiusClapeyron(), TD.FlatauPolynomial(), TD.TetensFormula())
        constants = ThermodynamicConstants(FT; saturation_vapor_pressure=fit)
        T = FT(280)
        δT = FT === Float32 ? FT(0.1) : FT(0.001)
        expected = (TD.saturation_vapor_pressure(T+δT, constants, TD.PlanarLiquidSurface()) -
                    TD.saturation_vapor_pressure(T-δT, constants, TD.PlanarLiquidSurface())) / (2δT)
        actual = @inferred TD.liquid_saturation_vapor_pressure_derivative(T, constants)
        @test actual ≈ expected rtol=FT === Float32 ? 0.001 : 1e-8
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
