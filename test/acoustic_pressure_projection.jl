include(joinpath(@__DIR__, "setup.jl"))

#####
##### The pressure the acoustic substep loop integrates, and the moisture's part of it
#####
##### The loop's thermodynamic unknown is the pressure perturbation in units of the thermodynamic
##### density, ϖ = (ρθ)′ + W, with W = − Σₐ Λₐ (ρqₐ)′ the moisture content of the pressure
##### (`PressureProjection`). These tests check the sensitivities Λ against the equation of state,
##### that moisture transport which moves no pressure is not read as pressure by the loop, that
##### the thermodynamic density keeps its own budget, and that the projection is exactly zero
##### without moisture.
#####

using Breeze, Oceananigans, Test
using Breeze.CompressibleEquations: ExplicitTimeStepping, SplitExplicitTimeDiscretization,
                                    moisture_pressure_sensitivities
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics, equilibrium_moisture_phase_fractions
using Oceananigans.TimeSteppers: time_step!

const AM = Breeze.AtmosphereModels
const TH = Breeze.Thermodynamics

# A test-only scheme: one liquid condensate mass that neither falls nor changes phase. With
# `declared = false` it does not declare the phase of that mass.
struct InertCondensate{declared} end

AM.prognostic_field_names(::InertCondensate) = (:ρqʳ,)
AM.moisture_prognostic_name(::InertCondensate) = :ρqᵛ
AM.condensate_liquid_fraction(::InertCondensate{true}, ::Val{:ρqʳ}) = 1
AM.compute_microphysical_tendencies!(::InertCondensate, model) = nothing
AM.microphysics_model_update!(::InertCondensate, model) = nothing
@inline AM.maybe_adjust_thermodynamic_state(state, ::InertCondensate, vapor, constants) = state
@inline AM.microphysical_state(::InertCondensate, ρ, μ::NamedTuple, state, velocities) = (; qʳ=μ.ρqʳ / ρ)
@inline AM.moisture_fractions(::InertCondensate, state::NamedTuple, vapor) =
    TH.MoistureMassFractions(vapor, state.qʳ, oftype(vapor, 0))
AM.liquid_mass_fraction(::InertCondensate, model) = model.microphysical_fields.qʳ
AM.ice_mass_fraction(::InertCondensate, model) = nothing

AM.materialize_microphysical_fields(::InertCondensate, grid, bcs) =
    (; ρqʳ=CenterField(grid; boundary_conditions=bcs.ρqʳ), qʳ=CenterField(grid), qᵛ=CenterField(grid))

@inline function AM.update_microphysical_auxiliaries!(μ, i, j, k, grid, ::InertCondensate, state, ρ, thermo, constants)
    @inbounds μ.qʳ[i, j, k] = state.qʳ
    @inbounds μ.qᵛ[i, j, k] = thermo.moisture_mass_fractions.vapor
    return nothing
end

host_values(f) = Float64.(Array(interior(f)))

# The pressure of the density-based state with dry density ρᵈ, thermodynamic density ρθ and
# partial densities ρᵛ, ρˡ, ρⁱ, as the compressible core diagnoses it.
function density_state_pressure(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ, pˢᵗ, solver, exponent, constants)
    ρ = ρᵈ + ρᵛ + ρˡ + ρⁱ
    q = TH.MoistureMassFractions(ρᵛ / ρ, ρˡ / ρ, ρⁱ / ρ)
    𝒰 = TH.LiquidIceDensityState(ρθ / ρᵈ, q, pˢᵗ, ρ, solver, exponent)
    return ρ * TH.mixture_gas_constant(q, constants) * TH.temperature(𝒰, constants)
end

@testset "Moisture sensitivities of the loop's pressure match the equation of state" begin
    constants = ThermodynamicConstants(Float64)
    grid = RectilinearGrid(CPU(), Float64; size=4, z=(0, 400), topology=(Flat, Flat, Bounded))
    formulation = AtmosphereModel(grid; dynamics=CompressibleDynamics(ExplicitTimeStepping())).formulation
    solver = Breeze.NewtonSolver(Float64; reltol=0.0, abstol=1e-11, maxiter=50)
    pˢᵗ, ρᵈ, T = 1e5, 1.1, 285.0

    compositions = ((0.012, 0.0, 0.0),      # vapor only
                    (0.012, 0.004, 0.0),    # warm cloud
                    (0.004, 0.002, 0.003))  # mixed phase

    for exponent in (TH.VaporEquivalentExponent(), TH.DryExponent()), (ρᵛ, ρˡ, ρⁱ) in compositions
        ρ = ρᵈ + ρᵛ + ρˡ + ρⁱ
        q = TH.MoistureMassFractions(ρᵛ / ρ, ρˡ / ρ, ρⁱ / ρ)
        θ = TH.with_temperature(TH.LiquidIceDensityState(0.0, q, pˢᵗ, ρ, solver, exponent), T, constants).potential_temperature
        ρθ = ρᵈ * θ

        𝒰 = TH.LiquidIceDensityState(θ, q, pˢᵗ, ρ, solver, exponent)
        Tᴸ, θ∂T∂θ = TH.temperature_and_sensitivity(𝒰, constants)
        Cᴸ = ρ * TH.mixture_gas_constant(q, constants) * θ∂T∂θ / ρθ
        Λᵛ, Λˡ, Λⁱ, Λᴴ = moisture_pressure_sensitivities(formulation, 𝒰, Tᴸ, ρᵈ, constants)
        Λᵈ = Λᴴ + (Λᵛ * ρᵛ + Λˡ * ρˡ + Λⁱ * ρⁱ) / ρᵈ

        p(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ) = density_state_pressure(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ, pˢᵗ, solver, exponent, constants)
        δθ, δ = 1e-4, 1e-6
        ∂p_∂ρθ = (p(ρᵈ, ρθ + δθ, ρᵛ, ρˡ, ρⁱ) - p(ρᵈ, ρθ - δθ, ρᵛ, ρˡ, ρⁱ)) / 2δθ
        ∂p_∂ρᵛ = (p(ρᵈ, ρθ, ρᵛ + δ, ρˡ, ρⁱ) - p(ρᵈ, ρθ, ρᵛ - δ, ρˡ, ρⁱ)) / 2δ
        ∂p_∂ρˡ = (p(ρᵈ, ρθ, ρᵛ, ρˡ + δ, ρⁱ) - p(ρᵈ, ρθ, ρᵛ, ρˡ - δ, ρⁱ)) / 2δ
        ∂p_∂ρⁱ = (p(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ + δ) - p(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ - δ)) / 2δ
        ∂p_∂ρᵈ = (p(ρᵈ + δ, ρθ, ρᵛ, ρˡ, ρⁱ) - p(ρᵈ - δ, ρθ, ρᵛ, ρˡ, ρⁱ)) / 2δ

        @test Tᴸ ≈ T
        @test Cᴸ ≈ ∂p_∂ρθ rtol=1e-6
        @test -Cᴸ * Λᵛ ≈ ∂p_∂ρᵛ rtol=1e-5
        @test -Cᴸ * Λˡ ≈ ∂p_∂ρˡ rtol=1e-5
        @test -Cᴸ * Λⁱ ≈ ∂p_∂ρⁱ rtol=1e-5

        # The dry density at fixed thermodynamic and moisture densities: the dependence at fixed
        # composition, plus the dilution of the moisture.
        @test Cᴸ * Λᵈ ≈ ∂p_∂ρᵈ rtol=1e-4 atol=1e-3

        # Vapor adds pressure like Rᵛ / Rᵈ of its mass in dry air (the moist potential temperature),
        # and condensate holds the latent heat the thermodynamic variable credits it with.
        @test Λᵛ < 0
        @test Λˡ < Λᵛ
        @test Λⁱ < Λˡ
        @test Λᴴ ≥ 0
    end
end

@testset "Phase fractions of a change of the prognostic moisture" begin
    q = TH.MoistureMassFractions(0.01, 0.002, 0.0)

    # Prognostic vapor: every scheme without diagnosed condensate.
    @test AM.prognostic_moisture_phase_fractions(1, 1, 1, nothing, nothing, nothing, q) == (1, 0, 0)

    # Equilibrium moisture: vapor in unsaturated air, cloud condensate in saturated air.
    @test equilibrium_moisture_phase_fractions(0.0, 0.0) == (1, 0, 0)
    @test equilibrium_moisture_phase_fractions(2e-3, 0.0) == (0, 1, 0)
    fᵛ, fˡ, fⁱ = equilibrium_moisture_phase_fractions(1e-3, 3e-3)
    @test fᵛ == 0
    @test fˡ ≈ 0.25
    @test fⁱ ≈ 0.75

    # Kessler's condensate masses are liquid.
    kessler = DCMIP2016KesslerMicrophysics()
    @test AM.condensate_liquid_fraction(kessler, Val(:ρqᶜˡ)) == 1
    @test AM.condensate_liquid_fraction(kessler, Val(:ρqʳ)) == 1
end

# A slab of 10 g/kg of condensate in air of one temperature and pressure, carried by a uniform
# wind in a periodic channel without gravity: the transport moves no pressure.
function condensate_channel(FT, time_discretization; microphysics = InertCondensate{true}(), U = 10)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(32, 4), halo=(3, 3), x=(0, 32e3), z=(0, 400),
                           topology=(Periodic, Flat, Bounded))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = CompressibleDynamics(time_discretization; reference_potential_temperature=290)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, advection=WENO(), microphysics)

    Rᵈ, Rᵛ = TH.dry_air_gas_constant(constants), TH.vapor_gas_constant(constants)
    p₀, T₀, ε = FT(1e5), FT(290), FT(0.005)
    slab(x, z) = ifelse(12e3 < x < 20e3, FT(0.01), FT(0))
    set!(model; ρᵈ = p₀ / ((Rᵈ + ε * Rᵛ) * T₀), T = T₀, u = FT(U),
                qᵛ = (x, z) -> ε * (1 - slab(x, z)) / (1 + ε), qʳ = slab)
    return model
end

@testset "Transport of condensate at uniform pressure is not read as pressure [$FT]" for FT in test_float_types()
    Δt, Nt, U, p₀ = 10, 30, 10, 1e5
    model = condensate_channel(FT, SplitExplicitTimeDiscretization(); U)
    ρθ = model.formulation.potential_temperature_density
    ∫ρθ⁰ = sum(host_values(ρθ))

    δu = δp = 0.0
    for _ in 1:Nt
        time_step!(model, Δt)
        δu = max(δu, maximum(abs, host_values(model.velocities.u) .- U))
        δp = max(δp, maximum(abs, host_values(model.dynamics.pressure) .- p₀) / p₀)
    end

    # Without the projection the loop reads the slab's content as about 25 hPa of pressure: the
    # wind departs by several tenths of a m/s and the pressure by several 10⁻³ of itself.
    @test δu < 0.05
    @test δp < 5e-4

    # The thermodynamic density takes its own tendency and the flux divergence the loop applied,
    # both in flux form: its integral over the closed channel is conserved.
    @test sum(host_values(ρθ)) ≈ ∫ρθ⁰ rtol=(FT == Float64 ? 1e-12 : 1e-5)

    # The content of the stage's last projection is that of the moisture the slab moved.
    content = host_values(model.timestepper.substepper.pressure_projection.content)
    @test maximum(abs, content) > 0
    @test all(isfinite, content)
end

@testset "The projection is exactly zero without moisture [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(16, 16), halo=(5, 5), x=(-5e3, 5e3), z=(0, 10e3),
                           topology=(Periodic, Flat, Bounded))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature=300)
    model = AtmosphereModel(grid; dynamics, advection=WENO())
    bubble(x, z) = 300 + 2 * max(0, 1 - sqrt(x^2 + (z - 2e3)^2) / 2e3)
    set!(model; θ = bubble, ρ = model.dynamics.reference_state.density)

    for _ in 1:3
        time_step!(model, 5)
    end

    projection = model.timestepper.substepper.pressure_projection
    @test all(iszero, host_values(projection.content))
    @test all(isfinite, host_values(projection.prognostic_moisture_sensitivity))
    @test maximum(abs, host_values(model.velocities.w)) > 0
end

@testset "A condensate mass must declare its phase for the projection" begin
    model = condensate_channel(Float64, SplitExplicitTimeDiscretization(); microphysics = InertCondensate{false}())
    @test_throws ArgumentError time_step!(model, 1)

    # The explicit time discretization has no loop, and needs no declaration.
    model = condensate_channel(Float64, ExplicitTimeStepping(); microphysics = InertCondensate{false}())
    time_step!(model, 0.1)
    @test all(isfinite, host_values(model.velocities.u))
end
