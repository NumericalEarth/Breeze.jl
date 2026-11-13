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
                                    moisture_pressure_sensitivities, refresh_linearization_basic_state!
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics, SaturationAdjustment, WarmPhaseEquilibrium,
                           adjust_thermodynamic_state
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

@testset "Sensitivities of the loop's pressure match the equation of state" begin
    constants = ThermodynamicConstants(Float64)
    pˢᵗ, ρᵈ, T = 1e5, 1.1, 285.0

    compositions = ((0.0, 0.0, 0.0),        # dry
                    (0.012, 0.0, 0.0),      # vapor only
                    (0.012, 0.004, 0.0),    # warm cloud
                    (0.012, 0.01, 0.005))   # heavy mixed-phase condensate

    # Every inversion the core can close with: the derivatives are those of the temperature the
    # selected solver returns, converged or not.
    solvers = (temperature_solver_variants(Float64)...,
               Breeze.NewtonSolver(Float64; reltol=0.0, abstol=1e-11, maxiter=50))

    for solver in solvers, exponent in (TH.VaporEquivalentExponent(), TH.DryExponent()), (ρᵛ, ρˡ, ρⁱ) in compositions
        ρ = ρᵈ + ρᵛ + ρˡ + ρⁱ
        q = TH.MoistureMassFractions(ρᵛ / ρ, ρˡ / ρ, ρⁱ / ρ)
        θ = TH.with_temperature(TH.LiquidIceDensityState(0.0, q, pˢᵗ, ρ, solver, exponent), T, constants).potential_temperature
        ρθ = ρᵈ * θ

        𝒰 = TH.LiquidIceDensityState(θ, q, pˢᵗ, ρ, solver, exponent)
        Tᴸ, θ∂T∂θ = TH.temperature_and_sensitivity(𝒰, constants)
        Cᴸ = ρ * TH.mixture_gas_constant(q, constants) * θ∂T∂θ / ρθ
        Λᵛ, Λˡ, Λⁱ, Λᵈ = moisture_pressure_sensitivities(𝒰, Tᴸ, Cᴸ, ρᵈ, constants)

        p(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ) = density_state_pressure(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ, pˢᵗ, solver, exponent, constants)
        δθ, δ = 1e-4, 1e-6
        ∂p_∂ρθ = (p(ρᵈ, ρθ + δθ, ρᵛ, ρˡ, ρⁱ) - p(ρᵈ, ρθ - δθ, ρᵛ, ρˡ, ρⁱ)) / 2δθ
        ∂p_∂ρᵛ = (p(ρᵈ, ρθ, ρᵛ + δ, ρˡ, ρⁱ) - p(ρᵈ, ρθ, ρᵛ - δ, ρˡ, ρⁱ)) / 2δ
        ∂p_∂ρˡ = (p(ρᵈ, ρθ, ρᵛ, ρˡ + δ, ρⁱ) - p(ρᵈ, ρθ, ρᵛ, ρˡ - δ, ρⁱ)) / 2δ
        ∂p_∂ρⁱ = (p(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ + δ) - p(ρᵈ, ρθ, ρᵛ, ρˡ, ρⁱ - δ)) / 2δ
        ∂p_∂ρᵈ = (p(ρᵈ + δ, ρθ, ρᵛ, ρˡ, ρⁱ) - p(ρᵈ - δ, ρθ, ρᵛ, ρˡ, ρⁱ)) / 2δ

        @test Cᴸ ≈ ∂p_∂ρθ rtol=1e-6
        @test -Cᴸ * Λᵛ ≈ ∂p_∂ρᵛ rtol=1e-5
        @test -Cᴸ * Λˡ ≈ ∂p_∂ρˡ rtol=1e-5
        @test -Cᴸ * Λⁱ ≈ ∂p_∂ρⁱ rtol=1e-5

        # The dry density at fixed thermodynamic density and fixed densities of the phases: the
        # dependence at fixed composition, plus the dilution of the moisture. Without moisture
        # the pressure does not depend on it, and the sensitivity is exactly zero.
        @test Cᴸ * Λᵈ ≈ ∂p_∂ρᵈ rtol=1e-4 atol=1e-2
        ρᵛ + ρˡ + ρⁱ == 0 && @test Λᵈ == 0

        # Vapor adds pressure like Rᵛ / Rᵈ of its mass in dry air (the moist potential temperature),
        # and condensate holds the latent heat the thermodynamic variable credits it with.
        @test Λᵛ < 0
        @test Λˡ < Λᵛ
        @test Λⁱ < Λˡ
    end
end

@testset "Kessler declares the phase of its condensate masses" begin
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
    @test all(isfinite, host_values(projection.vapor_sensitivity))
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

# Under saturation adjustment the projection covers unsaturated air, where the moisture density
# is the vapor and the cached sensitivities are the derivatives of the pressure the adjustment
# re-diagnoses. A saturated cell, whose vapor the scheme diagnoses, is not projected: the pressure
# it re-diagnoses is that of the adjusted partition, which responds less than the loop's
# fixed-composition coefficient.
@testset "Projection under saturation adjustment covers unsaturated air only" begin
    Oceananigans.defaults.FloatType = Float64
    constants = ThermodynamicConstants(Float64)
    solver = Breeze.NewtonSolver(Float64; reltol=0.0, abstol=1e-11, maxiter=50)
    adjustment = SaturationAdjustment(Float64; equilibrium=WarmPhaseEquilibrium(),
                                      solver=Breeze.SecantSolver(Float64; reltol=0.0, abstol=1e-11, maxiter=100))
    grid = RectilinearGrid(default_arch, Float64; size=4, z=(0, 400), topology=(Flat, Flat, Bounded))
    pˢᵗ, ρᵈ, T = 1e5, 1.1, 285.0
    ρᵛ⁺ = TH.saturation_specific_humidity(T, 1.0, constants, WarmPhaseEquilibrium())

    # The pressure after adjustment of the state with all its moisture as vapor.
    function adjusted_pressure(ρᵈ, ρθ, ρʷ)
        ρ = ρᵈ + ρʷ
        𝒰 = TH.LiquidIceDensityState(ρθ / ρᵈ, TH.MoistureMassFractions(ρʷ / ρ), pˢᵗ, ρ, solver)
        return TH.air_pressure(adjust_thermodynamic_state(𝒰, adjustment, constants), constants)
    end

    # 20 g/m³ of water at 285 K is about half cloud; 5 g/m³ is unsaturated.
    for ρʷ in (0.02, 0.005)
        saturated = ρʷ > ρᵛ⁺
        ρ = ρᵈ + ρʷ
        q = TH.MoistureMassFractions(min(ρʷ, ρᵛ⁺) / ρ, max(ρʷ - ρᵛ⁺, 0) / ρ, 0.0)
        θ = TH.with_temperature(TH.LiquidIceDensityState(0.0, q, pˢᵗ, ρ, solver), T, constants).potential_temperature
        ρθ = ρᵈ * θ

        model = AtmosphereModel(grid; thermodynamic_constants=constants,
                                dynamics=CompressibleDynamics(SplitExplicitTimeDiscretization(); standard_pressure=pˢᵗ),
                                formulation=LiquidIcePotentialTemperatureFormulation(temperature_solver=solver),
                                microphysics=adjustment)
        set!(model; ρᵈ, ρqᵉ=ρʷ, ρθ)

        substepper = model.timestepper.substepper
        refresh_linearization_basic_state!(substepper, model)
        projection = substepper.pressure_projection
        Cᴸ, Λᵛ, Λˡ, Λᵈ, qˡ = @allowscalar (substepper.linearization_pressure_coefficient[1, 1, 2],
                                           projection.vapor_sensitivity[1, 1, 2],
                                           projection.liquid_sensitivity[1, 1, 2],
                                           projection.dry_density_sensitivity[1, 1, 2],
                                           model.microphysical_fields.qˡ[1, 1, 2])

        if saturated
            @test qˡ > 1e-3
            @test Λᵛ == 0
            @test Λˡ == 0
            @test Λᵈ == 0
        else
            δ = 1e-6
            ∂p_∂ρʷ = (adjusted_pressure(ρᵈ, ρθ, ρʷ + δ) - adjusted_pressure(ρᵈ, ρθ, ρʷ - δ)) / 2δ
            ∂p_∂ρᵈ = (adjusted_pressure(ρᵈ + δ, ρθ, ρʷ) - adjusted_pressure(ρᵈ - δ, ρθ, ρʷ)) / 2δ

            @test qˡ == 0
            @test -Cᴸ * Λᵛ ≈ ∂p_∂ρʷ rtol=1e-5
            @test Cᴸ * Λᵈ ≈ ∂p_∂ρᵈ rtol=1e-4
        end
    end
end

# Uniform moistening of uniform air at rest between two open boundaries: the pressure rises
# uniformly, so nothing moves. The boundary cells' target has to hold the moisture content of
# the current substep, like the interior's pressure.
@testset "Open boundaries follow the moisture content of the substep [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch, FT; size=(16, 4), halo=(3, 3), x=(0, 16e3), z=(0, 400),
                           topology=(Bounded, Flat, Bounded))
    constants = ThermodynamicConstants(FT; gravitational_acceleration=0)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps=12); reference_potential_temperature=300)
    boundary_conditions = (; ρu = FieldBoundaryConditions(west=NormalFlowBoundaryCondition(FT(0)), east=NormalFlowBoundaryCondition(FT(0))),
                             ρᵈ = FieldBoundaryConditions(west=ValueBoundaryCondition(FT(1.1)), east=ValueBoundaryCondition(FT(1.1))),
                             ρθ = FieldBoundaryConditions(west=ValueBoundaryCondition(FT(330)), east=ValueBoundaryCondition(FT(330))))
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, boundary_conditions,
                            forcing=(; ρqᵗ=Returns(FT(1e-4))), advection=WENO())
    set!(model; ρᵈ=FT(1.1), ρθ=FT(330), ρqᵛ=FT(0.005))

    p⁰ = first(host_values(model.dynamics.pressure))
    for _ in 1:3
        time_step!(model, 6)
    end

    p = host_values(model.dynamics.pressure)
    @test minimum(p) > p⁰ + 10                        # the moistening has raised the pressure
    @test maximum(p) - minimum(p) < (FT == Float64 ? 1e-6 : 0.5)
    @test maximum(abs, host_values(model.velocities.u)) < (FT == Float64 ? 1e-8 : 1e-3)
end
