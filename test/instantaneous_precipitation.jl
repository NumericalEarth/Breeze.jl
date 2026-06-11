using Breeze
using Oceananigans
using Test

using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    LiquidIceDensityState,
    temperature,
    with_temperature,
    with_moisture,
    mixture_gas_constant,
    mixture_heat_capacity,
    saturation_specific_humidity

using Breeze.Microphysics:
    InstantaneousPrecipitation,
    SaturationAdjustment,
    WarmPhaseEquilibrium,
    adjust_thermodynamic_state,
    equilibrated_moisture_mass_fractions

using Breeze.AtmosphereModels: microphysics_model_update!
using Oceananigans.TimeSteppers: update_state!

# Tests for the `InstantaneousPrecipitation` scheme after it was refactored onto the shared #765
# density-based saturation-adjustment machinery: the once-per-step kernel now delegates condensation
# to `adjust_thermodynamic_state(::LiquidIceDensityState, ::SaturationAdjustment)` and rains out via
# `with_temperature`, instead of its own constant-density secant. These tests pin (a) numerical parity
# with the pre-refactor inline formulas and (b) the physical behavior: instantaneous condensation,
# rain-out with retained latent warming, and density-consistent saturation.

# Reference implementation of the *pre-refactor* large-scale-condensation scalar math: a constant-density θˡⁱ secant
# (run to tight convergence) plus the closed-form vapor-only rain-out. This is the ground truth the
# delegated path must reproduce.
function reference_ip_update(θ₀, qᵗ, ρ, pˢᵗ, constants, equilibrium)
    ℒˡ = constants.liquid.reference_latent_heat
    ℒⁱ = constants.ice.reference_latent_heat

    residual(T) = begin
        qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, equilibrium)
        q   = equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium)
        Rᵐ  = mixture_gas_constant(q, constants)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        κ   = Rᵐ / cᵖᵐ
        L   = (ℒˡ * q.liquid + ℒⁱ * q.ice) / cᵖᵐ
        p   = ρ * Rᵐ * T
        return (T - L) * (pˢᵗ / p)^κ - θ₀, q
    end

    Rᵗ  = mixture_gas_constant(MoistureMassFractions(qᵗ), constants)
    cᵖᵗ = mixture_heat_capacity(MoistureMassFractions(qᵗ), constants)
    γᵗ  = cᵖᵗ / (cᵖᵗ - Rᵗ)
    T₁  = θ₀^γᵗ * (ρ * Rᵗ / pˢᵗ)^(γᵗ - 1)
    r₁, q₁ = residual(T₁)

    cᵖ₁ = mixture_heat_capacity(q₁, constants)
    L₁  = (ℒˡ * q₁.liquid + ℒⁱ * q₁.ice) / cᵖ₁
    T₂  = T₁ + max(oftype(T₁, 0.01), γᵗ * L₁)
    r₂, q₂ = residual(T₂)

    T = T₂
    for _ in 1:40
        Δ = (T₂ - T₁) / (r₂ - r₁)
        Δ = ifelse(isfinite(Δ), Δ, zero(Δ))
        T₁ = T₂; r₁ = r₂
        T₂ = T₂ - r₂ * Δ
        r₂, q₂ = residual(T₂)
        T = T₂
    end

    _, q = residual(T)
    qᵛ⁺ = q.vapor
    qᶜ  = q.liquid + q.ice

    Rᵐf = mixture_gas_constant(MoistureMassFractions(qᵛ⁺), constants)
    cᵖf = mixture_heat_capacity(MoistureMassFractions(qᵛ⁺), constants)
    γf  = cᵖf / (cᵖf - Rᵐf)
    θᶠ  = T^(1 / γf) * (ρ * Rᵐf / pˢᵗ)^((1 - γf) / γf)

    return (; T, qᵛ⁺, qᶜ, θᶠ)
end

# The post-refactor scalar composition (mirrors the kernel body) using the shared #765 primitives.
function delegated_ip_update(θ₀, qᵗ, ρ, pˢᵗ, sa, constants)
    𝒰₀  = LiquidIceDensityState(θ₀, MoistureMassFractions(qᵗ), pˢᵗ, ρ)
    𝒰₁  = adjust_thermodynamic_state(𝒰₀, sa, constants)
    q   = 𝒰₁.moisture_mass_fractions
    T   = temperature(𝒰₁, constants)
    qᵛ⁺ = q.vapor
    qᶜ  = q.liquid + q.ice
    𝒰ᵥ  = with_moisture(𝒰₁, MoistureMassFractions(qᵛ⁺))
    θᶠ  = with_temperature(𝒰ᵥ, T, constants).potential_temperature
    return (; T, qᵛ⁺, qᶜ, θᶠ)
end

@testset "InstantaneousPrecipitation refactor parity vs pre-refactor inline formulas [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    pˢᵗ = FT(1e5)
    ρ   = FT(1)
    eq  = WarmPhaseEquilibrium()
    tol = FT == Float64 ? FT(1e-12) : FT(1e-6)
    sa  = SaturationAdjustment(FT; tolerance = tol, maxiter = 100, equilibrium = eq)
    rtol = FT == Float64 ? FT(1e-9) : FT(1e-4)

    # Supersaturated parcels spanning the warm range — both paths converge to the unique root.
    for (θ₀, qᵗ) in ((FT(300), FT(0.030)), (FT(295), FT(0.022)), (FT(305), FT(0.045)))
        new = delegated_ip_update(θ₀, qᵗ, ρ, pˢᵗ, sa, constants)
        ref = reference_ip_update(θ₀, qᵗ, ρ, pˢᵗ, constants, eq)
        @test new.qᶜ > 0                            # genuinely condensing
        @test new.T   ≈ ref.T    rtol = rtol
        @test new.qᵛ⁺ ≈ ref.qᵛ⁺  rtol = rtol
        @test new.qᶜ  ≈ ref.qᶜ   rtol = rtol
        @test new.θᶠ  ≈ ref.θᶠ   rtol = rtol
        @test new.θᶠ  > θ₀                          # latent warming retained after rain-out
    end

    # Subsaturated parcel: no condensation, θˡⁱ unchanged (round-trips), no precipitation.
    let θ₀ = FT(300), qᵗ = FT(0.001)
        new = delegated_ip_update(θ₀, qᵗ, ρ, pˢᵗ, sa, constants)
        @test new.qᶜ == 0
        @test new.qᵛ⁺ ≈ qᵗ rtol = rtol
        @test new.θᶠ  ≈ θ₀ rtol = rtol
    end
end

# Integration: drive the real once-per-step kernel through `microphysics_model_update!` on a
# compressible model and confirm the rain-out physics and the #765 density-consistency hold.
@testset "InstantaneousPrecipitation once-per-step kernel (integration)" begin
    arch = default_arch
    grid = RectilinearGrid(arch; size = (8, 8, 8), halo = (5, 5, 5),
                           x = (0, 1e3), y = (0, 1e3), z = (0, 1e3),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(Float64)
    θref(z) = 300.0 * exp(9.80616 * z / (1005 * 300.0))
    dyn = CompressibleDynamics(SplitExplicitTimeDiscretization();
                               surface_pressure = 1e5, standard_pressure = 1e5,
                               reference_potential_temperature = θref)
    pˢᵗ = 1e5
    eq  = WarmPhaseEquilibrium()

    function build_model(qᵗ)
        model = AtmosphereModel(grid; dynamics = dyn,
                                microphysics = InstantaneousPrecipitation(equilibrium = eq),
                                thermodynamic_constants = constants,
                                timestepper = :AcousticRungeKutta3)
        set!(model; ρ = model.dynamics.reference_state.density, θ = 300.0, qᵗ = qᵗ)
        update_state!(model)   # Δt invalid here ⇒ the precipitation kernel is a no-op; populates diagnostics
        return model
    end

    @testset "supersaturated ⇒ condense, rain out, retain warming" begin
        model = build_model(0.030)
        ρ   = Array(interior(model.dynamics.density))
        ρθ0 = Array(interior(model.formulation.potential_temperature_density))
        ρq0 = Array(interior(model.moisture_density))

        Δt = 1.0
        model.clock.last_Δt = Δt
        microphysics_model_update!(model.microphysics, model)

        ρθ1   = Array(interior(model.formulation.potential_temperature_density))
        ρq1   = Array(interior(model.moisture_density))
        prate = Array(interior(model.microphysical_fields.precipitation_rate))

        @test all(isfinite, ρθ1) && all(isfinite, ρq1)
        @test all(ρq1 .< ρq0)             # vapor condensed out
        @test all(ρθ1 .> ρθ0)             # latent warming retained ⇒ θˡⁱ rose
        @test all(prate .> 0)             # precipitation produced

        for I in eachindex(ρ)
            ρI = ρ[I]
            qv = ρq1[I] / ρI
            θᶠ = ρθ1[I] / ρI
            # Vapor-only post-state: diagnose its T and confirm it sits on the saturation curve at
            # the cell's own density ρ (the #765 density-consistency property).
            𝒰 = LiquidIceDensityState(θᶠ, MoistureMassFractions(qv), pˢᵗ, ρI)
            T = temperature(𝒰, constants)
            @test qv ≈ saturation_specific_humidity(T, ρI, constants, eq) atol = 1e-4
            # Water budget: all condensed vapor leaves as precipitation.
            @test prate[I] ≈ (ρq0[I] - ρq1[I]) / Δt rtol = 1e-6
        end
    end

    @testset "final update_state! preserves precipitation diagnostic" begin
        model = build_model(0.030)

        model.clock.last_Δt = 1.0
        microphysics_model_update!(model.microphysics, model)
        update_state!(model)

        prate = Array(interior(model.microphysical_fields.precipitation_rate))
        @test all(prate .> 0)
    end

    @testset "subsaturated ⇒ no-op" begin
        model = build_model(0.001)
        ρθ0 = Array(interior(model.formulation.potential_temperature_density))
        ρq0 = Array(interior(model.moisture_density))

        model.clock.last_Δt = 1.0
        microphysics_model_update!(model.microphysics, model)

        ρθ1   = Array(interior(model.formulation.potential_temperature_density))
        ρq1   = Array(interior(model.moisture_density))
        prate = Array(interior(model.microphysical_fields.precipitation_rate))

        @test ρθ1 ≈ ρθ0 rtol = 1e-8
        @test ρq1 ≈ ρq0 rtol = 1e-8
        @test all(prate .== 0)
    end
end
