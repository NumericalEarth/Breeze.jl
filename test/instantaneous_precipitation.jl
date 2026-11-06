include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Oceananigans
using Test

using Breeze.Thermodynamics:
    ThermodynamicConstants,
    MoistureMassFractions,
    LiquidIceDensityState,
    temperature,
    with_temperature,
    mixture_gas_constant,
    mixture_heat_capacity,
    vapor_equivalent_gas_constant,
    vapor_equivalent_heat_capacity,
    absolute_zero_latent_heat,
    saturation_specific_humidity

using Breeze.Microphysics:
    InstantaneousPrecipitation,
    SaturationAdjustment,
    WarmPhaseEquilibrium,
    adjust_thermodynamic_state,
    equilibrated_moisture_mass_fractions,
    rainout_state

using Breeze.AtmosphereModels: microphysics_model_update!
using Oceananigans.TimeSteppers: update_state!

# Condensation uses the shared density-based saturation adjustment. Rain-out then removes
# condensate mass, renormalizes the vapor, and recovers θ at the reduced density while keeping T.
# Compare the scalar thermodynamics with an independent secant and closed-form θ inversion,
# then check the actual model state after update_state! has recomputed density and temperature.

# Reference condensation at fixed density, followed by rain-out at fixed volume and dry mass.
# The θ inversion uses the exact definition θˡⁱ = (cᵖᵐ T − ℒˡ₀ qˡ − ℒⁱ₀ qⁱ) / (ĉᵖ Π̂),
# Π̂ = (p / pˢᵗ)^κ̂, evaluated with the density and composition after condensate removal.
function reference_ip_update(θ₀, qᵗ, ρ, pˢᵗ, constants, equilibrium)
    ℒˡ = constants.liquid.reference_latent_heat
    ℒⁱ = constants.ice.reference_latent_heat
    ℒˡ₀ = absolute_zero_latent_heat(constants, constants.liquid)
    ℒⁱ₀ = absolute_zero_latent_heat(constants, constants.ice)

    residual(T) = begin
        qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, equilibrium)
        q   = equilibrated_moisture_mass_fractions(T, qᵗ, qᵛ⁺, equilibrium)
        Rᵐ  = mixture_gas_constant(q, constants)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        ĉᵖ  = vapor_equivalent_heat_capacity(q, constants)
        κ̂   = vapor_equivalent_gas_constant(q, constants) / ĉᵖ
        p   = ρ * Rᵐ * T
        return (cᵖᵐ * T - ℒˡ₀ * q.liquid - ℒⁱ₀ * q.ice) / ĉᵖ * (pˢᵗ / p)^κ̂ - θ₀, q
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

    remaining_density = ρ * (1 - qᶜ)
    remaining_vapor = qᵛ⁺ / (1 - qᶜ)
    Rᵐf = mixture_gas_constant(MoistureMassFractions(remaining_vapor), constants)
    cᵖf = mixture_heat_capacity(MoistureMassFractions(remaining_vapor), constants)
    γf  = cᵖf / (cᵖf - Rᵐf)
    θᶠ  = T^(1 / γf) * (remaining_density * Rᵐf / pˢᵗ)^((1 - γf) / γf)

    return (; T, qᵛ⁺, qᶜ, θᶠ)
end

# The shared condensation and rain-out primitives used by the compressible kernel.
function delegated_ip_update(θ₀, qᵗ, ρ, pˢᵗ, sa, constants)
    𝒰₀  = LiquidIceDensityState(θ₀, MoistureMassFractions(qᵗ), pˢᵗ, ρ)
    𝒰₁  = adjust_thermodynamic_state(𝒰₀, sa, constants)
    q   = 𝒰₁.moisture_mass_fractions
    T   = temperature(𝒰₁, constants)
    qᵛ⁺ = q.vapor
    qᶜ  = q.liquid + q.ice
    ρᵈ  = ρ * (1 - qᵗ)
    𝒰ᵥ  = @inferred rainout_state(𝒰₁, qᵛ⁺, ρ * qᵛ⁺, ρᵈ)
    θᶠ  = with_temperature(𝒰ᵥ, T, constants).potential_temperature
    return (; T, qᵛ⁺, qᶜ, θᶠ)
end

@testset "InstantaneousPrecipitation scalar thermodynamics [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    pˢᵗ = FT(1e5)
    ρ   = FT(1)
    eq  = WarmPhaseEquilibrium()
    tol = FT == Float64 ? FT(1e-12) : FT(1e-6)
    sa  = SaturationAdjustment(FT; solver = SecantSolver(FT; abstol = tol, maxiter = 100), equilibrium = eq)
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

# Integration: test the state after rain-out and update_state!, including the density loss.
@testset "InstantaneousPrecipitation once-per-step kernel [$FT]" for FT in test_float_types()
    arch = default_arch
    grid = RectilinearGrid(arch, FT; size = (8, 8, 8), halo = (5, 5, 5),
                           x = (0, 1e3), y = (0, 1e3), z = (0, 1e3),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)
    θref(z) = 300.0 * exp(9.80616 * z / (1005 * 300.0))
    dyn = CompressibleDynamics(SplitExplicitTimeDiscretization();
                               base_pressure = 1e5, standard_pressure = 1e5,
                               reference_potential_temperature = θref)
    pˢᵗ = FT(1e5)
    eq  = WarmPhaseEquilibrium()
    temperature_tolerance = FT == Float64 ? 1e-4 : 2e-3
    roundtrip_tolerance = 100eps(FT)

    function build_model(qᵗ)
        model = AtmosphereModel(grid; dynamics = dyn,
                                microphysics = InstantaneousPrecipitation(FT; equilibrium = eq),
                                thermodynamic_constants = constants)
        set!(model; ρ = model.dynamics.reference_state.density, θ = 300.0, qᵗ = qᵗ)
        update_state!(model)   # Δt invalid here ⇒ the precipitation kernel is a no-op; populates diagnostics
        return model
    end

    @testset "supersaturated ⇒ condense, rain out, retain warming" begin
        model = build_model(0.030)
        ρt  = Array(interior(model.dynamics.total_density))  # mass fractions, EOS, saturation
        ρd  = Array(interior(model.dynamics.dry_density))    # ρθ = ρᵈθ is dry-coupled → θ recovery
        ρθ0 = Array(interior(model.formulation.potential_temperature_density))
        ρq0 = Array(interior(model.moisture_density))
        reference = reference_ip_update.(ρθ0 ./ ρd, ρq0 ./ ρt, ρt, pˢᵗ, Ref(constants), Ref(eq))

        Δt = 1.0
        model.clock.last_Δt = Δt
        microphysics_model_update!(model.microphysics, model)

        ρθ1   = Array(interior(model.formulation.potential_temperature_density))
        ρq1   = Array(interior(model.moisture_density))
        prate = Array(interior(model.microphysical_fields.precipitation_rate))
        ρt1 = Array(interior(model.dynamics.total_density))
        T1 = Array(interior(model.temperature))
        qv1 = ρq1 ./ ρt1
        surface_flux = bottom_precipitation_flux(model) |> compute!
        column_integrated_prate = Field(Integral(model.microphysical_fields.precipitation_rate, dims=3)) |> compute!

        @test all(isfinite, ρθ1) && all(isfinite, ρq1)
        @test all(ρq1 .< ρq0)             # vapor condensed out
        @test all(ρθ1 .> ρθ0)             # latent warming retained ⇒ θˡⁱ rose
        @test all(prate .> 0)             # precipitation produced
        @test all(Array(interior(surface_flux)) .> 0)
        @test Array(interior(surface_flux)) ≈ Array(interior(column_integrated_prate))

        # Condensate removal must retain the condensation temperature even after the model
        # diagnoses its smaller total density. Using the old density for θ spuriously cools
        # the cell and leaves it supersaturated; reconstructing T at the old density hides it.
        @test all(isapprox.(T1, getproperty.(reference, :T); atol = temperature_tolerance, rtol = 0))
        qsat = saturation_specific_humidity.(T1, ρt1, Ref(constants), Ref(eq))
        @test all(isapprox.(qv1, qsat; rtol = 1e-4))
        @test Array(interior(model.microphysical_fields.qᵛ)) ≈ qv1 rtol = roundtrip_tolerance

        # Only condensate mass leaves: dry mass is unchanged, total mass and vapor loss
        # agree with precipitation, and vapor is normalized by the remaining total mass.
        @test Array(interior(model.dynamics.dry_density)) == ρd
        @test all(ρt1 .< ρt)
        @test ρt1 ≈ ρt .- Δt .* prate rtol = roundtrip_tolerance
        @test prate ≈ (ρq0 .- ρq1) ./ Δt rtol = roundtrip_tolerance
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

        @test ρθ1 ≈ ρθ0 rtol = roundtrip_tolerance
        @test ρq1 ≈ ρq0 rtol = roundtrip_tolerance
        @test all(prate .== 0)
    end
end
