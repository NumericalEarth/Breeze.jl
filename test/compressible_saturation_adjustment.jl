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
    saturation_specific_humidity

using Breeze.Microphysics: SaturationAdjustment, adjust_thermodynamic_state, WarmPhaseEquilibrium
using Oceananigans.TimeSteppers: update_state!

# Regression tests for the compressible θˡⁱ density-based thermodynamic state
# (NumericalEarth/Breeze.jl#765): the temperature inversion and the saturation adjustment must be
# evaluated at the prognostic density ρ (with p = ρRᵐT), so the dynamics and the microphysics carry
# one self-consistent temperature and the equilibrium sits on the saturation curve at ρ.

@testset "Density-based θˡⁱ state (LiquidIceDensityState) [$FT]" for FT in test_float_types()
    constants = ThermodynamicConstants(FT)
    pˢᵗ = FT(1e5)
    ℒˡ = constants.liquid.reference_latent_heat
    ℒⁱ = constants.ice.reference_latent_heat
    rtol = FT == Float64 ? FT(1e-9) : FT(1e-4)
    qtol = FT == Float64 ? FT(1e-5) : FT(1e-3)

    moistures = (MoistureMassFractions(FT(0.020)),                          # vapor only
                 MoistureMassFractions(FT(0.018), FT(0.005), FT(0)),        # liquid condensate
                 MoistureMassFractions(FT(0.015), FT(0.003), FT(0.002)))    # mixed phase

    @testset "constant-density θˡⁱ→T inversion is self-consistent and round-trips" begin
        θ = FT(300); ρ = FT(1)
        for q in moistures
            𝒰 = LiquidIceDensityState(θ, q, pˢᵗ, ρ)
            T = temperature(𝒰, constants)
            Rᵐ = mixture_gas_constant(q, constants)
            cᵖᵐ = mixture_heat_capacity(q, constants)
            κ = Rᵐ / cᵖᵐ
            L = (ℒˡ * q.liquid + ℒⁱ * q.ice) / cᵖᵐ
            # self-consistent fixed point: T = (ρRᵐT/pˢᵗ)^κ θ + L  (and p = ρRᵐT)
            @test T ≈ (ρ * Rᵐ * T / pˢᵗ)^κ * θ + L  rtol=rtol
            # θˡⁱ round-trips: θ → T → θ
            @test with_temperature(𝒰, T, constants).potential_temperature ≈ θ  rtol=rtol
        end
    end

    @testset "density-consistent saturation adjustment" begin
        sa = SaturationAdjustment(FT; equilibrium = WarmPhaseEquilibrium())
        θ₀ = FT(300); ρ = FT(1)

        # Supersaturated, all-vapor parcel → condenses.
        𝒰₀ = LiquidIceDensityState(θ₀, MoistureMassFractions(FT(0.020)), pˢᵗ, ρ)
        𝒰₁ = adjust_thermodynamic_state(𝒰₀, sa, constants)
        T₁ = temperature(𝒰₁, constants)
        qᵛ = 𝒰₁.moisture_mass_fractions.vapor

        # The vapor lies on the saturation curve evaluated at the cell's OWN density ρ
        # (the crux of #765 — not against a stale reference pressure).
        @test qᵛ ≈ saturation_specific_humidity(T₁, ρ, constants, WarmPhaseEquilibrium())  atol=qtol
        @test 𝒰₁.potential_temperature == θ₀          # θˡⁱ held fixed (conserved) exactly
        @test 𝒰₁.density == ρ                          # density is carried directly
        @test 𝒰₁.moisture_mass_fractions.liquid > 0    # condensate produced

        # Subsaturated parcel → no condensation.
        𝒰dry = adjust_thermodynamic_state(
            LiquidIceDensityState(θ₀, MoistureMassFractions(FT(0.002)), pˢᵗ, ρ), sa, constants)
        @test 𝒰dry.moisture_mass_fractions.liquid == 0
    end

    @testset "fixes the κ·ΔL temperature inconsistency" begin
        θ = FT(300); ρ = FT(1)
        q = MoistureMassFractions(FT(0.018), FT(0.005), FT(0))
        Rᵐ = mixture_gas_constant(q, constants)
        cᵖᵐ = mixture_heat_capacity(q, constants)
        κ = Rᵐ / cᵖᵐ; γ = cᵖᵐ / (cᵖᵐ - Rᵐ)
        L = (ℒˡ * q.liquid + ℒⁱ * q.ice) / cᵖᵐ

        T★ = temperature(LiquidIceDensityState(θ, q, pˢᵗ, ρ), constants)
        T_noniter = θ^γ * (ρ * Rᵐ / pˢᵗ)^(γ - 1) + L   # the old non-iterated inversion
        @test T★ > T_noniter                            # the non-iterated inversion is biased cold
        @test isapprox(T★ - T_noniter, FT(1.39) * κ * L; rtol=FT(0.15))   # gap ≈ 1.39 κΔL (leading order)

        # The solver abstraction is honored: `nothing` returns the un-iterated closed form,
        # and `FixedIterations` (the unrolled, Reactant-safe form) matches the converged Newton.
        𝒰_noniter = LiquidIceDensityState(θ, q, pˢᵗ, ρ, nothing)
        @test temperature(𝒰_noniter, constants) ≈ T_noniter  rtol=rtol

        𝒰_fixed = LiquidIceDensityState(θ, q, pˢᵗ, ρ, FixedIterations(8))
        @test temperature(𝒰_fixed, constants) ≈ T★  rtol=rtol

        # An abstol-only Newton solver (no relative criterion) also converges to the root.
        𝒰_abstol = LiquidIceDensityState(θ, q, pˢᵗ, ρ, NewtonSolver(FT; reltol=0, abstol=1e-6, maxiter=20))
        @test temperature(𝒰_abstol, constants) ≈ T★  atol=FT(1e-4)

        # Control: with no condensate the inversion reduces to the dry closed form (no latent shift).
        qd = MoistureMassFractions(FT(0.020))
        Rd = mixture_gas_constant(qd, constants)
        cpd = mixture_heat_capacity(qd, constants)
        γd = cpd / (cpd - Rd)
        @test temperature(LiquidIceDensityState(θ, qd, pˢᵗ, ρ), constants) ≈
              θ^γd * (ρ * Rd / pˢᵗ)^(γd - 1)  rtol=rtol
    end
end

# Integration: a moist compressible `update_state!` must produce physical, self-consistent fields.
# In condensing cells the equilibrium vapor must sit on the saturation curve at the cell's OWN
# density ρ — which also requires the dynamics temperature and the saturation-adjusted temperature
# to agree (the κ·ΔL inconsistency of #765).
@testset "Moist compressible update_state! is density-consistent" begin
    arch = default_arch
    grid = RectilinearGrid(arch; size = (8, 8, 8), halo = (5, 5, 5),
                           x = (0, 2e3), y = (0, 2e3), z = (0, 2e3),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(Float64)
    θref(z) = 300.0 * exp(9.80616 * z / (1005 * 300.0))
    dyn = CompressibleDynamics(SplitExplicitTimeDiscretization();
                               surface_pressure = 1e5, standard_pressure = 1e5,
                               reference_potential_temperature = θref)
    model = AtmosphereModel(grid; dynamics = dyn,
                            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                            thermodynamic_constants = constants,
                            timestepper = :AcousticRungeKutta3)
    set!(model; ρ = model.dynamics.reference_state.density, θ = 300.0, qᵗ = 0.030)
    update_state!(model)

    T  = Array(interior(model.temperature))
    p  = Array(interior(model.dynamics.pressure))
    ρ  = Array(interior(model.dynamics.total_density))  # saturation adjustment saturates at total ρ
    qˡ = Array(interior(model.microphysical_fields.qˡ))
    qᵛ = Array(interior(model.microphysical_fields.qᵛ))

    @test all(isfinite, T) && all(>(0), T)
    @test all(isfinite, p) && all(>(0), p)
    @test maximum(qˡ) > 0                                   # condensation occurred

    eq = WarmPhaseEquilibrium()
    for I in eachindex(qˡ)
        qˡ[I] > 1e-6 || continue                            # saturated cells only
        @test qᵛ[I] ≈ saturation_specific_humidity(T[I], ρ[I], constants, eq)  atol = 1e-4
    end

    time_step!(model, 1e-3)
    @test all(isfinite, interior(model.temperature))
end

# `set!` density-input modes on the compressible core. Initializing with an in-situ temperature `T`
# exercises the θˡⁱ-from-T kernel, which needs the diagnosed total density ρ available (mass
# fractions) and weights ρθ = ρᵈθ by the dry density. The two input modes establish ρ differently:
#   `:ρ`  (total) — total_density ← ρ, dry_density ← ρ·(1-qᵗ)
#   `:ρᵈ` (dry)   — total_density ← ρᵈ/(1-qᵗ), dry_density ← ρᵈ
# Choosing ρᵈ = ρ·(1-qᵗ) makes the two columns identical, so they must produce the same state.
@testset "Moist compressible set! density modes (ρ vs ρᵈ) [$FT]" for FT in test_float_types()
    arch = default_arch
    grid = RectilinearGrid(arch; size = (8, 8, 8), halo = (5, 5, 5),
                           x = (0, 1e3), y = (0, 1e3), z = (0, 1e3),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(FT)

    make_model() = AtmosphereModel(grid;
        dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                        surface_pressure = 1e5, standard_pressure = 1e5,
                                        reference_potential_temperature = z -> 300.0),
        microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
        thermodynamic_constants = constants,
        timestepper = :AcousticRungeKutta3)

    T₀ = FT(300)
    qᵗ = FT(0.005)     # unsaturated at 300 K, 1e5 Pa — no condensation
    ρ₀ = FT(1.16)      # total air density [kg m⁻³]
    rtol = FT == Float64 ? FT(1e-6) : FT(1e-3)

    # Pull interiors to the CPU before reducing — `all(≈(ρ₀; rtol), ::CuArray)` would compile the
    # keyword-carrying closure into a GPU kernel (non-bitstype Symbol in the kwargs) and fail.
    cpu(field) = Array(interior(field))

    # Mode 1: total density given.
    model_ρ = make_model()
    set!(model_ρ; ρ = ρ₀, T = T₀, qᵗ = qᵗ)
    update_state!(model_ρ)

    ρθ_ρ = cpu(model_ρ.formulation.potential_temperature_density)
    ρt_ρ = cpu(model_ρ.dynamics.total_density)
    ρd_ρ = cpu(model_ρ.dynamics.dry_density)
    T_ρ  = cpu(model_ρ.temperature)
    @test all(isfinite, ρθ_ρ) && all(>(0), ρθ_ρ)            # not the ρθ = 0 bug
    @test all(≈(ρ₀; rtol), ρt_ρ)
    @test all(≈(ρ₀ * (1 - qᵗ); rtol), ρd_ρ)                  # dry excludes water
    # Physical, finite temperature — the ρθ = 0 staleness bug gave θ = ρθ/ρᵈ = 0 ⇒ T ≈ 0.
    @test all(t -> isfinite(t) && 200 < t < 400, T_ρ)

    # Mode 2: dry density given, chosen so the total recovers to ρ₀.
    model_ρᵈ = make_model()
    set!(model_ρᵈ; ρᵈ = ρ₀ * (1 - qᵗ), T = T₀, qᵗ = qᵗ)
    update_state!(model_ρᵈ)

    ρθ_ρᵈ = cpu(model_ρᵈ.formulation.potential_temperature_density)
    ρt_ρᵈ = cpu(model_ρᵈ.dynamics.total_density)
    ρd_ρᵈ = cpu(model_ρᵈ.dynamics.dry_density)
    T_ρᵈ  = cpu(model_ρᵈ.temperature)
    @test all(≈(ρ₀; rtol), ρt_ρᵈ)                            # ρ = ρᵈ/(1-qᵗ)
    @test all(≈(ρ₀ * (1 - qᵗ); rtol), ρd_ρᵈ)

    # The two input modes describe the same column → identical state.
    @test ρt_ρ ≈ ρt_ρᵈ
    @test ρd_ρ ≈ ρd_ρᵈ
    @test ρθ_ρ ≈ ρθ_ρᵈ
    @test T_ρ  ≈ T_ρᵈ
end
