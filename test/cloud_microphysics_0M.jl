using Breeze
using CloudMicrophysics
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

using Breeze.Thermodynamics:
    MoistureMassFractions,
    LiquidIceDensityState,
    mixture_heat_capacity,
    exner_function

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: ZeroMomentCloudMicrophysics

struct VaporOnlyNegativeMoistureCorrection end

Breeze.AtmosphereModels.negative_moisture_correction(::VaporOnlyNegativeMoistureCorrection) =
    Breeze.AtmosphereModels.VerticalBorrowing()

#####
##### Zero-moment microphysics tests
#####

@testset "ZeroMomentCloudMicrophysics construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    # Default construction
    μ0 = ZeroMomentCloudMicrophysics()
    @test μ0 isa BulkMicrophysics
    @test μ0.cloud_formation isa SaturationAdjustment

    # Custom parameters
    μ0_custom = ZeroMomentCloudMicrophysics(FT; τ_precip=500, qc_0=1e-3, S_0=0.01)
    @test μ0_custom isa BulkMicrophysics
    @test μ0_custom.categories.τ_precip == FT(500)
    @test μ0_custom.categories.qc_0 == FT(1e-3)
    @test μ0_custom.categories.S_0 == FT(0.01)
end

@testset "Standalone VerticalBorrowing corrects vapor columns [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 2), x=(0, 1), y=(0, 1), z=(0, 2),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, microphysics = ZeroMomentCloudMicrophysics())
    correction = VaporOnlyNegativeMoistureCorrection()

    ρ₀ = dynamics_density(model.dynamics)
    ρqᵛᵉ = model.moisture_density

    @allowscalar begin
        ρqᵛᵉ[1, 1, 1] = -FT(0.001) * ρ₀[1, 1, 1]
        ρqᵛᵉ[1, 1, 2] =  FT(0.003) * ρ₀[1, 1, 2]
    end

    initial_column_moisture = @allowscalar ρqᵛᵉ[1, 1, 1] + ρqᵛᵉ[1, 1, 2]

    Breeze.AtmosphereModels.fix_negative_moisture!(correction, model)

    @test @allowscalar ρqᵛᵉ[1, 1, 1] ≈ FT(0)
    @test @allowscalar ρqᵛᵉ[1, 1, 2] > 0
    @test @allowscalar ρqᵛᵉ[1, 1, 1] + ρqᵛᵉ[1, 1, 2] ≈ initial_column_moisture
end

@testset "ZeroMomentCloudMicrophysics time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = ZeroMomentCloudMicrophysics()

    model = AtmosphereModel(grid; dynamics, microphysics)

    # Set initial conditions with some moisture
    set!(model; θ=300, qᵗ=0.01)

    # Time step should succeed
    time_step!(model, 1)
    @test model.clock.time == 1
    @test model.clock.iteration == 1
end

@testset "ZeroMomentCloudMicrophysics precipitation rate diagnostic [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = ZeroMomentCloudMicrophysics()

    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.01)

    # Get precipitation rate diagnostic
    P = precipitation_rate(model, :liquid)
    @test P isa Field
    compute!(P)
    @test isfinite(maximum(P))

    # Ice precipitation not supported for 0M
    P_ice = precipitation_rate(model, :ice)
    @test P_ice === nothing
end

@testset "microphysical_thermodynamic_names defaults and opt-in" begin
    # Default: schemes do not source the thermodynamic prognostic
    sa = SaturationAdjustment()
    @test Breeze.AtmosphereModels.microphysical_thermodynamic_names(sa, nothing) == ()
    @test Breeze.AtmosphereModels.microphysical_thermodynamic_names(nothing, nothing) == ()

    # ZMCM opts in, per formulation (completed by Task 2 — @test_broken for now)
    grid = RectilinearGrid(default_arch; size=(1, 1, 2), x=(0, 1), y=(0, 1), z=(0, 1))
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    θ_model = AtmosphereModel(grid; dynamics, microphysics=ZeroMomentCloudMicrophysics())
    @test Breeze.AtmosphereModels.microphysical_thermodynamic_names(θ_model.microphysics, θ_model.formulation) == (:ρθ,)

    reference_state_e = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    e_model = AtmosphereModel(grid; dynamics=AnelasticDynamics(reference_state_e),
                              microphysics=ZeroMomentCloudMicrophysics(), formulation=:StaticEnergy)
    @test Breeze.AtmosphereModels.microphysical_thermodynamic_names(e_model.microphysics, e_model.formulation) == (:ρe,)
end

@testset "ZMCM precipitation tendencies retain latent heat [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    constants = ThermodynamicConstants(FT)
    microphysics = ZeroMomentCloudMicrophysics(FT; τ_precip=1000, qc_0=5e-4)

    pˢᵗ = FT(1e5)
    ρ = FT(11//10)
    θ = FT(300)
    ℒˡᵣ = constants.liquid.reference_latent_heat
    ℒⁱᵣ = constants.ice.reference_latent_heat
    microphysical_tendency = Breeze.AtmosphereModels.microphysical_tendency

    # Mixed-phase condensate above threshold
    q = MoistureMassFractions(FT(0.01), FT(2e-3), FT(1e-3))
    𝒰 = LiquidIceDensityState(θ, q, pˢᵗ, ρ)
    qᶜ = q.liquid + q.ice

    Gρqᵉ = microphysical_tendency(microphysics, Val(:ρqᵉ), ρ, nothing, 𝒰, constants)
    Gρθ  = microphysical_tendency(microphysics, Val(:ρθ),  ρ, nothing, 𝒰, constants)
    Gρe  = microphysical_tendency(microphysics, Val(:ρe),  ρ, nothing, 𝒰, constants)

    @test Gρqᵉ < 0   # water removed
    # Absolute magnitude from the documented 0M formula: dqᵉ/dt = -max(0, qᶜ - qc_0)/τ
    @test Gρqᵉ ≈ -ρ * (qᶜ - FT(5e-4)) / 1000
    @test Gρθ > 0    # warming retained
    @test Gρe > 0

    # Water sink and warming source derive from the same removal rate, with the
    # phase partition proportional to condensate:
    #   Gρθ = -Gρqᵉ ℒᶜ / (cᵖᵐ Π),   Gρe = -Gρqᵉ ℒᶜ,
    # where ℒᶜ is the condensate-weighted reference latent heat.
    ℒᶜ = (q.liquid * ℒˡᵣ + q.ice * ℒⁱᵣ) / qᶜ
    cᵖᵐ = mixture_heat_capacity(q, constants)
    Π = exner_function(𝒰, constants)
    @test Gρθ ≈ -Gρqᵉ * ℒᶜ / (cᵖᵐ * Π)
    @test Gρe ≈ -Gρqᵉ * ℒᶜ

    # Below the removal threshold: no precipitation, no spurious heating
    q₀ = MoistureMassFractions(FT(0.01), FT(1e-4), FT(0))
    𝒰₀ = LiquidIceDensityState(θ, q₀, pˢᵗ, ρ)
    @test microphysical_tendency(microphysics, Val(:ρqᵉ), ρ, nothing, 𝒰₀, constants) == 0
    @test microphysical_tendency(microphysics, Val(:ρθ),  ρ, nothing, 𝒰₀, constants) == 0
    @test microphysical_tendency(microphysics, Val(:ρe),  ρ, nothing, 𝒰₀, constants) == 0

    # Zero condensate: the phase-partition guard must not produce NaN
    qᵛ = MoistureMassFractions(FT(0.01))
    𝒰ᵛ = LiquidIceDensityState(θ, qᵛ, pˢᵗ, ρ)
    @test microphysical_tendency(microphysics, Val(:ρθ), ρ, nothing, 𝒰ᵛ, constants) == 0
end
