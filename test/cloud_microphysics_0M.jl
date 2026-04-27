using Breeze
using CloudMicrophysics
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

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
