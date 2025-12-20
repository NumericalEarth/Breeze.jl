using Breeze
using CloudMicrophysics
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: ZeroMomentCloudMicrophysics

#####
##### Zero-moment microphysics tests
#####

@testset "ZeroMomentCloudMicrophysics construction [$(FT)]" for FT in (Float32, Float64)
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

@testset "ZeroMomentCloudMicrophysics time-stepping [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)
    microphysics = ZeroMomentCloudMicrophysics()

    model = AtmosphereModel(grid; formulation, microphysics)

    # Set initial conditions with some moisture
    set!(model; θ=300, qᵗ=0.01)

    # Time step should succeed
    time_step!(model, 1)
    @test model.clock.time == 1
    @test model.clock.iteration == 1
end

@testset "ZeroMomentCloudMicrophysics precipitation rate diagnostic [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)
    microphysics = ZeroMomentCloudMicrophysics()

    model = AtmosphereModel(grid; formulation, microphysics)
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
