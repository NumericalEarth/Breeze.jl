using Breeze
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt:
    ZeroMomentCloudMicrophysics,
    OneMomentCloudMicrophysics

using Breeze.Microphysics: NonEquilibriumCloudFormation

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

    # Multiple time steps
    for _ in 1:5
        time_step!(model, 1)
    end
    @test model.clock.iteration == 6
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
    time_step!(model, 1)

    # Get precipitation rate diagnostic
    P = precipitation_rate(model, :liquid)
    @test P isa Field
    compute!(P)
    @test isfinite(maximum(P))

    # Ice precipitation not supported for 0M
    P_ice = precipitation_rate(model, :ice)
    @test P_ice === nothing
end

#####
##### One-moment microphysics tests
#####

@testset "OneMomentCloudMicrophysics construction [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    # Default construction (non-equilibrium)
    μ1 = OneMomentCloudMicrophysics()
    @test μ1 isa BulkMicrophysics
    @test μ1.cloud_formation isa NonEquilibriumCloudFormation
    @test μ1.cloud_formation.liquid isa CloudLiquid
    @test μ1.cloud_formation.ice === nothing

    # Check prognostic fields for non-equilibrium
    prog_fields = Breeze.AtmosphereModels.prognostic_field_names(μ1)
    @test :ρqᶜˡ in prog_fields
    @test :ρqʳ in prog_fields
end

@testset "OneMomentCloudMicrophysics with SaturationAdjustment [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    # Warm-phase saturation adjustment
    cloud_formation_warm = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    μ1_warm = OneMomentCloudMicrophysics(FT; cloud_formation=cloud_formation_warm)
    @test μ1_warm.cloud_formation isa SaturationAdjustment
    @test μ1_warm.cloud_formation.equilibrium isa WarmPhaseEquilibrium

    prog_fields_warm = Breeze.AtmosphereModels.prognostic_field_names(μ1_warm)
    @test :ρqʳ in prog_fields_warm
    @test :ρqᶜˡ ∉ prog_fields_warm  # cloud liquid is diagnostic for saturation adjustment

    # Mixed-phase saturation adjustment
    cloud_formation_mixed = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    μ1_mixed = OneMomentCloudMicrophysics(FT; cloud_formation=cloud_formation_mixed)
    @test μ1_mixed.cloud_formation isa SaturationAdjustment
    @test μ1_mixed.cloud_formation.equilibrium isa MixedPhaseEquilibrium

    prog_fields_mixed = Breeze.AtmosphereModels.prognostic_field_names(μ1_mixed)
    @test :ρqʳ in prog_fields_mixed
    @test :ρqˢ in prog_fields_mixed  # snow for mixed phase
end

@testset "OneMomentCloudMicrophysics non-equilibrium time-stepping [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)

    # Non-equilibrium (default)
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; formulation, microphysics)

    # Set initial conditions with some moisture
    set!(model; θ=300, qᵗ=0.015)

    # Check microphysical fields exist
    @test haskey(model.microphysical_fields, :ρqᶜˡ)
    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)

    # Time step should succeed
    time_step!(model, 1)
    @test model.clock.time == 1
    @test model.clock.iteration == 1

    # Multiple time steps
    for _ in 1:5
        time_step!(model, 1)
    end
    @test model.clock.iteration == 6
end

@testset "OneMomentCloudMicrophysics saturation adjustment time-stepping [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)

    # Warm-phase saturation adjustment
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; formulation, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    # Check microphysical fields exist (rain is prognostic, cloud liquid is diagnostic)
    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)

    # Time step should succeed
    time_step!(model, 1)
    @test model.clock.time == 1

    # Multiple time steps
    for _ in 1:5
        time_step!(model, 1)
    end
    @test model.clock.iteration == 6
end

@testset "OneMomentCloudMicrophysics mixed-phase time-stepping [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)

    # Mixed-phase saturation adjustment
    cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; formulation, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    # Check microphysical fields exist (rain and snow are prognostic)
    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :ρqˢ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qᶜⁱ)

    # Time step should succeed
    time_step!(model, 1)
    @test model.clock.time == 1

    # Multiple time steps
    for _ in 1:5
        time_step!(model, 1)
    end
    @test model.clock.iteration == 6
end

@testset "OneMomentCloudMicrophysics precipitation rate diagnostic [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)

    # Test for non-equilibrium scheme
    microphysics_ne = OneMomentCloudMicrophysics()
    model_ne = AtmosphereModel(grid; formulation, microphysics=microphysics_ne)
    set!(model_ne; θ=300, qᵗ=0.015)
    time_step!(model_ne, 1)

    P_ne = precipitation_rate(model_ne, :liquid)
    @test P_ne isa Field
    compute!(P_ne)
    @test isfinite(maximum(P_ne))

    # Test for saturation adjustment scheme
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics_sa = OneMomentCloudMicrophysics(FT; cloud_formation)
    model_sa = AtmosphereModel(grid; formulation, microphysics=microphysics_sa)
    set!(model_sa; θ=300, qᵗ=0.015)
    time_step!(model_sa, 1)

    P_sa = precipitation_rate(model_sa, :liquid)
    @test P_sa isa Field
    compute!(P_sa)
    @test isfinite(maximum(P_sa))

    # Ice precipitation not yet implemented
    P_ice = precipitation_rate(model_ne, :ice)
    @test P_ice === nothing
end

@testset "NonEquilibriumCloudFormation construction [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    # Default construction
    cloud_formation_default = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing)
    @test cloud_formation_default.liquid isa CloudLiquid
    @test cloud_formation_default.ice === nothing
    @test cloud_formation_default.liquid.τ_relax == FT(10.0)  # CloudMicrophysics default

    # With ice parameters
    cloud_formation_mixed = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    @test cloud_formation_mixed.liquid isa CloudLiquid
    @test cloud_formation_mixed.ice isa CloudIce

    # Build full microphysics with non-equilibrium cloud formation
    μ1 = OneMomentCloudMicrophysics(FT; cloud_formation=cloud_formation_default)
    @test μ1.cloud_formation isa NonEquilibriumCloudFormation
    @test μ1.cloud_formation.liquid.τ_relax == FT(10.0)
end
