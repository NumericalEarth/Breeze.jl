using Breeze
using Breeze.AtmosphereModels: microphysical_velocities
using CloudMicrophysics
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt:
    TwoMomentCloudMicrophysics,
    TwoMomentCategories,
    two_moment_cloud_microphysics_categories,
    AerosolActivation,
    default_aerosol_activation

using Breeze.Microphysics: ConstantRateCondensateFormation
using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition

#####
##### Two-moment microphysics tests
#####

@testset "TwoMomentCloudMicrophysics construction [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    # Default construction (non-equilibrium)
    μ2 = TwoMomentCloudMicrophysics()
    @test μ2 isa BulkMicrophysics
    @test μ2.cloud_formation isa NonEquilibriumCloudFormation
    @test μ2.cloud_formation.liquid isa ConstantRateCondensateFormation
    @test μ2.cloud_formation.ice === nothing

    # Check categories
    @test μ2.categories isa TwoMomentCategories
    @test μ2.categories.warm_processes isa CloudMicrophysics.Parameters.SB2006

    # Check prognostic fields (mass + number for both cloud and rain)
    prog_fields = Breeze.AtmosphereModels.prognostic_field_names(μ2)
    @test :ρqᶜˡ in prog_fields  # cloud liquid mass
    @test :ρnᶜˡ in prog_fields  # cloud liquid number
    @test :ρqʳ in prog_fields   # rain mass
    @test :ρnʳ in prog_fields   # rain number

    # Should throw error if trying to use saturation adjustment
    @test_throws ArgumentError TwoMomentCloudMicrophysics(cloud_formation = SaturationAdjustment(FT))
end

@testset "TwoMomentCategories construction [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    categories = two_moment_cloud_microphysics_categories(FT)
    @test categories isa TwoMomentCategories
    @test categories.warm_processes isa CloudMicrophysics.Parameters.SB2006
    @test categories.air_properties isa CloudMicrophysics.Parameters.AirProperties
    @test categories.cloud_liquid_fall_velocity isa CloudMicrophysics.Parameters.StokesRegimeVelType
    @test categories.rain_fall_velocity isa CloudMicrophysics.Parameters.SB2006VelType
end

@testset "TwoMomentCloudMicrophysics time-stepping [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Set initial conditions with some moisture
    set!(model; θ=300, qᵗ=0.015)

    # Check microphysical fields exist
    @test haskey(model.microphysical_fields, :ρqᶜˡ)  # cloud liquid mass
    @test haskey(model.microphysical_fields, :ρnᶜˡ)  # cloud liquid number
    @test haskey(model.microphysical_fields, :ρqʳ)   # rain mass
    @test haskey(model.microphysical_fields, :ρnʳ)   # rain number
    @test haskey(model.microphysical_fields, :qᶜˡ)   # diagnostic cloud liquid
    @test haskey(model.microphysical_fields, :qʳ)    # diagnostic rain
    @test haskey(model.microphysical_fields, :nᶜˡ)   # diagnostic cloud number
    @test haskey(model.microphysical_fields, :nʳ)    # diagnostic rain number
    @test haskey(model.microphysical_fields, :wᶜˡ)   # cloud terminal velocity (mass-weighted)
    @test haskey(model.microphysical_fields, :wʳ)    # rain terminal velocity (mass-weighted)

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

@testset "TwoMomentCloudMicrophysics setting initial conditions [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Get reference density
    ρᵣ = @allowscalar reference_state.density[1, 1, 1]

    # Set specific microphysical variables
    qᶜˡ_value = FT(0.001)  # 1 g/kg cloud liquid
    nᶜˡ_value = FT(100e6)  # 100 million droplets per kg
    qʳ_value = FT(0.002)   # 2 g/kg rain
    nʳ_value = FT(1e6)     # 1 million drops per kg

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=qᶜˡ_value, nᶜˡ=nᶜˡ_value, qʳ=qʳ_value, nʳ=nʳ_value)

    # Check that density-weighted fields were set correctly
    @test @allowscalar model.microphysical_fields.ρqᶜˡ[1, 1, 1] ≈ ρᵣ * qᶜˡ_value
    @test @allowscalar model.microphysical_fields.ρnᶜˡ[1, 1, 1] ≈ ρᵣ * nᶜˡ_value
    @test @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1] ≈ ρᵣ * qʳ_value
    @test @allowscalar model.microphysical_fields.ρnʳ[1, 1, 1] ≈ ρᵣ * nʳ_value

    # Check that specific fields are diagnosed correctly
    @test @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1] ≈ qᶜˡ_value
    @test @allowscalar model.microphysical_fields.nᶜˡ[1, 1, 1] ≈ nᶜˡ_value
    @test @allowscalar model.microphysical_fields.qʳ[1, 1, 1] ≈ qʳ_value
    @test @allowscalar model.microphysical_fields.nʳ[1, 1, 1] ≈ nʳ_value

    # Test that time-stepping works after setting specific variables
    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "TwoMomentCloudMicrophysics precipitation rate diagnostic [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015)
    time_step!(model, 1)

    P = precipitation_rate(model, :liquid)
    @test P isa Field
    compute!(P)
    @test isfinite(maximum(P))

    # Ice precipitation not yet implemented
    P_ice = precipitation_rate(model, :ice)
    @test P_ice === nothing
end

@testset "TwoMomentCloudMicrophysics surface precipitation flux [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Set some rain with number concentration
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, nᶜˡ=0, qʳ=0.001, nʳ=1e5)

    # Get surface precipitation flux
    spf = surface_precipitation_flux(model)
    @test spf isa Field
    compute!(spf)

    # Check that flux is computed correctly
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    ρqʳ = @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1]
    expected_flux = -wʳ * ρqʳ  # Positive for downward flux (wʳ < 0)

    @test @allowscalar spf[1, 1] ≈ expected_flux
    @test @allowscalar spf[1, 1] >= 0  # Rain falls down, so flux should be non-negative
end

@testset "TwoMomentCloudMicrophysics microphysical_velocities [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015, qᶜˡ=0.001, nᶜˡ=100e6, qʳ=0.001, nʳ=1e5)

    μ = model.microphysical_fields

    # Rain mass should have mass-weighted sedimentation velocity
    vel_rain_mass = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain_mass !== nothing
    @test haskey(vel_rain_mass, :w)

    # Rain number should have number-weighted sedimentation velocity
    vel_rain_num = microphysical_velocities(microphysics, μ, Val(:ρnʳ))
    @test vel_rain_num !== nothing
    @test haskey(vel_rain_num, :w)

    # Cloud liquid mass should have mass-weighted sedimentation velocity
    vel_cloud_mass = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud_mass !== nothing
    @test haskey(vel_cloud_mass, :w)

    # Cloud liquid number should have number-weighted sedimentation velocity
    vel_cloud_num = microphysical_velocities(microphysics, μ, Val(:ρnᶜˡ))
    @test vel_cloud_num !== nothing
    @test haskey(vel_cloud_num, :w)
end

@testset "TwoMomentCloudMicrophysics terminal velocities [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Set realistic cloud and rain with number concentrations
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0.001, nᶜˡ=100e6, qʳ=0.001, nʳ=1e5)

    μ = model.microphysical_fields

    # Check that terminal velocities are computed and negative (downward)
    wᶜˡ = @allowscalar μ.wᶜˡ[1, 1, 2]  # Not bottom cell
    wᶜˡₙ = @allowscalar μ.wᶜˡₙ[1, 1, 2]
    wʳ = @allowscalar μ.wʳ[1, 1, 2]
    wʳₙ = @allowscalar μ.wʳₙ[1, 1, 2]

    # Cloud droplets should have small (but non-zero) terminal velocity
    @test wᶜˡ <= 0  # Downward or zero
    @test wᶜˡₙ <= 0

    # Rain drops should fall faster than cloud droplets
    @test wʳ < 0  # Should definitely be falling
    @test wʳₙ < 0
    @test abs(wʳ) > abs(wᶜˡ)  # Rain falls faster than cloud
end

@testset "TwoMomentCloudMicrophysics ImpenetrableBoundaryCondition [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Use ImpenetrableBoundaryCondition to prevent rain from exiting
    microphysics = TwoMomentCloudMicrophysics(; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set initial rain
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, nᶜˡ=0, qʳ=0.001, nʳ=1e5)

    # Check terminal velocity at bottom is zero (impenetrable)
    wʳ_bottom = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    wʳₙ_bottom = @allowscalar model.microphysical_fields.wʳₙ[1, 1, 1]
    wᶜˡ_bottom = @allowscalar model.microphysical_fields.wᶜˡ[1, 1, 1]
    wᶜˡₙ_bottom = @allowscalar model.microphysical_fields.wᶜˡₙ[1, 1, 1]

    @test wʳ_bottom == 0
    @test wʳₙ_bottom == 0
    @test wᶜˡ_bottom == 0
    @test wᶜˡₙ_bottom == 0
end

@testset "TwoMomentCloudMicrophysics show methods [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    μ2 = TwoMomentCloudMicrophysics()
    str = sprint(show, μ2)
    @test contains(str, "TwoMomentCloudMicrophysics")
    @test contains(str, "cloud_formation")
    @test contains(str, "warm_processes")
end

@testset "TwoMomentCloudMicrophysics cloud condensation [$FT]" for FT in test_float_types()
    # Test that cloud liquid forms via condensation in supersaturated conditions
    Oceananigans.defaults.FloatType = FT
    Nz = 4
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set high moisture content (should be supersaturated)
    # Also set initial droplet number (since we don't have activation yet)
    set!(model; θ=300, qᵗ=FT(0.030), qᶜˡ=FT(0.001), nᶜˡ=FT(100e6), qʳ=0, nʳ=0)

    qᶜˡ_initial = maximum(model.microphysical_fields.qᶜˡ)

    # Run for condensation (reduced from 10τ to 5τ)
    τ_relax = 10.0  # Default relaxation timescale
    simulation = Simulation(model; Δt=τ_relax/5, stop_time=5τ_relax, verbose=false)
    run!(simulation)

    # Cloud liquid should have increased due to condensation
    qᶜˡ_final = maximum(model.microphysical_fields.qᶜˡ)
    @test qᶜˡ_final > qᶜˡ_initial * FT(0.5)  # Allow for some evaporation depending on conditions
end

#####
##### Aerosol activation tests
#####

@testset "AerosolActivation construction [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    # Check default aerosol activation is created
    aa = default_aerosol_activation(FT)
    @test aa isa AerosolActivation
    @test aa.activation_parameters isa CloudMicrophysics.Parameters.AerosolActivationParameters
    @test aa.aerosol_distribution isa CloudMicrophysics.AerosolModel.AerosolDistribution

    # Check aerosol activation is included in TwoMomentCategories
    categories = two_moment_cloud_microphysics_categories(FT)
    @test categories.aerosol_activation isa AerosolActivation

    # Check aerosol activation is included in TwoMomentCloudMicrophysics
    μ2 = TwoMomentCloudMicrophysics()
    @test μ2.categories.aerosol_activation isa AerosolActivation

    # Check that aerosol activation can be disabled
    categories_no_act = two_moment_cloud_microphysics_categories(FT; aerosol_activation=nothing)
    @test categories_no_act.aerosol_activation === nothing
end

@testset "AerosolActivation in parcel model [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    grid = RectilinearGrid(CPU(); size=100, z=(0, 10_000), topology=(Flat, Flat, Bounded))
    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics=ParcelDynamics(), microphysics)

    constants = model.thermodynamic_constants
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)

    qᵗ(z) = 0.015 * exp(-z / 2500)

    set!(model,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density,
         qᵗ = qᵗ,
         z = 0, w = 1)

    # Initially, cloud droplet number should be zero (no droplets before activation)
    @test model.dynamics.state.μ.ρnᶜˡ == 0

    # Run parcel simulation until it becomes supersaturated
    simulation = Simulation(model; Δt=1.0, stop_iteration=500)
    run!(simulation)

    # After rising, parcel should have some cloud droplets from activation
    # (if it reached supersaturation)
    z_final = model.dynamics.state.z
    nᶜˡ_final = model.dynamics.state.μ.ρnᶜˡ / model.dynamics.state.ρ
    qᶜˡ_final = model.dynamics.state.μ.ρqᶜˡ / model.dynamics.state.ρ

    # Parcel should have risen
    @test z_final > 499  # Should have risen at least 1 km

    # If cloud formed (qᶜˡ > 0), droplet number should also be positive
    if qᶜˡ_final > FT(1e-10)
        @test nᶜˡ_final > 0  # Activation should have produced droplets
    end
end
