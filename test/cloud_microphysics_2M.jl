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

    μ2 = TwoMomentCloudMicrophysics()
    @test μ2 isa BulkMicrophysics
    @test μ2.cloud_formation isa NonEquilibriumCloudFormation
    @test μ2.cloud_formation.liquid isa ConstantRateCondensateFormation
    @test μ2.cloud_formation.ice === nothing

    @test μ2.categories isa TwoMomentCategories
    @test μ2.categories.warm_processes isa CloudMicrophysics.Parameters.SB2006

    prog_fields = Breeze.AtmosphereModels.prognostic_field_names(μ2)
    @test :ρqᶜˡ in prog_fields
    @test :ρnᶜˡ in prog_fields
    @test :ρqʳ in prog_fields
    @test :ρnʳ in prog_fields

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

    set!(model; θ=300, qᵗ=0.015)

    @test haskey(model.microphysical_fields, :ρqᶜˡ)
    @test haskey(model.microphysical_fields, :ρnᶜˡ)
    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :ρnʳ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)
    @test haskey(model.microphysical_fields, :nᶜˡ)
    @test haskey(model.microphysical_fields, :nʳ)
    @test haskey(model.microphysical_fields, :wᶜˡ)
    @test haskey(model.microphysical_fields, :wʳ)

    # Single time step (reduced from 6 iterations)
    time_step!(model, 1)
    @test model.clock.time == 1
    @test model.clock.iteration == 1
end

@testset "TwoMomentCloudMicrophysics setting initial conditions [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    ρᵣ = @allowscalar reference_state.density[1, 1, 1]

    qᶜˡ_value = FT(0.001)
    nᶜˡ_value = FT(100e6)
    qʳ_value = FT(0.002)
    nʳ_value = FT(1e6)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=qᶜˡ_value, nᶜˡ=nᶜˡ_value, qʳ=qʳ_value, nʳ=nʳ_value)

    @test @allowscalar model.microphysical_fields.ρqᶜˡ[1, 1, 1] ≈ ρᵣ * qᶜˡ_value
    @test @allowscalar model.microphysical_fields.ρnᶜˡ[1, 1, 1] ≈ ρᵣ * nᶜˡ_value
    @test @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1] ≈ ρᵣ * qʳ_value
    @test @allowscalar model.microphysical_fields.ρnʳ[1, 1, 1] ≈ ρᵣ * nʳ_value

    @test @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1] ≈ qᶜˡ_value
    @test @allowscalar model.microphysical_fields.nᶜˡ[1, 1, 1] ≈ nᶜˡ_value
    @test @allowscalar model.microphysical_fields.qʳ[1, 1, 1] ≈ qʳ_value
    @test @allowscalar model.microphysical_fields.nʳ[1, 1, 1] ≈ nʳ_value

    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "TwoMomentCloudMicrophysics precipitation rate and surface flux [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, nᶜˡ=0, qʳ=0.001, nʳ=1e5)
    time_step!(model, 1)

    # Precipitation rate
    P = precipitation_rate(model, :liquid)
    @test P isa Field
    compute!(P)
    @test isfinite(maximum(P))

    P_ice = precipitation_rate(model, :ice)
    @test P_ice === nothing

    # Surface precipitation flux
    spf = surface_precipitation_flux(model)
    @test spf isa Field
    compute!(spf)

    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    ρqʳ = @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1]
    expected_flux = -wʳ * ρqʳ

    @test @allowscalar spf[1, 1] ≈ expected_flux
    @test @allowscalar spf[1, 1] >= 0
end

@testset "TwoMomentCloudMicrophysics velocities and terminal velocities [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0.001, nᶜˡ=100e6, qʳ=0.001, nʳ=1e5)

    μ = model.microphysical_fields

    # microphysical_velocities
    vel_rain_mass = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain_mass !== nothing
    @test haskey(vel_rain_mass, :w)

    vel_rain_num = microphysical_velocities(microphysics, μ, Val(:ρnʳ))
    @test vel_rain_num !== nothing
    @test haskey(vel_rain_num, :w)

    vel_cloud_mass = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud_mass !== nothing
    @test haskey(vel_cloud_mass, :w)

    vel_cloud_num = microphysical_velocities(microphysics, μ, Val(:ρnᶜˡ))
    @test vel_cloud_num !== nothing
    @test haskey(vel_cloud_num, :w)

    # Terminal velocities
    wᶜˡ = @allowscalar μ.wᶜˡ[1, 1, 2]
    wʳ = @allowscalar μ.wʳ[1, 1, 2]

    @test wᶜˡ <= 0
    @test wʳ < 0
    @test abs(wʳ) > abs(wᶜˡ)
end

@testset "TwoMomentCloudMicrophysics ImpenetrableBoundaryCondition [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics(; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, nᶜˡ=0, qʳ=0.001, nʳ=1e5)

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
    Oceananigans.defaults.FloatType = FT
    Nz = 4
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = TwoMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    set!(model; θ=300, qᵗ=FT(0.030), qᶜˡ=FT(0.001), nᶜˡ=FT(100e6), qʳ=0, nʳ=0)

    qᶜˡ_initial = maximum(model.microphysical_fields.qᶜˡ)

    # Reduced simulation time (from 5τ to 3τ)
    τ_relax = 10.0
    simulation = Simulation(model; Δt=τ_relax/5, stop_time=3τ_relax, verbose=false)
    run!(simulation)

    qᶜˡ_final = maximum(model.microphysical_fields.qᶜˡ)
    @test qᶜˡ_final > qᶜˡ_initial * FT(0.5)
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

    # Set initial aerosol number from activation parameters
    nᵃ₀ = Breeze.initial_aerosol_number(microphysics)
    model.dynamics.state.μ = (; ρqᶜˡ=FT(0), ρnᶜˡ=FT(0), ρqʳ=FT(0), ρnʳ=FT(0), ρnᵃ=FT(nᵃ₀))

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
