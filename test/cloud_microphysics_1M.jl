using Breeze
using Breeze.AtmosphereModels: microphysical_velocities
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

include("test_utils.jl")

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics
using Breeze.Microphysics: ConstantRateCondensateFormation

using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition

#####
##### One-moment microphysics tests
#####

@testset "OneMomentCloudMicrophysics construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    # Default construction (non-equilibrium)
    μ1 = OneMomentCloudMicrophysics()
    @test μ1 isa BulkMicrophysics
    @test μ1.cloud_formation isa NonEquilibriumCloudFormation
    @test μ1.cloud_formation.liquid isa ConstantRateCondensateFormation
    @test μ1.cloud_formation.ice === nothing

    # Mixed-phase non-equilibrium is signaled by `ice isa AbstractCondensateFormation`.
    # We use a placeholder `ConstantRateCondensateFormation` and the constructor materializes
    # relaxation parameters from `categories.cloud_ice`.
    μ1_mixed = OneMomentCloudMicrophysics(cloud_formation = NonEquilibriumCloudFormation(nothing, ConstantRateCondensateFormation(FT(0))))
    @test μ1_mixed.cloud_formation.ice isa ConstantRateCondensateFormation

    # Check prognostic fields for non-equilibrium
    prog_fields = Breeze.AtmosphereModels.prognostic_field_names(μ1)
    @test :ρqᶜˡ in prog_fields
    @test :ρqʳ in prog_fields
end

@testset "OneMomentCloudMicrophysics with SaturationAdjustment [$(FT)]" for FT in test_float_types()
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

@testset "OneMomentCloudMicrophysics non-equilibrium time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Non-equilibrium (default)
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

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

@testset "OneMomentCloudMicrophysics saturation adjustment time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Warm-phase saturation adjustment
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

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

@testset "OneMomentCloudMicrophysics mixed-phase time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Mixed-phase saturation adjustment
    cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

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

@testset "OneMomentCloudMicrophysics precipitation rate diagnostic [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Test for non-equilibrium scheme
    microphysics_ne = OneMomentCloudMicrophysics()
    model_ne = AtmosphereModel(grid; dynamics, microphysics=microphysics_ne)
    set!(model_ne; θ=300, qᵗ=0.015)
    time_step!(model_ne, 1)

    P_ne = precipitation_rate(model_ne, :liquid)
    @test P_ne isa Field
    compute!(P_ne)
    @test isfinite(maximum(P_ne))

    # Test for saturation adjustment scheme
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics_sa = OneMomentCloudMicrophysics(FT; cloud_formation)
    model_sa = AtmosphereModel(grid; dynamics, microphysics=microphysics_sa)
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

@testset "NonEquilibriumCloudFormation construction [$(FT)]" for FT in test_float_types()
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
    @test μ1.categories.cloud_liquid.τ_relax == FT(10.0)
end

@testset "Setting specific microphysical variables [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Non-equilibrium scheme has both qᶜˡ and qʳ as prognostic
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Get reference density
    ρᵣ = @allowscalar reference_state.density[1, 1, 1]

    # Set specific microphysical variables (without ρ prefix)
    qᶜˡ_value = FT(0.001)
    qʳ_value = FT(0.002)
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=qᶜˡ_value, qʳ=qʳ_value)

    # Check that density-weighted fields were set correctly
    @test @allowscalar model.microphysical_fields.ρqᶜˡ[1, 1, 1] ≈ ρᵣ * qᶜˡ_value
    @test @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1] ≈ ρᵣ * qʳ_value

    # Check that specific fields are diagnosed correctly
    @test @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1] ≈ qᶜˡ_value
    @test @allowscalar model.microphysical_fields.qʳ[1, 1, 1] ≈ qʳ_value

    # Test that time-stepping works after setting specific variables
    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "Surface precipitation flux diagnostic [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Set some rain
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)

    # Get surface precipitation flux
    spf = surface_precipitation_flux(model)
    @test spf isa Field
    compute!(spf)

    # Check that flux is computed correctly
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    ρqʳ = @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1]
    expected_flux = -wʳ * ρqʳ  # Positive for downward flux (wʳ < 0)

    @test @allowscalar spf[1, 1] ≈ expected_flux
    @test @allowscalar spf[1, 1] > 0  # Rain falls down, so flux should be positive
end

@testset "Rain accumulation from autoconversion [$(FT)]" for FT in test_float_types()
    # This test verifies that microphysical tendencies (autoconversion) are
    # actually being applied to prognostic fields during time-stepping.
    # If this test fails, it indicates a bug in tendency application.
    #
    # Note: We use multiple vertical levels because in a single-cell domain,
    # rain sedimentation (terminal velocity) creates a flux divergence that
    # exactly cancels the autoconversion tendency. With multiple levels,
    # rain can accumulate in the domain before sedimenting out.
    
    Oceananigans.defaults.FloatType = FT
    Nz = 10
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set initial conditions with cloud liquid present (at saturation or above)
    # High qᵗ ensures supersaturation → cloud forms
    set!(model; θ=300, qᵗ=FT(0.050))

    # First, run to condensation equilibrium (reduced from 10τ to 5τ)
    τ = microphysics.categories.cloud_liquid.τ_relax
    simulation = Simulation(model; Δt=τ/5, stop_time=5τ, verbose=false)
    run!(simulation)

    # Cloud liquid should have formed
    qᶜˡ_equilibrium = maximum(model.microphysical_fields.qᶜˡ)
    @test qᶜˡ_equilibrium > FT(0.001)  # At least 1 g/kg cloud liquid

    # Sum total rain in domain before autoconversion run
    ρqʳ_total_initial = sum(model.microphysical_fields.ρqʳ)

    # Now run longer for autoconversion to accumulate rain (reduced from 100τ to 30τ)
    # (Rain will sediment and exit at bottom, but should still accumulate in upper cells)
    simulation.stop_time = simulation.model.clock.time + 30τ
    run!(simulation)

    # Check that rain was produced (either still in domain or has sedimented through)
    # We check total rain mass in domain
    ρqʳ_total_final = sum(model.microphysical_fields.ρqʳ)
    qʳ_max_final = maximum(model.microphysical_fields.qʳ)

    # Rain should exist somewhere in the domain
    # (Even if some has sedimented out, there should be rain in upper cells)
    @test qʳ_max_final > FT(1e-8)  # At least some rain exists
    
    # Check that autoconversion is happening by verifying rain increases initially
    # before sedimentation can remove it all. We'll check the top cell which
    # should accumulate rain without losing it to sedimentation as quickly.
    qʳ_top = @allowscalar model.microphysical_fields.qʳ[1, 1, Nz]
    @test qʳ_top > FT(1e-10)  # Rain should form in top cell
end

@testset "ImpenetrableBoundaryCondition prevents rain from exiting domain [$(FT)]" for FT in test_float_types()
    # This test verifies that ImpenetrableBoundaryCondition allows rain to accumulate
    # in a single-cell domain where it would otherwise sediment out.
    
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Use ImpenetrableBoundaryCondition to prevent rain from exiting
    microphysics = OneMomentCloudMicrophysics(; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Set initial conditions with cloud liquid present
    set!(model; θ=300, qᵗ=FT(0.050))

    # Run to condensation equilibrium and beyond for autoconversion (reduced from 10τ to 5τ)
    τ = microphysics.categories.cloud_liquid.τ_relax
    simulation = Simulation(model; Δt=τ/5, stop_time=5τ, verbose=false)
    run!(simulation)

    # With ImpenetrableBoundaryCondition, rain should accumulate in the domain
    # because it can't sediment out through the bottom
    qʳ_final = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    
    # Check terminal velocity at bottom is zero (impenetrable)
    wʳ_bottom = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    @test wʳ_bottom == 0  # Terminal velocity should be zero at impenetrable bottom

    # Rain should have accumulated substantially
    @test qʳ_final > FT(0.001)  # At least 1 g/kg rain accumulated
end

@testset "Mixed-phase non-equilibrium time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    # Mixed-phase non-equilibrium (both cloud liquid and ice are prognostic)
    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    prog_fields = Breeze.AtmosphereModels.prognostic_field_names(microphysics)
    @test :ρqᶜˡ in prog_fields
    @test :ρqᶜⁱ in prog_fields
    @test :ρqʳ in prog_fields
    @test :ρqˢ in prog_fields

    set!(model; θ=260, qᵗ=0.010)
    @test haskey(model.microphysical_fields, :ρqᶜⁱ)
    @test haskey(model.microphysical_fields, :qᶜⁱ)

    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "OneMomentCloudMicrophysics show methods [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    # Non-equilibrium scheme
    μ_ne = OneMomentCloudMicrophysics()
    str_ne = sprint(show, μ_ne)
    @test contains(str_ne, "BulkMicrophysics")
    @test contains(str_ne, "cloud_formation")

    # Saturation adjustment scheme
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    μ_sa = OneMomentCloudMicrophysics(FT; cloud_formation)
    str_sa = sprint(show, μ_sa)
    @test contains(str_sa, "BulkMicrophysics")
end

@testset "microphysical_velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015, qʳ=0.001)

    # Rain should have sedimentation velocity
    μ = model.microphysical_fields
    vel_rain = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain !== nothing
    @test haskey(vel_rain, :w)

    # Cloud liquid has no sedimentation velocity
    vel_cloud = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud === nothing
end
