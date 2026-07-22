using Breeze
using Breeze.AtmosphereModels: microphysical_velocities, sedimentation_velocity, moisture_phase,
                               total_density, update_sedimentation_velocities!,
                               implicit_advection_velocities
using CloudMicrophysics
using CloudMicrophysics.Microphysics1M: conv_q_lcl_to_q_rai, accretion
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics
using Breeze.Microphysics: ConstantRateCondensateFormation

using Oceananigans.BoundaryConditions: ImpenetrableBoundaryCondition, fill_halo_regions!
using Oceananigans.Fields: ZeroField, ZFaceField
using Oceananigans.Operators: ℑzᵃᵃᶠ

struct MockSurfaceFluxTransportModel{G, D, V, M, A, W}
    grid :: G
    dynamics :: D
    velocities :: V
    microphysical_fields :: M
    advection :: A
    transport_w :: W
end

Breeze.AtmosphereModels.transport_velocities(model::MockSurfaceFluxTransportModel) =
    (; u = ZeroField(), v = ZeroField(), w = model.transport_w)

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

    μ1_vertical = OneMomentCloudMicrophysics(FT;
                                             negative_moisture_correction = Breeze.AtmosphereModels.VerticalBorrowing())
    @test μ1_vertical.negative_moisture_correction isa Breeze.AtmosphereModels.VerticalBorrowing

    # Mixed-phase non-equilibrium
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
    @test :ρqᶜˡ ∉ prog_fields_warm

    # Mixed-phase saturation adjustment
    cloud_formation_mixed = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    μ1_mixed = OneMomentCloudMicrophysics(FT; cloud_formation=cloud_formation_mixed)
    @test μ1_mixed.cloud_formation.equilibrium isa MixedPhaseEquilibrium

    prog_fields_mixed = Breeze.AtmosphereModels.prognostic_field_names(μ1_mixed)
    @test :ρqʳ in prog_fields_mixed
    @test :ρqˢ in prog_fields_mixed
end

@testset "OneMomentCloudMicrophysics non-equilibrium time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    @test haskey(model.microphysical_fields, :ρqᶜˡ)
    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)

    # Single time step (reduced from 6 iterations)
    time_step!(model, 1)
    @test model.clock.time == 1
    @test model.clock.iteration == 1
end

@testset "OneMomentCloudMicrophysics saturation adjustment time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qʳ)

    # Single time step (reduced from 6 iterations)
    time_step!(model, 1)
    @test model.clock.time == 1
end

@testset "OneMomentCloudMicrophysics mixed-phase time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.015)

    @test haskey(model.microphysical_fields, :ρqʳ)
    @test haskey(model.microphysical_fields, :ρqˢ)
    @test haskey(model.microphysical_fields, :qᶜˡ)
    @test haskey(model.microphysical_fields, :qᶜⁱ)

    # Single time step (reduced from 6 iterations)
    time_step!(model, 1)
    @test model.clock.time == 1
end

@testset "OneMomentCloudMicrophysics precipitation rate diagnostic [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 1_000), y=(0, 1_000), z=(0, 1_000))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    # Test non-equilibrium scheme only (saturation adjustment is tested elsewhere)
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015)
    time_step!(model, 1)

    P = precipitation_rate(model, :liquid)
    @test P isa Field
    compute!(P)
    @test isfinite(maximum(P))

    P_ice = precipitation_rate(model, :ice)
    @test P_ice === nothing
end

@testset "NonEquilibriumCloudFormation construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT

    cloud_formation_default = NonEquilibriumCloudFormation(CloudLiquid(FT), nothing)
    @test cloud_formation_default.liquid isa CloudLiquid
    @test cloud_formation_default.ice === nothing
    @test cloud_formation_default.liquid.τ_relax == FT(10.0)

    cloud_formation_mixed = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    @test cloud_formation_mixed.liquid isa CloudLiquid
    @test cloud_formation_mixed.ice isa CloudIce

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

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    ρᵣ = @allowscalar reference_state.density[1, 1, 1]

    qᶜˡ_value = FT(0.001)
    qʳ_value = FT(0.002)
    set!(model; θ=300, qᵗ=0.020, qᶜˡ=qᶜˡ_value, qʳ=qʳ_value)

    @test @allowscalar model.microphysical_fields.ρqᶜˡ[1, 1, 1] ≈ ρᵣ * qᶜˡ_value
    @test @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1] ≈ ρᵣ * qʳ_value
    @test @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1] ≈ qᶜˡ_value
    @test @allowscalar model.microphysical_fields.qʳ[1, 1, 1] ≈ qʳ_value

    time_step!(model, 1)
    @test model.clock.iteration == 1
end

@testset "Surface precipitation flux diagnostic [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_potential_temperature=300)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; ρ = 2, θ = 300, qᵗ = 0.020, qᶜˡ = 0.0001, qʳ = 0.005,
         enforce_mass_conservation = false)

    production = precipitation_rate(model, :liquid)
    compute!(production)

    categories = microphysics.categories
    qᶜˡ = @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ = @allowscalar total_density(model.dynamics)[1, 1, 1]
    expected_production = conv_q_lcl_to_q_rai(categories.rain.acnv1M, qᶜˡ) +
                          accretion(categories.cloud_liquid, categories.rain,
                                    categories.hydrometeor_velocities.blk1m.rain,
                                    categories.collisions, qᶜˡ, qʳ, ρ)
    @test @allowscalar production[1, 1, 1] ≈ expected_production

    spf = surface_precipitation_flux(model)
    @test spf isa Field
    compute!(spf)

    # The surface precipitation flux uses the advection scheme's face reconstruction.
    # For uniform condensate fields with Centered(order=2) advection, each
    # face-reconstructed tracer equals its cell-center value. The density is
    # face-interpolated (ℑz) to match the advection operator.
    wᶜˡ = @allowscalar model.microphysical_fields.wᶜˡ[1, 1, 1]
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qᶜˡ = @allowscalar model.microphysical_fields.qᶜˡ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))
    ρ_reference_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, model.dynamics.reference_state.density)
    expected_flux = -ρ_face * (wᶜˡ * qᶜˡ + wʳ * qʳ)

    @test ρ_face ≈ FT(2)
    @test !isapprox(ρ_face, ρ_reference_face)
    @test @allowscalar spf[1, 1] ≈ expected_flux
    @test @allowscalar spf[1, 1] > 0
end

@testset "Bounds-preserving WENO surface precipitation flux [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(6, 6, 6), extent=(100, 100, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    scalar_advection = (; ρqʳ = WENO(FT; order=5, bounds=(0, 1)))
    model = AtmosphereModel(grid; dynamics, microphysics, scalar_advection)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)
    flux = surface_precipitation_flux(model)
    compute!(flux)

    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))
    @test @allowscalar flux[1, 1] ≈ -ρ_face * wʳ * qʳ
end

@testset "Adaptive implicit sedimentation includes boundary outflow [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(6, 6, 6), extent=(100, 100, 1))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    adaptive_discretization = AdaptiveVerticallyImplicitDiscretization(FT; cfl=0.5)
    scalar_advection = (; ρqʳ = WENO(FT; order=5, time_discretization=adaptive_discretization))
    model = AtmosphereModel(grid; dynamics, microphysics, scalar_advection)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)
    advection = model.advection.ρqʳ
    td = Oceananigans.TimeSteppers.time_discretization(advection)
    Δt = FT(10)
    td.Δt[] = Δt

    velocities = implicit_advection_velocities(model.dynamics, model.velocities, :ρqʳ,
                                                model.microphysics, model.microphysical_fields)
    ρ = total_density(model.dynamics)
    diagonal = @allowscalar Oceananigans.Advection.implicit_advection_diagonal(1, 1, 1, grid,
                                                                               advection, velocities.w,
                                                                               Δt, Center(), Center(), ρ)
    @test diagonal > 0

    flux = surface_precipitation_flux(model)
    compute!(flux)
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, ρ)
    @test @allowscalar flux[1, 1] ≈ -ρ_face * wʳ * qʳ
end

@testset "Surface precipitation flux uses transport velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)

    transport_w = set!(ZFaceField(grid), FT(-1))
    mock_model = MockSurfaceFluxTransportModel(model.grid,
                                               model.dynamics,
                                               model.velocities,
                                               model.microphysical_fields,
                                               model.advection,
                                               transport_w)

    spf = surface_precipitation_flux(mock_model, microphysics)
    compute!(spf)

    wᵗ = @allowscalar transport_w[1, 1, 1]
    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    qʳ = @allowscalar model.microphysical_fields.qʳ[1, 1, 1]
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))

    expected_flux = -ρ_face * (wᵗ + wʳ) * qʳ
    sedimentation_only_flux = -ρ_face * wʳ * qʳ

    @test !isapprox(expected_flux, sedimentation_only_flux)
    @test @allowscalar spf[1, 1] ≈ expected_flux
end

# Consolidated simulation-based tests (reduced simulation times)
@testset "Rain accumulation from autoconversion [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 10
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    set!(model; θ=300, qᵗ=FT(0.050))

    # Reduced simulation time (from 5τ + 30τ = 35τ to just 10τ total)
    τ = microphysics.categories.cloud_liquid.τ_relax
    simulation = Simulation(model; Δt=τ/5, stop_time=10τ, verbose=false)
    run!(simulation)

    # Cloud liquid should have formed
    qᶜˡ_equilibrium = maximum(model.microphysical_fields.qᶜˡ)
    @test qᶜˡ_equilibrium > FT(0.001)

    # Rain should exist somewhere in the domain
    qʳ_max = maximum(model.microphysical_fields.qʳ)
    @test qʳ_max > FT(1e-10)
end

@testset "ImpenetrableBoundaryCondition prevents rain from exiting domain [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics(; precipitation_boundary_condition=ImpenetrableBoundaryCondition())
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    set!(model; θ=300, qᵗ=FT(0.050))

    # Reduced simulation time (from 10τ to 5τ)
    τ = microphysics.categories.cloud_liquid.τ_relax
    simulation = Simulation(model; Δt=τ/10, stop_time=5τ, verbose=false)
    run!(simulation)

    # Terminal velocity should be zero at impenetrable bottom
    wʳ_bottom = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    @test wʳ_bottom == 0
end

@testset "Mixed-phase non-equilibrium time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

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

    μ_ne = OneMomentCloudMicrophysics()
    str_ne = sprint(show, μ_ne)
    @test contains(str_ne, "BulkMicrophysics")
    @test contains(str_ne, "cloud_formation")
end

@testset "sedimentation_velocity, moisture_phase, and microphysical_velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015, qʳ=0.001)

    μ = model.microphysical_fields

    # sedimentation_velocity returns vertical velocity component fields
    w_rain = sedimentation_velocity(microphysics, μ, Val(:ρqʳ))
    @test w_rain !== nothing
    @test w_rain === μ.wʳ

    # WPNE1M has cloud liquid sedimentation
    w_cloud = sedimentation_velocity(microphysics, μ, Val(:ρqᶜˡ))
    @test w_cloud !== nothing
    @test w_cloud === μ.wᶜˡ

    # moisture_phase classification
    @test moisture_phase(microphysics, Val(:ρqʳ)) === Val(:liquid)
    @test moisture_phase(microphysics, Val(:ρqᶜˡ)) === Val(:liquid)

    # microphysical_velocities wraps sedimentation_velocity in a velocity tuple
    vel_rain = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain !== nothing
    @test haskey(vel_rain, :w)

    vel_cloud = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud !== nothing
    @test haskey(vel_cloud, :w)

    # Sedimentation velocity values should be negative downward
    wʳ = @allowscalar μ.wʳ[1, 1, 2]
    @test wʳ <= 0

    # Effective sedimentation velocities exist on model
    bsv = model.sedimentation_velocities
    @test bsv !== nothing
    @test haskey(bsv, :ρqᴸ)
    @test haskey(bsv, :ρqᴵ)

    # Validate effective liquid sedimentation: sign convention and mass-weighted averaging
    time_step!(model, 1)
    qʳ_val  = @allowscalar ℑzᵃᵃᶠ(1, 1, 2, grid, μ.qʳ)
    qᶜˡ_val = @allowscalar ℑzᵃᵃᶠ(1, 1, 2, grid, μ.qᶜˡ)
    wʳ_val  = @allowscalar μ.wʳ[1, 1, 2]
    wᶜˡ_val = @allowscalar μ.wᶜˡ[1, 1, 2]
    wᴸ_val  = @allowscalar bsv.ρqᴸ.w[1, 1, 2]

    @test qʳ_val + qᶜˡ_val > 0
    @test wᴸ_val <= 0
    @test wᴸ_val ≈ (wʳ_val * qʳ_val + wᶜˡ_val * qᶜˡ_val) / (qʳ_val + qᶜˡ_val)
end

@testset "Effective velocity includes diagnosed condensate [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), extent=(100, 100, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=280)
    dynamics = AnelasticDynamics(reference_state)
    cloud_formation = SaturationAdjustment(FT; equilibrium=WarmPhaseEquilibrium())
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=280, qᵗ=0.03, qʳ=0.001)
    μ = model.microphysical_fields
    qᶜˡ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qᶜˡ)
    qʳ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qʳ)
    wʳ = @allowscalar μ.wʳ[1, 1, 1]
    wᴸ = @allowscalar model.sedimentation_velocities.ρqᴸ.w[1, 1, 1]

    @test qᶜˡ > 0
    @test wʳ < 0
    @test wᴸ ≈ wʳ * qʳ / (qᶜˡ + qʳ)
    @test abs(wᴸ) < abs(wʳ)
end

@testset "Effective velocity uses face-collocated humidities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    μ = model.microphysical_fields

    lower_cell = (x, y, z) -> ifelse(z < 50, FT(0.002), FT(0))
    upper_cell = (x, y, z) -> ifelse(z < 50, FT(0), FT(0.004))
    set!(μ.qᶜˡ, lower_cell)
    set!(μ.qʳ, upper_cell)
    set!(μ.qˡ, (x, y, z) -> lower_cell(x, y, z) + upper_cell(x, y, z))
    set!(μ.wᶜˡ, FT(0))
    set!(μ.wʳ, FT(-2))
    fill_halo_regions!((μ.qᶜˡ, μ.qʳ, μ.qˡ, μ.wᶜˡ, μ.wʳ))
    update_sedimentation_velocities!(model.sedimentation_velocities, microphysics, μ)

    qᶜˡ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 2, grid, μ.qᶜˡ)
    qʳ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 2, grid, μ.qʳ)
    wᴸ = @allowscalar model.sedimentation_velocities.ρqᴸ.w[1, 1, 2]
    expected = FT(-2) * qʳ_face / (qᶜˡ_face + qʳ_face)

    @test qᶜˡ_face ≈ FT(0.001)
    @test qʳ_face ≈ FT(0.002)
    @test wᴸ ≈ expected

    # A transient negative stationary constituent must not shrink the denominator
    # below the positive moving mass and produce a super-terminal bulk velocity.
    set!(μ.qᶜˡ, FT(-0.001))
    set!(μ.qʳ, FT(0.002))
    set!(μ.qˡ, FT(0.001))
    fill_halo_regions!((μ.qᶜˡ, μ.qʳ, μ.qˡ))
    update_sedimentation_velocities!(model.sedimentation_velocities, microphysics, μ)

    bounded_wᴸ = @allowscalar model.sedimentation_velocities.ρqᴸ.w[1, 1, 2]
    @test bounded_wᴸ ≈ FT(-2)
end

@testset "Mixed-phase ice velocity and snow surface flux [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=250)
    dynamics = AnelasticDynamics(reference_state)
    cloud_formation = SaturationAdjustment(FT; equilibrium=MixedPhaseEquilibrium(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=250, qᵗ=0.01, qʳ=0.0005, qˢ=0.0005)
    μ = model.microphysical_fields
    qᶜⁱ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qᶜⁱ)
    qʳ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qʳ)
    qˢ = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, μ.qˢ)
    wʳ = @allowscalar μ.wʳ[1, 1, 1]
    wˢ = @allowscalar μ.wˢ[1, 1, 1]
    wᴵ = @allowscalar model.sedimentation_velocities.ρqᴵ.w[1, 1, 1]

    @test qᶜⁱ > 0
    @test wˢ < 0
    @test wᴵ ≈ wˢ * qˢ / (qᶜⁱ + qˢ)

    flux = surface_precipitation_flux(model)
    compute!(flux)
    ρ_face = @allowscalar ℑzᵃᵃᶠ(1, 1, 1, grid, total_density(model.dynamics))
    @test @allowscalar flux[1, 1] ≈ -ρ_face * (wʳ * qʳ + wˢ * qˢ)
end

@testset "Mixed-phase non-equilibrium snow field materialization [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, microphysics)

    # Snow terminal velocity field should exist
    @test haskey(model.microphysical_fields, :wˢ)

    # Snow sedimentation velocity dispatch
    μ = model.microphysical_fields
    vel_snow = microphysical_velocities(microphysics, μ, Val(:ρqˢ))
    @test vel_snow !== nothing
    @test haskey(vel_snow, :w)

    # Other tracers still have correct dispatch
    vel_rain = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain !== nothing

    # Cloud condensate velocity fields should exist
    @test haskey(model.microphysical_fields, :wᶜˡ)
    @test haskey(model.microphysical_fields, :wᶜⁱ)

    # Cloud liquid sedimentation velocity dispatch
    vel_cloud = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud !== nothing
    @test haskey(vel_cloud, :w)

    # Cloud ice sedimentation velocity dispatch
    vel_ice = microphysical_velocities(microphysics, μ, Val(:ρqᶜⁱ))
    @test vel_ice !== nothing
    @test haskey(vel_ice, :w)
end

@testset "MPNE1M snow processes time-stepping [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 10
    grid = RectilinearGrid(default_arch; size=(1, 1, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
                           topology=(Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=260)
    dynamics = AnelasticDynamics(reference_state)

    cloud_formation = NonEquilibriumCloudFormation(CloudLiquid(FT), CloudIce(FT))
    microphysics = OneMomentCloudMicrophysics(FT; cloud_formation)
    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)

    # Cold, supersaturated conditions → cloud ice should form via deposition
    set!(model; θ=260, qᵗ=FT(0.010))

    # Run for a few relaxation timescales
    τ = FT(1) / microphysics.cloud_formation.ice.rate
    simulation = Simulation(model; Δt=τ/5, stop_time=10τ, verbose=false)
    run!(simulation)

    # Cloud ice should have formed from deposition
    qᶜⁱ_max = maximum(model.microphysical_fields.qᶜⁱ)
    @test qᶜⁱ_max > FT(1e-6)

    # Snow should have formed from ice autoconversion
    qˢ_max = maximum(model.microphysical_fields.qˢ)
    @test qˢ_max > FT(0)

    # Model should complete without errors (all tendencies computed)
    @test model.clock.iteration > 0
end
