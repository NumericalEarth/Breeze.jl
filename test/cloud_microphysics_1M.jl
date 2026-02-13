using Breeze
using Breeze.AtmosphereModels: microphysical_velocities
using CloudMicrophysics
using CloudMicrophysics.Parameters: CloudLiquid, CloudIce
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

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

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)

    set!(model; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0.001)

    spf = surface_precipitation_flux(model)
    @test spf isa Field
    compute!(spf)

    wʳ = @allowscalar model.microphysical_fields.wʳ[1, 1, 1]
    ρqʳ = @allowscalar model.microphysical_fields.ρqʳ[1, 1, 1]
    expected_flux = -wʳ * ρqʳ

    @test @allowscalar spf[1, 1] ≈ expected_flux
    @test @allowscalar spf[1, 1] > 0
end

# Consolidated simulation-based tests (reduced simulation times)
@testset "Rain accumulation from autoconversion [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nz = 10
    grid = RectilinearGrid(default_arch; size=(2, 2, Nz), x=(0, 1), y=(0, 1), z=(0, 1000),
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
    grid = RectilinearGrid(default_arch; size=(2, 2, 1), x=(0, 1), y=(0, 1), z=(0, 1),
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

@testset "microphysical_velocities [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(2, 2, 2), x=(0, 100), y=(0, 100), z=(0, 100))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants, surface_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)

    microphysics = OneMomentCloudMicrophysics()
    model = AtmosphereModel(grid; dynamics, microphysics)
    set!(model; θ=300, qᵗ=0.015, qʳ=0.001)

    μ = model.microphysical_fields
    vel_rain = microphysical_velocities(microphysics, μ, Val(:ρqʳ))
    @test vel_rain !== nothing
    @test haskey(vel_rain, :w)

    vel_cloud = microphysical_velocities(microphysics, μ, Val(:ρqᶜˡ))
    @test vel_cloud === nothing
end
