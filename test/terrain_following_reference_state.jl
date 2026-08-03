include(joinpath(@__DIR__, "setup.jl"))

using Breeze

using CUDA: @allowscalar

using Oceananigans
using Oceananigans.Grids: XDirection, znode
using Oceananigans.Models: BoundaryConditionOperation
using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ
using Breeze.AtmosphereModels: surface_pressure
using Breeze.BoundaryConditions: surface_air_pressure
using Breeze.Thermodynamics: hydrostatic_pressure, dry_air_gas_constant, vapor_gas_constant,
                              potential_temperature_from_temperature, saturation_specific_humidity,
                              surface_density, ExnerReferenceState
using Test

@allowscalar begin
@testset "TerrainFollowing reference states" begin
    @testset "Terrain reference state matches continuous hydrostatic profile" begin
        # The terrain reference state pᵣ(i,j,k) must equal the continuous
        # hydrostatic pressure evaluated at the local physical height z(i,j,k).
        # A bug that initializes every column from sea-level pressure creates
        # O(ρgh) errors over terrain.
        Nx, Nz = 16, 8
        Lx, Lz = 100000.0, 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 1000.0
        a = 10000.0
        h(x) = h₀ * exp(-x^2 / a^2)
        materialize_terrain!(grid, h)

        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature=θ₀)
        model = AtmosphereModel(grid; dynamics)
        constants = model.thermodynamic_constants

        pᵣ = model.dynamics.reference_state.pressure

        # At each grid point, pᵣ should match the continuous profile
        # to within the discretization error of the Exner integration (O(Δz²))
        for i in 1:Nx, k in 1:Nz
            z = znode(i, 1, k, grid, Center(), Center(), Center())
            p_exact = hydrostatic_pressure(z, p₀, θ₀, pˢᵗ, constants)
            # Discrete Exner integration has O(Δz²) error; with Δz ≈ 1250 m
            # the accumulated error at the top is ~0.5%, so use 1% tolerance
            @test pᵣ[i, 1, k] ≈ p_exact rtol=1e-2
        end

        # Critical check: at a given k-level, pᵣ must NOT be constant across
        # columns (it should vary because physical heights differ). But at the
        # SAME physical height, values from different columns should agree closely.
        # Compare the flat column (i at domain edge) vs the mountain-top column.
        i_flat = 1    # far from mountain
        i_peak = Nx÷2 # near mountain peak
        z_flat_1 = znode(i_flat, 1, 1, grid, Center(), Center(), Center())
        z_peak_1 = znode(i_peak, 1, 1, grid, Center(), Center(), Center())

        # Physical heights differ, so pᵣ at k=1 should differ
        @test z_peak_1 > z_flat_1 + 100  # mountain is at least 100 m higher
        @test pᵣ[i_peak, 1, 1] < pᵣ[i_flat, 1, 1]  # higher altitude → lower pressure
    end

    @testset "Terrain reference state with θ(z) profile (Function dispatch)" begin
        # Same test but with a non-constant potential temperature profile,
        # exercising the numerically_integrated_hydrostatic_pressure path.
        Nx, Nz = 16, 16
        Lx, Lz = 100000.0, 10000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h₀ = 1000.0
        a = 10000.0
        h(x) = h₀ * exp(-x^2 / a^2)
        materialize_terrain!(grid, h)

        g_val = 9.80665
        N² = 1e-4
        θ₀ = 300.0
        p₀ = 101325.0
        pˢᵗ = 1e5
        θ_of_z(z) = θ₀ * exp(N² * z / g_val)

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature=θ_of_z)
        model = AtmosphereModel(grid; dynamics)
        constants = model.thermodynamic_constants

        pᵣ = model.dynamics.reference_state.pressure

        # At each grid point, pᵣ should match the continuous profile
        for i in 1:Nx, k in 1:Nz
            z = znode(i, 1, k, grid, Center(), Center(), Center())
            p_exact = hydrostatic_pressure(z, p₀, θ_of_z, pˢᵗ, constants)
            # Finer grid (Nz=16) so tighter tolerance than Nz=8 test
            @test pᵣ[i, 1, k] ≈ p_exact rtol=5e-3
        end

        # Mountain-top column should have lower pᵣ at k=1 than flat column
        i_flat = 1
        i_peak = Nx÷2
        @test pᵣ[i_peak, 1, 1] < pᵣ[i_flat, 1, 1]
    end

    @testset "Constant moist terrain reference state satisfies discrete hydrostatic balance" begin
        Nx, Nz = 8, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 200 * exp(-x^2 / 2000^2)
        materialize_terrain!(grid, h)

        θ_reference = 300.0
        qᵛ_reference = 0.012

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature = θ_reference,
                                        reference_vapor_mass_fraction = qᵛ_reference)
        model = AtmosphereModel(grid; dynamics)

        p_ref = model.dynamics.reference_state.pressure
        ρ_ref = model.dynamics.reference_state.density
        constants = model.thermodynamic_constants
        g = constants.gravitational_acceleration
        p₀ = dynamics.base_pressure
        pˢᵗ = dynamics.standard_pressure
        Rᵈ = dry_air_gas_constant(constants)
        Rᵛ = vapor_gas_constant(constants)
        cᵖᵈ = constants.dry_air.heat_capacity
        cᵖᵛ = constants.vapor.heat_capacity

        @test p_ref !== nothing
        @test ρ_ref !== nothing

        qᵛ_surface = qᵛ_reference
        qᵈ_surface = 1 - qᵛ_surface
        Rᵐ_surface = qᵈ_surface * Rᵈ + qᵛ_surface * Rᵛ
        cᵖᵐ_surface = qᵈ_surface * cᵖᵈ + qᵛ_surface * cᵖᵛ
        κ_surface = Rᵐ_surface / cᵖᵐ_surface
        T_surface₀ = θ_reference * (p₀ / pˢᵗ)^κ_surface

        for i in 1:Nx
            z_surface = znode(i, 1, 1, grid, Center(), Center(), Face())
            p_surface = p₀ * (1 - g * z_surface / (cᵖᵐ_surface * T_surface₀))^(cᵖᵐ_surface / Rᵐ_surface)
            T_surface = θ_reference * (p_surface / pˢᵗ)^κ_surface
            ρ_surface = p_surface / (Rᵐ_surface * T_surface)

            # Surface (bottom face) to first cell center spans half a cell.
            hydrostatic_residual = (p_ref[i, 1, 1] - p_surface) / (Δzᶜᶜᶜ(i, 1, 1, grid) / 2) +
                                   g * (ρ_ref[i, 1, 1] + ρ_surface) / 2
            @test abs(hydrostatic_residual) <= 1e-6
        end

        for i in 1:Nx, k in 2:Nz
            hydrostatic_residual = (p_ref[i, 1, k] - p_ref[i, 1, k - 1]) / Δzᶜᶜᶠ(i, 1, k, grid) +
                                   g * (ρ_ref[i, 1, k] + ρ_ref[i, 1, k - 1]) / 2
            @test abs(hydrostatic_residual) <= 1e-8
        end

        i_flat = 1
        i_peak = Nx ÷ 2
        @test p_ref[i_peak, 1, 1] < p_ref[i_flat, 1, 1]
    end

    @testset "Variable moist terrain reference state satisfies interior discrete hydrostatic balance" begin
        Nx, Nz = 8, 8
        Lx, Lz = 10000.0, 5000.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))

        h(x) = 200 * exp(-x^2 / 2000^2)
        materialize_terrain!(grid, h)

        θ_reference(z) = 300.0 + 0.01 * z
        qᵛ_reference(z) = 0.012 * exp(-z / 1000)

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        reference_potential_temperature = θ_reference,
                                        reference_vapor_mass_fraction = qᵛ_reference)
        model = AtmosphereModel(grid; dynamics)

        p_ref = model.dynamics.reference_state.pressure
        ρ_ref = model.dynamics.reference_state.density
        g = model.thermodynamic_constants.gravitational_acceleration

        @test p_ref !== nothing
        @test ρ_ref !== nothing

        for i in 1:Nx, k in 2:Nz
            hydrostatic_residual = (p_ref[i, 1, k] - p_ref[i, 1, k - 1]) / Δzᶜᶜᶠ(i, 1, k, grid) +
                                   g * (ρ_ref[i, 1, k] + ρ_ref[i, 1, k - 1]) / 2
            @test abs(hydrostatic_residual) <= 1e-8
        end

        i_flat = 1
        i_peak = Nx ÷ 2
        @test p_ref[i_peak, 1, 1] < p_ref[i_flat, 1, 1]
    end

    @testset "reference_state switch: default builds a reference, nothing disables it" begin
        Nx, Nz = 8, 6
        Lx, Lz = 10000.0, 5000.0
        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1)); formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz),
                               x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> 200 * exp(-x^2 / 2000^2))

        # Default (`reference_state = :auto`): a single 3D standard-atmosphere ExnerReferenceState.
        default_model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping()))
        @test default_model.dynamics.reference_state isa ExnerReferenceState
        @test size(default_model.dynamics.reference_state.pressure) == (Nx, 1, Nz)  # 3D on terrain
        @test all(isfinite, Array(interior(default_model.dynamics.reference_state.pressure)))
        @test all(ρ -> ρ > 0, Array(interior(default_model.dynamics.reference_state.density)))

        # Disabled (`reference_state = nothing`): no reference → full-pressure PGF/buoyancy via ::Nothing.
        off_model = AtmosphereModel(grid; dynamics = CompressibleDynamics(ExplicitTimeStepping(); reference_state=nothing))
        @test off_model.dynamics.reference_state === nothing

        # Disabling and supplying an explicit reference profile are mutually exclusive.
        @test_throws ArgumentError CompressibleDynamics(ExplicitTimeStepping();
                                                        reference_state=nothing,
                                                        reference_potential_temperature=300)

        # Invalid switches must not silently enable the automatic reference.
        @test_throws ArgumentError CompressibleDynamics(ExplicitTimeStepping(); reference_state=false)
        @test_throws ArgumentError CompressibleDynamics(ExplicitTimeStepping(); reference_state=:typo)
    end

    @testset "Cold start agrees with its own reference over terrain" begin
        # `HydrostaticallyBalancedDensity` integrates each column upward from the pressure at that
        # column's bottom face. On terrain that face is the terrain surface, so anchoring every
        # column at the single z = 0 datum instead leaves the initial state with essentially no
        # surface pressure gradient across the hill while the reference built from the same scalar
        # carries the full O(ρgh). The perturbation-form PGF differences the two, so the cold start
        # would begin with a ~20 kPa spurious perturbation at the summit.
        Nx, Nz = 12, 20
        Lx, Lz = 120000.0, 20000.0
        h₀, a = 2000.0, 20000.0
        θ₀, p₀ = 288.0, 101325.0

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length=Nz+1));
                                                        formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), x=(-Lx/2, Lx/2), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> h₀ * exp(-(x / a)^2))

        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        base_pressure=p₀, reference_potential_temperature=θ₀)
        model = AtmosphereModel(grid; dynamics)
        set!(model; θˡⁱ=θ₀, qᵗ=0, ρ=HydrostaticallyBalancedDensity(), compute_reference_state=true)

        p  = Array(interior(model.dynamics.pressure))[:, 1, 1]
        pᵣ = Array(interior(model.dynamics.reference_state.pressure))[:, 1, 1]

        # The reference carries a large horizontal pressure range across the hill, and the state
        # built at the same θ must reproduce it rather than flattening it.
        reference_range = maximum(pᵣ) - minimum(pᵣ)
        @test reference_range > 10000
        @test maximum(p) - minimum(p) ≈ reference_range rtol=1e-3
        @test maximum(abs, p .- pᵣ) < 1e-3 * reference_range

        # The per-column anchor is retained on the reference state, decreasing with terrain height.
        pˢ = Array(interior(model.dynamics.reference_state.surface_pressure))[:, 1, 1]
        @test maximum(pˢ) ≈ p₀ rtol=1e-3           # the lowest column sits near z = 0
        @test minimum(pˢ) < p₀ - 10000             # the summit column is ~2 km up

        # `compute_hydrostatic_pressure!` integrates from the same bottom face, so it must carry
        # the same terrain pressure gradient rather than starting every column at the datum.
        ph = CenterField(grid)
        compute_hydrostatic_pressure!(ph, model)
        ph₁ = Array(interior(ph))[:, 1, 1]
        @test maximum(ph₁) - minimum(ph₁) ≈ reference_range rtol=1e-2
    end

    @testset "Bulk fluxes diagnose live pressure at the terrain surface" begin
        Nx, Nz = 4, 12
        z_plateau = 1921.0
        p₀ = 100798.0
        θ₀ = 288.0
        T₀ = 300.0
        U = 5.0
        Cᴰ = 1e-3

        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, 10000, length=Nz+1));
                                                        formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), x=(-10000, 10000), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> z_plateau)

        drag = Breeze.BulkDrag(coefficient=Cᴰ, surface_temperature=T₀)
        ρu_bcs = FieldBoundaryConditions(bottom=drag)
        sensible_heat = Breeze.BulkSensibleHeatFlux(coefficient=Cᴰ, surface_temperature=T₀)
        ρθ_bcs = FieldBoundaryConditions(bottom=sensible_heat)
        vapor = Breeze.BulkVaporFlux(coefficient=Cᴰ, surface_temperature=T₀)
        ρqᵛ_bcs = FieldBoundaryConditions(bottom=vapor)
        dynamics = CompressibleDynamics(ExplicitTimeStepping();
                                        base_pressure=p₀, reference_potential_temperature=θ₀)
        boundary_conditions = (; ρu=ρu_bcs, ρθ=ρθ_bcs, ρqᵛ=ρqᵛ_bcs)
        model = AtmosphereModel(grid; dynamics, boundary_conditions)
        set!(model; θ=θ₀, qᵗ=0, ρ=HydrostaticallyBalancedDensity(), u=U,
             enforce_mass_conservation=false)
        Oceananigans.initialize!(model)

        constants = model.thermodynamic_constants
        model_fields = Oceananigans.fields(model)
        pˢ = surface_air_pressure(1, 1, grid, model_fields, constants, XDirection())
        pˢ_center = surface_air_pressure(1, 1, grid, model_fields, constants)
        pˢ_reference = model.dynamics.reference_state.surface_pressure[1, 1, 1]
        @test pˢ ≈ pˢ_reference rtol=5e-3
        @test pˢ_center ≈ pˢ rtol=1e-12
        @test pˢ < p₀ - 10000

        h = Breeze.BoundaryConditions.evaluation_height(1, 1, grid, nothing)
        z¹ = znode(1, 1, 1, grid, Center(), Center(), Center())
        zˢ = znode(1, 1, 1, grid, Center(), Center(), Face())
        @test h ≈ z¹ - zˢ

        bc_condition = Oceananigans.boundary_conditions(model.momentum.ρu).bottom.condition
        @test !hasproperty(bc_condition, :base_pressure)

        Jᵘ = Field(BoundaryConditionOperation(model.momentum.ρu, :bottom, model))
        compute!(Jᵘ)
        ρˢ = surface_density(pˢ, T₀, constants)
        Jᵘ_expected = -ρˢ * Cᴰ * U^2
        @test all(isapprox.(Array(interior(Jᵘ)), Jᵘ_expected; rtol=1e-6))

        Jᶿ = Field(BoundaryConditionOperation(model.formulation.potential_temperature_density,
                                                  :bottom, model))
        compute!(Jᶿ)
        pˢᵗ = model.dynamics.standard_pressure
        θˢ = potential_temperature_from_temperature(T₀, pˢ_center, pˢᵗ, constants)
        Jᶿ_expected = -ρˢ * Cᴰ * U * (θ₀ - θˢ)
        @test all(isapprox.(Array(interior(Jᶿ)), Jᶿ_expected; rtol=1e-6))

        Jᵛ = Field(BoundaryConditionOperation(model.moisture_density, :bottom, model))
        compute!(Jᵛ)
        qᵛˢ = saturation_specific_humidity(T₀, ρˢ, constants, PlanarLiquidSurface())
        Jᵛ_expected = ρˢ * Cᴰ * U * qᵛˢ
        @test all(isapprox.(Array(interior(Jᵛ)), Jᵛ_expected; rtol=1e-6))

        # The boundary condition must follow the evolving diagnostic pressure rather than
        # retaining the reference pressure it saw during model construction.
        set!(model; θ=θ₀, qᵗ=0, ρ=0.8, u=U, enforce_mass_conservation=false)
        updated_fields = Oceananigans.fields(model)
        pˢ_updated = surface_air_pressure(1, 1, grid, updated_fields, constants, XDirection())
        @test abs(pˢ_updated - pˢ_reference) > 1000

        compute!(Jᵘ)
        ρˢ_updated = surface_density(pˢ_updated, T₀, constants)
        Jᵘ_updated = -ρˢ_updated * Cᴰ * U^2
        @test all(isapprox.(Array(interior(Jᵘ)), Jᵘ_updated; rtol=1e-6))
    end

    @testset "Hydrostatic columns without a reference use the z = 0 datum" begin
        Nx, Nz = 8, 10
        p₀ = 101325.0
        θ₀ = 288.0
        z_faces = TerrainFollowingVerticalDiscretization(collect(range(0, 10000, length=Nz+1));
                                                        formulation = LinearDecay())
        grid = RectilinearGrid(default_arch; size=(Nx, Nz), x=(-40000, 40000), z=z_faces,
                               topology=(Periodic, Flat, Bounded))
        materialize_terrain!(grid, x -> 2000 * exp(-(x / 15000)^2))

        dynamics = CompressibleDynamics(ExplicitTimeStepping(); base_pressure=p₀, reference_state=nothing)
        model = AtmosphereModel(grid; dynamics)
        @test_throws ArgumentError surface_pressure(model.dynamics)

        set!(model; θ=θ₀, qᵗ=0, ρ=HydrostaticallyBalancedDensity(),
             enforce_mass_conservation=false)
        p₁ = Array(interior(model.dynamics.pressure))[:, 1, 1]
        @test maximum(p₁) - minimum(p₁) > 10000

        ph = CenterField(grid)
        compute_hydrostatic_pressure!(ph, model)
        ph₁ = Array(interior(ph))[:, 1, 1]
        @test maximum(ph₁) - minimum(ph₁) ≈ maximum(p₁) - minimum(p₁) rtol=2e-2
    end

end
end
