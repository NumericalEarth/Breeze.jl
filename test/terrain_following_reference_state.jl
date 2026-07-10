using Breeze
using Oceananigans.Architectures: CPU

# Run under `default_arch` when the test runner provides it (which routes to
# GPU when CUDA is functional, matching project convention); otherwise fall
# back to CPU() so this file can also be included directly with
# `julia --project=. test/<this file>.jl` during development.
@isdefined(default_arch) || (default_arch = CPU())

using CUDA: @allowscalar

using Oceananigans
using Oceananigans.Grids: znode
using Oceananigans.Operators: Δzᶜᶜᶠ, Δzᶜᶜᶜ
using Breeze.Thermodynamics: hydrostatic_pressure, dry_air_gas_constant, vapor_gas_constant
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

        pᵣ = model.dynamics.terrain_reference_pressure

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

        pᵣ = model.dynamics.terrain_reference_pressure

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
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
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

        p_ref = model.dynamics.terrain_reference_pressure
        ρ_ref = model.dynamics.terrain_reference_density
        constants = model.thermodynamic_constants
        g = constants.gravitational_acceleration
        p₀ = dynamics.surface_pressure
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
        grid = RectilinearGrid(CPU(); size=(Nx, Nz),
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

        p_ref = model.dynamics.terrain_reference_pressure
        ρ_ref = model.dynamics.terrain_reference_density
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

end
end
