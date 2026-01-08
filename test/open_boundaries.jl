using Breeze
using Breeze.AtmosphereModels: BoundaryMassFluxes, NoBoundaryMassFluxes, 
                               initialize_boundary_mass_fluxes, 
                               enforce_open_boundary_mass_conservation!
using Oceananigans
using Oceananigans.Advection: Centered
using Oceananigans.BoundaryConditions: Open, OpenBoundaryCondition, BoundaryCondition, PerturbationAdvection, FieldBoundaryConditions
using Test

# Type alias for open boundary conditions
const OpenBC = BoundaryCondition{<:Open}

@testset "Open boundaries" begin
    @testset "NoBoundaryMassFluxes for periodic domain" begin
        grid = RectilinearGrid(default_arch, size=(8, 8, 8), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Bounded))
        model = AtmosphereModel(grid)
        @test model.boundary_mass_fluxes isa NoBoundaryMassFluxes
    end

    # Use 2D grids (Flat in y) like the mountain_wave example
    @testset "BoundaryMassFluxes for 2D domain with open x-boundaries" begin
        FT = Oceananigans.defaults.FloatType
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        # Use a constant scalar value for simplicity in tests
        # (For physical accuracy, use a function-based BC in real applications)
        ρ₀ = FT(1.2) # representative surface density
        Uᵢ = FT(10)

        scheme = PerturbationAdvection(outflow_timescale=FT(1), inflow_timescale=FT(1))
        open_bc = OpenBoundaryCondition(ρ₀ * Uᵢ; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc))

        model = AtmosphereModel(grid; boundary_conditions)
        
        @test model.boundary_mass_fluxes isa BoundaryMassFluxes
        @test model.boundary_mass_fluxes.west !== nothing
        @test model.boundary_mass_fluxes.east !== nothing
        @test model.boundary_mass_fluxes.south === nothing
        @test model.boundary_mass_fluxes.north === nothing
    end

    @testset "Velocity fields inherit momentum boundary conditions" begin
        FT = Oceananigans.defaults.FloatType
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        ρ₀ = FT(1.2)
        Uᵢ = FT(10)

        scheme = PerturbationAdvection(outflow_timescale=FT(1), inflow_timescale=FT(1))
        open_bc = OpenBoundaryCondition(ρ₀ * Uᵢ; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc))

        model = AtmosphereModel(grid; boundary_conditions)

        # Check that velocity fields inherited the open boundary conditions
        u_bcs = model.velocities.u.boundary_conditions
        ρu_bcs = model.momentum.ρu.boundary_conditions

        @test u_bcs.west isa OpenBC
        @test u_bcs.east isa OpenBC
        @test typeof(ρu_bcs.west) == typeof(u_bcs.west)
        @test typeof(ρu_bcs.east) == typeof(u_bcs.east)
    end

    @testset "Model with open boundaries runs and enforces mass conservation" begin
        FT = Oceananigans.defaults.FloatType
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        ρ₀ = FT(1.2)
        Uᵢ = FT(10)

        scheme = PerturbationAdvection(outflow_timescale=FT(1), inflow_timescale=FT(1))
        open_bc = OpenBoundaryCondition(ρ₀ * Uᵢ; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc))

        model = AtmosphereModel(grid; boundary_conditions, advection=Centered())
        set!(model, u=Uᵢ)

        # Test that boundary mass fluxes container was created
        bmf = model.boundary_mass_fluxes
        @test bmf isa BoundaryMassFluxes

        # Test mass conservation enforcement (called internally during time-stepping)
        enforce_open_boundary_mass_conservation!(model, bmf)

        # Run a few time steps (which also tests the full integration)
        simulation = Simulation(model, Δt=0.001, stop_iteration=10)
        run!(simulation)

        @test model.clock.iteration == 10
    end
end
