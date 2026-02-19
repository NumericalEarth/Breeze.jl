using Breeze
using Breeze.AtmosphereModels: BoundaryMassFluxes, NoBoundaryMassFluxes,
                               initialize_boundary_mass_fluxes,
                               enforce_open_boundary_mass_conservation!
using Oceananigans
using Oceananigans.BoundaryConditions: Open, BoundaryCondition
using Test

const OpenBC = BoundaryCondition{<:Open}

@testset "Open boundaries [$(typeof(default_arch))]" begin

    @testset "NoBoundaryMassFluxes for periodic domain" begin
        grid = RectilinearGrid(default_arch, size=(8, 8, 8), extent=(1, 1, 1),
                               topology=(Periodic, Periodic, Bounded))
        model = AtmosphereModel(grid)
        @test model.boundary_mass_fluxes isa NoBoundaryMassFluxes
    end

    @testset "BoundaryMassFluxes for 2D domain with open x-boundaries" begin
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        Uᵢ = 10.0
        scheme = PerturbationAdvection(outflow_timescale=1.0, inflow_timescale=1.0)
        open_bc = OpenBoundaryCondition(Uᵢ; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc))

        model = AtmosphereModel(grid; boundary_conditions)

        @test model.boundary_mass_fluxes isa BoundaryMassFluxes
        @test model.boundary_mass_fluxes.west !== nothing
        @test model.boundary_mass_fluxes.east !== nothing
        @test model.boundary_mass_fluxes.south === nothing
        @test model.boundary_mass_fluxes.north === nothing
    end

    @testset "BoundaryMassFluxes for domain with open x and y boundaries" begin
        grid = RectilinearGrid(default_arch, size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                               topology=(Bounded, Bounded, Bounded))

        Uᵢ = 10.0
        scheme = PerturbationAdvection(outflow_timescale=1.0, inflow_timescale=1.0)
        open_bc = OpenBoundaryCondition(Uᵢ; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc),
                                 ρv = FieldBoundaryConditions(south=open_bc, north=open_bc))

        model = AtmosphereModel(grid; boundary_conditions)

        @test model.boundary_mass_fluxes isa BoundaryMassFluxes
        @test model.boundary_mass_fluxes.west !== nothing
        @test model.boundary_mass_fluxes.east !== nothing
        @test model.boundary_mass_fluxes.south !== nothing
        @test model.boundary_mass_fluxes.north !== nothing
    end

    @testset "Velocity fields inherit momentum boundary conditions" begin
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        scheme = PerturbationAdvection(outflow_timescale=1.0, inflow_timescale=1.0)
        open_bc = OpenBoundaryCondition(10.0; scheme)
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

    @testset "PerturbationMomentumAdvection with open boundaries" begin
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        # Create a simple density field for PMA
        ρ₀ = CenterField(grid)
        set!(ρ₀, 1.2) # uniform density

        Uᵢ = 10.0
        scheme = PerturbationMomentumAdvection(density=ρ₀, gravity_wave_speed=30.0)
        open_bc = OpenBoundaryCondition(Uᵢ; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc))

        model = AtmosphereModel(grid; boundary_conditions)

        @test model.boundary_mass_fluxes isa BoundaryMassFluxes

        # Check that the model can be constructed and BCs are correctly typed
        ρu_bcs = model.momentum.ρu.boundary_conditions
        @test ρu_bcs.west isa OpenBC
        @test ρu_bcs.east isa OpenBC
    end

    @testset "Mass conservation enforcement" begin
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        scheme = PerturbationAdvection(outflow_timescale=1.0, inflow_timescale=1.0)
        open_bc = OpenBoundaryCondition(10.0; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc))

        model = AtmosphereModel(grid; boundary_conditions)
        set!(model, u=10.0)

        bmf = model.boundary_mass_fluxes
        @test bmf isa BoundaryMassFluxes

        # Enforce mass conservation (should not error)
        enforce_open_boundary_mass_conservation!(model, bmf)
    end

    @testset "Model with open boundaries runs" begin
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        scheme = PerturbationAdvection(outflow_timescale=1.0, inflow_timescale=1.0)
        open_bc = OpenBoundaryCondition(10.0; scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=open_bc, east=open_bc))

        model = AtmosphereModel(grid; boundary_conditions, advection=Centered())
        set!(model, u=10.0)

        simulation = Simulation(model, Δt=0.001, stop_iteration=3)
        run!(simulation)

        @test model.clock.iteration == 3
    end

    @testset "Model with PerturbationMomentumAdvection runs" begin
        grid = RectilinearGrid(default_arch, size=(8, 8), x=(0, 1), z=(0, 1),
                               topology=(Bounded, Flat, Bounded))

        ρ₀ = CenterField(grid)
        set!(ρ₀, 1.2)

        Uᵢ = 10.0
        momentum_scheme = PerturbationMomentumAdvection(density=ρ₀, gravity_wave_speed=30.0)
        ρu_bc = OpenBoundaryCondition(Uᵢ; scheme=momentum_scheme)
        boundary_conditions = (; ρu = FieldBoundaryConditions(west=ρu_bc, east=ρu_bc))

        model = AtmosphereModel(grid; boundary_conditions, advection=Centered())
        set!(model, u=Uᵢ)

        simulation = Simulation(model, Δt=0.001, stop_iteration=3)
        run!(simulation)

        @test model.clock.iteration == 3
    end
end
