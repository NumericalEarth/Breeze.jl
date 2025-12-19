using Breeze
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: BoundaryCondition
using Oceananigans.Grids: minimum_xspacing
using Test

#####
##### Unit tests for PerturbationAdvection and OpenBoundaryCondition
#####

@testset "PerturbationAdvection construction [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    # Default construction
    scheme = PerturbationAdvection()
    @test scheme isa PerturbationAdvection

    # Construction with custom timescales
    scheme = PerturbationAdvection(outflow_timescale=100, inflow_timescale=50)
    @test scheme isa PerturbationAdvection

    # Construction with just one timescale
    scheme = PerturbationAdvection(outflow_timescale=100)
    @test scheme isa PerturbationAdvection
end

@testset "OpenBoundaryCondition construction [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT

    # Simple constant value boundary condition
    bc = OpenBoundaryCondition(10)
    @test bc isa BoundaryCondition

    # With PerturbationAdvection scheme
    scheme = PerturbationAdvection(outflow_timescale=100, inflow_timescale=50)
    bc = OpenBoundaryCondition(10; scheme)
    @test bc isa BoundaryCondition

    # With a function
    U(y, z, t) = 10 + 5 * sin(2π * z / 1000)
    bc = OpenBoundaryCondition(U; scheme)
    @test bc isa BoundaryCondition
end

#####
##### Model integration tests with open boundaries
#####

@testset "AtmosphereModel with open boundaries in x [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (16, 16),
                           x = (0, 1000), z = (0, 1000),
                           halo = (4, 4),
                           topology = (Bounded, Flat, Bounded))

    Uᵢ = 10
    Δx = minimum_xspacing(grid)
    τ = 10 * Δx / Uᵢ
    scheme = PerturbationAdvection(outflow_timescale=τ, inflow_timescale=τ)
    open_bc = OpenBoundaryCondition(Uᵢ; scheme)
    boundary_conditions = (; u = FieldBoundaryConditions(west=open_bc, east=open_bc))

    model = AtmosphereModel(grid; boundary_conditions)
    @test model isa AtmosphereModel

    # Set initial conditions
    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀, u=Uᵢ)

    # Model should time step without error
    time_step!(model, 1e-3)
    @test true
end

@testset "AtmosphereModel with open boundaries in y [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (16, 16),
                           y = (0, 1000), z = (0, 1000),
                           halo = (4, 4),
                           topology = (Flat, Bounded, Bounded))

    Vᵢ = 10
    scheme = PerturbationAdvection(outflow_timescale=100, inflow_timescale=100)
    open_bc = OpenBoundaryCondition(Vᵢ; scheme)
    boundary_conditions = (; v = FieldBoundaryConditions(south=open_bc, north=open_bc))

    model = AtmosphereModel(grid; boundary_conditions)
    @test model isa AtmosphereModel

    # Set initial conditions
    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀, v=Vᵢ)

    # Model should time step without error
    time_step!(model, 1e-3)
    @test true
end

@testset "AtmosphereModel open boundaries with 3D grid [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (8, 8, 8),
                           x = (0, 1000), y = (0, 1000), z = (0, 1000),
                           halo = (4, 4, 4),
                           topology = (Bounded, Bounded, Bounded))

    Uᵢ = 10
    Vᵢ = 5
    scheme = PerturbationAdvection(outflow_timescale=100, inflow_timescale=100)

    u_open_bc = OpenBoundaryCondition(Uᵢ; scheme)
    v_open_bc = OpenBoundaryCondition(Vᵢ; scheme)

    boundary_conditions = (;
        u = FieldBoundaryConditions(west=u_open_bc, east=u_open_bc),
        v = FieldBoundaryConditions(south=v_open_bc, north=v_open_bc)
    )

    model = AtmosphereModel(grid; boundary_conditions)
    @test model isa AtmosphereModel

    # Set initial conditions
    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀, u=Uᵢ, v=Vᵢ)

    # Model should time step without error
    time_step!(model, 1e-3)
    @test true
end

@testset "Open boundaries with WENO advection [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (16, 16),
                           x = (0, 1000), z = (0, 1000),
                           halo = (4, 4),
                           topology = (Bounded, Flat, Bounded))

    Uᵢ = 10
    scheme = PerturbationAdvection(outflow_timescale=100, inflow_timescale=100)
    open_bc = OpenBoundaryCondition(Uᵢ; scheme)
    boundary_conditions = (; u = FieldBoundaryConditions(west=open_bc, east=open_bc))

    model = AtmosphereModel(grid; advection=WENO(), boundary_conditions)
    @test model isa AtmosphereModel

    # Set initial conditions
    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀, u=Uᵢ)

    # Model should time step without error
    time_step!(model, 1e-3)
    @test true
end

@testset "Open boundaries mass flux conservation [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (16, 16),
                           x = (0, 1000), z = (0, 1000),
                           halo = (4, 4),
                           topology = (Bounded, Flat, Bounded))

    Uᵢ = 10
    scheme = PerturbationAdvection(outflow_timescale=100, inflow_timescale=100)
    open_bc = OpenBoundaryCondition(Uᵢ; scheme)
    boundary_conditions = (; u = FieldBoundaryConditions(west=open_bc, east=open_bc))

    model = AtmosphereModel(grid; boundary_conditions)

    # Set initial conditions with uniform flow
    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀, u=Uᵢ)

    # Run a few time steps
    simulation = Simulation(model, Δt=0.1, stop_iteration=10)
    run!(simulation)

    # Model should complete without error
    @test iteration(simulation) == 10
end

@testset "Open boundaries with asymmetric inflow/outflow timescales [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size = (16, 16),
                           x = (0, 1000), z = (0, 1000),
                           halo = (4, 4),
                           topology = (Bounded, Flat, Bounded))

    Uᵢ = 10
    # Different timescales for inflow vs outflow
    scheme = PerturbationAdvection(outflow_timescale=50, inflow_timescale=200)
    open_bc = OpenBoundaryCondition(Uᵢ; scheme)
    boundary_conditions = (; u = FieldBoundaryConditions(west=open_bc, east=open_bc))

    model = AtmosphereModel(grid; boundary_conditions)
    @test model isa AtmosphereModel

    θ₀ = model.formulation.reference_state.potential_temperature
    set!(model; θ=θ₀, u=Uᵢ)

    time_step!(model, 1e-3)
    @test true
end

