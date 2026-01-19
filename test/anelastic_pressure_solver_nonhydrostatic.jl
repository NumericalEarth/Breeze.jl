using Test
using Breeze
using Oceananigans

@testset "Pressure solver matches NonhydrostaticModel with ρᵣ == 1 [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    Nx = Ny = Nz = 32
    z = 0:(1/Nz):1
    grid = RectilinearGrid(default_arch; size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z)
    constants = ThermodynamicConstants(FT)
    reference_state = ReferenceState(grid, constants)

    dynamics = AnelasticDynamics(reference_state)
    parent(dynamics.reference_state.density) .= 1

    anelastic = AtmosphereModel(grid; thermodynamic_constants=constants, dynamics)
    boussinesq = NonhydrostaticModel(; grid)

    uᵢ = rand(size(grid)...)
    vᵢ = rand(size(grid)...)
    wᵢ = rand(size(grid)...)

    set!(anelastic, ρu=uᵢ, ρv=vᵢ, ρw=wᵢ)
    set!(boussinesq, u=uᵢ, v=vᵢ, w=wᵢ)

    ρu = anelastic.momentum.ρu
    ρv = anelastic.momentum.ρv
    ρw = anelastic.momentum.ρw
    δᵃ = Field(∂x(ρu) + ∂y(ρv) + ∂z(ρw))

    u = boussinesq.velocities.u
    v = boussinesq.velocities.v
    w = boussinesq.velocities.w
    δᵇ = Field(∂x(u) + ∂y(v) + ∂z(w))

    boussinesq_solver = boussinesq.pressure_solver
    anelastic_solver = anelastic.pressure_solver
    @test anelastic_solver.batched_tridiagonal_solver.a == boussinesq_solver.batched_tridiagonal_solver.a
    @test anelastic_solver.batched_tridiagonal_solver.b == boussinesq_solver.batched_tridiagonal_solver.b
    @test anelastic_solver.batched_tridiagonal_solver.c == boussinesq_solver.batched_tridiagonal_solver.c
    @test anelastic_solver.source_term == boussinesq_solver.source_term

    @test maximum(abs, δᵃ) < prod(size(grid)) * eps(FT)
    @test maximum(abs, δᵇ) < prod(size(grid)) * eps(FT)
    # When ρᵣ == 1, the kinematic pressure p'/ρᵣ equals the Boussinesq non-hydrostatic pressure
    @test anelastic.dynamics.pressure_anomaly == boussinesq.pressures.pNHS
end
