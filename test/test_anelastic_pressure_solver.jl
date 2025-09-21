include(joinpath(@__DIR__, "runtests_setup.jl"))

using Test
using Breeze
using Oceananigans
using Oceananigans.Fields: fill_halo_regions!

for FT in (Float32, Float64)
    @testset "Pressure solver matches NonhydrostaticModel with ρᵣ == 1 [$FT]" begin
        @info "Testing the anelastic pressure solver against NonhydrostaticModel [$FT]..."
        Nx = Ny = Nz = 32
        z = 0:(1/Nz):1
        grid = RectilinearGrid(FT; size=(Nx, Ny, Nz), x=(0, 1), y=(0, 1), z)
        thermo = AtmosphereThermodynamics(FT)
        constants = ReferenceStateConstants(FT; base_pressure=101325, potential_temperature=288)

        formulation = AnelasticFormulation(grid, constants, thermo)
        parent(formulation.reference_density) .= 1

        anelastic = AtmosphereModel(grid; thermodynamics=thermo, formulation)
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
        @test anelastic.nonhydrostatic_pressure == boussinesq.pressures.pNHS
    end

    @testset "Anelastic pressure solver recovers analytic solution [$FT]" begin
        @info "Test that anelastic pressure solver recovers analytic solution [$FT]..."
        grid = RectilinearGrid(FT; size=48, z=(0, 1), topology=(Flat, Flat, Bounded))
        thermo = AtmosphereThermodynamics(FT)
        constants = ReferenceStateConstants(FT; base_pressure=101325.0, potential_temperature=288.0)
        formulation = AnelasticFormulation(grid, constants, thermo)

        #=
        ρᵣ = 2 + cos(π z / 2)
        ∂z ρᵣ ∂z ϕ = ?

        ϕ = cos(π z)
        ⟹ ∂z ϕ = -π sin(π z)
        ⟹ (2 + cos(π z)) ∂z ϕ = -π (2 sin(π z) + cos(π z) sin(π z))
        ⟹ ∂z (1 + cos(π z / 2)) ∂z ϕ = -π² (2 cos(π z) + 2 cos²(π z) - 1)

        ϕ = z² / 2 - z³ / 3 = z² (1/2 - z/3)
        ∂z ϕ = z (1 - z) = z - z²
        ∂z² ϕ = 1 - 2z
        ⟹ z ∂z ϕ = z² - z³
        ⟹ ∂z (z ∂z ϕ) = 2 z - 3 z²

        R = ∂z ρw = 2 z - 3 z²
        ⟹ ρw = z² - z³
        =#

        set!(formulation.reference_density, z -> z)
        fill_halo_regions!(formulation.reference_density)
        model = AtmosphereModel(grid; thermodynamics=thermo, formulation)
        set!(model, ρw = z -> z^2 - z^3)

        ϕ_exact = CenterField(grid)
        set!(ϕ_exact, z -> z^2 / 2 - z^3 / 3 - 1 / 12)
        @test isapprox(ϕ_exact, model.nonhydrostatic_pressure, rtol=1e-3)
    end
end