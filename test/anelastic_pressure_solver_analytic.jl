using Test
using Breeze
using Oceananigans
using Oceananigans.Fields: fill_halo_regions!
using Statistics: mean

@testset "Anelastic pressure solver recovers analytic solution [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=48, z=(0, 1), topology=(Flat, Flat, Bounded))
    thermodynamics = ThermodynamicConstants()
    reference_state = ReferenceState(grid, thermodynamics, base_pressure=101325, potential_temperature=288)
    formulation = AnelasticFormulation(reference_state)

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

    set!(formulation.reference_state.density, z -> z)
    fill_halo_regions!(formulation.reference_state.density)
    model = AtmosphereModel(grid; thermodynamics, formulation)
    set!(model, ρw = z -> z^2 - z^3)

    # Test for zero mean
    atol = 10 * grid.Nz * eps(FT)
    ϕ = model.pressure
    @test mean(ϕ) ≈ 0 atol=atol

    # Test for exact solution
    ϕ_exact = CenterField(grid)
    set!(ϕ_exact, z -> z^2 / 2 - z^3 / 3 - 1 / 12)
    parent(ϕ_exact) .-= mean(ϕ_exact)

    @test isapprox(ϕ_exact, ϕ; rtol=1e-3)
end
