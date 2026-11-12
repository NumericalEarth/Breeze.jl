include(joinpath(@__DIR__, "setup.jl"))


using Breeze
using Oceananigans
using Oceananigans.Operators: divᶜᶜᶜ
using Test

# Uniform flow past a flat-faced cylinder (issue #1233)
function near_body_divergence(pressure_solver)
    underlying = RectilinearGrid(size = (32, 16, 16), x = (-8, 8), y = (-4, 4), z = (-4, 4),
                                 topology = (Periodic, Periodic, Bounded))
    body(x, y, z) = (y^2 + z^2 < 0.49) & (-3 < x < 3)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBoundary(body))

    reference_state = ReferenceState(grid, ThermodynamicConstants(), base_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state; pressure_solver)
    model = AtmosphereModel(grid; dynamics)
    set!(model; θ=300, u=10)

    for _ in 1:3
        time_step!(model, 0.01)
    end

    div = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, model.momentum...))
    compute!(div)
    x, y, z = xnodes(underlying, Center()), ynodes(underlying, Center()), znodes(underlying, Center())
    shifts = (-0.6, 0, 0.6)
    near = [any(body(x + a, y + b, z + c) for a in shifts, b in shifts, c in shifts) for x in x, y in y, z in z]
    inside = [body(x, y, z) for x in x, y in y, z in z]

    return maximum(abs, interior(div)[near .& .!inside])
end

# ρU / Δx ≈ 23 kg m⁻³ s⁻¹; the Fourier–tridiagonal solve alone leaves about 0.4 next to the body
@testset "AnelasticConjugateGradientSolver on an ImmersedBoundaryGrid" begin
    @test near_body_divergence(nothing) < 1e-5
    @test near_body_divergence(AnelasticConjugateGradientSolver(reltol=1e-10, abstol=1e-12)) < 1e-6
end

@testset "AnelasticConjugateGradientSolver without an immersed boundary" begin
    grid = RectilinearGrid(size = (16, 8, 8), x = (-8, 8), y = (-4, 4), z = (-4, 4),
                           topology = (Periodic, Periodic, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(), base_pressure=101325, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state; pressure_solver=AnelasticConjugateGradientSolver(reltol=1e-10))
    model = AtmosphereModel(grid; dynamics)
    set!(model; θ=300, u=(x, y, z) -> 10 + sin(π * x / 8))
    time_step!(model, 0.01)
    div = Field(KernelFunctionOperation{Center, Center, Center}(divᶜᶜᶜ, grid, model.momentum...))
    compute!(div)
    @test maximum(abs, interior(div)) < 1e-6
end
