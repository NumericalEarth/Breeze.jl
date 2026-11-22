using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans.Utils: getregion
using Oceananigans.MultiRegion: YinYangGrid
using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization, PressureLevelGrid, layer_pressure_thickness

#####
##### Moist HydrostaticDynamics with mixed-phase one-moment microphysics on a Yin-Yang grid of
##### pressure levels: the model builds, its coordinate is that of the grid of each panel, and it steps
#####
#
# The microphysics runs through the generic interface, which reads Δz as the physical depth
# Δp / (g ρ) of the grid. The surface pressure and the density of the dynamics must therefore be the
# arrays of the panels' grids, and the density must be the one diagnosed from the state: the
# first two testsets check that wiring before anything is stepped.

function pressure_level_coefficients(Nz; top_pressure = 2.5e4, p₀ = 1e5)
    η = range(1, top_pressure / p₀, length = Nz + 1)
    B = [max(0, (e - 0.4) / 0.6) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0; A[end] = top_pressure
    return A, B
end

@testset "Mixed-phase one-moment microphysics on a Yin-Yang PressureLevelGrid" begin
    # Panels of 48 × 16, whose overlap and halo stay off the poles
    Nz = 8
    A, B = pressure_level_coefficients(Nz)
    grid = YinYangGrid(CPU(), Float64; size = (48, 16, Nz), halo = (4, 4, 3), z = PressureLevelVerticalDiscretization(A, B))
    @test all(getregion(grid, r) isa PressureLevelGrid for r in 1:2)

    model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(), microphysics = mixed_phase_microphysics(),
                            formulation = moist_formulation(), momentum_advection = VectorInvariant())
    dynamics = model.dynamics

    @testset "The coordinate of the dynamics is that of the panels" begin
        for r in 1:2
            z = getregion(grid, r).z
            @test parent(getregion(dynamics.surface_pressure, r)) === parent(z.surface_pressure)
            @test parent(getregion(dynamics.density, r)) === parent(z.density)
        end
    end

    set!(model; pˢ = 1e5,
                θ = (λ, φ, z) -> 275 + 3e-3 * z + 2 * cosd(φ) * sind(2λ),
                u = (λ, φ, z) -> 10 * cosd(φ),
                v = (λ, φ, z) -> 2 * cosd(1.5φ) * sind(λ))
    set!(model; qᵗ = (λ, φ, z) -> 0.008 * exp(-z / 2500) * cosd(φ),
                qʳ = (λ, φ, z) -> (z < 2000) * 5e-4 * cosd(φ),
                qˢⁿ = (λ, φ, z) -> (1000 < z < 4000) * 5e-4 * cosd(φ))

    @testset "The density is that of the state" begin
        constants = model.thermodynamic_constants
        cᵖᵈ = constants.dry_air.heat_capacity
        for r in 1:2
            regional = getregion(model, r)
            rg = regional.grid
            ρ = regional.dynamics.density
            θᵨ = regional.dynamics.moisture.density_potential_temperature
            Πᶠ = regional.dynamics.interface_exner_function
            Nx, Ny, _ = size(rg)
            # ρ = Δp / (g Δz) with Δz = cᵖᵈ θᵨ ΔΠᶠ / g, to the rounding of the quotients (as in pressure_level_discretization.jl)
            @test all(isapprox(ρ[i, j, k], layer_pressure_thickness(i, j, k, rg) / (cᵖᵈ * θᵨ[i, j, k] * (Πᶠ[i, j, k] - Πᶠ[i, j, k+1]));
                               rtol = 8eps(Float64)) for i in 1:Nx, j in 1:Ny, k in 1:Nz)
        end
    end

    for _ in 1:2
        time_step!(model, 20)
    end

    @test model.clock.iteration == 2
    for r in 1:2
        regional = getregion(model, r)
        @test all(isfinite, Array(interior(regional.dynamics.surface_pressure)))
        @test all(>(0), Array(interior(regional.dynamics.density)))
        @test all(isfinite, Array(interior(regional.temperature)))
        @test all(≥(0), Array(interior(regional.dynamics.moisture.ledger.surface_rain)))
        @test maximum(Array(interior(regional.microphysical_fields.qⁱ))) > 0
    end
end
