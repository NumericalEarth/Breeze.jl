using Breeze
using Oceananigans
using Test

include("hydrostatic_moist_utils.jl")

using Oceananigans.Utils: getregion
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Operators: Δzᶜᶜᶜ
using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization
using Breeze.AtmosphereModels: surface_vapor_mass_change!

#####
##### Water vapor through the ground of moist HydrostaticDynamics on a Yin-Yang PressureLevelGrid:
##### the column water changes by the vapor that entered, and the surface pressure by its weight
#####

function surface_vapor_coefficients(Nz; top_pressure = 2.5e4, p₀ = 1e5)
    η = range(1, top_pressure / p₀, length = Nz + 1)
    B = [max(0, (e - 0.4) / 0.6) for e in η]
    A = [e * p₀ - b * p₀ for (e, b) in zip(η, B)]
    A[1] = 0; B[1] = 1; B[end] = 0; A[end] = top_pressure
    return A, B
end

# Column water ∑ ρqᵛ Δz (kg m⁻²) and surface pressure of the owned cells of panel r, on the host
function column_water_and_surface_pressure(model, r)
    regional = getregion(model, r)
    grid = regional.grid
    Nx, Ny, Nz = size(grid)
    ρqᵛ = Array(interior(regional.moisture_density))
    W = [sum(ρqᵛ[i, j, k] * Δzᶜᶜᶜ(i, j, k, grid) for k in 1:Nz) for i in 1:Nx, j in 1:Ny]
    pˢ = Array(interior(regional.dynamics.surface_pressure))[:, :, 1]
    return W, pˢ
end

@testset "Surface vapor through the total-mass coordinate of a Yin-Yang PressureLevelGrid" begin
    Nz = 8
    A, B = surface_vapor_coefficients(Nz)
    grid = YinYangGrid(CPU(), Float64; size = (48, 16, Nz), halo = (4, 4, 3), z = PressureLevelVerticalDiscretization(A, B))
    microphysics = Breeze.Microphysics.InstantaneousPrecipitation(Float64; equilibrium = MixedPhaseEquilibrium(Float64))
    model = AtmosphereModel(grid; dynamics = HydrostaticDynamics(), microphysics,
                            formulation = moist_formulation(), momentum_advection = VectorInvariant())

    set!(model; pˢ = 1e5, θ = (λ, φ, z) -> 275 + 3e-3 * z, u = (λ, φ, z) -> 10 * cosd(φ))
    set!(model; qᵗ = (λ, φ, z) -> 0.004 * exp(-z / 2500) * cosd(φ))

    # Evaporation and deposition (kg m⁻²), smaller than the vapor of the lowest layer
    E = Field{Center, Center, Nothing}(grid)
    set!(E, (λ, φ) -> 0.05 * sind(2λ) * cosd(φ))
    g = model.thermodynamic_constants.gravitational_acceleration

    before = [column_water_and_surface_pressure(model, r) for r in 1:2]
    T⁰ = [Array(interior(getregion(model.temperature, r))) for r in 1:2]

    surface_vapor_mass_change!(model.dynamics, model, E)

    for r in 1:2
        W⁰, pˢ⁰ = before[r]
        W¹, pˢ¹ = column_water_and_surface_pressure(model, r)
        Nx, Ny = size(W¹)
        Eʳ = Array(interior(getregion(E, r)))[1:Nx, 1:Ny, 1]
        @test maximum(abs, Eʳ) > 0.04
        # The water of every column changes by the vapor that entered or left it
        @test maximum(abs, (W¹ .- W⁰) .- Eʳ) < 1e-12 * maximum(W⁰)
        # and its surface pressure by the weight of that water
        @test maximum(abs, (pˢ¹ .- pˢ⁰) .- g .* Eʳ) < 1e-9
        # The vapor enters at the temperature of the lowest layer: the temperature barely changes
        T¹ = Array(interior(getregion(model.temperature, r)))
        @test all(isfinite, T¹)
        @test maximum(abs, T¹ .- T⁰[r]) < 0.1
    end

    # Without water through the ground the hook changes nothing but what update_state! itself does
    # (the overlap of the panels is interpolated again, and the layer density re-diagnosed)
    fill!(E, 0)
    Oceananigans.TimeSteppers.update_state!(model)
    reference = deepcopy(model)
    Oceananigans.TimeSteppers.update_state!(reference)
    surface_vapor_mass_change!(model.dynamics, model, E)
    for r in 1:2
        @test Array(interior(getregion(model.formulation.potential_temperature_density, r))) ==
              Array(interior(getregion(reference.formulation.potential_temperature_density, r)))
        @test Array(interior(getregion(model.dynamics.surface_pressure, r))) ==
              Array(interior(getregion(reference.dynamics.surface_pressure, r)))
    end
end
