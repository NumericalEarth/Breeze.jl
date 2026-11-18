include(joinpath(@__DIR__, "setup.jl"))

#####
##### Energy flux boundary conditions (the `ρE` key) with HydrostaticDynamics: an energy flux 𝒬
##### through the ground enters the lowest layer's ρθ as 𝒬 / (cᵖᵈ Π), so that the enthalpy of the
##### column changes by flux × time × area
#####

using Breeze
using Breeze.HydrostaticEquations: VerletSubsteps
using Oceananigans
using Oceananigans.Operators: Azᶜᶜᶜ, Vᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!, compute_flux_bc_tendencies!
using Test

include(joinpath(@__DIR__, "hydrostatic_test_utils.jl"))

const Nz = 12

energy_flux_bottom(𝒬) = (; ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(𝒬)))

function column_model(boundary_conditions; kw...)
    grid = RectilinearGrid(default_arch, FT; size=Nz, x=0, y=0, z=(0, 1), topology=(Flat, Flat, Bounded))
    A, B = hybrid_coefficients(Nz)
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B, surface_pressure=99000)
    model = hydrostatic_model(grid; dynamics, boundary_conditions, thermodynamic_constants=reference_model_constants(), kw...)
    set!(model; θ = z -> 290 + 80 * z^2)
    return model
end

column(field) = vec(Array(interior(field)))

# Enthalpy of the column per unit area, Σ cᵖᵈ T Δp / g = Σ cᵖᵈ Π ρθ Δz, in J m⁻²
column_enthalpy(model) = sum(cᵖᵈ .* column(model.dynamics.exner_function) .*
                             column(model.formulation.potential_temperature_density)) / Nz

#####
##### Tolerances, written before these tests were run
#####
#
#   1. Column at rest, constant flux 𝒬 through the ground (and, separately, through the model top):
#      the surface pressure of a single column does not change, so Π is fixed and the enthalpy per
#      unit area after n steps is its initial value plus 𝒬 n Δt (minus, for a flux out of the top)
#      within 1e-12 of the enthalpy (60 stage updates at a few eps), for SSP RK3 and the
#      split-explicit stepper. Only the layer next to the boundary changes.
#   2. The tendency that the boundary condition adds to the lowest layer is 𝒬 / (cᵖᵈ Π Δz) within
#      1e-13, relative: the form of the M-INT1 adapter, with Π the layer's Exner function.
#   3. Resting isentropic atmosphere on a latitude-longitude grid, flux varying with longitude and
#      latitude: the volume integral of cᵖᵈ Π ∂ₜ(ρθ) equals the area integral of 𝒬 within 1e-12,
#      relative.

@testset "Column enthalpy under an energy flux [$FT, $name]" for (name, kw) in
        (("SSP RK3", (;)), ("MIS3C + Verlet", split_explicit(:MIS3C, VerletSubsteps())))

    𝒬 = 150.0 # W m⁻², into the atmosphere through the ground
    Δt, steps = 300, 20

    model = column_model(energy_flux_bottom(𝒬); kw...)
    H⁰ = column_enthalpy(model)
    θ⁰ = column(model.formulation.potential_temperature)
    Π = column(model.dynamics.exner_function)

    # The tendency of the lowest layer, before any step
    update_state!(model)
    compute_flux_bc_tendencies!(model)
    G = column(model.timestepper.Gⁿ.ρθ)
    @test G[1] ≈ 𝒬 / (cᵖᵈ * Π[1] * (1 / Nz)) rtol=1e-13          # gate 2
    @test all(iszero, G[2:end])

    for _ in 1:steps
        time_step!(model, Δt)
    end

    ΔH = column_enthalpy(model) - H⁰
    @test abs(ΔH - 𝒬 * steps * Δt) < 1e-12 * H⁰                  # gate 1
    @test column(model.dynamics.exner_function) == Π

    θ = column(model.formulation.potential_temperature)
    @test θ[1] > θ⁰[1] + 1
    @test θ[2:end] ≈ θ⁰[2:end] rtol=1e-13

    @info "Energy flux through the ground [$name]" ΔH expected = 𝒬 * steps * Δt warming_of_lowest_layer = θ[1] - θ⁰[1]
end

@testset "Column enthalpy under an energy flux through the model top [$FT]" begin
    𝒬 = 40.0 # W m⁻², upward: out of the atmosphere
    Δt, steps = 300, 10
    model = column_model((; ρE = FieldBoundaryConditions(top=FluxBoundaryCondition(𝒬))))
    H⁰ = column_enthalpy(model)

    for _ in 1:steps
        time_step!(model, Δt)
    end

    @test abs(column_enthalpy(model) - H⁰ + 𝒬 * steps * Δt) < 1e-12 * H⁰
end

@testset "Area integral of an energy flux on a latitude-longitude grid [$FT]" begin
    grid = LatitudeLongitudeGrid(default_arch, FT; size=(24, 12, 6), longitude=(-180, 180), latitude=(-60, 60), z=(0, 1))
    𝒬 = Field{Center, Center, Nothing}(grid)
    set!(𝒬, (λ, φ) -> 100 + 80 * sind(2λ) * cosd(φ) + 30 * sind(φ))

    A, B = hybrid_coefficients(6)
    dynamics = HydrostaticDynamics(pressure_coefficient=A, surface_pressure_coefficient=B)
    model = hydrostatic_model(grid; dynamics, boundary_conditions=energy_flux_bottom(𝒬),
                              thermodynamic_constants=reference_model_constants())
    set!(model; θ = 300)
    update_state!(model)
    compute_flux_bc_tendencies!(model)

    cpu_grid = Oceananigans.Architectures.on_architecture(CPU(), grid)
    Π = Array(interior(model.dynamics.exner_function))
    G = Array(interior(model.timestepper.Gⁿ.ρθ))
    flux = Array(interior(𝒬))
    Nx, Ny, Nz′ = size(grid)

    heating = sum(cᵖᵈ * Π[i, j, k] * G[i, j, k] * Vᶜᶜᶜ(i, j, k, cpu_grid) for i in 1:Nx, j in 1:Ny, k in 1:Nz′)
    supplied = sum(flux[i, j, 1] * Azᶜᶜᶜ(i, j, 1, cpu_grid) for i in 1:Nx, j in 1:Ny)

    @test supplied > 1e15
    @test heating ≈ supplied rtol=1e-12                          # gate 3
    @test all(iszero, G[:, :, 2:end])
end
