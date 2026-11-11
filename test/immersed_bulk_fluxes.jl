include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.BoundaryConditions: BulkDrag, surface_layer_state, wall_air_pressure
using Breeze.Thermodynamics: surface_density, saturation_specific_humidity, potential_temperature_from_temperature,
                             PlanarLiquidSurface
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.BoundaryConditions: West, East, South, North, Bottom
using Oceananigans.Grids: XDirection, YDirection
using Oceananigans.TimeSteppers: compute_flux_bc_tendencies!
using Test

#####
##### Bulk fluxes on the facets of an immersed body: walls at x < 1/4 and y < 1/4 (seen from both
##### sides through the periodic boundaries) and a floor at z < 1/4.
#####

const C = 2e-3
const Tˢ = 300
const u, v = 1.0, 2.0
const Δ = 1 / 8

# The wall temperature of the heat flux varies in all three directions, so it is evaluated at the facet
heat_wall_temperature(x, y, z, t) = 299 + x + 2y + 4z

function immersed_box_model(formulation; bulk_fluxes=true)
    underlying = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                                 topology=(Periodic, Periodic, Bounded))
    body(x, y, z) = (x < 0.25) | (y < 0.25) | (z < 0.25)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBoundary(body))

    drag = BulkDrag(coefficient=C, surface_temperature=Tˢ)
    heat = BulkSensibleHeatFlux(coefficient=C, surface_temperature=heat_wall_temperature)
    vapor = BulkVaporFlux(coefficient=C, surface_temperature=Tˢ, surface_relative_humidity=1)
    thermodynamic_name = formulation === :StaticEnergy ? :ρs : :ρθ
    names = (:ρu, :ρv, thermodynamic_name, :ρqᵛ)
    boundary_conditions = bulk_fluxes ?
        NamedTuple{names}(FieldBoundaryConditions(immersed=bc) for bc in (drag, drag, heat, vapor)) : NamedTuple()

    model = AtmosphereModel(grid; formulation, boundary_conditions)
    # Walls spanning the periodic domain would block u and v under the projection
    set!(model; θ=290, u, v, enforce_mass_conservation=false)
    compute_flux_bc_tendencies!(model)
    return model
end

# Facet centers of the near-wall cell (i, j, k)
facet_center(::West, i, j, k)   = ((i - 1) * Δ, (j - 1/2) * Δ, (k - 1/2) * Δ)
facet_center(::East, i, j, k)   = (i * Δ, (j - 1/2) * Δ, (k - 1/2) * Δ)
facet_center(::South, i, j, k)  = ((i - 1/2) * Δ, (j - 1) * Δ, (k - 1/2) * Δ)
facet_center(::North, i, j, k)  = ((i - 1/2) * Δ, j * Δ, (k - 1/2) * Δ)
facet_center(::Bottom, i, j, k) = ((i - 1/2) * Δ, (j - 1/2) * Δ, (k - 1) * Δ)

# Probes: near-wall cell and its facets, each with the tangential speed of the scalar fluxes and
# of the drag on (ρu, ρv) built from active faces only. At the corner cell (3, 6, 3) the u face
# on the west wall is masked, so the floor sees u² from the single active u face.
probes = (((3, 6, 6), ((West(),   v,          (0, v)),)),
          ((8, 6, 6), ((East(),   v,          (0, v)),)),
          ((6, 3, 6), ((South(),  u,          (u, 0)),)),
          ((6, 8, 6), ((North(),  u,          (u, 0)),)),
          ((6, 6, 3), ((Bottom(), √(u^2+v^2), (√(u^2+v^2), √(u^2+v^2))),)),
          ((3, 6, 3), ((West(),   v,          (0, v)),
                       (Bottom(), √(u^2+v^2), (0, √(u^2+v^2))))))

@testset "Bulk fluxes on immersed facets [$formulation]" for formulation in (:LiquidIcePotentialTemperature, :StaticEnergy)
    model = immersed_box_model(formulation)
    G = model.timestepper.Gⁿ
    G₀ = immersed_box_model(formulation; bulk_fluxes=false).timestepper.Gⁿ
    ΔG(name) = interior(getproperty(G, name)) .- interior(getproperty(G₀, name))

    grid = model.grid
    constants = model.thermodynamic_constants
    state = surface_layer_state(model)
    thermodynamic_name = formulation === :StaticEnergy ? :ρs : :ρθ
    heat_condition = prognostic_fields(model)[thermodynamic_name].boundary_conditions.immersed.west.condition
    g = constants.gravitational_acceleration
    cᵖᵈ = constants.dry_air.heat_capacity

    for ((i, j, k), facets) in probes
        expected = Dict(:ρu => 0.0, :ρv => 0.0, :heat => 0.0, :ρqᵛ => 0.0)

        for (side, Ũ, (Ũᵘ, Ũᵛ)) in facets
            # Drag removes tangential momentum; u on the west wall of (3, 6, 3) is masked to zero
            for (name, direction, uₜ, Ũₜ) in ((:ρu, XDirection(), u, Ũᵘ), (:ρv, YDirection(), v, Ũᵛ))
                pᵘˢ = @allowscalar wall_air_pressure(i, j, k, grid, side, direction, state, constants)
                expected[name] -= surface_density(pᵘˢ, Tˢ, constants) * C * Ũₜ * uₜ / Δ
            end

            pˢ = @allowscalar wall_air_pressure(i, j, k, grid, side, nothing, state, constants)
            x, y, z = facet_center(side, i, j, k)
            Tʰ = heat_wall_temperature(x, y, z, 0)
            if formulation === :StaticEnergy
                zˢ = side isa Bottom ? z : (k - 1/2) * Δ
                Δϕ = @allowscalar(state.s[i, j, k]) - (cᵖᵈ * Tʰ + g * zˢ)
            else
                θ₀ = potential_temperature_from_temperature(Tʰ, pˢ, heat_condition.standard_pressure, constants, 0)
                Δϕ = @allowscalar(state.θ[i, j, k]) - θ₀
            end
            expected[:heat] -= surface_density(pˢ, Tʰ, constants) * C * Ũ * Δϕ / Δ

            ρˢ = surface_density(pˢ, Tˢ, constants)
            expected[:ρqᵛ] += ρˢ * C * Ũ * saturation_specific_humidity(Tˢ, ρˢ, constants, PlanarLiquidSurface()) / Δ
        end

        @test ΔG(:ρu)[i, j, k] ≈ expected[:ρu] atol=1e-12 rtol=1e-6
        @test ΔG(:ρv)[i, j, k] ≈ expected[:ρv] rtol=1e-6
        @test ΔG(thermodynamic_name)[i, j, k] > 0
        @test ΔG(thermodynamic_name)[i, j, k] ≈ expected[:heat] rtol=1e-6
        @test ΔG(:ρqᵛ)[i, j, k] ≈ expected[:ρqᵛ] rtol=1e-6
    end

    # A cell away from the body receives no immersed flux
    for name in (:ρu, :ρv, thermodynamic_name, :ρqᵛ)
        @test ΔG(name)[6, 6, 6] == 0
    end

    time_step!(model, 0.01)
    @test all(isfinite, interior(model.momentum.ρu))
    @test all(isfinite, interior(prognostic_fields(model)[thermodynamic_name]))
end
