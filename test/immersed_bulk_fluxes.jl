include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.BoundaryConditions: BulkDrag, surface_layer_state, wall_air_pressure
using Breeze.Thermodynamics: surface_density, saturation_specific_humidity, potential_temperature_from_temperature,
                             PlanarLiquidSurface
using GPUArraysCore: @allowscalar
using Oceananigans
using Oceananigans.BoundaryConditions: West, East, South, North, Bottom
using Oceananigans.Grids: XDirection, YDirection
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: compute_flux_bc_tendencies!
using Test

#####
##### Bulk fluxes on the facets of an immersed body: walls at x < 1/4 and y < 1/4 (seen from both
##### sides through the periodic boundaries) and a floor at z < 1/4.
#####

const Tˢ = 300
const C = 2e-3
const u, v = 1.0, 2.0

function immersed_box_model(formulation; bulk_fluxes=true)
    underlying = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 1), y=(0, 1), z=(0, 1),
                                 topology=(Periodic, Periodic, Bounded))
    body(x, y, z) = (x < 0.25) | (y < 0.25) | (z < 0.25)
    grid = ImmersedBoundaryGrid(underlying, GridFittedBoundary(body))

    drag = BulkDrag(coefficient=C, surface_temperature=Tˢ)
    heat = BulkSensibleHeatFlux(coefficient=C, surface_temperature=Tˢ)
    vapor = BulkVaporFlux(coefficient=C, surface_temperature=Tˢ, surface_relative_humidity=1)
    thermodynamic_name = formulation === :StaticEnergy ? :ρs : :ρθ
    names = (:ρu, :ρv, thermodynamic_name, :ρqᵛ)
    boundary_conditions = bulk_fluxes ?
        NamedTuple{names}(FieldBoundaryConditions(immersed=bc) for bc in (drag, drag, heat, vapor)) : NamedTuple()

    model = AtmosphereModel(grid; formulation, boundary_conditions)
    set!(model; θ=290, u, v, enforce_mass_conservation=false)
    compute_flux_bc_tendencies!(model)
    return model
end

# (facet, near-wall cell, grid spacing normal to the facet, tangential momentum components)
probes = ((West(),   (3, 6, 6), 1/8, ((:ρv, YDirection(), v),)),
          (East(),   (8, 6, 6), 1/8, ((:ρv, YDirection(), v),)),
          (South(),  (6, 3, 6), 1/8, ((:ρu, XDirection(), u),)),
          (North(),  (6, 8, 6), 1/8, ((:ρu, XDirection(), u),)),
          (Bottom(), (6, 6, 3), 1/8, ((:ρu, XDirection(), u), (:ρv, YDirection(), v))))

tangential_speed(::Union{West, East}) = v
tangential_speed(::Union{South, North}) = u
tangential_speed(::Bottom) = √(u^2 + v^2)

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

    for (side, (i, j, k), Δ, momentum) in probes
        Ũ = tangential_speed(side)

        # Drag removes tangential momentum
        for (name, direction, uₜ) in momentum
            ρˢ = surface_density(@allowscalar(wall_air_pressure(i, j, k, grid, side, direction, state, constants)), Tˢ, constants)
            @test ΔG(name)[i, j, k] ≈ - ρˢ * C * Ũ * uₜ / Δ rtol=1e-6
        end

        pˢ = @allowscalar wall_air_pressure(i, j, k, grid, side, nothing, state, constants)
        ρˢ = surface_density(pˢ, Tˢ, constants)

        # The 300 K wall heats the 290 K air
        if formulation === :StaticEnergy
            zˢ = side isa Bottom ? 0.25 : @allowscalar znode(i, j, k, grid, Center(), Center(), Center())
            Δϕ = @allowscalar(state.s[i, j, k]) - (cᵖᵈ * Tˢ + g * zˢ)
        else
            θ₀ = potential_temperature_from_temperature(Tˢ, pˢ, heat_condition.standard_pressure, constants, 0)
            Δϕ = @allowscalar(state.θ[i, j, k]) - θ₀
        end
        ΔGheat = ΔG(thermodynamic_name)[i, j, k]
        @test ΔGheat > 0
        @test ΔGheat ≈ - ρˢ * C * Ũ * Δϕ / Δ rtol=1e-6

        # The saturated wall moistens the dry air
        qᵛˢ = saturation_specific_humidity(Tˢ, ρˢ, constants, PlanarLiquidSurface())
        @test ΔG(:ρqᵛ)[i, j, k] ≈ ρˢ * C * Ũ * qᵛˢ / Δ rtol=1e-6
    end

    # Cells away from the facets probed here receive no immersed flux
    far = (6, 6, 6)
    for name in (:ρu, :ρv, thermodynamic_name, :ρqᵛ)
        @test ΔG(name)[far...] == 0
    end

    time_step!(model, 0.01)
    @test all(isfinite, interior(model.momentum.ρu))
    @test all(isfinite, interior(prognostic_fields(model)[thermodynamic_name]))
end
