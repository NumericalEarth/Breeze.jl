include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using Breeze.BoundaryConditions: BulkDrag, DefaultGustiness, TKEGustiness, materialize_gustiness, gustiness²,
                                  surface_layer_state
using Breeze.AtmosphereModels: total_density
using Oceananigans
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Models: boundary_condition_args
using Oceananigans.TurbulenceClosures: SmagorinskyLilly
using GPUArraysCore: @allowscalar
using Test

# A 2 km column with its first cell centre at 37 m, the first level of the RCEMIP sounding
column_grid(FT) = RectilinearGrid(default_arch, FT; size=27, z=(0, 1998), topology=(Flat, Flat, Bounded))

const Tˢ = 300 # sea surface temperature, K
const ΔT = 2   # the air near the surface is 2 K colder than the sea

surface_coefficient(FT) = PolynomialCoefficient(FT; roughness_length=FT(1.5e-4))

function bulk_flux_bcs(coefficient; gustiness = DefaultGustiness())
    drag = BulkDrag(; coefficient, gustiness, surface_temperature=Tˢ)
    return (ρu  = FieldBoundaryConditions(bottom=drag),
            ρv  = FieldBoundaryConditions(bottom=drag),
            ρθ  = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(; coefficient, gustiness, surface_temperature=Tˢ)),
            ρqᵗ = FieldBoundaryConditions(bottom=BulkVaporFlux(; coefficient, gustiness, surface_temperature=Tˢ)))
end

function surface_flux_model(grid; coefficient = surface_coefficient(eltype(grid)), gustiness = DefaultGustiness(),
                            closure = nothing, u = 0, e = 0)
    FT = eltype(grid)
    reference_state = ReferenceState(grid; base_pressure=101480, potential_temperature=Tˢ)
    dynamics = AnelasticDynamics(reference_state)
    model = AtmosphereModel(grid; dynamics, closure, boundary_conditions=bulk_flux_bcs(coefficient; gustiness))
    set!(model; θ = Tˢ - ΔT, qᵗ = FT(0.015), u)
    haskey(model.tracers, :ρe) && set!(model.tracers.ρe, e * total_density(model.dynamics))
    return model
end

# The bottom flux of a prognostic field, evaluated from the model state
function bottom_flux(model, field)
    condition = field.boundary_conditions.bottom.condition
    return @allowscalar getbc(condition, 1, 1, model.grid, boundary_condition_args(model)...)
end

bottom_gustiness(field) = field.boundary_conditions.bottom.condition.gustiness

thermodynamic_density(model) = Oceananigans.prognostic_fields(model).ρθ

# Latent and sensible heat fluxes into the air, W/m²
latent_heat_flux(model) = 2.5e6 * bottom_flux(model, model.moisture_density)
sensible_heat_flux(model) = 1004 * bottom_flux(model, thermodynamic_density(model))

surface_fluxes(model) = (bottom_flux(model, thermodynamic_density(model)), bottom_flux(model, model.moisture_density))

@testset "materialize_gustiness" begin
    # The placeholder becomes TKE gustiness under a prognostic TKE closure...
    @test materialize_gustiness(DefaultGustiness(), TKEBasedTurbulenceClosure()) isa TKEGustiness
    @test materialize_gustiness(DefaultGustiness(), (SmagorinskyLilly(), TKEBasedTurbulenceClosure())) isa TKEGustiness

    # ... and the constant default otherwise
    @test materialize_gustiness(DefaultGustiness(), nothing) === 0
    @test materialize_gustiness(DefaultGustiness(), SmagorinskyLilly()) === 0
    @test materialize_gustiness(DefaultGustiness(), (SmagorinskyLilly(),)) === 0

    # A gustiness that was given passes through unchanged
    tke_gustiness = TKEGustiness(coefficient=0.5, minimum=0.2)
    @test materialize_gustiness(0.7, TKEBasedTurbulenceClosure()) === 0.7
    @test materialize_gustiness(0.7, nothing) === 0.7
    @test materialize_gustiness(tke_gustiness, TKEBasedTurbulenceClosure()) === tke_gustiness

    # TKE gustiness needs the TKE of a TKEBasedTurbulenceClosure
    @test_throws ArgumentError materialize_gustiness(tke_gustiness, nothing)
    @test_throws ArgumentError materialize_gustiness(tke_gustiness, SmagorinskyLilly())

    @test_throws ArgumentError TKEGustiness(minimum=-1)
    @test_throws ArgumentError TKEGustiness(coefficient=-1)

    g = TKEGustiness()
    @test g.coefficient ≈ sqrt(2/3)
    @test g.minimum == 0.5
end

@testset "Default gustiness follows the closure [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = column_grid(FT)

    model = surface_flux_model(grid; closure=TKEBasedTurbulenceClosure(FT))
    @test bottom_gustiness(model.moisture_density) isa TKEGustiness{FT}
    @test bottom_gustiness(model.momentum.ρu) isa TKEGustiness{FT}
    @test bottom_gustiness(model.momentum.ρv) isa TKEGustiness{FT}
    @test bottom_gustiness(thermodynamic_density(model)) isa TKEGustiness{FT}

    for closure in (SmagorinskyLilly(), nothing)
        model = surface_flux_model(grid; closure)
        @test bottom_gustiness(model.moisture_density) === 0
        @test bottom_gustiness(model.momentum.ρu) === 0
        @test bottom_gustiness(thermodynamic_density(model)) === 0
    end

    # A user-supplied gustiness is never overridden
    model = surface_flux_model(grid; closure=TKEBasedTurbulenceClosure(FT), gustiness=1)
    @test bottom_gustiness(model.moisture_density) === 1
    @test bottom_gustiness(model.momentum.ρu) === 1
end

@testset "Transfer coefficient sees the gusty wind [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = column_grid(FT)
    Uᵍ = 1

    # Calm air with gustiness Uᵍ is the same surface layer as a wind of Uᵍ without gustiness:
    # the flux and its stability-dependent coefficient both see Ũ = √(u² + v² + Uᵍ²).
    calm = surface_flux_model(grid; gustiness=Uᵍ, u=0)
    windy = surface_flux_model(grid; gustiness=0, u=Uᵍ)

    @test all(surface_fluxes(calm) .≈ surface_fluxes(windy))

    # Free convection with a 1 m/s gustiness over a warmer sea gives a few W/m² sensible and a few tens
    # of W/m² latent (3.5 and 37 W/m² here), rather than vanishing (about 1e-7 W/m² when the unstable fit
    # folded back) or running away (about 400 W/m² latent when the coefficient ignored the gustiness)
    Q = sensible_heat_flux(calm)
    L = latent_heat_flux(calm)
    @test 1 < Q < 40
    @test 10 < L < 200

    # The same with a constant coefficient: the gustiness enters only the flux
    calm = surface_flux_model(grid; coefficient=1e-3, gustiness=Uᵍ, u=0)
    windy = surface_flux_model(grid; coefficient=1e-3, gustiness=0, u=Uᵍ)
    @test all(surface_fluxes(calm) .≈ surface_fluxes(windy))
end

@testset "TKE gustiness in a single column [$FT]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = column_grid(FT)
    closure = TKEBasedTurbulenceClosure(FT)

    # With e uniform, TKE gustiness is the constant gustiness c √e
    e = 3/2 # c √e = 1 m/s
    tke = surface_flux_model(grid; closure, e)
    constant = surface_flux_model(grid; closure, e, gustiness=sqrt(2e/3))
    gustiness = bottom_gustiness(tke.moisture_density)
    @test gustiness isa TKEGustiness
    @test @allowscalar(gustiness²(1, 1, 1, grid, gustiness, surface_layer_state(tke))) ≈ 1
    @test all(surface_fluxes(tke) .≈ surface_fluxes(constant))

    # Calm air: fluxes are nonzero and of free-convection size
    @test 1 < sensible_heat_flux(tke) < 40
    @test 10 < latent_heat_flux(tke) < 200

    # Vanishing TKE is held at the minimum gustiness
    tke = surface_flux_model(grid; closure, e=0)
    floor = surface_flux_model(grid; closure, e=0, gustiness=gustiness.minimum)
    @test all(surface_fluxes(tke) .≈ surface_fluxes(floor))
    @test latent_heat_flux(tke) > 0

    # The fluxes stay finite and positive as the column is stepped forward
    tke = surface_flux_model(grid; closure, e=FT(0.1))
    for _ in 1:3
        time_step!(tke, 1)
    end
    @test all(isfinite, interior(tke.tracers.ρe))
    @test sensible_heat_flux(tke) > 0
    @test latent_heat_flux(tke) > 0
end
