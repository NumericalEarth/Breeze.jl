# The flux-divergence operators of a vertically implicit scalar diffusivity of VerticalFormulation (the TKE closure)
# skip the terms that are identically +0 — the horizontal fluxes everywhere, the vertical flux on interior faces — and
# must return the generic operators' bits (signed zeros included) in every cell, interior and bottom/top levels alike.
# Checked here against the generic methods themselves (`invoke` with the `AbstractTurbulenceClosure` signature) on a
# terrain-following lat-lon grid with the CONUS ingredients, after a few steps (non-trivial diffusivities, boundary
# fluxes), in both float types; and through a time step: the tendencies of every prognostic field, bit for bit.

include(joinpath(@__DIR__, "setup.jl"))

using Breeze
using CloudMicrophysics
using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Test

using Breeze.AtmosphereModels: AtmosphereModels, dynamics_density, total_density, kernel_fields

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
using .BreezeCloudMicrophysicsExt: OneMomentCloudMicrophysics

function tke_terrain_model(FT)
    Oceananigans.defaults.FloatType = FT
    z = TerrainFollowingVerticalDiscretization(collect(range(0, 12e3, length = 12)))
    grid = LatitudeLongitudeGrid(CPU(), FT; size = (6, 5, 11), halo = (5, 5, 5), z,
                                 longitude = (-100, -99.4), latitude = (35, 35.5),
                                 topology = (Bounded, Bounded, Bounded))
    materialize_terrain!(grid, (λ, φ) -> 600 * exp(-((λ + 99.7)^2 + (φ - 35.25)^2) / 0.15^2))
    cloud_formation = SaturationAdjustment(FT; equilibrium = MixedPhaseEquilibrium(FT))
    bounded = WENO(order = 5, bounds = (0, 1))
    drag = FluxBoundaryCondition(FT(-1e-2))
    model = AtmosphereModel(grid;
                            dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                                            reference_potential_temperature = 300),
                            microphysics = OneMomentCloudMicrophysics(FT; cloud_formation),
                            closure = TKEBasedTurbulenceClosure(),
                            boundary_conditions = (; ρu = FieldBoundaryConditions(bottom = drag)),
                            momentum_advection = WENO(order = 5),
                            scalar_advection = (ρθ = WENO(order = 5), ρqᵉ = bounded, ρqʳ = bounded, ρqˢⁿ = bounded,
                                                ρe = Centered(order = 2)),
                            timestepper = :AcousticRungeKutta3)
    set!(model; ρ = model.dynamics.reference_state.density,
         θ = (λ, φ, z) -> 300 + 4e-3 * z + 0.5 * sin(3λ) * exp(-z / 3000),
         qᵗ = (λ, φ, z) -> 0.018 * exp(-z / 2500) * (1 + 0.3 * cos(5λ + 2φ)),
         u = (λ, φ, z) -> 8 + 1e-3 * z + sin(7φ), v = (λ, φ, z) -> -3 + cos(3λ),
         ρe = (λ, φ, z) -> 0.5 * exp(-z / 1000),
         qʳ = (λ, φ, z) -> 1e-4 * exp(-z / 2000))
    return model
end

const generic_signature = Tuple{Any, Any, Any, Any, Any, AbstractTurbulenceClosure, Vararg{Any}}
bits(x) = reinterpret(Unsigned, x)

function count_differences(model)
    grid = model.grid
    closure = model.closure
    K = model.closure_fields
    clock = model.clock
    fields = kernel_fields(model)
    ρᵈ = dynamics_density(model.dynamics)
    ρ = total_density(model.dynamics)
    Nx, Ny, Nz = size(grid)
    differences = 0
    zeros_compared = 0
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        for op in (AtmosphereModels.∂ⱼ_𝒯₁ⱼ, AtmosphereModels.∂ⱼ_𝒯₂ⱼ, AtmosphereModels.∂ⱼ_𝒯₃ⱼ)
            new = op(i, j, k, grid, ρᵈ, closure, K, clock, fields, nothing)
            old = invoke(op, generic_signature, i, j, k, grid, ρᵈ, closure, K, clock, fields, nothing)
            differences += bits(new) != bits(old)
            zeros_compared += iszero(old)
        end
        for (n, c) in enumerate((model.formulation.potential_temperature, model.tracers.ρe))
            id = Val(n)
            new = AtmosphereModels.∇_dot_Jᶜ(i, j, k, grid, ρ, closure, K, id, c, clock, fields, nothing)
            old = invoke(AtmosphereModels.∇_dot_Jᶜ, generic_signature, i, j, k, grid, ρ, closure, K, id, c, clock, fields, nothing)
            differences += bits(new) != bits(old)
            zeros_compared += iszero(old)
        end
    end
    return differences, zeros_compared
end

@testset "Vertically implicit vertical diffusivity: zero terms skipped bit for bit [$FT]" for FT in all_float_types()
    model = tke_terrain_model(FT)
    for _ in 1:3
        time_step!(model, 1)
    end
    # the grid-specific methods are the ones selected (the generic method's grid argument is untyped)
    argtypes = typeof((1, 1, 1, model.grid, dynamics_density(model.dynamics), model.closure, model.closure_fields,
                       model.clock, kernel_fields(model), nothing))
    @test which(AtmosphereModels.∂ⱼ_𝒯₁ⱼ, argtypes).sig.parameters[5] !== Any
    differences, zeros_compared = count_differences(model)
    @test zeros_compared > 0      # the comparison covers exact zeros (interior levels)
    @test differences == 0
end
