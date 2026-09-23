include(joinpath(@__DIR__, "setup.jl"))

using Test
using CUDA
using Breeze
using Oceananigans

using Oceananigans.Fields: set!
using Oceananigans.TimeSteppers: update_state!

gpu_drag_u(x, y, t, u, v) = -2f-3 * sqrt(u^2 + v^2) * u
gpu_drag_v(x, y, t, u, v) = -2f-3 * sqrt(u^2 + v^2) * v

function stable_native_model(architecture, λ)
    grid = RectilinearGrid(architecture, Float32; size=(12, 12, 12), halo=(5, 5, 5),
                           x=(0, 120), y=(0, 120), z=(0, 120))
    closure = SurfaceLayerDiffusivity(Float32; resolved_transport=:scheme_native,
                                      stability_strength=λ, support=2,
                                      minimum_scalar_fluxes=(ρθ=1f-8,))
    boundary_conditions = (
        ρu=FieldBoundaryConditions(bottom=FluxBoundaryCondition(gpu_drag_u,
                                                                field_dependencies=(:u, :v))),
        ρv=FieldBoundaryConditions(bottom=FluxBoundaryCondition(gpu_drag_v,
                                                                field_dependencies=(:u, :v))),
        ρE=FieldBoundaryConditions(bottom=FluxBoundaryCondition(-10f0)))
    model = AtmosphereModel(grid; closure, boundary_conditions,
                            momentum_advection=WENO(order=5), scalar_advection=WENO(order=5))
    set!(model; enforce_mass_conservation=false,
         θ=(x, y, z) -> 265 + 0.01z + 0.05sin(x / 11) * exp(-z / 30),
         u=(x, y, z) -> 5 + 0.05z + 0.1cos(y / 17),
         v=(x, y, z) -> 0.08cos(z / 9),
         w=(x, y, z) -> 0.02sin(x / 12) * z / 120)
    model.clock.time += 0.1f0
    model.clock.iteration += 1
    update_state!(model; compute_tendencies=false)
    return model
end

function stability_snapshot(model)
    fields = model.closure_fields
    return (; viscosity=Array(interior(fields.Kᵘ)),
            scalar_diffusivity=Array(interior(fields.tupled_tracer_diffusivities.ρθ)),
            inverse_obukhov_length=Array(interior(fields.inverse_obukhov_length)),
            stability_state=Array(interior(fields.stability_state)),
            momentum_stability_function=Array(interior(fields.momentum_stability_function[1])),
            scalar_stability_function=Array(interior(fields.scalar_stability_function[2])),
            surface_heat_flux=Array(interior(fields.surface_scalar_flux.ρθ)))
end

@testset "SLD stability correction GPU parity" begin
    if CUDA.functional()
        for λ in (0, 1)
            cpu = stability_snapshot(stable_native_model(CPU(), λ))
            gpu = stability_snapshot(stable_native_model(GPU(), λ))
            for name in keys(cpu)
                @test isapprox(getproperty(gpu, name), getproperty(cpu, name);
                               rtol=8f-5, atol=8f-6)
                @test all(isfinite, getproperty(gpu, name))
            end
            @test all(==(1), gpu.stability_state)
            @test λ == 0 ? all(==(1), gpu.momentum_stability_function) :
                           all(>(1), gpu.momentum_stability_function)
        end
        model = stable_native_model(GPU(), 1)
        for n in 1:3
            Oceananigans.time_step!(model, 0.1f0)
        end
        @test all(isfinite, Array(interior(model.closure_fields.Kᵘ)))
        @test model.closure_fields.previous_update_iteration[] == model.clock.iteration
    else
        @test_skip "CUDA is not functional"
    end
end
