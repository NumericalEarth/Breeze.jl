using Test
using CUDA
using Breeze
using Oceananigans

using Oceananigans.Fields: set!
using Oceananigans.TimeSteppers: update_state!

CUDA.functional() || error("CUDA is required for this bounded GPU validation")

function model_for_native_flux(architecture, scalar_advection)
    grid = RectilinearGrid(architecture, Float32; size=(12, 12, 12),
                           halo=(5, 5, 5), extent=(120, 120, 120))
    closure = SurfaceLayerDiffusivity(Float32;
                                      resolved_transport=:scheme_native,
                                      support=2,
                                      minimum_scalar_fluxes=(ρθ=1f-8,))
    model = AtmosphereModel(grid; closure,
                            momentum_advection=WENO(order=5),
                            scalar_advection)
    set!(model; enforce_mass_conservation=false,
         θ=(x, y, z) -> 300 + 0.15sin(z / 11) + 0.02cos(x / 13),
         u=(x, y, z) -> 2 + 0.1sin(z / 7) + 0.03cos(y / 17),
         v=(x, y, z) -> -1 + 0.08cos(z / 9) + 0.02sin(x / 15),
         w=(x, y, z) -> 0.03sin(x / 12) - 0.02cos(y / 19))
    model.clock.time += 0.1f0
    model.clock.iteration += 1
    update_state!(model; compute_tendencies=false)
    return model
end

function flux_snapshot(model)
    fields = model.closure_fields
    return (; viscosity=Array(interior(fields.Kᵘ)),
            scalar_diffusivity=Array(interior(fields.tupled_tracer_diffusivities.ρθ)),
            native_u=Array(interior(fields.scheme_u_flux[1])),
            native_scalar=Array(interior(fields.scheme_scalar_flux.ρθ[1])),
            correction_u=Array(interior(fields.numerical_u_correction[1])),
            correction_scalar=Array(interior(fields.numerical_scalar_correction.ρθ[1])))
end

function compare_cpu_gpu(scalar_advection)
    cpu = flux_snapshot(model_for_native_flux(CPU(), scalar_advection))
    gpu = flux_snapshot(model_for_native_flux(GPU(), scalar_advection))
    for name in keys(cpu)
        @test isapprox(getproperty(gpu, name), getproperty(cpu, name); rtol=8f-5, atol=8f-6)
        @test all(isfinite, getproperty(gpu, name))
    end
end

@testset "scheme-native SLD GPU parity" begin
    compare_cpu_gpu(WENO(order=5))
    compare_cpu_gpu(WENO(order=5, bounds=(299f0, 301f0)))
end
