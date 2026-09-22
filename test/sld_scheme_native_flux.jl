using Test
using Breeze
using Oceananigans

using Breeze.TurbulenceClosures: native_u_transport, native_v_transport,
                                  native_scalar_transport, exponential_mean_and_covariance,
                                  exponential_filter_weight
using Breeze.AtmosphereModels: reconstructed_fields, dynamics_density,
                                update_completed_step_closure_state!
using Oceananigans.Advection: _advective_momentum_flux_Wu,
                                _advective_momentum_flux_Wv,
                                _advective_tracer_flux_z,
                                bounded_tracer_flux_divergence_z
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑzᵃᵃᶠ, Azᶜᶜᶠ
using Oceananigans.Fields: set!, CenterField
using Oceananigans.TimeSteppers: update_state!

function native_test_model(scheme; FT=Float32, scalar_advection=scheme)
    grid = RectilinearGrid(CPU(), FT; size=(12, 12, 12),
                           halo=(5, 5, 5), extent=(120, 120, 120))
    closure = SurfaceLayerDiffusivity(FT; resolved_transport=:scheme_native,
                                      support=2, minimum_scalar_fluxes=(ρθ=1f-8,))
    model = AtmosphereModel(grid; closure, momentum_advection=scheme,
                            scalar_advection)
    set!(model; enforce_mass_conservation=false,
         θ=(x, y, z) -> 300 + 0.15sin(z / 11) + 0.02cos(x / 13),
         u=(x, y, z) -> 2 + 0.1sin(z / 7) + 0.03cos(y / 17),
         v=(x, y, z) -> -1 + 0.08cos(z / 9) + 0.02sin(x / 15),
         w=(x, y, z) -> 0.03sin(x / 12) - 0.02cos(y / 19))
    return model
end

@testset "scheme-native vertical operator and binding" begin
    for scheme in (WENO(order=9), WENO(order=5), Centered(order=2))
        model = native_test_model(scheme)
        grid = model.grid
        closure_fields = model.closure_fields
        ρ = dynamics_density(model.dynamics)
        c = reconstructed_fields(model, model.advection).ρθ
        @test keys(model.closure.advection) == keys(model.advection)
        @test typeof(model.closure.advection.momentum) == typeof(model.advection.momentum)
        @test maximum(Array(interior(ρ))) > minimum(Array(interior(ρ)))
        @test model.velocities.w[2, 3, 2] > 0
        @test model.velocities.w[7, 8, 2] < 0
        for (slot, face) in enumerate((2, 3)), i in (2, 7), j in (3, 8)
            denominator = Azᶜᶜᶠ(i, j, face, grid) * ℑzᵃᵃᶠ(i, j, face, grid, ρ)
            expected_u = ℑxᶜᵃᵃ(i, j, face, grid, _advective_momentum_flux_Wu,
                                 model.advection.momentum, model.momentum.ρw,
                                 model.velocities.u) / denominator
            expected_v = ℑyᵃᶜᵃ(i, j, face, grid, _advective_momentum_flux_Wv,
                                 model.advection.momentum, model.momentum.ρw,
                                 model.velocities.v) / denominator
            expected_c = _advective_tracer_flux_z(i, j, face, grid,
                                                    model.advection.ρθ,
                                                    model.velocities.w, c) /
                         Azᶜᶜᶠ(i, j, face, grid)
            @test closure_fields.scheme_u_flux[slot][i, j, 1] ≈ expected_u rtol=1f-5
            @test closure_fields.scheme_v_flux[slot][i, j, 1] ≈ expected_v rtol=1f-5
            @test closure_fields.scheme_scalar_flux.ρθ[slot][i, j, 1] ≈ expected_c rtol=1f-5
            @test native_u_transport(i, j, face, grid, model.advection.momentum,
                                     model.momentum.ρw, model.velocities.u, ρ) ≈ expected_u
            @test native_v_transport(i, j, face, grid, model.advection.momentum,
                                     model.momentum.ρw, model.velocities.v, ρ) ≈ expected_v
            @test native_scalar_transport(i, j, face, grid, model.advection.ρθ,
                                          model.velocities.w, c) ≈ expected_c
            @test closure_fields.numerical_u_correction[slot][i, j, 1] ≈
                  expected_u - closure_fields.uw_product_mean[slot][i, j, 1] rtol=1f-5
            @test closure_fields.numerical_scalar_correction.ρθ[slot][i, j, 1] ≈
                  expected_c - closure_fields.scalar_w_product_mean.ρθ[slot][i, j, 1] rtol=1f-5
        end
        @test isfinite(closure_fields.Kᵘ[2, 3, 2])
    end
end

@testset "per-scalar schemes and nonzero mean transport" begin
    model = native_test_model(WENO(order=9);
        scalar_advection=(ρθ=Centered(order=2), ρqᵛ=WENO(order=5)))
    @test typeof(model.closure.advection.ρθ) == typeof(model.advection.ρθ)
    @test typeof(model.closure.advection.ρqᵛ) == typeof(model.advection.ρqᵛ)
    @test typeof(model.closure.advection.ρθ) != typeof(model.closure.advection.momentum)

    grid = RectilinearGrid(CPU(), Float32; size=(8, 8, 8),
                           halo=(5, 5, 5), extent=(80, 80, 80))
    uniform = AtmosphereModel(grid;
        closure=SurfaceLayerDiffusivity(Float32; resolved_transport=:scheme_native,
                                        support=2),
        advection=Centered(order=2))
    set!(uniform; enforce_mass_conservation=false, θ=300, u=3, v=-2, w=0.02)
    fields = uniform.closure_fields
    @test fields.scheme_scalar_flux.ρθ[1][2, 3, 1] ≈ 6f0 atol=1f-4
    for slot in (1, 2)
        @test abs(fields.numerical_scalar_correction.ρθ[slot][2, 3, 1]) < 2f-5
        @test abs(fields.numerical_u_correction[slot][2, 3, 1]) < 2f-6
        @test abs(fields.resolved_scalar_flux.ρθ[slot][2, 3, 1]) < 2f-5
    end
    set!(uniform; enforce_mass_conservation=false, θ=300, u=3, v=-2, w=0)
    @test all(iszero, Array(interior(uniform.closure_fields.scheme_scalar_flux.ρθ[1])))
end

@testset "bounded limiter matches actual vertical divergence and refreshes" begin
    scheme = WENO(order=5, bounds=(299f0, 301f0))
    model = native_test_model(WENO(order=5); scalar_advection=(ρθ=scheme,))
    grid = model.grid
    ρ = dynamics_density(model.dynamics)
    θ = model.formulation.potential_temperature
    bounded = model.advection.ρθ
    old_flux = model.closure_fields.scheme_scalar_flux.ρθ[1][2, 3, 1]
    # A sharp near-wall profile activates Oceananigans' rescaling limiter.
    profile = CenterField(grid)
    set!(profile, (x, y, z) -> z < 20 ? 299f0 : 301f0)
    set!(model.formulation.potential_temperature_density, ρ * profile)
    model.clock.time += 0.1f0
    model.clock.iteration += 1
    update_state!(model; compute_tendencies=false)
    limiter = bounded.bounds.limiter
    limiter_before = Array(interior(limiter))
    @test minimum(limiter_before) < 1f0
    for i in (2, 7), j in (3, 8), cell in (2, 3)
        lower = cell
        upper = cell + 1
        expected = bounded_tracer_flux_divergence_z(i, j, cell, grid,
            bounded, ρ, model.velocities.w, θ)
        measured = ℑzᵃᵃᶠ(i, j, upper, grid, ρ) *
                   Azᶜᶜᶠ(i, j, upper, grid) *
                   native_scalar_transport(i, j, upper, grid, bounded,
                                           model.velocities.w, θ) -
                   ℑzᵃᵃᶠ(i, j, lower, grid, ρ) *
                   Azᶜᶜᶠ(i, j, lower, grid) *
                   native_scalar_transport(i, j, lower, grid, bounded,
                                           model.velocities.w, θ)
        @test measured ≈ expected rtol=2f-5 atol=2f-5
    end
    @test model.closure_fields.previous_update_iteration[] == model.clock.iteration
    α = exponential_filter_weight(0.1f0, model.closure.filter_timescale)
    expected_filtered = (1 - α) * old_flux +
        α * native_scalar_transport(2, 3, 2, grid, bounded, model.velocities.w, θ)
    @test model.closure_fields.scheme_scalar_flux.ρθ[1][2, 3, 1] ≈
          expected_filtered rtol=2f-5
end

@testset "scheme-native configuration and covariance baseline" begin
    @test_throws ArgumentError SurfaceLayerDiffusivity(Float32; resolved_transport=:other)
    @test_throws ArgumentError SurfaceLayerDiffusivity(Float32;
        resolved_transport=:scheme_native, resolved_flux_factor=2)
    grid = RectilinearGrid(CPU(), Float32; size=(8, 8, 8), extent=(80, 80, 80))
    baseline = AtmosphereModel(grid; closure=SurfaceLayerDiffusivity(Float32),
                               advection=Centered(order=2))
    @test baseline.closure.advection === nothing
    @test baseline.closure.resolved_transport isa Val{:covariance}
    @test all(iszero, Array(interior(baseline.closure_fields.numerical_u_correction[1])))
    @test_throws ArgumentError AtmosphereModel(grid;
        closure=(SurfaceLayerDiffusivity(Float32; resolved_transport=:scheme_native),),
        advection=Centered(order=2))
end

@testset "Float32 centered covariance plus filtered correction" begin
    α = Float32(-expm1(-0.1 / 300))
    mean_c = Float32(300)
    mean_w = Float32(0.05)
    covariance = 0f0
    correction = 0f0
    reference_mean_c = Float64(mean_c)
    reference_mean_w = Float64(mean_w)
    reference_covariance = 0.0
    reference_correction = 0.0
    for n in 1:90000
        c = Float32(300 + 0.1sin(0.02n))
        w = Float32(0.05 + 0.02sin(0.02n + 0.4))
        numerical = Float32(0.001w * sin(0.005n))
        statistics = exponential_mean_and_covariance(mean_c, mean_w,
                                                       covariance, c, w, α)
        mean_c = statistics.mean_x
        mean_w = statistics.mean_y
        covariance = statistics.covariance
        correction = (1f0 - α) * correction + α * numerical
        δc = Float64(c) - reference_mean_c
        δw = Float64(w) - reference_mean_w
        reference_covariance = (1 - Float64(α)) *
                               (reference_covariance + Float64(α) * δc * δw)
        reference_mean_c += Float64(α) * δc
        reference_mean_w += Float64(α) * δw
        reference_correction = (1 - Float64(α)) * reference_correction +
                               Float64(α) * Float64(numerical)
    end
    @test sign(covariance) == sign(reference_covariance)
    @test isapprox(covariance + correction,
                   reference_covariance + reference_correction; rtol=3f-3)
end
