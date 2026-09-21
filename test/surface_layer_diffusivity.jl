include(joinpath(@__DIR__, "setup.jl"))

using Test
using Adapt: adapt
import Oceananigans
using Breeze
using Breeze.TurbulenceClosures: exponential_filter_weight, exponential_mean_and_covariance,
    momentum_surface_layer_properties, scalar_surface_layer_properties, support_weight,
    SurfaceLayerDiffusivityFields, SurfaceLayerDiffusivityDeviceFields
using Breeze.AtmosphereModels: compute_tendencies!, dynamics_thermodynamic_fields,
    update_completed_step_closure_state!
using Oceananigans: fields
using Oceananigans.BoundaryConditions: getbc
using Oceananigans.Fields: location
using Oceananigans.Grids: Center, Face, znodes
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization

@testset "SurfaceLayerDiffusivity construction [$(FT)]" for FT in test_float_types()
    closure = SurfaceLayerDiffusivity(FT;
        filter_timescale=100,
        minimum_scalar_fluxes=(ρθ=1e-8, ρqᵗ=1e-12),
        support=2)
    @test closure.filter_timescale === FT(100)
    @test closure.minimum_scalar_fluxes.ρθ === FT(1e-8)
    @test closure.minimum_scalar_fluxes.ρqᵗ === FT(1e-12)
    @test closure.support == 2
    @test summary(closure) == "SurfaceLayerDiffusivity{VerticallyImplicitTimeDiscretization}"

    explicit = SurfaceLayerDiffusivity(ExplicitTimeDiscretization(), FT)
    @test summary(explicit) == "SurfaceLayerDiffusivity{ExplicitTimeDiscretization}"

    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; filter_timescale=0)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; filter_timescale=Inf)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; von_karman_constant=NaN)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; turbulent_prandtl_number=0)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; minimum_friction_velocity=-1)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; minimum_scalar_fluxes=(ρθ=-1,))
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; support=3)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; maximum_viscosity=-1)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; maximum_viscosity=NaN)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; maximum_diffusivity=-1)
    @test_throws ArgumentError SurfaceLayerDiffusivity(FT; maximum_diffusivity=NaN)
end

@testset "SurfaceLayerDiffusivity device closure fields" begin
    if CUDA.functional()
        Oceananigans.defaults.FloatType = Float32
        grid = RectilinearGrid(GPU(); size=(4, 4, 4), extent=(50, 50, 50))
        boundary_conditions = (;
            ρu=FieldBoundaryConditions(bottom=FluxBoundaryCondition(-0.04f0)),
            ρv=FieldBoundaryConditions(bottom=FluxBoundaryCondition(0f0)))
        closure = SurfaceLayerDiffusivity(Float32;
            support=2, minimum_scalar_fluxes=(ρθ=1f-8,))
        model = AtmosphereModel(grid; closure, boundary_conditions, advection=nothing)
        set!(model; θ=300, u=1, v=0, w=0)
        host_fields = model.closure_fields
        device_fields = CUDA.cudaconvert(host_fields)
        @test device_fields isa SurfaceLayerDiffusivityDeviceFields
        @test isbitstype(typeof(device_fields))
        @test propertynames(device_fields) == (:Kᵘ, :tupled_tracer_diffusivities)
        @test host_fields.previous_update_time isa Base.RefValue
        @test host_fields.previous_update_iteration isa Base.RefValue
        Oceananigans.time_step!(model, 0.1f0)
        @test all(isfinite, Array(interior(host_fields.Kᵘ)))
    end
end

@testset "SurfaceLayerDiffusivity CPU model integration" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(CPU(); size=(4, 4, 4), extent=(50, 50, 50))
    ρu_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(-0.04))
    ρv_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(0.0))
    boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs)

    closure = SurfaceLayerDiffusivity(Float64)
    model = AtmosphereModel(grid; closure, boundary_conditions, advection=nothing)
    set!(model; θ=300, u=(x, y, z) -> z / 10, v=0, w=0)
    @test model.closure_fields isa SurfaceLayerDiffusivityFields
    kernel_fields = adapt(CPU(), model.closure_fields)
    @test kernel_fields isa SurfaceLayerDiffusivityDeviceFields
    @test propertynames(kernel_fields) == (:Kᵘ, :tupled_tracer_diffusivities)
    @test kernel_fields.Kᵘ[1, 1, 2] == model.closure_fields.Kᵘ[1, 1, 2]
    @test keys(kernel_fields.tupled_tracer_diffusivities) ==
          keys(model.closure_fields.tupled_tracer_diffusivities)
    @test model.closure_fields.previous_update_time isa Base.RefValue
    @test model.closure_fields.previous_update_iteration isa Base.RefValue

    viscosity = Array(interior(model.closure_fields.Kᵘ, 1, 1, :))
    @test location(model.closure_fields.Kᵘ) === (Center, Center, Face)
    @test length(viscosity) == grid.Nz + 1
    @test viscosity[2] > 0
    @test viscosity[[1, 3, 4, 5]] == zeros(4)
    @test model.closure_fields.momentum_active[1][1, 1, 1] == 1
    @test model.closure_fields.momentum_active[2][1, 1, 1] == 0
    @test model.closure_fields.surface_u_flux[1, 1, 1] < 0
    surface_density = ℑzᵃᵃᶠ(1, 1, 1, grid, model.dynamics.reference_state.density)
    @test model.closure_fields.surface_u_flux[1, 1, 1] * surface_density ≈ -0.04

    # Neutral manufactured log profile. The baseline deliberately uses physical face height,
    # so the discrete stress differs from u★² by z_f / logarithmic_mean(z₁, z₂).
    κ = closure.von_karman_constant
    u★ = sqrt(abs(model.closure_fields.surface_u_flux[1, 1, 1]))
    zᶜ = znodes(grid, Center())
    zᶠ = znodes(grid, Face())
    hᶜ = zᶜ .- zᶠ[1]
    hᶠ = zᶠ .- zᶠ[1]
    log_model = AtmosphereModel(grid; closure, boundary_conditions, advection=nothing)
    set!(log_model; θ=300,
         u=(x, y, z) -> u★ / κ * log(max(z - zᶠ[1], hᶜ[1])), v=0, w=0)
    log_viscosity = log_model.closure_fields.Kᵘ[1, 1, 2]
    discrete_gradient = u★ / κ * log(hᶜ[2] / hᶜ[1]) / (hᶜ[2] - hᶜ[1])
    modeled_log_stress = -log_viscosity * discrete_gradient
    @test log_viscosity ≈ κ * u★ * hᶠ[2]
    @test modeled_log_stress ≈
          -u★^2 * hᶠ[2] * log(hᶜ[2] / hᶜ[1]) / (hᶜ[2] - hᶜ[1])

    scalar_boundary_conditions = merge(boundary_conditions, (;
        ρE=FieldBoundaryConditions(bottom=FluxBoundaryCondition(-0.1))))
    scalar_closure = SurfaceLayerDiffusivity(Float64;
        minimum_scalar_fluxes=(ρθ=1e-8,), maximum_diffusivity=0.2)
    scalar_model = AtmosphereModel(grid; closure=scalar_closure,
                                   boundary_conditions=scalar_boundary_conditions,
                                   advection=nothing)
    set!(scalar_model; θ=300, u=1, v=0, w=0)
    θ_diffusivity = scalar_model.closure_fields.tupled_tracer_diffusivities.ρθ
    vapor_diffusivity = scalar_model.closure_fields.tupled_tracer_diffusivities.ρqᵛ
    @test location(θ_diffusivity) === (Center, Center, Face)
    @test θ_diffusivity[1, 1, 2] == 0.2
    @test scalar_model.closure_fields.scalar_active.ρθ[1][1, 1, 1] == 1
    @test scalar_model.closure_fields.diffusivity_cap_active.ρθ[1][1, 1, 1] == 1
    @test all(iszero, interior(vapor_diffusivity))
    @test scalar_model.closure_fields.scalar_active.ρqᵛ[1][1, 1, 1] == 0
    scalar_surface_density = ℑzᵃᵃᶠ(1, 1, 1, grid,
                                    scalar_model.dynamics.reference_state.density)
    materialized_θ_flux = getbc(
        scalar_model.closure_fields.scalar_boundary_conditions.ρθ,
        1, 1, grid, scalar_model.clock, fields(scalar_model),
        dynamics_thermodynamic_fields(scalar_model.dynamics))
    @test scalar_model.closure_fields.surface_scalar_flux.ρθ[1, 1, 1] *
          scalar_surface_density ≈ materialized_θ_flux

    two_face = SurfaceLayerDiffusivity(Float64; support=2)
    two_face_model = AtmosphereModel(grid; closure=two_face, boundary_conditions,
                                     advection=nothing)
    set!(two_face_model; θ=300, u=(x, y, z) -> z / 10, v=0, w=0)
    two_face_viscosity = Array(interior(two_face_model.closure_fields.Kᵘ, 1, 1, :))
    @test two_face_viscosity[2] > 0
    @test two_face_viscosity[3] > 0
    @test two_face_viscosity[2] ≈ two_face_viscosity[3]
    @test two_face_viscosity[[1, 4, 5]] == zeros(3)

    capped = SurfaceLayerDiffusivity(Float64; maximum_viscosity=0.1)
    capped_model = AtmosphereModel(grid; closure=capped, boundary_conditions,
                                   advection=nothing)
    set!(capped_model; θ=300, u=(x, y, z) -> z / 10, v=0, w=0)
    @test capped_model.closure_fields.Kᵘ[1, 1, 2] == 0.1
    @test capped_model.closure_fields.viscosity_cap_active[1][1, 1, 1] == 1

    # Only one physical-time update is allowed per completed iteration. Changing the sampled
    # state and calling the hook again at the same iteration must leave the filter unchanged.
    closure_fields = model.closure_fields
    model.clock.time = 10
    model.clock.iteration = 1
    set!(model.velocities.u, (x, y, z) -> x / 10)
    set!(model.velocities.w, (x, y, z) -> x / 20)
    update_completed_step_closure_state!(closure_fields, model.closure, model)
    once = copy(Array(interior(closure_fields.u_mean[1])))
    set!(model.velocities.u, 100)
    update_completed_step_closure_state!(closure_fields, model.closure, model)
    @test Array(interior(closure_fields.u_mean[1])) == once
    @test closure_fields.previous_update_time[] == 10
    @test closure_fields.previous_update_iteration[] == 1

    # The stable centered recurrence reproduces the exponentially weighted covariance.
    # This direct two-sample reference also checks variable-state collocation for a
    # horizontally uniform field.
    covariance_closure = SurfaceLayerDiffusivity(Float64; filter_timescale=100)
    covariance_model = AtmosphereModel(grid; closure=covariance_closure,
                                       boundary_conditions, advection=nothing)
    set!(covariance_model; θ=300, u=1, v=0, w=0)
    covariance_model.clock.time = 100
    covariance_model.clock.iteration = 1
    set!(covariance_model.velocities.u, 3)
    set!(covariance_model.velocities.w, 2)
    update_completed_step_closure_state!(covariance_model.closure_fields,
                                         covariance_model.closure, covariance_model)
    α = 1 - exp(-1)
    expected_u = (1 - α) * 1 + α * 3
    expected_w = α * 2
    expected_uw = α * 6
    @test covariance_model.closure_fields.u_mean[1][1, 1, 1] ≈ expected_u
    @test covariance_model.closure_fields.w_mean[1][1, 1, 1] ≈ expected_w
    @test covariance_model.closure_fields.uw_product_mean[1][1, 1, 1] ≈ expected_uw
    @test covariance_model.closure_fields.resolved_u_flux[1][1, 1, 1] ≈
          expected_uw - expected_u * expected_w

    # A time-dependent materialized wall flux is sampled once at the accepted end of an
    # actual SSPRK3 step. Intermediate stages retain iteration zero and cannot shorten T.
    timed_flux(x, y, t) = -0.01 * (1 + t)
    timed_boundary_conditions = (
        ρu=FieldBoundaryConditions(bottom=FluxBoundaryCondition(timed_flux)),
        ρv=FieldBoundaryConditions(bottom=FluxBoundaryCondition(0.0)))
    timed_closure = SurfaceLayerDiffusivity(Float64; filter_timescale=0.2)
    timed_model = AtmosphereModel(grid; closure=timed_closure,
                                  boundary_conditions=timed_boundary_conditions,
                                  advection=nothing)
    set!(timed_model; θ=300, u=1, v=0, w=0)
    initial_filtered_flux = timed_model.closure_fields.surface_u_flux[1, 1, 1]
    Oceananigans.time_step!(timed_model, 0.2)
    expected_filtered_flux = exp(-1) * initial_filtered_flux +
                             (1 - exp(-1)) * 1.2initial_filtered_flux
    @test timed_model.clock.time == 0.2
    @test timed_model.clock.iteration == 1
    @test timed_model.closure_fields.previous_update_time[] == 0.2
    @test timed_model.closure_fields.previous_update_iteration[] == 1
    @test timed_model.closure_fields.surface_u_flux[1, 1, 1] ≈ expected_filtered_flux

    # Exercise the production-default vertically implicit operator. With identical wall fluxes,
    # the SLD-control difference is an interior redistribution and conserves column momentum.
    implicit_model = AtmosphereModel(grid; closure, boundary_conditions, advection=nothing)
    implicit_control = AtmosphereModel(grid; closure=nothing, boundary_conditions,
                                       advection=nothing)
    nonlinear_u(x, y, z) = (z - zᶠ[1])^2 / 100
    set!(implicit_model; θ=300, u=nonlinear_u, v=0, w=0)
    set!(implicit_control; θ=300, u=nonlinear_u, v=0, w=0)
    Oceananigans.time_step!(implicit_model, 0.2)
    Oceananigans.time_step!(implicit_control, 0.2)
    implicit_difference = Array(interior(implicit_model.momentum.ρu)) .-
                          Array(interior(implicit_control.momentum.ρu))
    @test sum(implicit_difference) ≈ 0 atol=1000eps(Float64)
    @test maximum(abs, implicit_difference) > 0

    scalar_implicit_model = AtmosphereModel(
        grid; closure=scalar_closure, boundary_conditions=scalar_boundary_conditions,
        advection=nothing)
    scalar_implicit_control = AtmosphereModel(
        grid; closure=nothing, boundary_conditions=scalar_boundary_conditions,
        advection=nothing)
    nonlinear_θ(x, y, z) = 300 + (z - zᶠ[1])^2 / 1000
    set!(scalar_implicit_model; θ=nonlinear_θ, u=1, v=0, w=0)
    set!(scalar_implicit_control; θ=nonlinear_θ, u=1, v=0, w=0)
    Oceananigans.time_step!(scalar_implicit_model, 0.2)
    Oceananigans.time_step!(scalar_implicit_control, 0.2)
    scalar_ρθ = Breeze.AtmosphereModels.prognostic_fields(scalar_implicit_model).ρθ
    control_ρθ = Breeze.AtmosphereModels.prognostic_fields(scalar_implicit_control).ρθ
    scalar_implicit_difference = Array(interior(scalar_ρθ)) .- Array(interior(control_ρθ))
    @test sum(scalar_implicit_difference) ≈ 0 atol=1000eps(Float64)
    @test maximum(abs, scalar_implicit_difference) > 0

    # Restore the complete model state, refresh only derived quantities at the same iteration,
    # then continue both branches. Covariances and the filter clock must evolve identically.
    continued_model = AtmosphereModel(grid; closure, boundary_conditions, advection=nothing)
    set!(continued_model; θ=300, u=nonlinear_u, v=0, w=0)
    Oceananigans.time_step!(continued_model, 0.2)
    checkpoint_state = deepcopy(Oceananigans.prognostic_state(continued_model))
    restarted_model = AtmosphereModel(grid; closure, boundary_conditions, advection=nothing)
    set!(restarted_model; θ=300, u=nonlinear_u, v=0, w=0)
    Oceananigans.restore_prognostic_state!(restarted_model, checkpoint_state)
    Oceananigans.TimeSteppers.update_state!(continued_model)
    Oceananigans.TimeSteppers.update_state!(restarted_model)
    Oceananigans.time_step!(continued_model, 0.2)
    Oceananigans.time_step!(restarted_model, 0.2)
    @test Oceananigans.prognostic_state(continued_model) ==
          Oceananigans.prognostic_state(restarted_model)

    # The full filter state and update clock are checkpoint state, unlike diagnostic-only
    # diffusivity fields that can be reconstructed from an instantaneous model state.
    state = deepcopy(Oceananigans.prognostic_state(closure_fields))
    saved_mean = copy(Array(interior(closure_fields.u_mean[1])))
    set!(closure_fields.u_mean[1], -999)
    closure_fields.previous_update_time[] = -1
    closure_fields.previous_update_iteration[] = -1
    Oceananigans.restore_prognostic_state!(closure_fields, state)
    @test Array(interior(closure_fields.u_mean[1])) == saved_mean
    @test closure_fields.previous_update_time[] == 10
    @test closure_fields.previous_update_iteration[] == 1

    closure_restarted_model = AtmosphereModel(grid; closure, boundary_conditions, advection=nothing)
    set!(closure_restarted_model; θ=300, u=(x, y, z) -> z / 10, v=0, w=0)
    Oceananigans.restore_prognostic_state!(closure_restarted_model.closure_fields, state)
    for evolved_model in (model, closure_restarted_model)
        evolved_model.clock.time = 20
        evolved_model.clock.iteration = 2
        set!(evolved_model.velocities.u, 4)
        set!(evolved_model.velocities.w, -1)
        update_completed_step_closure_state!(evolved_model.closure_fields,
                                             evolved_model.closure, evolved_model)
    end
    @test Oceananigans.prognostic_state(model.closure_fields) ==
          Oceananigans.prognostic_state(closure_restarted_model.closure_fields)

    # Interior closure flux is conservative and does not alter the independently applied wall
    # momentum flux: subtracting an otherwise identical no-closure tendency has zero column sum.
    # Use the explicit form here because vertically implicit fluxes are applied by the solver,
    # rather than stored in Gⁿ.
    explicit_closure = SurfaceLayerDiffusivity(ExplicitTimeDiscretization(), Float64)
    explicit_model = AtmosphereModel(grid; closure=explicit_closure, boundary_conditions,
                                     advection=nothing)
    control = AtmosphereModel(grid; closure=nothing, boundary_conditions, advection=nothing)
    set!(explicit_model; θ=300, u=(x, y, z) -> z / 10, v=0, w=0)
    set!(control; θ=300, u=(x, y, z) -> z / 10, v=0, w=0)
    compute_tendencies!(explicit_model)
    compute_tendencies!(control)
    difference = Array(interior(explicit_model.timestepper.Gⁿ.ρu)) .-
                 Array(interior(control.timestepper.Gⁿ.ρu))
    @test sum(difference) ≈ 0 atol=100eps(Float64)
    @test maximum(abs, difference) > 0

    periodic_grid = RectilinearGrid(CPU(); size=4, z=(0, 50),
                                    topology=(Flat, Flat, Periodic))
    @test_throws ArgumentError AtmosphereModel(periodic_grid; closure)

    short_grid = RectilinearGrid(CPU(); size=2, z=(0, 50),
                                 topology=(Flat, Flat, Bounded))
    @test_throws ArgumentError AtmosphereModel(short_grid;
        closure=SurfaceLayerDiffusivity(Float64; support=2))

    @test_throws ArgumentError AtmosphereModel(grid; formulation=:StaticEnergy, closure,
                                               boundary_conditions, advection=nothing)
end

@testset "Exponential filter and support" begin
    @test exponential_filter_weight(0.0, 100.0) == 0
    @test exponential_filter_weight(100.0, 100.0) ≈ 1 - exp(-1)
    α₁ = exponential_filter_weight(30.0, 100.0)
    α₂ = exponential_filter_weight(70.0, 100.0)
    filtered = (1 - α₂) * ((1 - α₁) * 2 + α₁ * 5) + α₂ * 5
    @test filtered ≈ 2exp(-1) + 5(1 - exp(-1))

    @test support_weight(1, 1) == 0
    @test support_weight(2, 1) == 1
    @test support_weight(3, 1) == 0
    @test support_weight(2, 2) == 1
    @test support_weight(3, 2) == 0.5
    @test support_weight(4, 2) == 0
end

@testset "Stable Float32 exponential covariance" begin
    α = exponential_filter_weight(0.1f0, 300f0)
    x_mean = 300f0
    shifted_x_mean = x_mean + 512f0
    y_mean = 0.05f0
    covariance = 0f0
    shifted_covariance = 0f0
    reference_x_mean = Float64(x_mean)
    reference_shifted_x_mean = Float64(shifted_x_mean)
    reference_y_mean = Float64(y_mean)
    reference_covariance = 0.0
    reference_shifted_covariance = 0.0
    reference_α = Float64(α)

    for n in 1:90000
        time = 0.1n
        x = Float32(300 + 0.1sin(2π * time / 37))
        shifted_x = x + 512f0
        y = Float32(0.05 + 0.02sin(2π * time / 37 + 0.4))

        statistics = exponential_mean_and_covariance(
            x_mean, y_mean, covariance, x, y, α)
        shifted_statistics = exponential_mean_and_covariance(
            shifted_x_mean, y_mean, shifted_covariance, shifted_x, y, α)
        x_mean = statistics.mean_x
        shifted_x_mean = shifted_statistics.mean_x
        y_mean = statistics.mean_y
        covariance = statistics.covariance
        shifted_covariance = shifted_statistics.covariance

        reference_x = Float64(x)
        reference_shifted_x = Float64(shifted_x)
        reference_y = Float64(y)
        δx = reference_x - reference_x_mean
        shifted_δx = reference_shifted_x - reference_shifted_x_mean
        δy = reference_y - reference_y_mean
        reference_covariance = (1 - reference_α) *
            (reference_covariance + reference_α * δx * δy)
        reference_shifted_covariance = (1 - reference_α) *
            (reference_shifted_covariance + reference_α * shifted_δx * δy)
        reference_x_mean += reference_α * δx
        reference_shifted_x_mean += reference_α * shifted_δx
        reference_y_mean += reference_α * δy
    end

    @test covariance isa Float32
    @test covariance > 0
    @test covariance ≈ reference_covariance rtol=5e-4
    @test shifted_covariance ≈ reference_shifted_covariance rtol=1e-3
    @test shifted_covariance ≈ covariance rtol=1e-3
end

@testset "Momentum deficit constitutive law" begin
    closure = SurfaceLayerDiffusivity(Float64; minimum_friction_velocity=1e-6)
    no_resolved = momentum_surface_layer_properties(0.0, 0.0, -0.04, 0.0,
                                                    12.5, 1.0, closure)
    @test no_resolved.valid
    @test no_resolved.friction_velocity ≈ 0.2
    @test no_resolved.deficit ≈ 1
    @test no_resolved.viscosity ≈ 1

    partial = momentum_surface_layer_properties(-0.02, 0.0, -0.04, 0.0,
                                                12.5, 1.0, closure)
    @test partial.deficit ≈ 0.5
    @test partial.viscosity ≈ 0.5

    full = momentum_surface_layer_properties(-0.04, 0.0, -0.04, 0.0,
                                             12.5, 1.0, closure)
    over = momentum_surface_layer_properties(-0.08, 0.0, -0.04, 0.0,
                                             12.5, 1.0, closure)
    @test full.viscosity == 0
    @test over.viscosity == 0

    transverse = momentum_surface_layer_properties(0.0, -0.02, -0.04, 0.0,
                                                   12.5, 1.0, closure)
    @test transverse.deficit ≈ 1
    @test transverse.transverse_resolved_stress ≈ 0.02

    countergradient = momentum_surface_layer_properties(0.02, 0.0, -0.04, 0.0,
                                                        12.5, 1.0, closure)
    @test countergradient.deficit ≈ 1.5
    @test countergradient.viscosity ≈ 1.5

    calm = momentum_surface_layer_properties(0.0, 0.0, 0.0, 0.0,
                                             12.5, 1.0, closure)
    @test !calm.valid
    @test calm.viscosity == 0
end

@testset "Signed scalar deficit and guards" begin
    closure = SurfaceLayerDiffusivity(Float64; minimum_friction_velocity=1e-6)
    for surface_flux in (-0.1, 0.1)
        partial = scalar_surface_layer_properties(surface_flux / 2, surface_flux, 0.2,
                                                  12.5, 1.0, 1e-8, closure)
        @test partial.valid
        @test partial.deficit ≈ 0.5
        @test partial.diffusivity ≈ 0.5

        opposite = scalar_surface_layer_properties(-surface_flux, surface_flux, 0.2,
                                                   12.5, 1.0, 1e-8, closure)
        @test opposite.deficit ≈ 2
        @test opposite.diffusivity ≈ 2
    end

    zero_flux = scalar_surface_layer_properties(0.0, 0.0, 0.2,
                                                12.5, 1.0, 1e-8, closure)
    tiny_flux = scalar_surface_layer_properties(0.0, 0.5e-8, 0.2,
                                                12.5, 1.0, 1e-8, closure)
    @test !zero_flux.valid
    @test !tiny_flux.valid
    @test zero_flux.diffusivity == 0
    @test tiny_flux.diffusivity == 0

    capped_closure = SurfaceLayerDiffusivity(Float64; maximum_diffusivity=0.25)
    capped = scalar_surface_layer_properties(0.0, 0.1, 0.2,
                                             12.5, 1.0, 1e-8, capped_closure)
    @test capped.diffusivity == 0.25
end

@testset "Float32 guard and cap arithmetic" begin
    closure = SurfaceLayerDiffusivity(Float32;
        minimum_friction_velocity=1f-4,
        maximum_viscosity=0.3f0,
        maximum_diffusivity=0.2f0)
    inactive_momentum = momentum_surface_layer_properties(
        0f0, 0f0, -0.5f-8, 0f0, 12.5f0, 1f0, closure)
    capped_momentum = momentum_surface_layer_properties(
        0f0, 0f0, -0.04f0, 0f0, 12.5f0, 1f0, closure)
    inactive_scalar = scalar_surface_layer_properties(
        0f0, 1f-8, 0.2f0, 12.5f0, 1f0, 1f-8, closure)
    capped_scalar = scalar_surface_layer_properties(
        0f0, -0.1f0, 0.2f0, 12.5f0, 1f0, 1f-8, closure)
    @test !inactive_momentum.valid
    @test inactive_momentum.viscosity === 0f0
    @test capped_momentum.viscosity === 0.3f0
    @test capped_momentum.cap_active
    @test !inactive_scalar.valid
    @test inactive_scalar.diffusivity === 0f0
    @test capped_scalar.diffusivity === 0.2f0
    @test capped_scalar.cap_active
    @test all(isfinite, (inactive_momentum.viscosity, capped_momentum.viscosity,
                         inactive_scalar.diffusivity, capped_scalar.diffusivity))
end
