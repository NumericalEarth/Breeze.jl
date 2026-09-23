include(joinpath(@__DIR__, "setup.jl"))

using Test
using Adapt: adapt
import Oceananigans
using Breeze
using Breeze.TurbulenceClosures: momentum_surface_layer_properties,
    scalar_surface_layer_properties, obukhov_stability_properties,
    surface_layer_stability_functions, SurfaceLayerDiffusivityFields
using Breeze.AtmosphereModels: compute_tendencies!, prognostic_fields,
    update_completed_step_closure_state!
using Oceananigans.Grids: Face, znodes
using Oceananigans.Operators: ℑzᵃᵃᶠ
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: with_tracers

# A cooled surface: downward heat flux (W m⁻², routed onto ρθ) and a quadratic drag.
stable_drag_u(x, y, t, u, v) = -2e-3 * sqrt(u^2 + v^2) * u
stable_drag_v(x, y, t, u, v) = -2e-3 * sqrt(u^2 + v^2) * v

function stable_boundary_conditions(; heat_flux=-10.0, moisture=nothing)
    ρu = FieldBoundaryConditions(bottom=FluxBoundaryCondition(stable_drag_u,
                                                              field_dependencies=(:u, :v)))
    ρv = FieldBoundaryConditions(bottom=FluxBoundaryCondition(stable_drag_v,
                                                              field_dependencies=(:u, :v)))
    ρE = FieldBoundaryConditions(bottom=FluxBoundaryCondition(heat_flux))
    bcs = (; ρu, ρv, ρE)
    return isnothing(moisture) ? bcs :
           merge(bcs, (; ρqᵗ=FieldBoundaryConditions(bottom=FluxBoundaryCondition(moisture))))
end

stable_θ(x, y, z) = 265 + 0.01z + 0.02sin(x / 7) * exp(-z / 20)
stable_u(x, y, z) = 5 + 0.05z + 0.1cos(y / 9)

function stable_model(grid, λ; resolved_transport=:covariance, support=1, heat_flux=-10.0,
                      advection=nothing, boundary_conditions=stable_boundary_conditions(; heat_flux),
                      time_discretization=VerticallyImplicitTimeDiscretization(),
                      initial_w=0, kw...)
    closure = SurfaceLayerDiffusivity(time_discretization, Float64; stability_strength=λ,
                                      resolved_transport, support, filter_timescale=10,
                                      minimum_scalar_fluxes=(ρθ=1e-8,), kw...)
    model = isnothing(advection) ?
        AtmosphereModel(grid; closure, boundary_conditions, advection) :
        AtmosphereModel(grid; closure, boundary_conditions, momentum_advection=advection,
                        scalar_advection=advection)
    set!(model; enforce_mass_conservation=false, θ=stable_θ, u=stable_u, v=0, w=initial_w)
    return model
end

# Independent reference for the local inverse Obukhov length of column (i, j).
function reference_inverse_obukhov_length(model, i, j)
    fields = model.closure_fields
    κ = model.closure.von_karman_constant
    g = model.thermodynamic_constants.gravitational_acceleration
    θ₀ = model.dynamics.reference_state.potential_temperature
    stress = hypot(fields.surface_u_flux[i, j, 1], fields.surface_v_flux[i, j, 1])
    return -κ * g * fields.surface_scalar_flux.ρθ[i, j, 1] / (θ₀ * sqrt(stress)^3)
end

@testset "Stability correction construction and adaptation [$FT]" for FT in all_float_types()
    neutral = SurfaceLayerDiffusivity(FT)
    @test neutral.stability_strength === FT(0)
    @test neutral.momentum_stability_parameter === FT(4.8)
    @test neutral.scalar_stability_parameter === FT(7.8)

    closure = SurfaceLayerDiffusivity(FT; stability_strength=1.5,
                                      momentum_stability_parameter=5,
                                      scalar_stability_parameter=8,
                                      minimum_scalar_fluxes=(ρθ=1e-8,))
    @test closure.stability_strength === FT(1.5)
    @test isbitstype(typeof(closure))
    @test adapt(CPU(), closure) === closure
    reconstructed = with_tracers((:ρθ, :ρqᵛ), closure)
    for name in (:stability_strength, :momentum_stability_parameter, :scalar_stability_parameter)
        @test getproperty(reconstructed, name) === getproperty(closure, name)
    end
    shown = sprint(show, closure)
    @test occursin("stability_strength: 1.5", shown)
    @test occursin("momentum_stability_parameter: 5", shown)
    @test occursin("scalar_stability_parameter: 8", shown)

    for name in (:stability_strength, :momentum_stability_parameter, :scalar_stability_parameter)
        for value in (-1, -1e-100, Inf, NaN)
            @test_throws ArgumentError SurfaceLayerDiffusivity(FT; (name => value,)...)
        end
    end
    @test_throws ArgumentError SurfaceLayerDiffusivity(Float32; stability_strength=1e100)
end

@testset "Local Obukhov length and gradient functions [$FT]" for FT in all_float_types()
    closure = SurfaceLayerDiffusivity(FT; stability_strength=1, minimum_friction_velocity=1e-4)
    κ, g, θ₀ = FT(0.4), FT(9.81), FT(263.5)
    stress_u, stress_v = FT(-0.03), FT(0.04)   # u★² = 0.05
    u★ = sqrt(FT(0.05))
    properties(heat_flux; guard=FT(1e-8), u=stress_u, v=stress_v) =
        obukhov_stability_properties(u, v, heat_flux, guard, θ₀, g, closure)

    cooling = properties(FT(-0.02))
    expected = -κ * g * FT(-0.02) / (θ₀ * u★^3)
    @test cooling.inverse_obukhov_length ≈ expected rtol=10eps(FT)
    @test cooling.inverse_obukhov_length > 0
    @test cooling.stable && !cooling.unstable && cooling.state == 1

    heating = properties(FT(0.02))
    @test heating.inverse_obukhov_length ≈ -expected rtol=10eps(FT)
    @test heating.unstable && !heating.stable && heating.state == -1

    # Zero, guarded, and nonfinite heat flux and calm stress give the neutral limit.
    for inactive in (properties(FT(0)), properties(FT(-0.5e-8)), properties(FT(NaN)),
                     properties(FT(-Inf)), properties(FT(-0.02); u=FT(0), v=FT(0)),
                     properties(FT(-0.02); u=FT(NaN)))
        @test inactive.inverse_obukhov_length === zero(FT)
        @test inactive.state == 0
    end
    # The zero-flux limit is continuous: 1/L → 0 linearly in the heat flux.
    small = properties(FT(-1e-6)).inverse_obukhov_length
    @test small ≈ expected * FT(1e-6) / FT(0.02) rtol=100eps(FT)

    z = FT(12.5)
    φ = surface_layer_stability_functions(z, cooling.inverse_obukhov_length, closure)
    ζ = z * cooling.inverse_obukhov_length
    @test φ.momentum ≈ 1 + FT(4.8) * ζ rtol=10eps(FT)
    @test φ.scalar ≈ 1 + FT(7.8) * ζ rtol=10eps(FT)
    @test φ.scalar > φ.momentum > 1
    # Only the stable branch is defined: unstable and neutral columns use φ = 1.
    @test surface_layer_stability_functions(z, heating.inverse_obukhov_length, closure) ==
          (; momentum=one(FT), scalar=one(FT))
    @test surface_layer_stability_functions(z, zero(FT), closure) ==
          (; momentum=one(FT), scalar=one(FT))

    doubled = SurfaceLayerDiffusivity(FT; stability_strength=2)
    φ₂ = surface_layer_stability_functions(z, cooling.inverse_obukhov_length, doubled)
    @test φ₂.momentum - 1 ≈ 2 * (φ.momentum - 1) rtol=10eps(FT)
    @test φ₂.scalar - 1 ≈ 2 * (φ.scalar - 1) rtol=10eps(FT)

    # λ = 0 is exactly neutral, even at extreme stability; λ > 0 overflow suppresses exactly.
    neutral = SurfaceLayerDiffusivity(FT)
    for inverse_length in (cooling.inverse_obukhov_length, floatmax(FT), -floatmax(FT))
        @test surface_layer_stability_functions(z, inverse_length, neutral) ===
              (; momentum=one(FT), scalar=one(FT))
    end
    extreme = surface_layer_stability_functions(z, floatmax(FT), closure)
    @test extreme.momentum == Inf && extreme.scalar == Inf

    # Coefficients are divided by φ; deficits, validity, and the resolved projection are not.
    for resolved in FT.((-0.02, 0, 0.01))
        base = momentum_surface_layer_properties(resolved, FT(0), -stress_u, -stress_v,
                                                 z, one(FT), closure)
        corrected = momentum_surface_layer_properties(resolved, FT(0), -stress_u, -stress_v,
                                                      z, one(FT), closure, φ.momentum)
        @test corrected.viscosity ≈ base.viscosity / φ.momentum rtol=10eps(FT)
        @test corrected.deficit === base.deficit
        @test corrected.parallel_resolved_stress === base.parallel_resolved_stress
        @test corrected.valid === base.valid
        suppressed = momentum_surface_layer_properties(resolved, FT(0), -stress_u, -stress_v,
                                                       z, one(FT), closure, extreme.momentum)
        @test suppressed.viscosity === zero(FT)

        scalar_base = scalar_surface_layer_properties(resolved, FT(-0.02), u★, z, one(FT),
                                                      FT(1e-8), closure)
        scalar_corrected = scalar_surface_layer_properties(resolved, FT(-0.02), u★, z, one(FT),
                                                           FT(1e-8), closure, φ.scalar)
        @test scalar_corrected.diffusivity ≈ scalar_base.diffusivity / φ.scalar rtol=10eps(FT)
        @test scalar_corrected.deficit === scalar_base.deficit
        @test all(isfinite, (corrected.viscosity, scalar_corrected.diffusivity))
    end
end

@testset "Local stable Obukhov length in a CPU model" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(CPU(); size=(4, 4, 6), halo=(3, 3, 3), x=(0, 40), y=(0, 40), z=(0, 60))
    neutral = stable_model(grid, 0)
    stable = stable_model(grid, 1)
    zᶠ = znodes(grid, Face())
    face_height = zᶠ[2] - zᶠ[1]
    for model in (neutral, stable)
        fields = model.closure_fields
        @test fields isa SurfaceLayerDiffusivityFields
        for j in 1:grid.Ny, i in 1:grid.Nx
            inverse_length = reference_inverse_obukhov_length(model, i, j)
            @test inverse_length > 0
            @test fields.inverse_obukhov_length[i, j, 1] ≈ inverse_length rtol=1e-12
            @test fields.stability_state[i, j, 1] == 1
        end
    end
    # The stored filtered fluxes are density-consistent kinematic fluxes.
    surface_density = ℑzᵃᵃᶠ(1, 1, 1, grid, stable.dynamics.reference_state.density)
    cᵖᵈ = stable.thermodynamic_constants.dry_air.heat_capacity
    Π₀ = 1  # the reference surface potential temperature equals its temperature at p₀ = pˢᵗ
    @test stable.closure_fields.surface_scalar_flux.ρθ[1, 1, 1] * surface_density ≈
          -10 / (cᵖᵈ * Π₀) rtol=1e-2

    # Identical wall and filter state; only the coefficients carry 1/φ.
    for name in (:surface_u_flux, :surface_v_flux, :inverse_obukhov_length, :stability_state)
        @test interior(getproperty(stable.closure_fields, name)) ==
              interior(getproperty(neutral.closure_fields, name))
    end
    @test interior(stable.closure_fields.surface_scalar_flux.ρθ) ==
          interior(neutral.closure_fields.surface_scalar_flux.ρθ)
    for j in 1:grid.Ny, i in 1:grid.Nx
        ζ = face_height * stable.closure_fields.inverse_obukhov_length[i, j, 1]
        φᵐ = 1 + 4.8ζ
        φʰ = 1 + 7.8ζ
        @test stable.closure_fields.momentum_stability_function[1][i, j, 1] ≈ φᵐ
        @test stable.closure_fields.scalar_stability_function[1][i, j, 1] ≈ φʰ
        @test neutral.closure_fields.momentum_stability_function[1][i, j, 1] == 1
        @test neutral.closure_fields.scalar_stability_function[1][i, j, 1] == 1
        @test stable.closure_fields.Kᵘ[i, j, 2] ≈ neutral.closure_fields.Kᵘ[i, j, 2] / φᵐ
        @test stable.closure_fields.tupled_tracer_diffusivities.ρθ[i, j, 2] ≈
              neutral.closure_fields.tupled_tracer_diffusivities.ρθ[i, j, 2] / φʰ
        @test stable.closure_fields.Kᵘ[i, j, 2] > 0
    end
    @test interior(stable.closure_fields.momentum_deficit[1]) ==
          interior(neutral.closure_fields.momentum_deficit[1])

    # Zero heat flux: the local length is infinite (1/L = 0) and λ = 1 is exactly neutral.
    zero_bcs = stable_boundary_conditions(; heat_flux=0.0)
    calm_stable = stable_model(grid, 1; boundary_conditions=zero_bcs)
    calm_neutral = stable_model(grid, 0; boundary_conditions=zero_bcs)
    @test all(iszero, interior(calm_stable.closure_fields.inverse_obukhov_length))
    @test all(iszero, interior(calm_stable.closure_fields.stability_state))
    @test interior(calm_stable.closure_fields.Kᵘ) == interior(calm_neutral.closure_fields.Kᵘ)

    # An upward heat flux is recorded as unstable and is left uncorrected.
    heating_bcs = stable_boundary_conditions(; heat_flux=10.0)
    heated = stable_model(grid, 1; boundary_conditions=heating_bcs)
    heated_neutral = stable_model(grid, 0; boundary_conditions=heating_bcs)
    @test all(<(0), interior(heated.closure_fields.inverse_obukhov_length))
    @test all(==(-1), interior(heated.closure_fields.stability_state))
    @test interior(heated.closure_fields.Kᵘ) == interior(heated_neutral.closure_fields.Kᵘ)
    @test interior(heated.closure_fields.tupled_tracer_diffusivities.ρθ) ==
          interior(heated_neutral.closure_fields.tupled_tracer_diffusivities.ρθ)
end

@testset "Stability correction preserves native levels, wall flux, and conservation" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(CPU(); size=(4, 4, 32), halo=(3, 3, 3), x=(0, 40), y=(0, 40), z=(0, 400))
    for support in (1, 2)
        model = stable_model(grid, 1; support)
        viscosity = Array(interior(model.closure_fields.Kᵘ, 1, 1, :))
        diffusivity = Array(interior(model.closure_fields.tupled_tracer_diffusivities.ρθ, 1, 1, :))
        @test length(viscosity) == 33
        @test length(diffusivity) == 33
        active = support == 1 ? [2] : [2, 3]
        inactive = setdiff(1:33, active)
        @test all(>(0), viscosity[active])
        @test all(>(0), diffusivity[active])
        @test all(iszero, viscosity[inactive])
        @test all(iszero, diffusivity[inactive])
    end

    # Explicit closure tendencies redistribute momentum and heat within each column, leaving
    # the independently applied wall fluxes (identical in both models) unchanged.
    small_grid = RectilinearGrid(CPU(); size=(4, 4, 6), halo=(3, 3, 3), x=(0, 40), y=(0, 40), z=(0, 60))
    explicit = stable_model(small_grid, 1; time_discretization=ExplicitTimeDiscretization())
    control = AtmosphereModel(small_grid; closure=nothing, advection=nothing,
                              boundary_conditions=stable_boundary_conditions())
    set!(control; θ=stable_θ, u=stable_u, v=0, w=0)
    compute_tendencies!(explicit)
    compute_tendencies!(control)
    for name in (:ρu, :ρθ)
        difference = Array(interior(getproperty(explicit.timestepper.Gⁿ, name))) .-
                     Array(interior(getproperty(control.timestepper.Gⁿ, name)))
        @test maximum(abs, difference) > 0
        @test maximum(abs, sum(difference; dims=3)) ≤ 1e-10 * maximum(abs, difference)
    end

    # The same holds for the production vertically implicit operator.
    implicit = stable_model(small_grid, 1)
    implicit_control = AtmosphereModel(small_grid; closure=nothing, advection=nothing,
                                       boundary_conditions=stable_boundary_conditions())
    set!(implicit_control; θ=stable_θ, u=stable_u, v=0, w=0)
    Oceananigans.time_step!(implicit, 0.2)
    Oceananigans.time_step!(implicit_control, 0.2)
    ρθ_difference = Array(interior(prognostic_fields(implicit).ρθ)) .-
                    Array(interior(prognostic_fields(implicit_control).ρθ))
    @test maximum(abs, ρθ_difference) > 0
    @test sum(ρθ_difference) ≈ 0 atol=1e4 * eps(Float64) * 300
end

@testset "Stability correction with scheme-native transport" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(CPU(); size=(6, 6, 8), halo=(3, 3, 3), x=(0, 60), y=(0, 60), z=(0, 80))
    initial_w(x, y, z) = 0.02sin(x / 6) * z / 80
    neutral = stable_model(grid, 0; resolved_transport=:scheme_native, advection=WENO(order=5),
                           initial_w)
    stable = stable_model(grid, 1; resolved_transport=:scheme_native, advection=WENO(order=5),
                          initial_w)
    for model in (neutral, stable)
        model.clock.time += 0.5
        model.clock.iteration += 1
        update_state!(model; compute_tendencies=false)
    end
    fields = stable.closure_fields
    neutral_fields = neutral.closure_fields
    # Factor-one identity: the driving resolved flux is covariance plus numerical correction,
    # and the stability correction neither enters nor alters the native flux filters.
    @test stable.closure.resolved_flux_factor == 1
    for name in (:resolved_u_flux, :scheme_u_flux, :numerical_u_correction)
        @test interior(getproperty(fields, name)[1]) == interior(getproperty(neutral_fields, name)[1])
    end
    @test interior(fields.numerical_scalar_correction.ρθ[1]) ==
          interior(neutral_fields.numerical_scalar_correction.ρθ[1])
    @test maximum(abs, interior(fields.numerical_u_correction[1])) > 0
    for j in 1:grid.Ny, i in 1:grid.Nx
        φᵐ = fields.momentum_stability_function[1][i, j, 1]
        φʰ = fields.scalar_stability_function[1][i, j, 1]
        @test φᵐ > 1
        @test fields.Kᵘ[i, j, 2] ≈ neutral_fields.Kᵘ[i, j, 2] / φᵐ
        @test fields.tupled_tracer_diffusivities.ρθ[i, j, 2] ≈
              neutral_fields.tupled_tracer_diffusivities.ρθ[i, j, 2] / φʰ
        resolved = fields.resolved_u_flux[1][i, j, 1] + fields.numerical_u_correction[1][i, j, 1]
        surface_u = fields.surface_u_flux[i, j, 1]
        surface_v = fields.surface_v_flux[i, j, 1]
        resolved_v = fields.resolved_v_flux[1][i, j, 1] + fields.numerical_v_correction[1][i, j, 1]
        stress = hypot(surface_u, surface_v)
        parallel = (resolved * surface_u + resolved_v * surface_v) / stress
        z = znodes(grid, Face())[2]
        expected = 0.4 * sqrt(stress) * z * max(0, 1 - parallel / stress) / φᵐ
        @test fields.Kᵘ[i, j, 2] ≈ expected
    end
end

@testset "Stability correction checkpoint pickup" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(CPU(); size=(4, 4, 6), halo=(3, 3, 3), x=(0, 40), y=(0, 40), z=(0, 60))
    for resolved_transport in (:covariance, :scheme_native)
        advection = resolved_transport === :scheme_native ? WENO(order=5) : nothing
        continued = stable_model(grid, 1; resolved_transport, advection)
        Oceananigans.time_step!(continued, 0.2)
        state = deepcopy(Oceananigans.prognostic_state(continued))
        closure_state = state.closure_fields
        for name in (:inverse_obukhov_length, :stability_state,
                     :momentum_stability_function, :scalar_stability_function)
            @test hasproperty(closure_state, name)
        end
        restarted = stable_model(grid, 1; resolved_transport, advection)
        Oceananigans.restore_prognostic_state!(restarted, state)
        update_state!(restarted)
        for n in 1:2
            Oceananigans.time_step!(continued, 0.2)
            Oceananigans.time_step!(restarted, 0.2)
        end
        @test Oceananigans.prognostic_state(continued) == Oceananigans.prognostic_state(restarted)
        @test all(>(0), interior(continued.closure_fields.inverse_obukhov_length))

        # A checkpoint written before the stability diagnostics existed restores the filters;
        # the diagnostics are recomputed from the restored filtered wall fluxes.
        legacy_names = Tuple(name for name in keys(closure_state)
                             if !(name in (:inverse_obukhov_length, :stability_state,
                                           :momentum_stability_function,
                                           :scalar_stability_function)))
        legacy_state = merge(state, (; closure_fields=NamedTuple{legacy_names}(
            map(name -> getproperty(closure_state, name), legacy_names))))
        legacy = stable_model(grid, 1; resolved_transport, advection)
        reference = stable_model(grid, 1; resolved_transport, advection)
        Oceananigans.restore_prognostic_state!(legacy, legacy_state)
        Oceananigans.restore_prognostic_state!(reference, state)
        update_state!(legacy)
        update_state!(reference)
        @test interior(legacy.closure_fields.Kᵘ) == interior(reference.closure_fields.Kᵘ)
        @test interior(legacy.closure_fields.inverse_obukhov_length) ==
              interior(reference.closure_fields.inverse_obukhov_length)
    end
end

@testset "Stability correction rejects unsupported formulations" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(CPU(); size=(4, 4, 6), halo=(3, 3, 3), x=(0, 40), y=(0, 40), z=(0, 60))
    stable_closure = SurfaceLayerDiffusivity(Float64; stability_strength=1,
                                             minimum_scalar_fluxes=(ρθ=1e-8,))
    neutral_closure = SurfaceLayerDiffusivity(Float64; minimum_scalar_fluxes=(ρθ=1e-8,))
    bcs = stable_boundary_conditions()

    # Moist: condensate-bearing microphysics, or a wall moisture flux in a dry-microphysics model.
    @test_throws "dry models" AtmosphereModel(grid; closure=stable_closure, advection=nothing,
        boundary_conditions=bcs, microphysics=SaturationAdjustment())
    moist_bcs = stable_boundary_conditions(; moisture=1e-5)
    @test_throws "no wall moisture flux" AtmosphereModel(grid; closure=stable_closure,
        advection=nothing, boundary_conditions=moist_bcs)
    # No active heat-flux guard means no defined local Obukhov length.
    @test_throws "active ρθ wall flux" AtmosphereModel(grid; advection=nothing, boundary_conditions=bcs,
        closure=SurfaceLayerDiffusivity(Float64; stability_strength=1))
    # Compressible dynamics needs its own buoyancy-flux definition.
    function compressible_stable_model()
        dynamics = CompressibleDynamics(; reference_potential_temperature=265)
        model = AtmosphereModel(grid; dynamics, closure=stable_closure, advection=nothing,
                                boundary_conditions=bcs)
        set!(model; θ=stable_θ, u=stable_u)
        return model
    end
    @test_throws "AnelasticDynamics only" compressible_stable_model()

    # The neutral closure remains available for all of them.
    for (microphysics, boundary_conditions) in ((SaturationAdjustment(), bcs), (nothing, moist_bcs))
        model = AtmosphereModel(grid; closure=neutral_closure, advection=nothing,
                                boundary_conditions, microphysics)
        @test model.closure.stability_strength == 0
    end
end

@testset "Stable CPU integration" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(CPU(); size=(6, 6, 8), halo=(3, 3, 3), x=(0, 60), y=(0, 60), z=(0, 80))
    models = map((0, 1)) do λ
        model = stable_model(grid, λ; resolved_transport=:scheme_native, advection=WENO(order=5))
        for n in 1:10
            Oceananigans.time_step!(model, 0.25)
        end
        model
    end
    neutral, stable = models
    fields = stable.closure_fields
    @test stable.clock.iteration == 10
    @test fields.previous_update_iteration[] == 10
    for field in (stable.velocities.u, stable.velocities.w, prognostic_fields(stable).ρθ,
                  fields.Kᵘ, fields.tupled_tracer_diffusivities.ρθ, fields.inverse_obukhov_length)
        @test all(isfinite, interior(field))
    end
    @test all(==(1), interior(fields.stability_state))
    @test all(>(1), interior(fields.momentum_stability_function[1]))
    @test all(>(1), interior(fields.scalar_stability_function[1]))
    # Weaker surface-layer mixing retains more near-wall shear in the stable run.
    Δu(model) = sum(interior(model.velocities.u, :, :, 2) .- interior(model.velocities.u, :, :, 1))
    @test Δu(stable) > Δu(neutral)
    @test maximum(interior(fields.Kᵘ)) < maximum(interior(neutral.closure_fields.Kᵘ))
end
