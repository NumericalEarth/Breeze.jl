using Breeze
using Breeze: SpecificForcing
using Breeze.AtmosphereModels: is_density_tendency_forcing
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.Forcings: MultipleForcings, Relaxation
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: update_state!
using Test

#####
##### Construction and dispatch
#####

@testset "SpecificForcing construction [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # Specific key wraps user value in SpecificForcing and stores under the ρ-key
    model = AtmosphereModel(grid; forcing=(; θ=Returns(FT(-1e-5))))
    @test haskey(model.forcing, :ρθ)
    @test model.forcing.ρθ isa SpecificForcing
    @test !haskey(model.forcing, :θ)

    # Density key keeps current behavior (no wrap)
    model = AtmosphereModel(grid; forcing=(; ρθ=Returns(FT(-1e-5))))
    @test !(model.forcing.ρθ isa SpecificForcing)

    # Explicit SpecificForcing under a ρ-key materializes correctly
    sf = SpecificForcing(Returns(FT(-1e-5)))
    model = AtmosphereModel(grid; forcing=(; ρθ=sf))
    @test model.forcing.ρθ isa SpecificForcing
end

@testset "Mixed specific + density keys merge into MultipleForcings [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    model = AtmosphereModel(grid; forcing=(; θ=Returns(FT(-1e-5)),
                                             ρθ=Returns(FT(-1e-5))))
    @test model.forcing.ρθ isa MultipleForcings
    @test length(model.forcing.ρθ.forcings) == 2
end

@testset "Tuple under specific key wraps each element [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    f1 = Returns(FT(-1e-5))
    f2 = Returns(FT(-2e-5))
    model = AtmosphereModel(grid; forcing=(; θ=(f1, f2)))
    @test model.forcing.ρθ isa MultipleForcings
    @test length(model.forcing.ρθ.forcings) == 2
    @test all(f isa SpecificForcing for f in model.forcing.ρθ.forcings)
end

#####
##### Numerical correctness: anelastic Center- and Face-located forcings
#####

@testset "Specific θ forcing produces ρᵣ * F tendency [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    F_θ = FT(-1e-5)  # specific potential-temperature tendency, K/s
    model = AtmosphereModel(grid; forcing=(; θ=Returns(F_θ)))
    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model; θ=θ₀)

    # The kernel writes ρᵣ * F_θ into Gρθ
    update_state!(model)
    Gρθ = interior(model.timestepper.Gⁿ.ρθ) |> Array
    ρᵣ = interior(model.dynamics.reference_state.density) |> Array
    expected = ρᵣ .* F_θ
    @test maximum(abs.(Gρθ .- expected)) < eps(FT) * 100
end

@testset "Unprefixed-tracer forcing wraps in SpecificForcing [$(FT)]" for FT in test_float_types()
    # A user tracer named `:c` (no `ρ` prefix) is itself in specific form, so any forcing
    # supplied under that name is wrapped in SpecificForcing and Breeze applies ρᵣ at
    # kernel time — same convention as the specific alias of a ρ-prefixed prognostic.
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    F = FT(1e-5)
    model = AtmosphereModel(grid; tracers=:c, forcing=(; c=Returns(F)))
    @test model.forcing.c isa SpecificForcing

    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model; θ=θ₀)
    update_state!(model)

    Gρc = interior(model.timestepper.Gⁿ.c) |> Array
    ρᵣ = interior(model.dynamics.reference_state.density) |> Array
    @test maximum(abs.(Gρc .- ρᵣ .* F)) < eps(FT) * 100
end

@testset "Specific w forcing interpolates ρᵣ to Face [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    F_w = FT(1e-3)  # specific vertical-momentum tendency, m/s²
    model = AtmosphereModel(grid; forcing=(; w=Returns(F_w)))
    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model; θ=θ₀)

    update_state!(model)
    Gρw = interior(model.timestepper.Gⁿ.ρw) |> Array
    ρᵣ = interior(model.dynamics.reference_state.density) |> Array

    # ρᵣ lives at Center; interpolate to Face for interior k = 2, 3 (away from boundaries)
    ρ_face_2 = (ρᵣ[1, 1, 1] + ρᵣ[1, 1, 2]) / 2
    @test Gρw[1, 1, 2] ≈ ρ_face_2 * F_w rtol=10 * eps(FT)
end

@testset "Specific-key path matches manual ρᵣ * field under ρ-key [$(FT)]" for FT in test_float_types()
    # Verifies that the new ergonomic
    #     forcing = (; θ = Forcing(field))
    # produces bit-identical model evolution to the old pattern
    #     set!(field, ρᵣ * field); forcing = (; ρθ = Forcing(field)).
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    # A nontrivial height-varying specific tendency (K/s).
    F_specific_profile(z) = FT(-1e-5) * (1 + z / 100)

    # Path A: manual ρᵣ multiply, supplied under :ρθ (the pre-change idiom)
    model_A = AtmosphereModel(grid; forcing = (; ρθ = (x, y, z, t) -> 0))  # placeholder
    ρᵣ = model_A.dynamics.reference_state.density
    F_density_field = Field{Nothing, Nothing, Center}(grid)
    set!(F_density_field, z -> F_specific_profile(z))
    set!(F_density_field, ρᵣ * F_density_field)
    model_A = AtmosphereModel(grid; forcing = (; ρθ = Forcing(F_density_field)))

    # Path B: specific tendency supplied under :θ
    F_specific_field = Field{Nothing, Nothing, Center}(grid)
    set!(F_specific_field, z -> F_specific_profile(z))
    model_B = AtmosphereModel(grid; forcing = (; θ = Forcing(F_specific_field)))

    θ₀ = model_A.dynamics.reference_state.potential_temperature
    set!(model_A; θ=θ₀)
    set!(model_B; θ=θ₀)

    Δt = FT(1)
    for _ in 1:5
        time_step!(model_A, Δt)
        time_step!(model_B, Δt)
    end

    ρθ_A = interior(prognostic_fields(model_A).ρθ) |> Array
    ρθ_B = interior(prognostic_fields(model_B).ρθ) |> Array
    @test maximum(abs.(ρθ_A .- ρθ_B)) < eps(FT) * 1000 * maximum(abs.(ρθ_A))
end

@testset "Mixed θ + ρθ contributions sum [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    F_θ_specific = FT(-1e-5)              # gets multiplied by ρᵣ
    F_ρθ_density = FT(-1e-5)              # added directly
    model = AtmosphereModel(grid; forcing=(; θ=Returns(F_θ_specific),
                                             ρθ=Returns(F_ρθ_density)))
    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model; θ=θ₀)

    update_state!(model)
    Gρθ = interior(model.timestepper.Gⁿ.ρθ) |> Array
    ρᵣ = interior(model.dynamics.reference_state.density) |> Array
    expected = ρᵣ .* F_θ_specific .+ F_ρθ_density
    @test maximum(abs.(Gρθ .- expected)) < eps(FT) * 100
end

@testset "Specific-keyed Relaxation relaxes the specific variable [$(FT)]" for FT in test_float_types()
    # Regression test for the Oceananigans ≥ v0.110.7 Relaxation rework (Oceananigans
    # #5620): `materialize_forcing(::Relaxation, field, name, …)` now captures `field`
    # as the relaxed variable instead of binding `name`. SpecificForcing must therefore
    # materialize its inner forcing against the *specific* field — otherwise a
    # specific-keyed relaxation (e.g. a Davies fringe on θ) computes
    # rate·(θ_target − ρθ) instead of rate·(θ_target − θ).
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    rate = FT(1/100)

    # Center path: θ relaxation. With uniform θ and zero velocity the tendency is
    # forcing-only, so Gρθ = ρᵣ · rate · (θ_target − θ).
    θ_target = FT(310)
    model = AtmosphereModel(grid; forcing=(; θ=Relaxation(; rate, target=θ_target)))
    θ₀ = model.dynamics.reference_state.potential_temperature
    set!(model; θ=θ₀)
    update_state!(model)

    Gρθ = interior(model.timestepper.Gⁿ.ρθ) |> Array
    ρᵣ = interior(model.dynamics.reference_state.density) |> Array
    θ = interior(model.formulation.potential_temperature) |> Array
    expected = ρᵣ .* rate .* (θ_target .- θ)
    @test maximum(abs.(Gρθ .- expected)) < 100 * eps(FT) * maximum(abs.(expected))

    # Face path: u relaxation. ρᵣ varies only in z, so its x-Face interpolation equals
    # the Center value and Gρu = ρᵣ · rate · (u_target − u).
    u_target = FT(10)
    u₀ = FT(2)
    model = AtmosphereModel(grid; forcing=(; u=Relaxation(; rate, target=u_target)))
    set!(model; θ=θ₀, u=u₀)
    update_state!(model)

    Gρu = interior(model.timestepper.Gⁿ.ρu) |> Array
    expected_u = ρᵣ .* rate .* (u_target - u₀)
    @test maximum(abs.(Gρu .- expected_u)) < 100 * eps(FT) * maximum(abs.(expected_u))
end

#####
##### Compressible: ρ varies in (x, y, z, t)
#####

@testset "Specific θ forcing under compressible dynamics [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch;
                           size = (8, 8, 8), halo = (5, 5, 5),
                           x = (0, 100), y = (0, 100), z = (0, 100),
                           topology = (Periodic, Periodic, Bounded))

    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 2,
                                                                    damping = NoDivergenceDamping());
                                    reference_potential_temperature = FT(300),
                                    surface_pressure = FT(1e5),
                                    standard_pressure = FT(1e5))

    F_θ = FT(-1e-5)
    model = AtmosphereModel(grid; dynamics, forcing=(; θ=Returns(F_θ)))
    @test model.forcing.ρθ isa SpecificForcing

    set!(model, ρ = (x, y, z) -> FT(1.2),
                θ = (x, y, z) -> FT(300))
    update_state!(model)

    Gρθ = interior(model.timestepper.Gⁿ.ρθ) |> Array
    ρ = interior(Breeze.AtmosphereModels.dynamics_density(model.dynamics)) |> Array
    expected = ρ .* F_θ
    @test maximum(abs.(Gρθ .- expected)) < eps(FT) * 100
end

#####
##### Error paths
#####

# A user-defined density-tendency forcing — Breeze should reject it under a specific key.
struct ExampleDensityTendency end
@inline (::ExampleDensityTendency)(i, j, k, grid, clock, fields) = zero(grid)
Breeze.AtmosphereModels.is_density_tendency_forcing(::ExampleDensityTendency) = true

@testset "User-defined density-tendency forcing under specific key errors" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    bad = ExampleDensityTendency()
    @test is_density_tendency_forcing(bad)
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; θ=bad))
    # Same check inside a tuple
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; θ=(bad, Returns(0.0))))
end

@testset "Built-in Breeze forcings are not density-tendency forcings" begin
    # After the refactor SubsidenceForcing and GeostrophicForcing both return specific
    # tendencies (the ρ multiply happens in SpecificForcing), so the trait must be false.
    subsidence = SubsidenceForcing(z -> -0.01)
    geostrophic_u = geostrophic_forcings(z -> -10.0, z -> 0.0).u
    @test !is_density_tendency_forcing(subsidence)
    @test !is_density_tendency_forcing(geostrophic_u)
    @test !is_density_tendency_forcing(Returns(0.0))
end

@testset "Built-in Breeze forcings under ρ-key error with migration message" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    subsidence = SubsidenceForcing(z -> -0.01)
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; ρθ=subsidence))

    geostrophic_u = geostrophic_forcings(z -> -10.0, z -> 0.0).u
    @test_throws ArgumentError AtmosphereModel(grid;
                                               coriolis=FPlane(f=1e-4),
                                               forcing=(; ρu=geostrophic_u))
end

@testset "Unknown specific key errors" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; foo=Returns(0.0)))
end
