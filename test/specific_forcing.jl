using Breeze
using Breeze: SpecificForcing
using Breeze.AtmosphereModels: is_density_tendency_forcing
using Oceananigans: Oceananigans, prognostic_fields
using Oceananigans.Forcings: MultipleForcings
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

@testset "Density-tendency forcing under specific key errors" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    subsidence = SubsidenceForcing(z -> -0.01)
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; θ=subsidence))

    # Same check inside a tuple
    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; θ=(subsidence, Returns(0.0))))

    # GeostrophicForcing too
    geostrophic = geostrophic_forcings(z -> -10.0, z -> 0.0).ρu
    @test_throws ArgumentError AtmosphereModel(grid;
                                               coriolis=FPlane(f=1e-4),
                                               forcing=(; u=geostrophic))

    # Trait sanity
    @test is_density_tendency_forcing(subsidence)
    @test is_density_tendency_forcing(geostrophic)
    @test !is_density_tendency_forcing(Returns(0.0))
end

@testset "Unknown specific key errors" begin
    Oceananigans.defaults.FloatType = Float64
    grid = RectilinearGrid(default_arch; size=(4, 4, 4), x=(0, 100), y=(0, 100), z=(0, 100))

    @test_throws ArgumentError AtmosphereModel(grid; forcing=(; foo=Returns(0.0)))
end
