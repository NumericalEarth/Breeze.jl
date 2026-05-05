using Breeze
using Oceananigans
using Test

test_thermodynamics = (:StaticEnergy, :LiquidIcePotentialTemperature)

@testset "Time stepping with TurbulenceClosures [$(FT)]" for FT in test_float_types()
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 100), y=(0, 100), z=(0, 100))
    vitd = VerticallyImplicitTimeDiscretization()

    closures = (
        ScalarDiffusivity(ν=1, κ=2),
        ScalarDiffusivity(vitd, ν=1),
        SmagorinskyLilly(),
        DynamicSmagorinsky(),
        AnisotropicMinimumDissipation(),
    )

    for closure in closures
        @testset let closure=closure
            model = AtmosphereModel(grid; closure)
            time_step!(model, 1)
            @test true
        end
    end

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants)
    dynamics = AnelasticDynamics(reference_state)
    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()
    discretizations = (vitd, etd)

    @testset "AtmosphereModel with $formulation thermodynamics [$FT]" for formulation in test_thermodynamics

        @testset "Implicit diffusion solver with ScalarDiffusivity [$formulation, $(FT), $(typeof(disc))]" for disc in discretizations
            closure = ScalarDiffusivity(disc, ν=1, κ=1)
            model = AtmosphereModel(grid; dynamics, formulation, closure, tracers=:ρc)
            # Set uniform specific energy for no diffusion
            θ₀ = model.dynamics.reference_state.potential_temperature
            cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
            e₀ = cᵖᵈ * θ₀
            set!(model; e=e₀)
            ρe₀ = deepcopy(static_energy_density(model))
            time_step!(model, 1)
            # Use rtol for implicit solver which may have small numerical effects
            @test isapprox(static_energy_density(model), ρe₀, rtol=1e-5)
        end

        @testset "Closure flux affects momentum tendency [$formulation, $(FT)]" begin
            closure = ScalarDiffusivity(ν=1e4)
            model = AtmosphereModel(grid; dynamics, formulation, advection=nothing, closure)
            set!(model; ρu = (x, y, z) -> exp((z - 50)^2 / (2 * 20^2)))
            Breeze.AtmosphereModels.compute_tendencies!(model)
            Gρu = model.timestepper.Gⁿ.ρu
            @test maximum(abs, Gρu) > 0
        end

        @testset "SmagorinskyLilly with velocity gradients [$formulation, $(FT)]" begin
            model = AtmosphereModel(grid; dynamics, formulation, closure=SmagorinskyLilly())
            θ₀ = model.dynamics.reference_state.potential_temperature
            set!(model; θ=θ₀, ρu = (x, y, z) -> z / 100)
            Breeze.AtmosphereModels.update_state!(model)
            @test maximum(abs, model.closure_fields.νₑ) > 0
        end

        @testset "DynamicSmagorinsky with velocity gradients [$formulation, $(FT)]" begin
            model = AtmosphereModel(grid; dynamics, formulation, closure=DynamicSmagorinsky())
            θ₀ = model.dynamics.reference_state.potential_temperature
            # Mean shear plus a triad wave in (x,y) so the test filter sees
            # horizontal structure. Without xy variation Lᵢⱼ vanishes and
            # 𝒥ᴸᴹ sits at the minimum_numerator floor regardless of whether
            # `step_closure_prognostics!` runs.
            ρuᵢ(x, y, z) = z / 100 + 0.1 * sin(2π * x / 100) * cos(2π * y / 100)
            set!(model; θ=θ₀, ρu=ρuᵢ)
            # `initialize_closure_fields!` seeds 𝒥ᴸᴹ with a spatial mean
            # uniform across the domain. A successful call to
            # `step_closure_prognostics!` during the time step must advance
            # 𝒥ᴸᴹ with per-cell local values, making it spatially varying.
            # If the hook is called at the wrong clock stage, Oceananigans's
            # LASD kernel gates it out and 𝒥ᴸᴹ stays at its seed.
            𝒥ᴸᴹ_seed = Array(interior(model.closure_fields.𝒥ᴸᴹ)) |> copy
            time_step!(model, 1)
            𝒥ᴸᴹ_after = Array(interior(model.closure_fields.𝒥ᴸᴹ))
            @test maximum(abs, model.closure_fields.νₑ) > 0
            @test 𝒥ᴸᴹ_after != 𝒥ᴸᴹ_seed
        end

        @testset "AnisotropicMinimumDissipation with velocity gradients [$formulation, $(FT)]" begin
            model = AtmosphereModel(grid; dynamics, formulation, closure=AnisotropicMinimumDissipation())
            set!(model; ρu = (x, y, z) -> z / 100)
            Breeze.AtmosphereModels.update_state!(model)
            @test haskey(model.closure_fields, :νₑ) || haskey(model.closure_fields, :κₑ)
        end

        # Test LES scalar diffusion with advection=nothing
        # This isolates the effect of the closure on scalar fields
        les_closures = (SmagorinskyLilly(), AnisotropicMinimumDissipation())

        @testset "LES scalar diffusion without advection [$formulation, $(FT), $(nameof(typeof(closure)))]" for closure in les_closures
            model = AtmosphereModel(grid; dynamics, formulation, closure, advection=nothing, tracers=:ρc)

            # Set random velocity field to trigger non-zero eddy diffusivity
            Ξ(x, y, z) = randn()

            # Set scalar gradients for energy, moisture, and passive tracer
            θ₀ = model.dynamics.reference_state.potential_temperature
            z₀, dz = 50, 10
            gaussian(z) = exp(- (z - z₀)^2 / 2dz^2)
            θᵢ(x, y, z) = θ₀ + 10 * gaussian(z)
            qᵗᵢ(x, y, z) = 0.01 + 1e-3 * gaussian(z)
            ρcᵢ(x, y, z) = gaussian(z)
            set!(model; θ = θᵢ, ρqᵗ = qᵗᵢ, ρc = ρcᵢ, ρu = Ξ, ρv = Ξ, ρw = Ξ)

            # Store initial scalar fields (using copy of data to avoid reference issues)
            ρe₀ = static_energy_density(model) |> Field |> interior |> Array
            ρqᵛ₀ = model.moisture_density |> interior |> Array
            ρc₀ = model.tracers.ρc |> interior |> Array

            # Take a time step
            time_step!(model, 1)

            ρe₁ = static_energy_density(model) |> Field |> interior |> Array
            ρqᵛ₁ = model.moisture_density |> interior |> Array
            ρc₁ = model.tracers.ρc |> interior |> Array

            # Scalars should change due to diffusion (not advection since advection=nothing)
            # Use explicit maximum difference check instead of ≈ to handle Float32
            @test maximum(abs, ρe₁ - ρe₀) > 0
            @test maximum(abs, ρqᵛ₁ - ρqᵛ₀) > 0
            @test maximum(abs, ρc₁ - ρc₀) > 0
        end
    end
end
