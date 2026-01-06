using Breeze
using Oceananigans
using Test

test_thermodynamics = (:StaticEnergy, :LiquidIcePotentialTemperature)

@testset "Vertically implicit diffusion correctness [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    Nz = 16
    Lz = 100
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz))
    vitd = VerticallyImplicitTimeDiscretization()
    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()

    # Use a Gaussian profile centered in the domain for testing diffusion
    z₀ = Lz / 2
    σ = Lz / 8
    gaussian(z) = exp(-(z - z₀)^2 / (2σ^2))

    @testset "Implicit scalar diffusion smooths vertical gradients" begin
        κ = FT(10)
        closure = VerticalScalarDiffusivity(vitd; κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        # Set tracer with vertical gradient
        set!(model; ρc = (x, y, z) -> gaussian(z))
        ρc = model.tracers.ρc

        # Measure initial variance
        var₀ = sum(interior(ρc) .^ 2)

        # Take several time steps
        for _ in 1:10
            time_step!(model, 1)
        end

        # Variance should decrease due to diffusion
        var₁ = sum(interior(ρc) .^ 2)
        @test var₁ < var₀
    end

    @testset "Implicit and explicit diffusion produce similar results" begin
        κ = FT(1)
        Δt = FT(0.1)  # Small timestep for stability with explicit
        Nt = 5

        # Create models with implicit and explicit discretization
        implicit_closure = VerticalScalarDiffusivity(vitd; κ)
        explicit_closure = VerticalScalarDiffusivity(etd; κ)

        implicit_model = AtmosphereModel(grid; closure=implicit_closure, advection=nothing, tracers=:ρc)
        explicit_model = AtmosphereModel(grid; closure=explicit_closure, advection=nothing, tracers=:ρc)

        # Set same initial condition
        set!(implicit_model; ρc = (x, y, z) -> gaussian(z))
        set!(explicit_model; ρc = (x, y, z) -> gaussian(z))

        # Step both models
        for _ in 1:Nt
            time_step!(implicit_model, Δt)
            time_step!(explicit_model, Δt)
        end

        # Results should be similar (within numerical tolerance)
        ρc_implicit = Array(interior(implicit_model.tracers.ρc))
        ρc_explicit = Array(interior(explicit_model.tracers.ρc))
        @test isapprox(ρc_implicit, ρc_explicit, rtol=0.01)
    end

    @testset "Implicit viscosity smooths momentum gradients" begin
        ν = FT(10)
        closure = VerticalScalarDiffusivity(vitd; ν)
        model = AtmosphereModel(grid; closure, advection=nothing)

        # Set momentum with vertical gradient (ρu varies with z)
        set!(model; ρu = (x, y, z) -> gaussian(z))
        ρu = model.momentum.ρu

        # Measure initial variance
        var₀ = sum(interior(ρu) .^ 2)

        # Take several time steps
        for _ in 1:10
            time_step!(model, 1)
        end

        # Variance should decrease due to diffusion
        var₁ = sum(interior(ρu) .^ 2)
        @test var₁ < var₀
    end

    @testset "Implicit diffusion with both ν and κ" begin
        ν = FT(5)
        κ = FT(10)
        closure = VerticalScalarDiffusivity(vitd; ν, κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        # Set both momentum and tracer with gradients
        set!(model; ρu = (x, y, z) -> gaussian(z), ρc = (x, y, z) -> gaussian(z))
        ρu = model.momentum.ρu
        ρc = model.tracers.ρc

        var_u₀ = sum(interior(ρu) .^ 2)
        var_c₀ = sum(interior(ρc) .^ 2)

        for _ in 1:10
            time_step!(model, 1)
        end

        var_u₁ = sum(interior(ρu) .^ 2)
        var_c₁ = sum(interior(ρc) .^ 2)

        # Both should decrease
        @test var_u₁ < var_u₀
        @test var_c₁ < var_c₀

        # Tracer should diffuse more (κ > ν)
        # Relative change should be larger for tracer
        rel_change_u = (var_u₀ - var_u₁) / var_u₀
        rel_change_c = (var_c₀ - var_c₁) / var_c₀
        @test rel_change_c > rel_change_u
    end
end

@testset "Time stepping with TurbulenceClosures [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 100), y=(0, 100), z=(0, 100))
    vitd = VerticallyImplicitTimeDiscretization()

    closures = (
        ScalarDiffusivity(ν=1, κ=2),
        ScalarDiffusivity(vitd, ν=1),
        SmagorinskyLilly(),
        AnisotropicMinimumDissipation()
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
            ρqᵗ₀ = model.moisture_density |> interior |> Array
            ρc₀ = model.tracers.ρc |> interior |> Array

            # Take a time step
            time_step!(model, 1)

            ρe₁ = static_energy_density(model) |> Field |> interior |> Array
            ρqᵗ₁ = model.moisture_density |> interior |> Array
            ρc₁ = model.tracers.ρc |> interior |> Array

            # Scalars should change due to diffusion (not advection since advection=nothing)
            # Use explicit maximum difference check instead of ≈ to handle Float32
            @test maximum(abs, ρe₁ - ρe₀) > 0
            @test maximum(abs, ρqᵗ₁ - ρqᵗ₀) > 0
            @test maximum(abs, ρc₁ - ρc₀) > 0
        end
    end
end
