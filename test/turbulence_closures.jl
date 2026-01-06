using Breeze
using Oceananigans
using Test

test_thermodynamics = (:StaticEnergy, :LiquidIcePotentialTemperature)

@testset "Vertically implicit diffusion correctness [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    Nz = 32
    Lz = FT(100)
    grid = RectilinearGrid(default_arch; size=(4, 4, Nz), x=(0, 100), y=(0, 100), z=(0, Lz))
    vitd = VerticallyImplicitTimeDiscretization()
    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()

    # Cosine profile: c(z) = cos(π z / Lz)
    # Analytical solution for diffusion: c(z,t) = cos(π z / Lz) * exp(-κ (π/Lz)² t)
    # This satisfies zero-flux BCs at z=0 and z=Lz
    k = FT(π) / Lz
    cosine(z) = cos(k * z)

    # Analytical decay factor after time t with diffusivity κ
    analytical_decay(κ, t) = exp(-κ * k^2 * t)

    @testset "Implicit scalar diffusion matches analytical solution" begin
        κ = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        set!(model; ρc = (x, y, z) -> cosine(z))
        ρc₀ = sum(interior(model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        # Compare numerical decay to analytical decay
        ρc₁ = sum(interior(model.tracers.ρc) .^ 2)
        numerical_decay = sqrt(ρc₁ / ρc₀)
        expected_decay = analytical_decay(κ, t_final)

        @test isapprox(numerical_decay, expected_decay, rtol=0.05)
    end

    @testset "Implicit and explicit diffusion match analytical solution" begin
        κ = FT(1)
        Δt = FT(0.5)
        Nt = 10
        t_final = Δt * Nt

        implicit_closure = VerticalScalarDiffusivity(vitd; κ)
        explicit_closure = VerticalScalarDiffusivity(etd; κ)

        implicit_model = AtmosphereModel(grid; closure=implicit_closure, advection=nothing, tracers=:ρc)
        explicit_model = AtmosphereModel(grid; closure=explicit_closure, advection=nothing, tracers=:ρc)

        set!(implicit_model; ρc = (x, y, z) -> cosine(z))
        set!(explicit_model; ρc = (x, y, z) -> cosine(z))

        ρc₀_implicit = sum(interior(implicit_model.tracers.ρc) .^ 2)
        ρc₀_explicit = sum(interior(explicit_model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(implicit_model, Δt)
            time_step!(explicit_model, Δt)
        end

        ρc₁_implicit = sum(interior(implicit_model.tracers.ρc) .^ 2)
        ρc₁_explicit = sum(interior(explicit_model.tracers.ρc) .^ 2)

        numerical_decay_implicit = sqrt(ρc₁_implicit / ρc₀_implicit)
        numerical_decay_explicit = sqrt(ρc₁_explicit / ρc₀_explicit)
        expected_decay = analytical_decay(κ, t_final)

        # Both should match analytical solution
        @test isapprox(numerical_decay_implicit, expected_decay, rtol=0.05)
        @test isapprox(numerical_decay_explicit, expected_decay, rtol=0.05)

        # And they should match each other closely
        @test isapprox(numerical_decay_implicit, numerical_decay_explicit, rtol=0.01)
    end

    @testset "Implicit viscosity matches analytical solution" begin
        ν = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; ν)
        model = AtmosphereModel(grid; closure, advection=nothing)

        set!(model; ρu = (x, y, z) -> cosine(z))
        ρu₀ = sum(interior(model.momentum.ρu) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        ρu₁ = sum(interior(model.momentum.ρu) .^ 2)
        numerical_decay = sqrt(ρu₁ / ρu₀)
        expected_decay = analytical_decay(ν, t_final)

        @test isapprox(numerical_decay, expected_decay, rtol=0.05)
    end

    @testset "Implicit diffusion with both ν and κ matches analytical solutions" begin
        ν = FT(5)
        κ = FT(10)
        Δt = FT(1)
        Nt = 10
        t_final = Δt * Nt

        closure = VerticalScalarDiffusivity(vitd; ν, κ)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        set!(model; ρu = (x, y, z) -> cosine(z), ρc = (x, y, z) -> cosine(z))

        ρu₀ = sum(interior(model.momentum.ρu) .^ 2)
        ρc₀ = sum(interior(model.tracers.ρc) .^ 2)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        ρu₁ = sum(interior(model.momentum.ρu) .^ 2)
        ρc₁ = sum(interior(model.tracers.ρc) .^ 2)

        numerical_decay_u = sqrt(ρu₁ / ρu₀)
        numerical_decay_c = sqrt(ρc₁ / ρc₀)
        expected_decay_u = analytical_decay(ν, t_final)
        expected_decay_c = analytical_decay(κ, t_final)

        @test isapprox(numerical_decay_u, expected_decay_u, rtol=0.05)
        @test isapprox(numerical_decay_c, expected_decay_c, rtol=0.05)
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
