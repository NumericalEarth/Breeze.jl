using Breeze
using Oceananigans
using Test

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

    etd = Oceananigans.TurbulenceClosures.ExplicitTimeDiscretization()
    discretizations = (vitd, etd)
    @testset "Implicit diffusion solver with ScalarDiffusivity [$(FT), $(typeof(disc))]" for disc in discretizations
        closure = ScalarDiffusivity(disc, ν=1, κ=1)
        model = AtmosphereModel(grid; closure, tracers=:ρc)
        # Set uniform specific energy for no diffusion
        θ₀ = model.formulation.reference_state.potential_temperature
        cᵖᵈ = model.thermodynamic_constants.dry_air.heat_capacity
        e₀ = cᵖᵈ * θ₀
        set!(model; e=e₀)
        ρe₀ = deepcopy(static_energy_density(model))
        time_step!(model, 1)
        # Use rtol for implicit solver which may have small numerical effects
        @test isapprox(static_energy_density(model), ρe₀, rtol=1e-5)
    end

    @testset "Closure flux affects momentum tendency [$(FT)]" begin
        closure = ScalarDiffusivity(ν=1e4)
        model = AtmosphereModel(grid; advection=nothing, closure)
        set!(model; ρu = (x, y, z) -> exp((z - 50)^2 / (2 * 20^2)))
        Breeze.AtmosphereModels.compute_tendencies!(model)
        Gρu = model.timestepper.Gⁿ.ρu
        @test maximum(abs, Gρu) > 0
    end

    @testset "SmagorinskyLilly with velocity gradients [$(FT)]" begin
        model = AtmosphereModel(grid; closure=SmagorinskyLilly())
        θ₀ = model.formulation.reference_state.potential_temperature
        set!(model; θ=θ₀, ρu = (x, y, z) -> z / 100)
        Breeze.AtmosphereModels.update_state!(model)
        @test maximum(abs, model.closure_fields.νₑ) > 0
    end

    @testset "AnisotropicMinimumDissipation with velocity gradients [$(FT)]" begin
        model = AtmosphereModel(grid; closure=AnisotropicMinimumDissipation())
        set!(model; ρu = (x, y, z) -> z / 100)
        Breeze.AtmosphereModels.update_state!(model)
        @test haskey(model.closure_fields, :νₑ) || haskey(model.closure_fields, :κₑ)
    end

    # Test LES scalar diffusion with advection=nothing
    # This isolates the effect of the closure on scalar fields
    les_closures = (SmagorinskyLilly(), AnisotropicMinimumDissipation())

    @testset "LES scalar diffusion without advection [$(FT), $(nameof(typeof(closure)))]" for closure in les_closures
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        # Set random velocity field to trigger non-zero eddy diffusivity
        Ξ(x, y, z) = randn()

        # Set scalar gradients for energy, moisture, and passive tracer
        θ₀ = model.formulation.reference_state.potential_temperature
        z₀, dz = 50, 10
        gaussian(z) = exp(- (z - z₀)^2 / 2dz^2)
        θᵢ(x, y, z) = θ₀ + 10 * gaussian(z)
        qᵗᵢ(x, y, z) = 0.01 + 1e-3 * gaussian(z)
        ρcᵢ(x, y, z) = gaussian(z)
        set!(model; θ = θᵢ, ρqᵗ = qᵗᵢ, ρc = ρcᵢ, ρu = Ξ, ρv = Ξ, ρw = Ξ)

        # Store initial scalar fields (using copy of data to avoid reference issues)
        ρe₀ = copy(interior(static_energy_density(model)))
        ρqᵗ₀ = copy(interior(model.moisture_density))
        ρc₀ = copy(interior(model.tracers.ρc))

        # Take a time step
        time_step!(model, 1)

        # Scalars should change due to diffusion (not advection since advection=nothing)
        # Use explicit maximum difference check instead of ≈ to handle Float32
        @test maximum(abs, interior(static_energy_density(model)) .- ρe₀) > 0
        @test maximum(abs, interior(model.moisture_density) .- ρqᵗ₀) > 0
        @test maximum(abs, interior(model.tracers.ρc) .- ρc₀) > 0
    end
end
