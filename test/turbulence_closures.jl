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
        ρe₀ = 3e5
        set!(model; ρe=ρe₀)
        ρe₀ = deepcopy(model.energy_density)
        time_step!(model, 1)

        ϵ = sqrt(eps(FT))
        @test model.momentum.ρu ≈ XFaceField(grid)
        @test model.momentum.ρv ≈ YFaceField(grid)
        @test model.momentum.ρw ≈ ZFaceField(grid) atol=ϵ # use atol bc fields are close to 0
        @test model.moisture_density ≈ CenterField(grid)
        @test model.tracers.ρc ≈ CenterField(grid)
        @test model.energy_density ≈ ρe₀
    end

    @testset "Closure flux affects momentum tendency [$(FT)]" begin
        closure = ScalarDiffusivity(ν=1e4)
        model = AtmosphereModel(grid; advection=nothing, closure)
        set!(model; ρu = (x, y, z) -> exp((z - 50)^2 / (10 * 20^2)))
        Breeze.AtmosphereModels.compute_tendencies!(model)
        Gρu = model.timestepper.Gⁿ.ρu
        @test maximum(abs, Gρu) > 0
    end

    @testset "SmagorinskyLilly with velocity gradients [$(FT)]" begin
        model = AtmosphereModel(grid; closure=SmagorinskyLilly())
        set!(model; ρu = (x, y, z) -> z / 100)
        Breeze.AtmosphereModels.update_state!(model)
        @test maximum(abs, model.closure_fields.νₑ) > 0
    end

    @testset "AnisotropicMinimumDissipation with velocity gradients [$(FT)]" begin
        model = AtmosphereModel(grid; closure=AnisotropicMinimumDissipation())
        set!(model; ρu = (x, y, z) -> z / 100)
        Breeze.AtmosphereModels.update_state!(model)
        @test haskey(model.closure_fields, :νₑ) || haskey(model.closure_fields, :κₑ)
    end

    # Test scalar diffusion with advection=nothing
    # This isolates the effect of the closure on scalar fields
    @testset "Scalar diffusion without advection [$(FT)]" begin
        closure = ScalarDiffusivity(ν=1e4, κ=1e4)
        model = AtmosphereModel(grid; closure, advection=nothing, tracers=:ρc)

        # Set scalar gradients for energy, moisture, and passive tracer
        set!(model; ρe = (x, y, z) -> 3e5 + 1e3 * z)
        set!(model; ρqᵗ = (x, y, z) -> 0.01 * z / 100)
        set!(model; ρc = (x, y, z) -> z / 100)

        # Store initial scalar fields
        ρe₀ = deepcopy(model.energy_density)
        ρqᵗ₀ = deepcopy(model.moisture_density)
        ρc₀ = deepcopy(model.tracers.ρc)

        # Take a time step
        time_step!(model, 1)

        # Scalars should change due to diffusion (not advection since advection=nothing)
        @test !(model.energy_density ≈ ρe₀)
        @test !(model.moisture_density ≈ ρqᵗ₀)
        @test !(model.tracers.ρc ≈ ρc₀)
    end

    # Test SmagorinskyLilly scalar diffusion (energy only, since Smagorinsky
    # computes diffusivity from strain rate and applies it to all scalars)
    @testset "SmagorinskyLilly energy diffusion [$(FT)]" begin
        closure = SmagorinskyLilly()
        model = AtmosphereModel(grid; closure, advection=nothing)

        # Set velocity gradient to trigger eddy diffusivity
        set!(model; ρu = (x, y, z) -> z / 100)

        # Set energy gradient
        set!(model; ρe = (x, y, z) -> 3e5 + 1e3 * z)

        # Store initial energy
        ρe₀ = deepcopy(model.energy_density)

        # Take a time step
        time_step!(model, 1)

        # Energy should change due to LES diffusion
        @test !(model.energy_density ≈ ρe₀)
    end
end
