using Breeze
using GPUArraysCore: @allowscalar
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
    @testset "Implicit diffusion solver with ScalarDiffusivity [$(FT), $(disc)]" for FT in (Float32, Float64), disc in discretizations
        closure = ScalarDiffusivity(disc, ν=1, κ=1)
        model = AtmosphereModel(grid; closure, tracers=:ρc)
        ρe₀ = 3e5
        set!(model; ρe=ρe₀)
        ρe₀ = deepcopy(model.energy_density)
        time_step!(model, 1)

        ϵ = sqrt(eps(FT))
        @allowscalar begin
            @test model.momentum.ρu ≈ XFaceField(grid) atol=ϵ
            @test model.momentum.ρv ≈ YFaceField(grid) atol=ϵ
            @test model.momentum.ρw ≈ ZFaceField(grid) atol=100ϵ # large bc of non-zero buoyancy
            @test model.moisture_density ≈ CenterField(grid) atol=ϵ
            @test model.tracers.ρc ≈ CenterField(grid) atol=ϵ
            @test model.energy_density ≈ ρe₀ rtol=10ϵ
        end
    end
end
