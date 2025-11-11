using Breeze
using GPUArraysCore: @allowscalar
using Oceananigans
using Test

const TC = Breeze.TurbulenceClosures

@testset "Time stepping with TurbulenceClosures [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(8, 8, 8), x=(0, 100), y=(0, 100), z=(0, 100))

    # Try two closures from Oceananigans
    for closure in (SmagorinskyLilly(), AnisotropicMinimumDissipation())
        @testset let closure=closure
            model = AtmosphereModel(grid; closure=closure)
            @test try
                time_step!(model, 1)
                true
            catch
                false
            end
        end
    end
    Oceananigans.defaults.FloatType = Float64
end

