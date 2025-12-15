using Breeze
using Oceananigans
using Test

@testset "AtmosphereModel tracers advection [$(FT)]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(default_arch; size=(16, 8, 8), x=(0, 1_000), y=(0, 500), z=(0, 500))
    model = AtmosphereModel(grid; tracers=(:a, :b))
    set!(model; u = 1)
    # Initialize tracer with an x-gradient
    Lx = FT(1_000)
    set!(model.tracers.a, (x, y, z) -> sin(2π * x / Lx))
    set!(model.tracers.b, (x, y, z) -> cos(2π * x / Lx))

    @test try
        time_step!(model, 1)
        true
    catch
        false
    end
end
