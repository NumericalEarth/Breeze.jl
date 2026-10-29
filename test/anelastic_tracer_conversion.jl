# Tracer density <-> specific conversions on anelastic models: the reference density has parent
# size (1, 1, Nz+2Hz) and must be broadcast, not linearly indexed (task 13 NaN regression).

using Breeze
using Oceananigans
using Test

@testset "Anelastic tracer stays finite [$FT]" for FT in (Float32, Float64)
    Oceananigans.defaults.FloatType = FT
    grid = RectilinearGrid(CPU(), FT; size=(12, 10, 8), x=(0, 24e3), y=(0, 20e3), z=(0, 8e3), halo=(5, 5, 5),
                           topology=(Bounded, Bounded, Bounded))
    model = AtmosphereModel(grid; tracers=:c, momentum_advection=WENO(FT; order=5), scalar_advection=WENO(FT; order=5))
    blob(x, y, z) = exp(-((x - 10e3)^2 + (y - 9e3)^2) / 4e6 - (z - 3e3)^2 / 1e6)
    set!(model; θ=(x, y, z) -> 300 + 0.003z, u=8, v=-5, c=blob)
    for _ in 1:8
        time_step!(model, FT(2))
    end
    for (name, f) in pairs(Oceananigans.prognostic_fields(model))
        @test all(isfinite, interior(f))
    end
    @test maximum(interior(model.tracers.c)) > 0
    Oceananigans.defaults.FloatType = Float64
end
