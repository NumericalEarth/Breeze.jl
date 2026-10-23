# Benchmark band forcings on the GPU.
#
# Usage: julia --project=validation/turbulence_forcing validation/turbulence_forcing/benchmark_band_forcing.jl
#
# Compares, per Runge-Kutta stage:
#   1. computing the amplitudes of ConstantPowerBandForcing with partial Fourier transforms
#      (the implementation in Breeze), and
#   2. three FFT-based Poisson solves, each of which is one forward and one inverse 3D FFT:
#      a lower bound for an FFT-based implementation, which transforms u, v, w forward
#      and the three forcing components back,
# and the cost of a full time step with and without band forcing.

using Breeze
using Oceananigans
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!
using CUDA
using Printf
using Statistics: median

using Breeze.Forcings: compute_band_forcing!

function time_median(f, trials=20)
    f(); f() # warm up
    CUDA.synchronize()
    times = map(1:trials) do _
        CUDA.@elapsed f()
    end
    return median(times)
end

for FT in (Float64, Float32), N in (64, 128, 256)
    grid = RectilinearGrid(GPU(), FT; size=(N, N, N), extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

    band = ConstantPowerBandForcing(grid; energy_injection_rate=0.1)
    stochastic = StochasticBandForcing(grid; energy_injection_rate=0.1, correlation_time=0.5)

    models = map((nothing, band, stochastic)) do f
        forcing = isnothing(f) ? NamedTuple() : (u=f, v=f, w=f)
        model = AtmosphereModel(grid; dynamics=BoussinesqDynamics(grid), closure=ScalarDiffusivity(ν=1e-3), forcing)
        set!(model, u=(x, y, z) -> sin(z) + randn() / 10, v=(x, y, z) -> sin(x), w=(x, y, z) -> sin(y))
        model
    end

    model = models[2]
    partial_transforms = time_median() do
        for c in 1:3
            compute_band_forcing!(band, c, model)
        end
    end

    solver = FFTBasedPoissonSolver(grid)
    ϕ = CenterField(grid)
    fft_lower_bound = time_median() do
        for _ in 1:3
            solve!(ϕ, solver)
        end
    end

    Δt = 1e-3
    step_times = map(models) do model
        time_median(() -> time_step!(model, Δt), 10)
    end

    @printf("%s %4d³: band amplitudes %.3f ms/stage, 3 FFT Poisson solves %.3f ms; time step: unforced %.2f ms, constant power %.2f ms (+%.1f%%), stochastic %.2f ms (+%.1f%%)\n",
            FT, N, 1e3 * partial_transforms, 1e3 * fft_lower_bound,
            1e3 * step_times[1], 1e3 * step_times[2], 100 * (step_times[2] / step_times[1] - 1),
            1e3 * step_times[3], 100 * (step_times[3] / step_times[1] - 1))
end
