# Statistically stationary homogeneous isotropic turbulence forced by a band forcing.
#
# Usage: julia --project=validation/turbulence_forcing validation/turbulence_forcing/forced_turbulence.jl [N] [forcing] [arch] [stop_time] [ν]
#   N:         grid points per direction (default 32)
#   forcing:   "constant" (ConstantPowerBandForcing) or "stochastic" (StochasticBandForcing)
#   arch:      "CPU" or "GPU"
#   stop_time: in units of L = 2π, ε = 0.1 (default 100)
#   ν:         kinematic viscosity (default 0.02)

using Breeze
using Oceananigans
using CUDA
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Operators: ∂xᶜᶜᶜ, ∂yᶠᶠᶜ, ∂zᶠᶜᶠ, ∂xᶠᶠᶜ, ∂yᶜᶜᶜ, ∂zᶜᶠᶠ, ∂xᶠᶜᶠ, ∂yᶜᶠᶠ, ∂zᶜᶜᶜ
using Printf
using Statistics

N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 32
forcing_name = length(ARGS) ≥ 2 ? ARGS[2] : "constant"
arch = length(ARGS) ≥ 3 && ARGS[3] == "GPU" ? GPU() : CPU()
stop_time = length(ARGS) ≥ 4 ? parse(Float64, ARGS[4]) : 100.0


ε = 0.1
L = 2π
ν = length(ARGS) ≥ 5 ? parse(Float64, ARGS[5]) : 0.02
Δx = L / N

grid = RectilinearGrid(arch; size=(N, N, N), extent=(L, L, L), topology=(Periodic, Periodic, Periodic))

band_forcing = forcing_name == "stochastic" ?
    StochasticBandForcing(grid; energy_injection_rate=ε, correlation_time=0.5, seed=1) :
    ConstantPowerBandForcing(grid; energy_injection_rate=ε)

model = AtmosphereModel(grid; dynamics = BoussinesqDynamics(grid),
                        closure = ScalarDiffusivity(ν=ν),
                        forcing = (u=band_forcing, v=band_forcing, w=band_forcing))

# A flow with random amplitudes in the forced modes, plus noise, with velocity variance U² per component
function random_band_flow(U)
    wavevectors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1))
    amplitudes = map(wavevectors) do n
        a = randn(3) .+ im .* randn(3)
        a .- n .* (sum(n .* a) / sum(abs2, n))
    end
    scale = U / sqrt(sum(sum(abs2, a) for a in amplitudes) * 2 / 3)
    component(c) = (x, y, z) -> scale * 2 * sum(real(a[c] * exp(im * (n[1] * x + n[2] * y + n[3] * z)))
                                                for (n, a) in zip(wavevectors, amplitudes)) + U / 10 * randn()
    return (u = component(1), v = component(2), w = component(3))
end

set!(model; random_band_flow(0.5)...)

u, v, w = model.velocities
squared(i, j, k, grid, ∂, φ) = ∂(i, j, k, grid, φ)^2
gradient_squared(loc, ∂, φ) = Field(KernelFunctionOperation{loc...}(squared, grid, ∂, φ))

gradients = [gradient_squared((Center, Center, Center), ∂xᶜᶜᶜ, u),
             gradient_squared((Face, Face, Center),     ∂yᶠᶠᶜ, u),
             gradient_squared((Face, Center, Face),     ∂zᶠᶜᶠ, u),
             gradient_squared((Face, Face, Center),     ∂xᶠᶠᶜ, v),
             gradient_squared((Center, Center, Center), ∂yᶜᶜᶜ, v),
             gradient_squared((Center, Face, Face),     ∂zᶜᶠᶠ, v),
             gradient_squared((Face, Center, Face),     ∂xᶠᶜᶠ, w),
             gradient_squared((Center, Face, Face),     ∂yᶜᶠᶠ, w),
             gradient_squared((Center, Center, Center), ∂zᶜᶜᶜ, w)]

function statistics(model)
    u, v, w = model.velocities
    u², v², w² = mean(u^2), mean(v^2), mean(w^2)
    for g in gradients
        compute!(g)
    end
    dissipation = ν * sum(mean, gradients)
    return (; t = model.clock.time, E = (u² + v² + w²) / 2, dissipation, u², v², w²)
end

series = []
Δt = 0.25 * Δx
simulation = Simulation(model; Δt, stop_time, verbose=false)

function record!(sim)
    s = statistics(sim.model)
    push!(series, s)
    if length(series) % 20 == 1
        @printf("t = %6.2f, E = %.4f, D = %.4f, u′ = (%.3f, %.3f, %.3f), wall time %s\n",
                s.t, s.E, s.dissipation, sqrt(s.u²), sqrt(s.v²), sqrt(s.w²), prettytime(sim.run_wall_time))
    end
end

add_callback!(simulation, record!, TimeInterval(0.5))
run!(simulation)

# Statistics over the second half of the run
second_half = filter(s -> s.t ≥ stop_time / 2, series)
E = mean(s.E for s in second_half)
D = mean(s.dissipation for s in second_half)
ΔE = second_half[end].E - second_half[1].E
τ = second_half[end].t - second_half[1].t
u′² = 2E / 3
λ = sqrt(15ν * u′² / D)
Re_λ = sqrt(u′²) * λ / ν
η = (ν^3 / D)^(1/4)

@printf("\n%s forcing, %d³, ν = %.3g, ε = %.3g\n", forcing_name, N, ν, ε)
@printf("⟨D⟩ = %.4f, ⟨D⟩ / ε = %.3f, budget (⟨D⟩ + ΔE / τ) / ε = %.4f\n", D, D / ε, (D + ΔE / τ) / ε)
@printf("u′ = %.4f, λ = %.4f, Re_λ = %.1f, η = %.4f, kmax η = %.2f, Δx / η = %.2f\n",
        sqrt(u′²), λ, Re_λ, η, π / Δx * η, Δx / η)
@printf("⟨u²⟩ : ⟨v²⟩ : ⟨w²⟩ = %.3f : %.3f : %.3f\n", (mean(getproperty(s, n) for s in second_half) / u′² for n in (:u², :v², :w²))...)
