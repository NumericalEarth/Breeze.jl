# Reverse-mode gradient of the mean kinetic energy of an anelastic model after a few steps
# with respect to the initial potential temperature, with Reactant + Enzyme, checked
# against central finite differences computed with the CPU model.
#
#   julia --project anelastic_gradient.jl
using Breeze
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using Reactant: @trace
using Enzyme
using CUDA
using GPUArraysCore: @allowscalar
using Statistics: mean
using Printf
using Random

Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", "cpu"))
Random.seed!(42)

const t₀ = time()
stage(s) = (@printf("[%6.1f s, maxrss %.1f GB] %s\n", time() - t₀, Sys.maxrss() / 1e9, s); flush(stdout))

N  = parse(Int, get(ENV, "N", "8"))
Ns = parse(Int, get(ENV, "NSTEPS", "4"))
Δt = 1.0

function anelastic_model(arch)
    grid = RectilinearGrid(arch; size=(N, N, N), halo=(5, 5, 5), extent=(1e3, 1e3, 1e3),
                           topology=(Periodic, Periodic, Bounded))
    reference_state = ReferenceState(grid, ThermodynamicConstants(); base_pressure=1e5, potential_temperature=300)
    dynamics = AnelasticDynamics(reference_state)
    return AtmosphereModel(grid; dynamics, advection=WENO(order=5))
end

kinetic_energy(model) = mean(interior(model.velocities.u) .^ 2) +
                        mean(interior(model.velocities.v) .^ 2) +
                        mean(interior(model.velocities.w) .^ 2)

function loss(model, θ_init, Δt, Ns)
    set!(model; θ=θ_init, u=0, v=0, w=0)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Ns
        time_step!(model, Δt)
    end
    return kinetic_energy(model) / 2
end

function grad_loss(model, dmodel, θ_init, dθ_init, Δt, Ns)
    parent(dθ_init) .= 0
    _, J = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                           loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel),
                           Enzyme.Duplicated(θ_init, dθ_init),
                           Enzyme.Const(Δt),
                           Enzyme.Const(Ns))
    return dθ_init, J
end

θ₀ = 300 .+ randn(N, N, N)

# CPU reference
cpu_model = anelastic_model(CPU())
cpu_θ = CenterField(cpu_model.grid)
function cpu_loss(θ)
    set!(cpu_θ, θ)
    set!(cpu_model; θ=cpu_θ, u=0, v=0, w=0)
    for _ in 1:Ns
        time_step!(cpu_model, Δt)
    end
    return kinetic_energy(cpu_model) / 2
end
J_cpu = cpu_loss(θ₀)
stage(@sprintf("CPU J = %.12e", J_cpu))

model = anelastic_model(ReactantState())
stage("Reactant model built")
θ_init = CenterField(model.grid)
dθ_init = CenterField(model.grid)
set!(θ_init, θ₀)
dmodel = Enzyme.make_zero(model)

compiled_loss = Reactant.@compile raise=true sync=true loss(model, θ_init, Δt, Ns)
J_reactant = Float64(compiled_loss(model, θ_init, Δt, Ns))
stage(@sprintf("Reactant forward J = %.12e (CPU %.12e, rel. diff %.2e)", J_reactant, J_cpu, abs(J_reactant - J_cpu) / J_cpu))

compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(model, dmodel, θ_init, dθ_init, Δt, Ns)
stage("gradient compiled")
dθ, J_ad = compiled_grad(model, dmodel, θ_init, dθ_init, Δt, Ns)
∂J∂θ = @allowscalar Array(interior(dθ))
stage(@sprintf("Reactant AD J = %.12e, max|∂J/∂θ| = %.6e", Float64(J_ad), maximum(abs, ∂J∂θ)))

h = 1e-4
println("     i  j  k        AD               FD (CPU)          rel. error")
max_error = 0.0
for (i, j, k) in ((1, 1, 1), (3, 5, 2), (4, 4, 4), (8, 2, 7), (5, 7, N), argmax(abs.(∂J∂θ)).I)
    θ⁺ = copy(θ₀); θ⁺[i, j, k] += h
    θ⁻ = copy(θ₀); θ⁻[i, j, k] -= h
    fd = (cpu_loss(θ⁺) - cpu_loss(θ⁻)) / 2h
    ad = ∂J∂θ[i, j, k]
    error = abs(ad - fd) / max(abs(fd), 1e-30)
    global max_error = max(max_error, error)
    @printf("  %4d %2d %2d  % .10e  % .10e  %.2e\n", i, j, k, ad, fd, error)
end
stage(@sprintf("max relative error AD vs FD: %.2e", max_error))
