using Breeze
using Printf

Nx = Ny = Nz = 16
x = y = z = (0, 10_000)
grid = RectilinearGrid(size = (Nx, Ny, Nz); x, y, z)

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
model = AtmosphereModel(grid; microphysics, advection = WENO(order=5), tracers = tuple(:ρc))

θ₀ = model.formulation.reference_state.potential_temperature
θᵢ(x, y, z) = θ₀ + 1e-2 * rand()
ρqᵗᵢ(x, y, z) = 2e-2 * rand()
ρcᵢ(x, y, z) = rand()
set!(model, θ=θᵢ, ρqᵗ=ρqᵗᵢ, ρc=ρcᵢ)

simulation = Simulation(model; Δt=10, stop_iteration=100)

ρE = Field(Integral(model.energy_density))
ρQᵗ = Field(Integral(model.moisture_density))
ρC = Field(Integral(model.tracers.ρc))

ρE₀ = first(ρE)
ρQᵗ₀ = first(ρQᵗ)
ρC₀ = first(ρC)

function progress(sim)
    compute!(ρE)
    compute!(ρQᵗ)
    compute!(ρC)

    δρE  = (first(ρE) - ρE₀) / ρE₀
    δρQᵗ = (first(ρQᵗ) - ρQᵗ₀) / ρQᵗ₀
    δρC  = (first(ρC) - ρC₀) / ρC₀

    msg = @sprintf("Iter: %d, t: %s, δ∫ρe: %.2e, δ∫ρqᵗ: %.2e, δ∫ρc: %.2e",
                    iteration(sim), prettytime(sim), δρE, δρQᵗ, δρC)

    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(1))
run!(simulation)

compute!(ρE)
compute!(ρQᵗ)
compute!(ρC)

δρE  = (first(ρE) - ρE₀) / ρE₀
δρQᵗ = (first(ρQᵗ) - ρQᵗ₀) / ρQᵗ₀
δρC  = (first(ρC) - ρC₀) / ρC₀

@show δρE, δρQᵗ, δρC