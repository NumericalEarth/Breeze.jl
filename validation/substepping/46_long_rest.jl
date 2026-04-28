#####
##### validation/substepping/46_long_rest.jl
#####
##### Long rest-atmosphere test (200 outer steps × Δt = 20 s = 4000 s simulated
##### time) at the new defaults. Confirms no slow secondary growth.
#####

using Breeze
using Oceananigans
using Oceananigans.TimeSteppers: update_state!
using CUDA
using Printf

const arch = CUDA.functional() ? GPU() : CPU()

const T₀ = 250.0
const Lz = 30e3
const Lh = 100e3
const Nx = Ny = 16
const Nz = 64
const G  = 9.80665
const cᵖᵈ = 1005.0

θ_ref(z) = T₀ * exp(G * z / (cᵖᵈ * T₀))

function build_default_model()
    grid = RectilinearGrid(arch;
                           size = (Nx, Ny, Nz), halo = (5, 5, 5),
                           x = (0, Lh), y = (0, Lh), z = (0, Lz),
                           topology = (Periodic, Periodic, Bounded))
    constants = ThermodynamicConstants(eltype(grid))
    td  = SplitExplicitTimeDiscretization()       # use new defaults
    dyn = CompressibleDynamics(td;
                               reference_potential_temperature = θ_ref,
                               surface_pressure = 1e5,
                               standard_pressure = 1e5)
    return AtmosphereModel(grid; dynamics = dyn,
                           thermodynamic_constants = constants,
                           timestepper = :AcousticRungeKutta3)
end

function set_rest_state!(model)
    ref = model.dynamics.reference_state
    Rᵈ  = Breeze.dry_air_gas_constant(model.thermodynamic_constants)
    parent(model.dynamics.density) .= parent(ref.density)
    ρθ_field = Breeze.AtmosphereModels.thermodynamic_density(model.formulation)
    parent(ρθ_field) .= parent(ref.pressure) ./ (Rᵈ .* parent(ref.exner_function))
    fill!(parent(model.velocities.u), 0)
    fill!(parent(model.velocities.v), 0)
    fill!(parent(model.velocities.w), 0)
    update_state!(model)
end

model = build_default_model()
set_rest_state!(model)

Δt = 20.0
n_steps = 200
sample_every = 10

@info @sprintf("Long rest-atmosphere test: Δt = %.1f s, %d steps (%.1f s simulated)",
               Δt, n_steps, n_steps * Δt)

samples = NTuple{2, Float64}[]
for n in 1:n_steps
    time_step!(model, Δt)
    if n % sample_every == 0 || n == 1
        wmax = Float64(maximum(abs, interior(model.velocities.w)))
        umax = Float64(maximum(abs, interior(model.velocities.u)))
        push!(samples, (n * Δt, wmax))
        @info @sprintf("  step %4d  t=%6.1fs  max|w|=%.3e  max|u|=%.3e", n, n * Δt, wmax, umax)
    end
end

envelope = isempty(samples) ? 0.0 : maximum(s[2] for s in samples)
final_wmax = isempty(samples) ? 0.0 : samples[end][2]

@info "===================="
@info @sprintf("Envelope (max over all samples): %.3e m/s", envelope)
@info @sprintf("Final max|w|:                    %.3e m/s", final_wmax)

# Pass: envelope and final stay within 100× machine ε ~ 1e-12 m/s for a
# well-behaved rest atmosphere.
if envelope <= 1e-10 && final_wmax <= 1e-12
    @info "✓ PASS: rest atmosphere stays at machine ε."
else
    @warn "✗ FAIL: envelope or final exceeds 1e-10 m/s."
end
