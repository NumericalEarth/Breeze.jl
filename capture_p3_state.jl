using Breeze
using Breeze: TetensFormula
using Breeze.Thermodynamics: hydrostatic_density, hydrostatic_temperature
using Oceananigans
using Oceananigans.Units
using CUDA
using Serialization

Oceananigans.defaults.FloatType = Float32
const FT = Float32

const Nx, Ny, Nz = 168, 168, 40
const Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers
const N_STEPS = parse(Int, get(ENV, "N_STEPS", "100"))
const THREE_MOMENT = parse(Bool, get(ENV, "THREE_MOMENT", "false"))
const FIXED_DT = 2.0

θ₀, θᵖ, zᵖ, Tᵖ = 300, 343, 12000, 213
qᵛ_max = 0.014
zˢ, uˢ, uᶜ = 5kilometers, 30, 15
Δθ, rᵇʰ, rᵇᵛ, zᵇ = 3, 10kilometers, 1500, 1500
xᵇ, yᵇ = Lx / 2, Ly / 2

grid = RectilinearGrid(GPU(), size=(Nx, Ny, Nz),
                       x=(0, Lx), y=(0, Ly), z=(0, Lz),
                       halo=(5, 5, 5),
                       topology=(Periodic, Periodic, Bounded))

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())
reference_state = ReferenceState(grid, constants,
                                 surface_pressure=100000,
                                 potential_temperature=300)
dynamics = AnelasticDynamics(reference_state)

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

θ_bg(z) = let θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4),
              θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    (z ≤ zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

function qᵛ_col(z)
    ℋ = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z ≤ zᵖ) + 1/4 * (z > zᵖ)
    p₀ = reference_state.surface_pressure
    pˢᵗ = reference_state.standard_pressure
    T = hydrostatic_temperature(z, p₀, θ_bg, pˢᵗ, constants)
    ρ = hydrostatic_density(z, p₀, θ_bg, pˢᵗ, constants)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return min(ℋ * qᵛ⁺, qᵛ_max)
end

qᵛ_column = Field{Nothing, Nothing, Center}(grid)
set!(qᵛ_column, qᵛ_col)

function u_bg(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) ≤ 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

function θᵢ(x, y, z)
    θ̄ = θ_bg(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos(π * R / 2)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_bg(z)

microphysics = PredictedParticlePropertiesMicrophysics(FT; three_moment_ice = THREE_MOMENT)
weno = WENO(order=9, minimum_buffer_upwind_order=3)
upwind = UpwindBiased(order=1)
scalar_advection = if THREE_MOMENT
    (ρθ = weno, ρqᵛ = weno,
     ρqᶜˡ = upwind, ρnᶜˡ = upwind,
     ρqʳ  = upwind, ρnʳ  = upwind,
     ρqⁱ  = upwind, ρnⁱ  = upwind,
     ρqᶠ  = upwind, ρbᶠ  = upwind,
     ρz̃ⁱ  = upwind, ρqʷⁱ = upwind,
     ρsˢᵃᵗ = upwind)
else
    (ρθ = weno, ρqᵛ = weno,
     ρqᶜˡ = upwind, ρnᶜˡ = upwind,
     ρqʳ  = upwind, ρnʳ  = upwind,
     ρqⁱ  = upwind, ρnⁱ  = upwind,
     ρqᶠ  = upwind, ρbᶠ  = upwind,
     ρqʷⁱ = upwind,
     ρsˢᵃᵗ = upwind)
end
model = AtmosphereModel(grid; dynamics, microphysics,
                        momentum_advection = weno, scalar_advection,
                        thermodynamic_constants=constants)
set!(model, θ=θᵢ, qᵛ=qᵛ_column, u=uᵢ)

simulation = Simulation(model; Δt=FIXED_DT, stop_iteration=N_STEPS)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

CUDA.synchronize()
run!(simulation)
CUDA.synchronize()

pf = Oceananigans.prognostic_fields(model)
state = Dict{Symbol, Array{Float32, 3}}()
for name in keys(pf)
    fld = pf[name]
    state[name] = Array(interior(fld))
end

out_path = get(ENV, "STATE_OUT", "p3_state_prep6d.bin")
serialize(out_path, state)
println("STATE_SAVED $out_path")
