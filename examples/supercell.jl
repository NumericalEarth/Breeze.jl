#include("../src/AtmosphereModels/compute_hydrostatic_pressure.jl")
using Breeze
using Oceananigans.Units
using Statistics
using Printf
using CairoMakie

using Oceananigans.Grids: znode
using Oceananigans: Center, Face
using Oceananigans.Operators: Δzᶜᶜᶜ, Δzᶜᶜᶠ, ℑzᵃᵃᶠ
using Breeze.Thermodynamics: dry_air_gas_constant
using CUDA

using CloudMicrophysics
import Breeze: Breeze

# Access extension module and define aliases to avoid namespace conflicts
if !isdefined(@__MODULE__, :BreezeCloudMicrophysicsExt)
    const BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
end

if !isdefined(@__MODULE__, :BreezeZeroMomentCloudMicrophysics)
    const BreezeZeroMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.ZeroMomentCloudMicrophysics
end

if !isdefined(@__MODULE__, :BreezeOneMomentCloudMicrophysics)
    const BreezeOneMomentCloudMicrophysics  = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics
end

const c = Center()
const f = Face()

# Supercell simulation
# Reference:
# Klemp et al. (2015): "Idealized global nonhydrostatic atmospheric test cases on a reduced-radius sphere"

#  grid configuration
Nx, Ny, Nz = 336, 336, 40
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers

grid = RectilinearGrid(GPU(),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))


# Problem parameters and initial conditions
p₀, θ₀ = 100000, 300
thermo = ThermodynamicConstants()
reference_state = ReferenceState(grid, thermo, base_pressure=p₀, potential_temperature=θ₀)
formulation = AnelasticFormulation(reference_state)


θ_tr = 343
z_tr = 12000
q₀ = 14e-3
T_tr = 213
z_s = 5kilometers
u_s = 30
u_c = 15
g = thermo.gravitational_acceleration
cᵖᵈ = thermo.dry_air.heat_capacity
Rᵈ = dry_air_gas_constant(thermo)
θ̄(x, y, z) = (θ₀ + (θ_tr-θ₀) * (z / z_tr)^(5/4)) * (z <= z_tr) + θ_tr * exp(g/(cᵖᵈ*T_tr) * (z - z_tr)) * (z > z_tr)
RHᵢ(z) = (1 - 3/4 * (z / z_tr)^(5/4)) * (z <= z_tr) + 1/4 * (z > z_tr)
uᵢ(x, y, z) = (u_s*(z/z_s)- u_c) * (z < (z_s-1000))  + ((-4/5 + 3 *(z/z_s) - 5/4 *(z/z_s)^2) * u_s - u_c) * (abs(z - z_s) < 1000) + (u_s - u_c) * (z > (z_s+1000))

# Warm bubble potential temperature perturbation (Eq. 17–18)
Δθ = 3           # K amplitude
r_h = 10kilometers
r_z = 1500       # m vertical radius
z_c = 1500       # m bubble center height
x_c = Lx / 2
y_c = Ly / 2

θᵢ(x, y, z) = begin
    θ_base = θ̄(x, y, z)
    r = sqrt((x - x_c)^2 + (y - y_c)^2)
    Rθ = sqrt((r / r_h)^2 + ((z - z_c) / r_z)^2)
    θ_pert = Rθ < 1 ? Δθ * cos((π / 2) * Rθ)^2 : 0
    return θ_base + θ_pert
end

   
# Atmosphere model setup

microphysics = BreezeOneMomentCloudMicrophysics()

model = AtmosphereModel(grid;formulation, microphysics, advection = WENO(order=5))
set!(model, θ = θᵢ)

ph = Breeze.AtmosphereModels.compute_hydrostatic_pressure!(CenterField(grid), model)
T = model.temperature

# Saturation mixing ratio (kg/kg) and water vapor initial condition
qᵛˢ = CenterField(grid)
ph_host = Array(parent(ph)) # bring to CPU to avoid GPU scalar indexing
T_host = Array(parent(T))
qᵛˢ_host = similar(ph_host)

for k in axes(qᵛˢ_host, 3), j in axes(qᵛˢ_host, 2), i in axes(qᵛˢ_host, 1)
    z = znode(i, j, k, grid, c, c, c)
    T_eq = @inbounds T_host[i, j, k]
    p_eq = @inbounds ph_host[i, j, k]
    local saturation = 380 / p_eq * exp(17.27 * ((T_eq - 273) / (T_eq - 36)))
    @inbounds qᵛˢ_host[i, j, k] = RHᵢ(z) * saturation
end

copyto!(parent(qᵛˢ), qᵛˢ_host)

set!(model, qᵗ = qᵛˢ, θ = θᵢ, u = uᵢ)

θ = Breeze.AtmosphereModels.PotentialTemperatureField(model)
qˡ = model.microphysical_fields.qᶜˡ
qⁱ = model.microphysical_fields.qᶜⁱ
qᵛ = model.microphysical_fields.qᵛ
qᵗ = model.specific_moisture


simulation = Simulation(model; Δt=2, stop_time=2hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

function progress(sim)
    u, v, w = sim.model.velocities
    qᵗ = sim.model.specific_moisture
    ρe = energy_density(sim.model)
    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|u,v,w|: (%.2f, %.2f, %.2f) m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w))
    msg *= @sprintf(", max(qᵗ): %.2e, extrema(ρe): (%.3e, %.3e)",
                    maximum(qᵗ), minimum(ρe), maximum(ρe))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

outputs = merge(model.velocities, model.tracers, (; θ, qˡ, qᵛ))

filename = "supercell.jld2"

ow = JLD2Writer(model, outputs; filename,
                schedule = TimeInterval(5minutes),
                overwrite_existing = true)


simulation.output_writers[:jld2] = ow

run!(simulation)
