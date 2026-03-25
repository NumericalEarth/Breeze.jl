using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znodes

using CairoMakie
using CUDA
using Printf

Oceananigans.defaults.FloatType = Float32

Nx, Ny, Nz = 168, 168, 40
Lx, Ly, Lz = 168kilometers, 168kilometers, 20kilometers

grid = RectilinearGrid(GPU(),
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 100000,
                                 potential_temperature = 300)

dynamics = AnelasticDynamics(reference_state)

θ₀ = 300       # K - surface potential temperature
θᵖ = 343       # K - tropopause potential temperature
zᵖ = 12000     # m - tropopause height
Tᵖ = 213       # K - tropopause temperature
zˢ = 5kilometers  # m - shear layer height
uˢ = 30           # m/s - maximum shear wind speed
uᶜ = 15           # m/s - storm motion (Galilean translation speed)

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z <= zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

ℋ_background(z) = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z <= zᵖ) + 1/4 * (z > zᵖ)

function u_background(z)
    uˡ = uˢ * (z / zˢ) - uᶜ
    uᵗ = (-4/5 + 3 * (z / zˢ) - 5/4 * (z / zˢ)^2) * uˢ - uᶜ
    uᵘ = uˢ - uᶜ
    return (z < (zˢ - 1000)) * uˡ +
           (abs(z - zˢ) <= 1000) * uᵗ +
           (z > (zˢ + 1000)) * uᵘ
end

Δθ = 3              # K - perturbation amplitude
rᵇʰ = 10kilometers  # m - bubble horizontal radius
rᵇᵛ = 1500          # m - bubble vertical radius
zᵇ = 1500           # m - bubble center height
xᵇ = Lx / 2         # m - bubble center x-coordinate
yᵇ = Ly / 2         # m - bubble center y-coordinate

function θᵢ(x, y, z)
    θ̄ = θ_background(z)
    r = sqrt((x - xᵇ)^2 + (y - yᵇ)^2)
    R = sqrt((r / rᵇʰ)^2 + ((z - zᵇ) / rᵇᵛ)^2)
    θ′ = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, y, z) = u_background(z)

microphysics = DCMIP2016KesslerMicrophysics()
advection = WENO(order=5)
model = AtmosphereModel(grid; dynamics, microphysics, advection, thermodynamic_constants=constants)

ℋᵢ(x, y, z) = ℋ_background(z)
set!(model, θ=θᵢ, ℋ=ℋᵢ, u=uᵢ)

function many_time_steps!(model, Nt=10)
    for n = 1:Nt
        time_step!(model, 0.1)
    end
    return nothing
end

@time many_time_steps!(model)
@time many_time_steps!(model)
@time many_time_steps!(model)
