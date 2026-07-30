# # Baroclinic wave in a Cartesian channel
#
# This example simulates the growth of a baroclinic wave in a midlatitude
# ``\beta``-plane channel following the standardized test case of
# [UllrichEtAl2015b](@citet). A single zonally-uniform jet in
# thermal-wind balance with a meridional temperature gradient is seeded
# with a localized zonal-wind perturbation that triggers baroclinic
# instability, producing growing Rossby waves over roughly ten days.
#
# ## Physical setup
#
# The background zonal wind (Eq. 1 of [UllrichEtAl2015b](@citet)) is
#
# ```math
# u(y, \eta) = -u_0 \sin^2\!\!\left(\frac{\pi y}{L_y}\right)
#   \ln\eta \; \exp\!\left\{-\left(\frac{\ln\eta}{b}\right)^{\!2}\right\}
# ```
#
# where ``\eta = p/p_0`` is the pressure coordinate. The wind vanishes
# at the surface, the lateral boundaries, and near the model top,
# with a jet peaking at ~30 m/s near ``\eta \approx 0.24``.
#
# Temperature is derived from thermal-wind balance
# ``f_0\,\partial u/\partial\!\ln\eta = R^d\,\partial T/\partial y``,
# and the geopotential from hydrostatic integration. A Newton iteration
# inverts ``\Phi(y, \eta) = gz`` to recover the pressure coordinate at
# each height, giving a properly balanced initial state in ``z``
# coordinates.
#
# ### Perturbation (Eq. 12)
#
# A localized Gaussian zonal-wind perturbation centered at
# ``(x_c, y_c) = (2000\,\text{km}, 2500\,\text{km})`` seeds the
# instability.

using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie
using CUDA

# ## URJ15 parameters
#
# All values from Tables 1–2 of [UllrichEtAl2015b](@citet).

Oceananigans.defaults.FloatType = Float32
Oceananigans.defaults.gravitational_acceleration = 9.80616

constants = ThermodynamicConstants(;
    gravitational_acceleration = Oceananigans.defaults.gravitational_acceleration,
    dry_air_heat_capacity = 1004.5,
    dry_air_molar_mass = 8.314462618 / 287.0)

g   = constants.gravitational_acceleration
Rᵈ  = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
κ   = Rᵈ / cᵖᵈ

a  = 6.371229e6   # m — Earth radius
Ω  = 7.29212e-5   # s⁻¹ — Earth rotation rate
p₀ = 1e5          # Pa — surface pressure
T₀ = 288          # K — reference temperature
Γ  = 0.005        # K/m — lapse rate
b  = 2            # vertical width parameter
u₀ = 35           # m/s — reference zonal wind speed
ΔT = 4.8e5        # K — empirical stratospheric temperature parameter
ηₜ = 0.2          # tropopause η level
κ_T = Rᵈ * Γ / g  # ≈ 0.146

## Coriolis (φ₀ = 45° N)
φ₀ = π / 4
f₀ = 2Ω * sin(φ₀)
β₀ = 2Ω * cos(φ₀) / a

## Perturbation parameters
uₚ = 1            # m/s
Lₚ = 600kilometers
xc = 2000kilometers
yc = 2500kilometers

# ## Domain and grid
#
# Standard URJ15 configuration: 40 000 km × 6 000 km × 30 km,
# with ``\Delta x = \Delta y = 100`` km and ``\Delta z = 1`` km.

Lx = 40_000kilometers
Ly = 6_000kilometers
Lz = 30kilometers

Nx = 400
Ny = 60
Nz = 30

grid = RectilinearGrid(GPU();
                       size = (Nx, Ny, Nz),
                       halo = (5, 5, 5),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Bounded, Bounded))

# ## Analytic initial conditions
#
# The URJ15 balanced state is defined in ``\eta`` coordinates. We
# convert to height coordinates by inverting the geopotential
# ``\Phi(y, \eta) = gz`` with Newton iteration.

## Mean pressure coordinate from the lapse-rate atmosphere
α_exp = g / (Rᵈ * Γ)
η_mean(z) = (1 - Γ * z / T₀)^α_exp

## Horizontally averaged temperature (Eqs. 4–5)
function T_bar(η)
    T = T₀ * η^κ_T
    return ifelse(η < ηₜ, T + ΔT * (ηₜ - η)^5, T)
end

## Balanced zonal wind (Eq. 1)
function urj15_u(y, η)
    s = log(η)
    return -u₀ * sin(π * y / Ly)^2 * s * exp(-(s / b)^2)
end

## Temperature perturbation from thermal-wind balance
## ∂T/∂y = (f₀/Rᵈ) ∂u/∂(ln η)
## Integrating in y with zero y-mean:
function T_prime(y, η)
    s = log(η)
    G = (1 - 2 * s^2 / b^2) * exp(-(s / b)^2)
    I = y / 2 - Ly / (4π) * sin(2π * y / Ly) - Ly / 4
    return -(f₀ * u₀ / Rᵈ) * I * G
end

## Full temperature
T_full(y, η) = T_bar(η) + T_prime(y, η)

## Mean geopotential (hydrostatic integration of T_bar from η=1)
function Φ_bar(η)
    Φ = (g * T₀ / Γ) * (1 - η^κ_T)
    if η < ηₜ
        ## Stratospheric contribution from ΔT(ηₜ - η')⁵
        Φ -= Rᵈ * ΔT * (ηₜ^5 * log(η / ηₜ)
                         - 5 * ηₜ^4 * (η - ηₜ)
                         + 5 * ηₜ^3 * (η^2 - ηₜ^2)
                         - 10/3 * ηₜ^2 * (η^3 - ηₜ^3)
                         + 5/4 * ηₜ * (η^4 - ηₜ^4)
                         - 1/5 * (η^5 - ηₜ^5))
    end
    return Φ
end

## Geopotential perturbation (hydrostatic integration of T_prime)
## ∫₁^η T'/(η') dη' = (f₀ u₀/Rᵈ) I(y) ln(η) exp(-(ln η/b)²)  [exact]
function Φ_prime(y, η)
    s = log(η)
    return f₀ * u₀ * (y / 2 - Ly / (4π) * sin(2π * y / Ly) - Ly / 4) * s * exp(-(s / b)^2)
end

## Total geopotential
Φ_total(y, η) = Φ_bar(η) + Φ_prime(y, η)

## Newton iteration: find η(y, z) such that Φ(y, η) = gz
function η_at(y, z)
    target = g * z
    η = η_mean(z)
    for _ in 1:10
        Φ = Φ_total(y, η)
        T = T_full(y, η)
        dΦdη = -Rᵈ * T / η
        η = η - (Φ - target) / dΦdη
        η = clamp(η, 1e-8, 1.0)
    end
    return η
end

## Height-coordinate fields
function virtual_temperature(y, z)
    return T_full(y, η_at(y, z))
end

function pressure(y, z)
    return p₀ * η_at(y, z)
end

density(y, z) = pressure(y, z) / (Rᵈ * virtual_temperature(y, z))

potential_temperature_bg(y, z) = virtual_temperature(y, z) * (p₀ / pressure(y, z))^κ

function balanced_u(y, z)
    return urj15_u(y, η_at(y, z))
end

z_jet = T₀ / Γ * (1 - exp(log(0.24) / α_exp))
@info @sprintf("Peak jet speed: %.1f m/s at z ≈ %.0f m", balanced_u(Ly/2, z_jet), z_jet)

# ### Perturbation (Eq. 12)

function zonal_velocity(x, y, z)
    u_bg   = balanced_u(y, z)
    u_pert = uₚ * exp(-((x - xc)^2 + (y - yc)^2) / Lₚ^2)
    return u_bg + u_pert
end

function potential_temperature(x, y, z)
    return potential_temperature_bg(y, z)
end

function initial_density(x, y, z)
    return density(y, z)
end

# ## Model configuration
#
# The ``\beta``-plane Coriolis parameter is centered at ``y_0 = L_y/2``
# where ``f = f_0 = 2\Omega\sin(45°)``. Since Oceananigans evaluates
# ``f = f_0^{\rm oc} + \beta y`` (no offset), we set
# ``f_0^{\rm oc} = f_0 - \beta L_y/2``.

coriolis = FPlane(f=f₀)

T₀ᵣ = 250
θᵣ(z) = T₀ᵣ * exp(g * z / (cᵖᵈ * T₀ᵣ))

dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization();
                                surface_pressure = p₀,
                                reference_potential_temperature = θᵣ)

model = AtmosphereModel(grid; dynamics, coriolis,
                        thermodynamic_constants = constants,
                        advection = WENO())

# ## Set initial conditions

set!(model; θ=potential_temperature, u=zonal_velocity, ρ=initial_density)

# ## Time-stepping
#
# We run for 15 days as recommended by [UllrichEtAl2015b](@citet).

Δt = 10minutes
stop_time = 15days

simulation = Simulation(model; Δt, stop_time)
conjure_time_step_wizard!(simulation; cfl=1.4, max_Δt=10minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Progress callback

function progress(sim)
    u, v, w = sim.model.velocities
    @info @sprintf("Iter %5d | t = %s | Δt = %s | max|u| = %.1f m/s | max|w| = %.4f m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(50))

# ## Output

using Oceananigans.Operators: ζ₃ᶠᶠᶜ
u, v, w = model.velocities
ζ = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, model.grid, u, v)

θ = PotentialTemperature(model)
p = model.dynamics.pressure

outputs = merge(model.velocities, (; ζ, θ, p))

for k in (1, 15)
    filename = "cartesian_baroclinic_wave_k$k"
    ow = JLD2Writer(model, outputs; filename,
                    indices = (:, :, k),
                    schedule = TimeInterval(6hours),
                    overwrite_existing = true)

    simulation.output_writers[Symbol(filename)] = ow
end

# ## Run

run!(simulation)

# ## Visualization

p_ts = FieldTimeSeries("cartesian_baroclinic_wave_k1.jld2", "p")
times = p_ts.times

snapshot_days = [5, 10, 15]
snapshot_indices = [argmin(abs.(times .- d * 86400)) for d in snapshot_days]

fig = Figure(size = (1800, 550))

for (col, (day, idx)) in enumerate(zip(snapshot_days, snapshot_indices))
    pn = view(p_ts[idx], :, :, 1)

    ax = Axis(fig[1, 2col - 1];
              title = "Surface pressure — day $day",
              xlabel = "x (m)",
              ylabel = "y (m)",
              aspect = DataAspect())

    hm = heatmap!(ax, pn; colormap = :viridis)
    Colorbar(fig[1, 2col], hm; label = "p (Pa)", height = Relative(0.7))
end

save("cartesian_baroclinic_wave_pressure.png", fig)
nothing #hide

# ![](cartesian_baroclinic_wave_pressure.png)
