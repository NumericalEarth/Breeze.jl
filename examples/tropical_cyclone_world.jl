# # Tropical Cyclone World (Cronin & Chavas, 2019)
#
# This example implements the rotating radiative-convective equilibrium (RCE) experiment
# from [Cronin and Chavas (2019)](@cite Cronin2019). The experiment demonstrates that
# tropical cyclones can form and persist even in completely dry atmospheres, challenging
# the conventional wisdom that moisture is essential for TC dynamics.
#
# The key innovation is the surface wetness parameter β, which controls the transition
# from completely dry (β = 0, no evaporation) to fully moist (β = 1) conditions.
# [Cronin and Chavas (2019)](@citet Cronin2019) found that TCs form in both limits,
# with a "no-storms-land" at intermediate β where spontaneous genesis does not occur.
# This script defaults to β = 1 (moist), which produces robust spontaneous TC genesis
# at moderate resolution. The simulation approximates the paper's 100-day nonrotating
# RCE spinup with an equilibrated initial temperature profile (dry adiabat in the
# troposphere, isothermal stratosphere) and uses warm-phase saturation adjustment
# microphysics for the moist case.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units

using CairoMakie
using CUDA
using Printf
using Random

Random.seed!(2019)

# ## Domain and grid
#
# The paper uses a 1152 km × 1152 km doubly-periodic domain with 2 km horizontal
# resolution and a 28 km model top. We use 8 km horizontal resolution for testing.
# The vertical grid follows the paper's Section 2a specification:
# 64 levels in the lowest kilometer, 500 m spacing above 3.5 km,
# and a linear transition in between. For testing we use coarser vertical
# resolution (16 levels in the lowest km, 2000 m spacing above).

arch = GPU()
Lx = Ly = 1152e3
H = 28e3
Nx = Ny = 144

Δz_fine = 1000 / 16   # 62.5 m (paper: 1000/64 ≈ 15.6 m)
Δz_coarse = 2000      # m (paper: 500 m)

z = PiecewiseStretchedDiscretization(
    z  = [0, 1000, 3500, H],
    Δz = [Δz_fine, Δz_fine, Δz_coarse, Δz_coarse])

Nz = length(z) - 1

grid = RectilinearGrid(arch;
                       size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z,
                       halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and dynamics
#
# We use the anelastic formulation with a reference state derived from the
# surface potential temperature T₀ = 300 K and standard surface pressure.

T₀ = 300
constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = T₀)

dynamics = AnelasticDynamics(reference_state)
coriolis = FPlane(f = 3e-4)

# ## Surface fluxes
#
# Following the paper's bulk formulas (Eqs. 2-4), with drag coefficient
# Cᴰ = 1.5 × 10⁻³ and gustiness v★ = 1 m/s. The surface wetness parameter β
# scales the moisture flux coefficient.

β = 1
Cᴰ = 1.5e-3
Cᵀ = Cᴰ
v★ = 1

ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=v★))
ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient=Cᴰ, gustiness=v★))

ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient=Cᵀ,
                                                                gustiness=v★,
                                                                surface_temperature=T₀))

ρqᵗ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient=Cᵀ * β,
                                                          gustiness=v★,
                                                          surface_temperature=T₀))

boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs)
nothing #hide

# ## Radiative forcing
#
# The paper (Eq. 1) prescribes a piecewise radiative tendency: constant cooling
# at Ṫ = 1 K/day for T > Tᵗˢ (troposphere), and Newtonian relaxation toward Tᵗˢ
# with timescale τᵣ = 20 days for T ≤ Tᵗˢ (stratosphere). We apply this as an
# energy forcing on ρe, so that Breeze handles the conversion to ρθ tendency.

Tᵗˢ = 210
Ṫ  = 1 / day
τᵣ = 20days

FT = eltype(grid)
ρᵣ = reference_state.density
cᵖᵈ = constants.dry_air.heat_capacity

forcing_params = (; Tᵗˢ=FT(Tᵗˢ), Ṫ=FT(Ṫ), τ=FT(τᵣ), ρᵣ, cₚ=FT(cᵖᵈ))

@inline function piecewise_T_forcing(i, j, k, grid, clock, model_fields, p)
    @inbounds T = model_fields.T[i, j, k]
    @inbounds ρ = p.ρᵣ[i, j, k]
    ∂T∂t = ifelse(T > p.Tᵗˢ, -p.Ṫ, (p.Tᵗˢ - T) / p.τ)
    return ρ * p.cₚ * ∂T∂t
end

ρe_forcing = Forcing(piecewise_T_forcing;
                     discrete_form = true,
                     parameters = forcing_params)

# ## Sponge layer
#
# Rayleigh damping in the upper 3 km prevents spurious wave reflections
# from the rigid lid.

sponge_mask = GaussianMask{:z}(center = H - 1500, width = 3000)
ρw_sponge = Relaxation(rate = 1/60, mask = sponge_mask)
forcing = (; ρe=ρe_forcing, ρw=ρw_sponge)
nothing #hide

# ## Model
#
# We use 9th-order WENO advection and warm-phase saturation adjustment microphysics.

advection = WENO(order=9)
microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())

model = AtmosphereModel(grid; dynamics, coriolis, advection,
                        microphysics, forcing, boundary_conditions)

# ## Initial conditions
#
# We initialize with an equilibrated temperature profile: a dry adiabat in the
# troposphere (θ = θ₀) transitioning to an isothermal stratosphere at Tᵗˢ = 210 K.
# This approximates the paper's 100-day nonrotating RCE spinup. Small random
# perturbations in the lowest kilometer trigger convection.

θ₀ = T₀
g = constants.gravitational_acceleration
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
κ = Rᵈ / cᵖᵈ
pˢᵗ = reference_state.standard_pressure
Π₀ = (101325 / pˢᵗ)^κ

# Analytical Exner function for a hydrostatic constant-θ atmosphere
Π(z) = Π₀ - g * z / (cᵖᵈ * θ₀)

function θ_equilibrium(z)
    T_adiabat = θ₀ * Π(z)
    return ifelse(T_adiabat > Tᵗˢ, θ₀, Tᵗˢ / Π(z))
end

δθ = 1//2  # K
zδ = 1000  # m

θᵢ(x, y, z) = θ_equilibrium(z) + δθ * (2rand() - 1) * (z < zδ)

q₀ = 15e-3 # surface specific humidity (kg/kg)
Hq = 3000   # moisture scale height (m)
δq = 1e-4   # perturbation amplitude (kg/kg)
qᵢ(x, y, z) = max(0, β * q₀ * exp(-z / Hq) + δq * (2rand() - 1) * (z < zδ))

set!(model, θ=θᵢ, qᵗ=qᵢ)

# ## Simulation
#
# We run for 10 days, which is sufficient for moist TC genesis and intensification.

simulation = Simulation(model; Δt=10, stop_time=10days)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)

function progress(sim)
    wmax = maximum(abs, w)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    θmin, θmax = extrema(θ)
    msg = @sprintf("Iter %d, t = %s, Δt = %s, max|u,v,w| = (%.1f, %.1f, %.1f) m/s, θ ∈ [%.1f, %.1f] K",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   umax, vmax, wmax, θmin, θmax)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(500))

# Horizontally-averaged profiles for comparison with the paper's Figure 3.

avg_outputs = (θ = Average(θ, dims=(1, 2)),
               u = Average(u, dims=(1, 2)),
               v = Average(v, dims=(1, 2)))

function save_parameters(file, model)
    file["parameters/β"] = β
    file["parameters/T₀"] = T₀
    file["parameters/Tᵗˢ"] = Tᵗˢ
    file["parameters/Ṫ"] = Ṫ
    file["parameters/f₀"] = 3e-4
    file["parameters/Cᴰ"] = Cᴰ
    file["parameters/Nx"] = Nx
    file["parameters/Nz"] = Nz
end

simulation.output_writers[:profiles] = JLD2Writer(model, avg_outputs;
                                                  filename = "tc_world_profiles.jld2",
                                                  schedule = TimeInterval(1hour),
                                                  init = save_parameters,
                                                  overwrite_existing = true)

# Surface fields for tracking TC development.

surface_outputs = (u = view(u, :, :, 1),
                   v = view(v, :, :, 1),
                   θ = view(θ, :, :, 1))

simulation.output_writers[:surface] = JLD2Writer(model, surface_outputs;
                                                 filename = "tc_world_surface.jld2",
                                                 schedule = TimeInterval(30minutes),
                                                 overwrite_existing = true)

# ## Run

run!(simulation)

# ## Results: mean profile evolution
#
# We visualize the evolution of horizontally-averaged profiles,
# for comparison with Figure 3 in [Cronin and Chavas (2019)](@cite Cronin2019).

θt = FieldTimeSeries("tc_world_profiles.jld2", "θ")
ut = FieldTimeSeries("tc_world_profiles.jld2", "u")
vt = FieldTimeSeries("tc_world_profiles.jld2", "v")

times = θt.times
Nt = length(times)

fig = Figure(size=(900, 400), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (m)")
axu = Axis(fig[1, 2], xlabel="u (m/s)", ylabel="z (m)")
axv = Axis(fig[1, 3], xlabel="v (m/s)", ylabel="z (m)")

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(n, length(default_colours))] for n in 1:Nt]

for n in 1:Nt
    label = n == 1 ? "initial" : "t = $(prettytime(times[n]))"
    lines!(axθ, θt[n], color=colors[n], label=label)
    lines!(axu, ut[n], color=colors[n])
    lines!(axv, vt[n], color=colors[n])
end

axislegend(axθ, position=:rb, labelsize=10)

fig[0, :] = Label(fig, "TC World (β = $β): mean profile evolution",
                  fontsize=16, tellwidth=false)

save("tc_world_profiles.png", fig) #src
fig

# ## Surface wind speed snapshots
#
# Snapshots of the surface wind speed field at early, middle, and late times
# show the evolution of convective organization and TC formation.

u_ts = FieldTimeSeries("tc_world_surface.jld2", "u")
v_ts = FieldTimeSeries("tc_world_surface.jld2", "v")
times = u_ts.times
Nt = length(times)

speed(u, v) = @at (Center, Center, Center) sqrt(u^2 + v^2)

un = XFaceField(u_ts.grid)
vn = YFaceField(u_ts.grid)
U = Field(speed(un, vn))

function compute_speed!(n)
    parent(un) .= parent(u_ts[n])
    parent(vn) .= parent(v_ts[n])
    compute!(U)
    return Array(interior(U, :, :, 1))
end

Umax = maximum(maximum(compute_speed!(n)) for n in 1:Nt)

fig = Figure(size=(1200, 400), fontsize=12)

indices = [1, max(1, Nt ÷ 2), Nt]
local hm
for (i, idx) in enumerate(indices)
    ax = Axis(fig[1, i];
              xlabel = "x (m)",
              ylabel = i == 1 ? "y (m)" : "",
              title = "t = $(prettytime(times[idx]))",
              aspect = 1)
    hm = heatmap!(ax, compute_speed!(idx); colormap=:speed, colorrange=(0, Umax))
end
Colorbar(fig[1, length(indices) + 1], hm; label="Surface wind speed (m/s)")

fig[0, :] = Label(fig, "TC World (β = $β): surface wind speed",
                   fontsize=16, tellwidth=false)

save("tc_world_surface_winds.png", fig) #src
fig

# ## Animation of surface wind speed

fig = Figure(size=(600, 550), fontsize=14)
ax = Axis(fig[1, 1]; xlabel="x (m)", ylabel="y (m)", aspect=1)

n = Observable(1)
title = @lift "TC World (β = $β) — t = $(prettytime(times[$n]))"
Un = @lift compute_speed!($n)

hm = heatmap!(ax, Un; colormap=:speed, colorrange=(0, Umax))
Colorbar(fig[1, 2], hm; label="Wind speed (m/s)")
fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

CairoMakie.record(fig, "tc_world.mp4", 1:Nt, framerate=16) do nn
    n[] = nn
end
nothing #hide

# ![](tc_world.mp4)

# ## Discussion
#
# [Cronin and Chavas (2019)](@citet Cronin2019) found that tropical cyclones form
# in both dry (β = 0) and moist (β = 1) limits, with a "no-storms-land" at
# intermediate surface wetness (β ≈ 0.01-0.3) where spontaneous TC genesis does
# not occur. Dry TCs have smaller outer radii but similar-sized convective cores,
# and TC intensity decreases as the surface is dried.
#
# The radiative forcing is implemented as a piecewise temperature tendency (Eq. 1):
# constant cooling at 1 K/day in the troposphere (T > Tᵗˢ) and Newtonian relaxation
# in the stratosphere (T ≤ Tᵗˢ). Surface fluxes follow bulk formulas with constant
# drag and heat exchange coefficients. The f-plane Coriolis parameter f₀ = 3 × 10⁻⁴ s⁻¹
# and a Rayleigh damping sponge layer in the upper 3 km prevent spurious reflections.
#
# For full reproduction of the paper's results, use 2 km horizontal resolution
# (`Nx = Ny = 576`, `scale_factor = 1` in `stretched_vertical_faces`) and run
# for at least 70 days. The dry case (β = 0) requires finer resolution and longer
# integration times for spontaneous genesis.
