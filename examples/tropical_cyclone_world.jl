# # Tropical Cyclone World (Cronin & Chavas, 2019)
#
# This example implements the rotating radiative-convective equilibrium (RCE) experiment
# from [Cronin2019](@citet). The experiment demonstrates that tropical cyclones can form
# and persist even in completely dry atmospheres, challenging the conventional wisdom
# that moisture is essential for TC dynamics.
#
# The key innovation is the surface wetness parameter β, which controls the transition
# from completely dry (β = 0, no evaporation) to fully moist (β = 1) conditions.
# [Cronin2019](@citet) found that TCs form in both limits, with a "no-storms-land" at
# intermediate β where spontaneous genesis does not occur.  This script defaults to β =
# 1 (moist), which produces robust spontaneous TC genesis at moderate resolution. The
# simulation approximates the paper's 100-day nonrotating RCE spinup with an
# equilibrated initial temperature profile (dry adiabat in the troposphere, isothermal
# stratosphere) and uses warm-phase saturation adjustment microphysics for the moist
# case.

using Breeze
using Breeze.Thermodynamics: compute_reference_state!
using Oceananigans: Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode, Center

using CairoMakie
using CUDA
using Printf
using Random

Random.seed!(2019)
Oceananigans.defaults.FloatType = Float32

# ## Domain and grid
#
# A 288 km × 288 km doubly-periodic domain with 3 km horizontal resolution
# and a 28 km model top. The vertical grid uses 62.5 m spacing in the lowest
# kilometer, 2000 m spacing above 3.5 km, and a smooth transition in between.

arch = GPU()
paper_Lx = paper_Ly = 1152e3
paper_Nx = paper_Ny = 576
Lx = Ly = paper_Lx / 4
Nx = Ny = 96
# Lx = Ly = paper_Lx / 8 |> Int 
H = 28e3

Δz_fine = 1000 / 16   # 62.5 m (paper: 1000/64 ≈ 15.6 m)
Δz_coarse = 2000      # m (paper: 500 m)

z = PiecewiseStretchedDiscretization(
    z  = [0, 1000, 3500, H],
    Δz = [Δz_fine, Δz_fine, Δz_coarse, Δz_coarse])

Nz = length(z) - 1

grid = RectilinearGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       x = (0, Lx), y = (0, Ly), z,
                       topology = (Periodic, Periodic, Bounded))

@show grid

# ## Reference state and dynamics
#
# We use the anelastic formulation with a reference state initialized from
# the surface potential temperature T₀ = 300 K and standard surface pressure.
# The reference state is then adjusted to match the initial temperature and
# moisture profiles. This adjustment is critical for tall domains: without it,
# the constant-θ adiabat reference state diverges from the actual atmosphere
# in the stratosphere (T_ref ≈ 26 K vs T_actual = 210 K at 28 km), producing
# catastrophic buoyancy forces.

T₀ = 300
constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 101325,
                                 potential_temperature = T₀,
                                 vapor_mass_fraction = 0)

# Define equilibrium temperature and moisture profiles for adjustment and initialization
Tᵗˢ = 210
cᵖᵈ = constants.dry_air.heat_capacity
g = constants.gravitational_acceleration
Rᵈ = Breeze.Thermodynamics.dry_air_gas_constant(constants)
κ = Rᵈ / cᵖᵈ
pˢᵗ = reference_state.standard_pressure
Π₀ = (101325 / pˢᵗ)^κ

# Analytical Exner function for a hydrostatic constant-θ atmosphere
Π(z) = Π₀ - g * z / (cᵖᵈ * T₀)

β = 1
q₀ = 15e-3 # surface specific humidity (kg/kg)
Hq = 3000   # moisture scale height (m)

Tᵇᵍ(z) = max(Tᵗˢ, T₀ * Π(z))
qᵇᵍ(z) = max(0, β * q₀ * exp(-z / Hq))

# Adjust reference state to match actual profiles
compute_reference_state!(reference_state, Tᵇᵍ, qᵇᵍ, constants)

dynamics = AnelasticDynamics(reference_state)
coriolis = FPlane(f = 3e-4)

# ## Surface fluxes
#
# Following the paper's bulk formulas (Eqs. 2-4), with drag coefficient
# Cᴰ = 1.5 × 10⁻³ and gustiness v★ = 1 m/s. The surface wetness parameter β
# scales the moisture flux coefficient.

Cᴰ = Cᵀ = 1.5e-3
Uᵍ = 1

ρu_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ))
ρv_bcs = FieldBoundaryConditions(bottom = BulkDrag(coefficient = Cᴰ, gustiness = Uᵍ))

ρe_bcs = FieldBoundaryConditions(bottom = BulkSensibleHeatFlux(coefficient = Cᵀ,
                                                               gustiness = Uᵍ,
                                                               surface_temperature = T₀))

ρqᵗ_bcs = FieldBoundaryConditions(bottom = BulkVaporFlux(coefficient = β*Cᵀ,
                                                         gustiness = Uᵍ,
                                                         surface_temperature = T₀))

boundary_conditions = (; ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs)
nothing #hide

# ## Radiative forcing
#
# The paper (Eq. 1) prescribes a piecewise radiative tendency: constant cooling
# at Ṫ = 1 K/day for T > Tᵗˢ (troposphere), and Newtonian relaxation toward Tᵗˢ
# with timescale τᵣ = 20 days for T ≤ Tᵗˢ (stratosphere). We apply this as an
# energy forcing on ρe, so that Breeze handles the conversion to ρθ tendency.

Ṫ  = 1 / day
τᵣ = 20days

FT = eltype(grid)
ρᵣ = reference_state.density

forcing_params = (; Tᵗˢ, Ṫ, τᵣ, ρᵣ, cᵖᵈ)

@inline function piecewise_T_forcing(i, j, k, grid, clock, model_fields, p)
    @inbounds T = model_fields.T[i, j, k]
    @inbounds ρ = p.ρᵣ[i, j, k]
    ∂T∂t = ifelse(T > p.Tᵗˢ, -p.Ṫ, (p.Tᵗˢ - T) / p.τᵣ)
    return ρ * p.cᵖᵈ * ∂T∂t
end

ρe_forcing = Forcing(piecewise_T_forcing;
                     discrete_form = true,
                     parameters = forcing_params)

# ## Sponge layer
#
# Rayleigh damping with a Gaussian profile centered at 26 km (width 2 km)
# prevents spurious wave reflections from the rigid lid.

sponge_mask = GaussianMask{:z}(center=26000, width=2000)
ρw_sponge = Relaxation(rate=1/100, mask=sponge_mask)

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
# troposphere transitioning to an isothermal stratosphere at Tᵗˢ = 210 K.
# This approximates the paper's 100-day nonrotating RCE spinup. Small random
# perturbations in the lowest kilometer trigger convection.
#
# **Important:** After `compute_reference_state!`, we must use `set!(model, T=...)` rather than
# `set!(model, θ=...)`. The `compute_reference_state!` call recomputes the reference pressure,
# which changes the Exner function used to convert θ → T. Setting θ directly
# would produce incorrect temperatures in the stratosphere.

δT = 1//2  # K perturbation amplitude
zδ = 1000  # m perturbation depth
δq = 1e-4  # moisture perturbation amplitude (kg/kg)

Tᵢ(x, y, z) = Tᵇᵍ(z) + δT * (2rand() - 1) * (z < zδ)
qᵗᵢ(x, y, z) = max(0, qᵇᵍ(z) + δq * (2rand() - 1) * (z < zδ))

set!(model, T = Tᵢ, qᵗ = qᵗᵢ)

# ## Simulation
#
# We run for 8 days, which is sufficient for moist TC genesis and intensification.

simulation = Simulation(model; Δt=1, stop_time=8days)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)
s₀ = Field(sqrt(u^2 + v^2), indices = (:, :, 1))

function progress(sim)
    compute!(s₀)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)
    s₀max = maximum(s₀)
    θmin, θmax = extrema(θ)
    msg = @sprintf("(%d) t = %s, Δt = %s",
                   iteration(sim), prettytime(sim, false), prettytime(sim.Δt, false))
    msg *= @sprintf(", s₀ = %.1f m/s, max|U| ≈ (%d, %d, %d) m/s, θ ∈ [%d, %d] K",
                    s₀max, umax, vmax, wmax, floor(θmin), ceil(θmax))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# Horizontally-averaged profiles.

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
                                                  schedule = TimeInterval(12hour),
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
# Evolution of horizontally-averaged potential temperature and velocity profiles.

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

Legend(fig[1, 4], axθ, labelsize=10)

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
Ulim = Umax / 2 

fig = Figure(size=(1200, 400), fontsize=12)

hms = []
indices = round.(Int, [Nt / 3, 2Nt / 3, Nt])

for (i, idx) in enumerate(indices)
    xlabel = i == 1 ? "x (m)" : ""
    ylabel = i == 1 ? "y (m)" : ""
    title = "t = $(prettytime(times[idx]))"
    ax = Axis(fig[1, i]; aspect = 1, xlabel, ylabel, title)
    hm = heatmap!(ax, compute_speed!(idx); colormap=:speed, colorrange=(0, Umax / 2))
    push!(hms, hm)
end

Colorbar(fig[1, length(indices) + 1], hms[end]; label="Surface wind speed (m/s)")

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

hm = heatmap!(ax, Un; colormap=:speed, colorrange=(0, Umax / 2))
Colorbar(fig[1, 2], hm; label="Wind speed (m/s)")
fig[0, :] = Label(fig, title, fontsize=16, tellwidth=false)

CairoMakie.record(fig, "tc_world.mp4", 1:Nt, framerate=16) do nn
    n[] = nn
end
nothing #hide

# ![](tc_world.mp4)

# ## Discussion
#
# This example demonstrates spontaneous tropical cyclone genesis in a rotating
# radiative-convective equilibrium setup, following [Cronin2019](@citet).
# The surface wetness parameter β controls moisture availability: β = 1 (default)
# produces robust moist TC genesis, while β = 0 yields dry TCs.
#
# The radiative forcing is a piecewise temperature tendency: constant cooling
# at 1 K/day in the troposphere (T > Tᵗˢ) and Newtonian relaxation toward Tᵗˢ
# with timescale τᵣ = 20 days in the stratosphere. Surface fluxes use bulk
# formulas with drag coefficient Cᴰ = 1.5 × 10⁻³ and gustiness 1 m/s.
# The f-plane Coriolis parameter is f₀ = 3 × 10⁻⁴ s⁻¹.
