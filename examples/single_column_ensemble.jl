# # Single-column ensemble: a forest of independent columns
#
# Breeze can run an `AtmosphereModel` in "single column mode" on a grid with
# `topology = (Flat, Flat, Bounded)`. All horizontal terms vanish, the anelastic
# pressure solve and vertical-velocity stepping are skipped (`w ≡ 0`), and vertical
# transport is carried entirely by the turbulence closure. When the horizontal
# dimensions are given a size greater than one — with `ColumnEnsembleSize` — the model
# becomes a *forest* of independent columns advanced concurrently in a single kernel
# launch, with no coupling between them. This is ideal for ensembles: parameter sweeps,
# closure calibration, and boundary-layer scheme development.
#
# Here we build an ensemble of columns that differ only in their vertical diffusivity —
# the classic calibration knob — and watch each column mix a warm near-surface layer to a
# different depth.

using Breeze
using Oceananigans
using Oceananigans.Grids: ColumnEnsembleSize
using Oceananigans.Units
using CairoMakie

# ## A forest of columns
#
# `ColumnEnsembleSize(Nz, ensemble=(N, 1), Hz)` lays out `N` independent columns in the
# (Flat) x-direction. The horizontal halos are zero, so the columns never exchange
# information.

Nz = 48
N = 5

grid = RectilinearGrid(size = ColumnEnsembleSize(Nz=Nz, ensemble=(N, 1), Hz=3),
                       z = (0, 3kilometers),
                       topology = (Flat, Flat, Bounded))

# ## Reference state and dynamics
#
# The columns share one anelastic reference state (a dry, neutrally stratified background).

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=290)
dynamics = AnelasticDynamics(reference_state)

# ## Per-column vertical diffusivity
#
# The closure is an *array* of closures — one per column. Each column mixes with its own
# constant vertical diffusivity, spanning two orders of magnitude.

κs = [1, 3, 10, 30, 100]  # m² s⁻¹, one per column
closures = [VerticalScalarDiffusivity(ν=κ, κ=κ) for κ in κs, j in 1:1]

model = AtmosphereModel(grid; dynamics, closure=closures)

# ## Initial condition
#
# Every column starts from the same state: a stably stratified background with a warm
# near-surface layer that the turbulent mixing will erode.

θ★ = 290       # background potential temperature (K)
Γ = 0.004      # background lapse rate (K m⁻¹)
Δθ = 6         # surface warm anomaly (K)
h = 400        # anomaly scale height (m)

θᵢ(z) = θ★ + Γ * z + Δθ * exp(-z / h)
set!(model, θ = θᵢ)

# ## Run the ensemble
#
# All `N` columns step forward together.

simulation = Simulation(model, Δt=20, stop_time=6hours)
run!(simulation)

# ## Visualize
#
# We plot the potential temperature profile of each column. Columns with larger
# diffusivity mix the warm surface layer deeper, producing a taller, better-mixed layer —
# exactly the sensitivity a calibration would exploit.

θ = model.temperature  # temperature ≈ θ for this dry, near-surface case

set_theme!(fontsize=14, linewidth=2.5)
fig = Figure(size=(700, 600))
ax = Axis(fig[1, 1]; xlabel="Temperature (K)", ylabel="Altitude (km)",
          title="A forest of single columns with different vertical mixing")

z_ticks_km = 0:0.5:3
ax.yticks = ((z_ticks_km .* 1000), string.(z_ticks_km))

colors = cgrad(:viridis, N, categorical=true)
for i in 1:N
    lines!(ax, view(θ, i, 1, :); color=colors[i], label="κ = $(κs[i]) m²/s")
end
axislegend(ax; position=:rb, framevisible=false)

fig
