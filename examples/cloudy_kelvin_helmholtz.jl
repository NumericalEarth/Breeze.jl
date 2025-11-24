# # Cloudy Kelvin-Helmholtz instability
#
# This example sets up a two-dimensional (``x``–``z``) Kelvin–Helmholtz instability
# in a moist, stably stratified atmosphere.
#
# The configuration is intentionally simple but reasonably "meteorological":
#
# - We impose horizontal wind ``U(z)`` with a shear layer.
# - We impose a stably stratified potential temperature profile ``θ(z)`` with
#   a specified dry [Brunt–Väisälä frequency](https://en.wikipedia.org/wiki/Brunt–Väisälä_frequency) ``N``.
# - We embed a Gaussian moisture layer ``q(z)`` centered on the shear layer.
#
# As the shear layer rolls up, the moist layer is advected and deformed,
# producing billow-like patterns reminiscent of observed "wave clouds".
# Breeze encapsulates much of this thermodynamics for us via the
# `AtmosphereModel` and saturation adjustment.

using Breeze
using Oceananigans.Units
using CairoMakie
using Printf

# ## Domain and grid
#
# We use a 2D ``x``–``z`` slice with periodic boundaries in ``x`` and rigid, impermeable
# boundaries at the top and bottom.
#
# Grid resolution is modest but enough to clearly resolve the Kelvin-Helmholtz billows and
# rolled-up moisture filament.

Nx = 384   # horizontal resolution
Nz = 128   # vertical resolution

Lx = 10e3  # domain length
Lz =  3e3  # domain height

grid = RectilinearGrid(; size = (Nx, Nz), x = (0, Lx), z = (0, Lz),
                         topology = (Periodic, Flat, Bounded))

# ## Model and microphysics
# We construct the AtmosphereModel model with saturation adjustment microphysics.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
model = AtmosphereModel(grid; advection=WENO(order=5), microphysics)

# ## Background thermodynamic state
#
# We set a reference potential temperature ``θ₀`` and a linear ``θ`` gradient
# that corresponds to a desired dry Brunt–Väisälä frequency ``N``. For a dry
# atmosphere,
#
# ```math
# N² = \frac{g}{θ₀} \frac{∂θ}{∂z} ,
# ```
#
# We initialize the potential temperature that gives constant Brunt–Väisälä frequency,
# representative of mid-tropospheric stability. The (dry) Brunt–Väisälä frequency is
#
# ```math
# N² = \frac{g}{θ} \frac{∂θ}{∂z}
# ```
#
# and thus, for constant ``N²`` the above implies ``θ = θ₀ \exp{(N² z / g)}``.

thermo = ThermodynamicConstants()
g = thermo.gravitational_acceleration
θ₀ = model.formulation.reference_state.potential_temperature
N = 0.01                  # target dry Brunt–Väisälä frequency (s⁻¹)
θᵇ(z) = θ₀ * exp(N^2 * z / g)

# ## Shear and moisture profiles
#
# We want:
#
# - A shear layer centered at height ``z₀`` with the zonal flow transitioning from a lower
#   speed ``U_{\rm bot}`` to an upper speed ``U_{\rm top}``.
# - A moist layer centered at the same height with a Gaussian profile.
#
# The above  mimics a moist, stably stratified layer embedded in stronger flow
# above and weaker flow below.

# First, we set up the shear layer using a ``\tanh`` profile:

z₀    = 1e3     # center of shear & moist layer (m)
Δzᶸ   = 150     # shear layer half-thickness (m)
U_top = 25      # upper-layer wind (m/s)
U_bot =  5      # lower-layer wind (m/s)
uᵇ(z) = U_bot + (U_top - U_bot) * (1 + tanh((z - z₀) / Δzᶸ)) / 2

# For the moisture layer, we use a Gaussian in ``z`` centered at ``z₀``:

q_max = 0.012  # peak specific humidity (kg/kg)
Δz_q = 200     # moist layer half-width (m)
qᵇ(z) = q_max * exp(-(z - z₀)^2 / 2Δz_q^2)

# ## The Kelvin-Helmholtz instability
#
# The Miles–Howard criterion tells us that Kelvin–Helmholtz instability
# occurs where the Richardson number,
#
# ```math
# Ri = \frac{N²}{(∂uᵇ/∂z)²}
# ```
#
# is less than 1/4 [Miles1961, Howard1961](@cite). With the parameters chosen
# above this is the case.
#
# Let's plot the initial state as well as the Richardson number.

z = znodes(grid, Center())

dudz = @. (U_top - U_bot) * sech((z - z₀) / Δzᶸ)^2 / 2Δzᶸ
Ri = N^2 ./ dudz.^2

using CairoMakie

fig = Figure(size=(800, 500))

axu = Axis(fig[1, 1], xlabel = "uᵇ (m/s)", ylabel = "z (m)", title = "Zonal velocity")
axq = Axis(fig[1, 2], xlabel = "qᵇ (kg/kg)", title="Total liquid")
axθ = Axis(fig[1, 3], xlabel = "θᵇ (K)", title="Potential temperature")
axR = Axis(fig[1, 4], xlabel = "Ri", ylabel="z (m)", title="Richardson number")

lines!(axu, uᵇ.(z), z)
lines!(axq, qᵇ.(z), z)
lines!(axθ, θᵇ.(z), z)
lines!(axR, Ri, z)
lines!(axR, [1/4, 1/4], [0, Lz], linestyle = :dash, color = :black)
xlims!(axR, 0, 0.8)
axR.xticks = 0:0.25:1

for ax in (axq, axθ, axR)
    ax.yticksvisible = false
    ax.yticklabelsvisible = false
    ax.ylabelvisible = false
end

fig

# ## Define initial conditions
#
# We initialize the model via Oceananigans `set!`, adding also a bit of random noise.

δθ = 0.01
δu = 1e-3
δq = 0.05 * q_max

θᵢ(x, z) = θᵇ(z) + δθ * rand()
qᵗᵢ(x, z) = qᵇ(z) + δq * rand()
uᵢ(x, z) = uᵇ(z) + δu * rand()

set!(model; u=uᵢ, qᵗ=qᵗᵢ, θ=θᵢ)

# ## Set up and run the simulation
#
# We construct a simulation and use the time-step wizard to keep the CFL number under control.

stop_time = 12minutes   # total simulation time
simulation = Simulation(model; Δt=1, stop_time)
conjure_time_step_wizard!(simulation; cfl = 0.7)

# We also add a progress callback:

function progress(sim)
    u, v, w = model.velocities
    max_w = maximum(abs, w)
    @info @sprintf("iteration: %d, time: %s, Δt: %s, max|w|: %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), max_w)
    return nothing
end

add_callback!(simulation, progress, TimeInterval(1minute))

# ## Output
# We save the model velocities, the cross-stream component of vorticity, ``ξ = ∂_z u - ∂_x w``,
# the potential temperatures and the specific humidities (vapour, liquid, ice).
u, v, w = model.velocities
ξ = ∂z(u) - ∂x(w)
θ = PotentialTemperatureField(model)
outputs = merge(model.velocities, model.microphysical_fields, (; ξ, θ))

filename = "wave_clouds.jld2"

output_writer = JLD2Writer(model, outputs;
                           filename,
                           schedule = TimeInterval(4),
                           overwrite_existing = true)

simulation.output_writers[:fields] = output_writer

# ## Run!
# Now we are ready to run the simulation.
run!(simulation)

# ## Read output and visualize

# We load the saved output as Oceananigans' `FieldTimeSeries`

ξt = FieldTimeSeries(filename, "ξ")
θt = FieldTimeSeries(filename, "θ")
qˡt = FieldTimeSeries(filename, "qˡ")

times = ξt.times
Nt = length(ξt)

# and then use CairoMakie to plot and animate the output.

n = Observable(Nt)

ξn = @lift ξt[$n]
θn = @lift θt[$n]
qˡn = @lift qˡt[$n]

fig = Figure(size=(800, 800), fontsize=14)

axξ = Axis(fig[1, 1], ylabel="z (m)", title = "Vorticity", titlesize = 20)
axl = Axis(fig[2, 1], ylabel="z (m)", title = "Liquid mass fraction", titlesize = 20)
axθ = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)", title = "Potential temperature", titlesize = 20)

hmξ = heatmap!(axξ, ξn, colormap = :balance, colorrange = (-0.25, 0.25))
hml = heatmap!(axl, qˡn, colormap = Reverse(:Blues_4), colorrange = (0, 0.003))
hmθ = heatmap!(axθ, θn, colormap = :thermal, colorrange = (θ₀, θ₀+ dθdz * grid.Lz))

Colorbar(fig[1, 2], hmξ, label = "s⁻¹", vertical = true)
Colorbar(fig[2, 2], hml, label = "kg/kg", vertical = true)
Colorbar(fig[3, 2], hmθ, label = "Κ", vertical = true)

fig

# We can also make a movie:

CairoMakie.record(fig, "wave_clouds.mp4", 1:Nt, framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](wave_clouds.mp4)
