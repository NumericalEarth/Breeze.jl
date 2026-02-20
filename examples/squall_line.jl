# # Two-dimensional idealized squall line
#
# This example simulates the development of a two-dimensional idealized squall line,
# following the Rotunno-Klemp-Weisman (RKW) framework [RotunnoEtAl1988](@cite).
# Squall lines are organized lines of deep convection that propagate via
# the interaction of the surface cold pool — produced by rain evaporation —
# with the ambient low-level wind shear [WeismanRotunno2004](@cite).
#
# The thermodynamic sounding is the same Weisman-Klemp profile used in the
# splitting supercell example [KlempEtAl2015](@cite), but the wind profile
# is simplified to unidirectional low-level shear, which favors line-organized
# convection rather than supercellular rotation.
#
# This case is a useful stress test for warm-rain microphysics: rain evaporating
# into the dry rear-inflow jet produces the largest evaporation tendencies of any
# standard convective benchmark.
#
# ## Physical setup
#
# ### Thermodynamic sounding
#
# The background potential temperature follows the Weisman-Klemp piecewise profile
# (Equation 14 in [KlempEtAl2015](@citet)):
#
# ```math
# θ(z) = \begin{cases}
#     θ_0 + (θ_{\rm tr} - θ_0) \left(\frac{z}{z_{\rm tr}}\right)^{5/4} & z \leq z_{\rm tr} \\
#     θ_{\rm tr} \exp\left(\frac{g}{c_p^d T_{\rm tr}} (z - z_{\rm tr})\right) & z > z_{\rm tr}
# \end{cases}
# ```
#
# with surface potential temperature ``θ_0 = 300 \, {\rm K}``,
# tropopause values ``θ_{\rm tr} = 343 \, {\rm K}``, ``z_{\rm tr} = 12 \, {\rm km}``,
# and ``T_{\rm tr} = 213 \, {\rm K}``.
#
# ### Wind profile
#
# Unlike the supercell case, the squall line uses unidirectional linear shear
# confined to the lowest 2.5 km:
#
# ```math
# u(z) = \begin{cases}
#     U_s \, z / z_s - u_c & z < z_s \\
#     U_s - u_c & z \geq z_s
# \end{cases}
# ```
#
# where ``U_s = 20 \, {\rm m/s}`` is the shear magnitude (optimal for RKW balance),
# ``z_s = 2.5 \, {\rm km}`` is the shear depth, and ``u_c = 10 \, {\rm m/s}``
# is a Galilean translation to keep the storm centered in the domain.
#
# ### Warm bubble perturbation
#
# A line-parallel warm bubble triggers convection:
#
# ```math
# θ'(x, z) = \begin{cases}
#     Δθ \cos^2\left(\frac{π}{2} R\right) & R < 1 \\
#     0 & R \geq 1
# \end{cases}
# ```
#
# where ``R = \sqrt{(x - x_c)^2 / R_x^2 + (z - z_c)^2 / R_z^2}`` is the
# normalized distance, ``Δθ = 2 \, {\rm K}``, ``R_x = 10 \, {\rm km}``,
# ``R_z = 1.5 \, {\rm km}``, and the bubble is centered at ``z_c = 1.5 \, {\rm km}``.

using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Oceananigans: Oceananigans
using Oceananigans.Units

using CairoMakie
using Printf

# ## Domain and grid
#
# The domain is 600 km × 20 km with 600 × 40 grid points, giving
# 1 km horizontal resolution and 500 m vertical resolution.
# The 2D setup uses `(Periodic, Flat, Bounded)` topology.

Oceananigans.defaults.FloatType = Float32

Nx, Nz = 600, 40
Lx, Lz = 600kilometers, 20kilometers

grid = RectilinearGrid(CPU(),
                       size = (Nx, Nz),
                       x = (0, Lx),
                       z = (0, Lz),
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))

# ## Reference state and dynamics
#
# We define the anelastic reference state with surface pressure ``p_0 = 1000 \, {\rm hPa}``
# and reference potential temperature ``θ_0 = 300 \, {\rm K}``, using the Tetens formula
# for saturation vapor pressure (as in the supercell intercomparison).

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = 100000,
                                 potential_temperature = 300)

dynamics = AnelasticDynamics(reference_state)

# ## Background atmosphere profiles
#
# The atmospheric stratification parameters follow the Weisman-Klemp sounding.

θ₀ = 300       # K — surface potential temperature
θᵖ = 343       # K — tropopause potential temperature
zᵖ = 12000     # m — tropopause height
Tᵖ = 213       # K — tropopause temperature
nothing #hide

# Wind shear parameters for the squall line environment. The shear is confined
# to the lowest 2.5 km, which is shallower than the supercell case (5 km) and
# promotes line-oriented convection rather than rotating updrafts.

zˢ = 2.5kilometers  # shear layer depth
Uˢ = 20             # m/s — shear magnitude
uᶜ = 10             # m/s — Galilean translation (≈ Uˢ/2)
nothing #hide

# Extract thermodynamic constants:

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity
nothing #hide

# Background potential temperature profile (Equation 14 in [KlempEtAl2015](@citet)):

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z <= zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

# Relative humidity profile (decreases with height, 25% above the tropopause):

ℋ_background(z) = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z <= zᵖ) + 1/4 * (z > zᵖ)

# Unidirectional zonal wind with linear shear below ``zˢ``:

u_background(z) = ifelse(z < zˢ, Uˢ * z / zˢ - uᶜ, Uˢ - uᶜ)

# ## Warm bubble perturbation
#
# The line thermal has a slightly weaker amplitude (2 K) than the supercell case (3 K),
# which is standard for squall line simulations.

Δθ = 2              # K — perturbation amplitude
Rₓ = 10kilometers   # m — bubble horizontal radius
Rz = 1500           # m — bubble vertical radius
zᵇ = 1500           # m — bubble center height
xᵇ = Lx / 2         # m — bubble center x-coordinate
nothing #hide

# The total initial potential temperature combines the background profile with the
# cosine-squared warm bubble:

function θᵢ(x, z)
    θ̄ = θ_background(z)
    R = sqrt(((x - xᵇ) / Rₓ)^2 + ((z - zᵇ) / Rz)^2)
    θ′ = ifelse(R < 1, Δθ * cos((π / 2) * R)^2, 0.0)
    return θ̄ + θ′
end

uᵢ(x, z) = u_background(z)

# ## Visualization of initial conditions
#
# We plot the background potential temperature, relative humidity, and wind profiles.

θ_profile = set!(Field{Nothing, Nothing, Center}(grid), z -> θ_background(z))
ℋ_profile = set!(Field{Nothing, Nothing, Center}(grid), z -> ℋ_background(z) * 100)
u_profile = set!(Field{Nothing, Nothing, Center}(grid), z -> u_background(z))

fig = Figure(size=(1000, 400), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θ (K)", ylabel="z (km)", title="Potential temperature")
lines!(axθ, θ_profile, linewidth=2, color=:magenta)
hlines!(axθ, [zᵖ / 1000], color=:gray, linestyle=:dash)

axℋ = Axis(fig[1, 2], xlabel="ℋ (%)", ylabel="z (km)", title="Relative humidity")
lines!(axℋ, ℋ_profile, linewidth=2, color=:dodgerblue)
hlines!(axℋ, [zᵖ / 1000], color=:gray, linestyle=:dash)

axu = Axis(fig[1, 3], xlabel="u (m/s)", ylabel="z (km)", title="Wind profile")
lines!(axu, u_profile, linewidth=2, color=:orangered)
hlines!(axu, [zˢ / 1000], color=:gray, linestyle=:dash)
vlines!(axu, [0], color=:black, linestyle=:dot)

save("squall_line_initial_conditions.png", fig) #src
fig

# ## Model setup
#
# We use DCMIP2016 Kessler microphysics with high-order WENO advection,
# and a sponge layer on vertical momentum to prevent spurious reflections
# from the rigid lid.

microphysics = DCMIP2016KesslerMicrophysics()
advection = WENO(order=9, minimum_buffer_upwind_order=3)

sponge_center = 18000
sponge_width = 2000
sponge_mask(x, z) = exp(-(z - sponge_center)^2 / (2 * sponge_width^2))
ρw_sponge = Relaxation(rate=1/30, mask=sponge_mask)
forcing = (; ρw=ρw_sponge)

model = AtmosphereModel(grid; dynamics, microphysics, advection,
                        thermodynamic_constants=constants, forcing)

# ## Model initialization
#
# We initialize with the Weisman-Klemp sounding, unidirectional shear,
# and the line-parallel warm bubble perturbation.

ℋᵢ = set!(CenterField(grid), (x, z) -> ℋ_background(z))

set!(model, θ=θᵢ, ℋ=ℋᵢ, u=uᵢ)

# ## Simulation
#
# Run for 4 hours with adaptive time stepping (CFL = 0.7):

simulation = Simulation(model; Δt=2, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

# ## Output and progress
#
# We set up callbacks to monitor simulation health. In particular, we track
# `min(qʳ)` to detect negativity issues in the microphysics.

θˡⁱ = liquid_ice_potential_temperature(model)
qᶜˡ = model.microphysical_fields.qᶜˡ
qʳ = model.microphysical_fields.qʳ
qᵛ = model.microphysical_fields.qᵛ
u, v, w = model.velocities

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall time: %s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed))

    msg *= @sprintf(", max|u|: %.2f m/s, max w: %.2f m/s, min w: %.2f m/s",
                    maximum(abs, u), maximum(w), minimum(w))

    msg *= @sprintf(", max(qᵛ): %.2e, max(qᶜˡ): %.2e, max(qʳ): %.2e, min(qʳ): %.2e",
                    maximum(qᵛ), maximum(qᶜˡ), maximum(qʳ), minimum(qʳ))
    @info msg

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# Save vertical slices (x-z) for animation. The 2D domain naturally provides
# the full storm cross-section at each output time.

outputs = (; w, qᶜˡ, qʳ, θ=θˡⁱ)

slices_filename = "squall_line_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(model, outputs;
                                                filename = slices_filename,
                                                schedule = TimeInterval(5minutes),
                                                overwrite_existing = true)

run!(simulation)

# ## Animation: vertical cross-sections
#
# We create a 3-panel animation showing the squall line structure in the x-z plane:
# - Vertical velocity ``w``: reveals the updraft, downdraft, and rear-inflow jet
# - Cloud liquid ``qᶜˡ``: outlines the cloud boundary and anvil
# - Rain ``qʳ``: shows the precipitation shaft and evaporation zone
#
# A well-developed squall line should exhibit a leading convective updraft,
# a trailing stratiform rain region, and a surface cold pool that drives
# propagation via the RKW mechanism [RotunnoEtAl1988](@cite).

w_ts = FieldTimeSeries(slices_filename, "w")
qᶜˡ_ts = FieldTimeSeries(slices_filename, "qᶜˡ")
qʳ_ts = FieldTimeSeries(slices_filename, "qʳ")

times = w_ts.times
Nt = length(times)

wlim = maximum(abs, w_ts) / 2
qᶜˡlim = maximum(qᶜˡ_ts) / 4
qʳlim = maximum(qʳ_ts) / 4

fig = Figure(size=(1200, 400), fontsize=12)

axw = Axis(fig[1, 1], xlabel="x (km)", ylabel="z (km)", title="w (m/s)")
axqᶜˡ = Axis(fig[1, 2], xlabel="x (km)", ylabel="z (km)", title="qᶜˡ (kg/kg)")
axqʳ = Axis(fig[1, 3], xlabel="x (km)", ylabel="z (km)", title="qʳ (kg/kg)")

n = Observable(1)
w_n = @lift w_ts[$n]
qᶜˡ_n = @lift qᶜˡ_ts[$n]
qʳ_n = @lift qʳ_ts[$n]
title = @lift "Squall line at t = " * prettytime(times[$n])

hmw = heatmap!(axw, w_n, colormap=:balance, colorrange=(-wlim, wlim))
hmqᶜˡ = heatmap!(axqᶜˡ, qᶜˡ_n, colormap=:dense, colorrange=(0, qᶜˡlim))
hmqʳ = heatmap!(axqʳ, qʳ_n, colormap=:amp, colorrange=(0, qʳlim))

Colorbar(fig[2, 1], hmw, vertical=false)
Colorbar(fig[2, 2], hmqᶜˡ, vertical=false)
Colorbar(fig[2, 3], hmqʳ, vertical=false)

fig[0, :] = Label(fig, title, fontsize=14, tellwidth=false)

CairoMakie.record(fig, "squall_line.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
nothing #hide

# ![](squall_line.mp4)
