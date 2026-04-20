# # Neutral atmospheric boundary layer (ABL)
#
# This canonical setup is based on the paper by [Moeng1994](@citet), which was a demonstration
# case for the NCAR LES subgrid-scale model development [Sullivan1994](@cite).
# Sometimes, this model configuration is called "conventionally" neutral [Pedersen2014](@cite)
# or "conditionally" neutral [Berg2020](@cite), which indicate an idealized dry, shear-driven
# atmospheric boundary layer, capped by a stable inversion layer, without any surface heating.
# Forcings come from a specified constant geostrophic wind (i.e., a specified background
# pressure gradient) and Coriolis forces; the temperature lapse rate in the free atmosphere
# is maintained with a sponge layer.
#
# In lieu of more sophisticated surface layer modeling in this example, we
# impose a fixed friction velocity at the bottom boundary.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units
using CUDA
using Printf
using Random
using CairoMakie

Random.seed!(1994)
if CUDA.functional()
    CUDA.seed!(1994)
end

# ## Domain and grid
#
# For faster time to solution, we reduce the numerical precision to Float32.

arch = GPU()
Oceananigans.defaults.FloatType = Float32;

# Simulation "S" (shear-driven ABL) domain setup from [Moeng1994](@citet):

Nx = Ny = Nz = 96
x = y = (0, 3000)
z = (0, 1000)

grid = RectilinearGrid(arch; x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and formulation

p₀ = 1e5   # Pa
θ₀ = 300   # K

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants,
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)

dynamics = AnelasticDynamics(reference_state)

# Capping inversion for "S" simulation, as in the paper by [Moeng1994](@citet):
# The base of the inversion is at 468 m and has a thickness of 6 grid levels,
# over which the potential temperature increases by 8 K. Above the cap, the
# lapse rate is 3 K/km.
Δz  = first(zspacings(grid))
zᵢ₁ = 468        # m
zᵢ₂ = zᵢ₁ + 6Δz  # m
Γᵢ  = 8 / 6Δz    # K/m
Γᵗᵒᵖ = 0.003     # K/m
θᵣ(z) = z < zᵢ₁ ? θ₀ :
        z < zᵢ₂ ? θ₀ + Γᵢ * (z - zᵢ₁) :
        θ₀ + Γᵢ * (zᵢ₂ - zᵢ₁) + Γᵗᵒᵖ * (z - zᵢ₂)
nothing #hide

# ## Surface momentum flux (drag)
#
# For testing, we prescribe the surface shear stress. In practice, however,
# this is not known a priori. A surface layer scheme (i.e., a wall model) will
# dynamically update ``u_★`` based on environmental conditions that include
# surface roughness and heat fluxes.

u★ = 0.5  # m/s, _result_ from simulation "S" by Moeng and Sullivan (1994)
q₀ = Breeze.Thermodynamics.MoistureMassFractions{eltype(grid)} |> zero
ρ₀ = Breeze.Thermodynamics.density(θ₀, p₀, q₀, constants)
nothing #hide

# A bulk drag parameterization is applied with friction velocity:

@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / max(sqrt(ρu^2 + ρv^2), p.ρ₀ * 1e-6)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / max(sqrt(ρu^2 + ρv^2), p.ρ₀ * 1e-6)

ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρu_bcs = FieldBoundaryConditions(bottom=ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_drag_bc)

# ## Sponge layer
#
# To enforce an upper-air temperature gradient, we introduce a sponge layer with Gaussian weighting
# that corresponds to an effective depth of approximately 500 m. At `|z - zᵗᵒᵖ| = 500`,
# `exp(-0.5 * (500/sponge_width)^2) = 0.04 ~ 0`. The sponge rate (inverse timescale) is an ad hoc
# value; a higher sponge rate (shorter damping time scale) made no difference in this case, and a
# weaker sponge rate may be used.
sponge_width = 200  # m
sponge_rate = 0.01  # 1/s
sponge_mask = GaussianMask{:z}(center = last(z), width = sponge_width)

# We damp out any vertical motions near the top boundary.

ρw_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask) # relaxes to 0 by default

# We relax temperature to the initial profile.

ρθᵣ = Field{Nothing, Nothing, Center}(grid)
set!(ρθᵣ, z -> θᵣ(z))
set!(ρθᵣ, reference_state.density * ρθᵣ)

ρθᵣ_data = interior(ρθᵣ, 1, 1, :)

@inline function ρθ_sponge_fun(i, j, k, grid, clock, model_fields, p)
    zₖ = znode(k, grid, Center())
    return @inbounds p.rate * p.mask(0, 0, zₖ) * (p.target[k] - model_fields.ρθ[i, j, k])
end

ρθ_sponge = Forcing(
    ρθ_sponge_fun; discrete_form = true,
    parameters = (rate = sponge_rate, mask = sponge_mask, target = ρθᵣ_data)
)

# ## Assembling all the forcings

coriolis = FPlane(f=1e-4)

uᵍ, vᵍ = 15, 0  # m/s, simulation "S" by Moeng and Sullivan (1994)
geostrophic = geostrophic_forcings(uᵍ, vᵍ)

ρu_forcing = geostrophic.ρu
ρv_forcing = geostrophic.ρv
ρw_forcing = ρw_sponge
ρθ_forcing = ρθ_sponge

forcing = (; ρu=ρu_forcing, ρv=ρv_forcing, ρw=ρw_forcing, ρθ=ρθ_forcing)
nothing #hide

# ## Model setup

advection = WENO(order=9)   # WENO(order=5), Centered(order=6) are too dissipative

closure = SmagorinskyLilly()

model = AtmosphereModel(grid; dynamics, coriolis, advection, forcing, closure,
                        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs))

# ## Initial conditions

# We add velocity and temperature perturbations to help initiate turbulence.
δu = δv = 0.01  # m/s
δθ = 0.1        # K
zδ = 400        # m, < zᵢ₁

ϵ() = rand() - 1/2
uᵢ(x, y, z) =   uᵍ  + δu * ϵ() * (z < zδ)
vᵢ(x, y, z) =   vᵍ  + δv * ϵ() * (z < zδ)
θᵢ(x, y, z) = θᵣ(z) + δθ * ϵ() * (z < zδ)

set!(model, θ=θᵢ, u=uᵢ, v=vᵢ)

# ## Simulation and output
#
# We run the simulation for 5 hours with adaptive time-stepping.

simulation = Simulation(model; Δt=0.5, stop_time=5hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ### Progress monitor
#
# We add a progress callback to monitor the simulation.

u, v, w = model.velocities
θ = liquid_ice_potential_temperature(model)
νₑ = model.closure_fields.νₑ

## For keeping track of the computational expense
wall_clock = time_ns()

function progress(sim)
    wmax = maximum(abs, sim.model.velocities.w)
    elapsed = 1e-9 * (time_ns() - wall_clock)
    msg = @sprintf("Iter: %d, t: % 12s, Δt: %s, elapsed: %s; max|w|: %.2e m/s",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed), wmax)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ### Horizontal averaging
#
# Profiles of horizontally averaged quantities are output every 10 minutes for
# statistical analysis. All outputs are at cell centers.

# Note: Higher-order moments are computed at the location of the _first_ field.
# E.g., u * w results in a BinaryOperation at (Face, Center, Center).
avg_outputs_varlist = (;
    θ, νₑ,
    uu = u^2, vv = v^2, ww = w^2,
    uw = u*w, vw = v*w, θw = θ*w,           # second-order moments for fluxes
    uuw = u^2*w, vvw = v^2*w, www = w^3,    # third-order moments to calculate turbulent transport
    νₑ³ = νₑ^3,                             # SGS dissipation — note: |S̄|² = νₑ² / (Cₛ Δ)⁴) with Smagorinsky model
)
outputs = merge(model.velocities, model.tracers, avg_outputs_varlist)
nothing #hide

# After computing the output quantity and prior to calculating the slab average,
# staggered quantities are interpolated to cell centers.
avg_outputs = NamedTuple(name => Average(@at((Center, Center, Center), outputs[name]), dims=(1, 2))
                         for name in keys(outputs))

∂z_outputs = (; ∂z_u=u, ∂z_v=v, ∂z_θ=θ)
avg_∂z_outputs = NamedTuple(name => Average(@at((Center, Center, Center), ∂z(∂z_outputs[name])), dims=(1, 2))
                            for name in keys(∂z_outputs))

avg_filename = "abl_averages.jld2"
avg_output_interval = 10minutes
simulation.output_writers[:averages] = JLD2Writer(model, merge(avg_outputs, avg_∂z_outputs);
                                                  filename = avg_filename,
                                                  schedule = AveragedTimeInterval(avg_output_interval),
                                                  overwrite_existing = true)

# ### Instantaneous slices for animation

# Find the `k`-index closest to z = 100 m.

z = znodes(grid, Center())
k₁₀₀ = searchsortedfirst(z, 100)
@info "Saving slices at z = $(z[k₁₀₀]) m (k = $k₁₀₀)"

# Find the `j`-index closest to the domain center.

y = ynodes(grid, Center())
jmid = Ny ÷ 2
@info "Saving slices at y = $(y[jmid]) m (j = $jmid)"

# Set up the output writer.

slice_fields = (; u, v, w, θ)
slice_outputs = (
    u_xy = view(u, :, :, k₁₀₀),
    v_xy = view(v, :, :, k₁₀₀),
    w_xy = view(w, :, :, k₁₀₀),
    u_xz = view(u, :, jmid, :),
    w_xz = view(w, :, jmid, :),
    θ_xz = view(θ, :, jmid, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "abl_slices.jld2",
                                                schedule = TimeInterval(5minutes),
                                                overwrite_existing = true)

# ### Go time

run!(simulation)

# ## Load output and visualize

# Let's load the saved output.

u_ts  = FieldTimeSeries(avg_filename, "u")
v_ts  = FieldTimeSeries(avg_filename, "v")
w_ts  = FieldTimeSeries(avg_filename, "w")
θ_ts  = FieldTimeSeries(avg_filename, "θ")
uu_ts = FieldTimeSeries(avg_filename, "uu")
vv_ts = FieldTimeSeries(avg_filename, "vv")
ww_ts = FieldTimeSeries(avg_filename, "ww")
uw_ts = FieldTimeSeries(avg_filename, "uw")
vw_ts = FieldTimeSeries(avg_filename, "vw")
θw_ts = FieldTimeSeries(avg_filename, "θw")
νₑ_ts = FieldTimeSeries(avg_filename, "νₑ")
∂z_u_ts = FieldTimeSeries(avg_filename, "∂z_u")
∂z_v_ts = FieldTimeSeries(avg_filename, "∂z_v")
∂z_θ_ts = FieldTimeSeries(avg_filename, "∂z_θ")

grid = u_ts.grid
times = u_ts.times
Nt = length(times)

ρᵣ = Oceananigans.on_architecture(CPU(), reference_state.density)

# Compute diagnostics at each saved time. First, we create a few empty timeseries to save the computed diagnostics.

loc = (nothing, nothing, Center())

WS_mean_ts = FieldTimeSeries(loc, grid, times)
WD_mean_ts = FieldTimeSeries(loc, grid, times)
 uu_var_ts = FieldTimeSeries(loc, grid, times)
 vv_var_ts = FieldTimeSeries(loc, grid, times)
 ww_var_ts = FieldTimeSeries(loc, grid, times)
 uw_res_ts = FieldTimeSeries(loc, grid, times)
 vw_res_ts = FieldTimeSeries(loc, grid, times)
 θw_res_ts = FieldTimeSeries(loc, grid, times)
 uw_sgs_ts = FieldTimeSeries(loc, grid, times)
 vw_sgs_ts = FieldTimeSeries(loc, grid, times)
 θw_sgs_ts = FieldTimeSeries(loc, grid, times)

# and then we loop over all saved fields and compute what we want.

for n in 1:Nt
    u_n = u_ts[n]
    v_n = v_ts[n]
    w_n = w_ts[n]
    θ_n = θ_ts[n]
    νₑ_n = νₑ_ts[n]
    uu_n = uu_ts[n]
    vv_n = vv_ts[n]
    ww_n = ww_ts[n]
    uw_n = uw_ts[n]
    vw_n = vw_ts[n]
    θw_n = θw_ts[n]
    ∂z_u_n = ∂z_u_ts[n]
    ∂z_v_n = ∂z_v_ts[n]
    ∂z_θ_n = ∂z_θ_ts[n]

    WS_mean_ts[n] .= sqrt(u_n^2 + v_n^2)
    interior(WD_mean_ts[n]) .= @. mod(270 - atand($interior(v_n), $interior(u_n)), 360)
    uu_var_ts[n] .= (uu_n - u_n^2) / u★^2
    vv_var_ts[n] .= (vv_n - v_n^2) / u★^2
    ww_var_ts[n] .= (ww_n - w_n^2) / u★^2
    uw_res_ts[n] .= ρᵣ * (uw_n - u_n * w_n) / (ρ₀ * u★^2)
    vw_res_ts[n] .= ρᵣ * (vw_n - v_n * w_n) / (ρ₀ * u★^2)
    θw_res_ts[n] .= θw_n - θ_n * w_n

    uw_sgs_ts[n] .= -ρᵣ * νₑ_n * ∂z_u_n / (ρ₀ * u★^2)
    vw_sgs_ts[n] .= -ρᵣ * νₑ_n * ∂z_v_n / (ρ₀ * u★^2)
    θw_sgs_ts[n] .= -νₑ_n * ∂z_θ_n / closure.Pr  # not normalized
end

# Define a colormap for each time.

cmap = cgrad(:viridis)
colors = [cmap[(n-1)/max(Nt-1, 1)] for n in 1:Nt]
nothing #hide

# Note that the AveragedTimeInterval schedule outputs the initial snapshot
# followed by averaged snapshots at the requested output interval.
smart_label(n) = prettytime(n * avg_output_interval)
labels = [n == 1 ? "initial condition" : smart_label(n-1) for n in 1:Nt]
nothing #hide

# Finally, we are ready to plot.

plot_interval = 1hour  # should be a multiple of avg_output_interval
plot_skip = Int(plot_interval / avg_output_interval)
nothing #hide

# First we plot the mean profiles (wind speed, wind direction, potential temperature).

fig1 = Figure(size=(1000, 500), fontsize=14)

ax1a = Axis(fig1[1, 1], xlabel="√(U² + V²) (m/s)", ylabel="z (m)", title="Horizontal wind speed")
ax1b = Axis(fig1[1, 2], xlabel="WD (° from N)", ylabel="z (m)", title="Wind direction")
ax1c = Axis(fig1[1, 3], xlabel="θ (K)", ylabel="z (m)", title="Potential temperature")

for n in 1:plot_skip:Nt
    lines!(ax1a, WS_mean_ts[n], color=colors[n], label=labels[n])
    lines!(ax1b, WD_mean_ts[n], color=colors[n])
    lines!(ax1c,       θ_ts[n], color=colors[n])
end

linkyaxes!(ax1a, ax1b, ax1c)
hideydecorations!(ax1b, grid=false)
hideydecorations!(ax1c, grid=false)

Legend(fig1[1, 4], ax1a, "Time", framevisible=false)
fig1

# Next, we plot the velocity variances normalized by ``u_★^2``

fig2 = Figure(size=(1000, 500), fontsize=14)

ax2a = Axis(fig2[1, 1], xlabel="⟨u′u′⟩ / u_★²", ylabel="z (m)", title="u variance")
ax2b = Axis(fig2[1, 2], xlabel="⟨v′v′⟩ / u_★²", ylabel="z (m)", title="v variance")
ax2c = Axis(fig2[1, 3], xlabel="⟨w′w′⟩ / u_★²", ylabel="z (m)", title="w variance")

for n in 1:plot_skip:Nt
    lines!(ax2a, uu_var_ts[n], color=colors[n], label=labels[n])
    lines!(ax2b, vv_var_ts[n], color=colors[n])
    lines!(ax2c, ww_var_ts[n], color=colors[n])
end

linkyaxes!(ax2a, ax2b, ax2c)
hideydecorations!(ax2b, grid=false)
hideydecorations!(ax2c, grid=false)

Legend(fig2[1, 4], ax2a, "Time", framevisible=false)
fig2

# Last, we plot the resolved and the SGS fluxes.

fig3 = Figure(size=(1000, 500), fontsize=14)

ax3a = Axis(fig3[1, 1], xlabel="τˣ / ρ₀u_★²", ylabel="z (m)", title="x-momentum flux")
ax3b = Axis(fig3[1, 2], xlabel="τʸ / ρ₀u_★²", ylabel="z (m)", title="y-momentum flux")
ax3c = Axis(fig3[1, 3], xlabel="Jᶿ (K m/s)", ylabel="z (m)", title="Potential temperature flux")

for n in 1:plot_skip:Nt
    lines!(ax3a, uw_res_ts[n] + uw_sgs_ts[n], color=colors[n], label=labels[n])
    lines!(ax3a, uw_res_ts[n], color=colors[n], linestyle=:dash)
    lines!(ax3a, uw_sgs_ts[n], color=colors[n], linestyle=:dot)

    lines!(ax3b, vw_res_ts[n] + vw_sgs_ts[n], color=colors[n])
    lines!(ax3b, vw_res_ts[n], color=colors[n], linestyle=:dash)
    lines!(ax3b, vw_sgs_ts[n], color=colors[n], linestyle=:dot)

    lines!(ax3c, θw_res_ts[n] + θw_sgs_ts[n], color=colors[n])
    lines!(ax3c, θw_res_ts[n], color=colors[n], linestyle=:dash)
    lines!(ax3c, θw_sgs_ts[n], color=colors[n], linestyle=:dot)
end

for ax in (ax3a, ax3b, ax3c)
    vlines!(ax, 0, color=:grey, linewidth=0.5)
end

## Legends: line style (inside panel a) and time (right)
style_entries = [LineElement(color=:black, linestyle=:solid),
                 LineElement(color=:black, linestyle=:dash),
                 LineElement(color=:black, linestyle=:dot)]
axislegend(ax3a, style_entries, ["total", "resolved", "SGS"],
           position=:lt, framevisible=false)

linkyaxes!(ax3a, ax3b, ax3c)
hideydecorations!(ax3b, grid=false)
hideydecorations!(ax3c, grid=false)

Legend(fig3[1, 4], ax3a, "Time", framevisible=false)
fig3
