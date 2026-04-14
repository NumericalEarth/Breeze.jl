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
# The base of the inversion is at 468 m, has a thickness of 6 grid levels, over which the potential
# temperature increases by 8 K. Above the cap, the lapse rate is 3 K/km.
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

# A bulk drag parameterization is applied with friction velocity

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

ρw_sponge = Relaxation(rate = sponge_rate, mask = sponge_mask) # relax to 0 by default

# We relax temperature to the initial profile.

ρᵣ = reference_state.density
ρθˡⁱᵣ = Field{Nothing, Nothing, Center}(grid)
set!(ρθˡⁱᵣ, z -> θᵣ(z))
set!(ρθˡⁱᵣ, ρᵣ * ρθˡⁱᵣ)

ρθˡⁱᵣ_data = interior(ρθˡⁱᵣ, 1, 1, :)

@inline function ρθ_sponge_fun(i, j, k, grid, clock, model_fields, p)
    zₖ = znode(k, grid, Center())
    return @inbounds p.rate * p.mask(0, 0, zₖ) * (p.target[k] - model_fields.ρθ[i, j, k])
end

ρθ_sponge = Forcing(
    ρθ_sponge_fun; discrete_form = true,
    parameters = (rate = sponge_rate, mask = sponge_mask, target = ρθˡⁱᵣ_data)
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

advection = WENO(order=9)           # WENO(order=5), Centered(order=6) are dissipative

closure = SmagorinskyLilly(C=0.18)  # Sullivan et al. (1994)

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

# ## Simulation
#
# We run the simulation for 5 hours with adaptive time-stepping.

simulation = Simulation(model; Δt=0.5, stop_time=5hours)
conjure_time_step_wizard!(simulation, cfl=0.7)

Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Output and progress
#
# We add a progress callback and output the hourly time-averages of the horizontally-averaged
# profiles for post-processing.

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

# Output averaged products for calculating statistics

avg_outputs_varlist = (;
    θ, νₑ,
    uu = u*u, vv = v*v, ww = w*w,
    uw = u*w, vw = v*w, θw = θ*w,       # second-order moments for fluxes
    uuw = u*u*w, vvw=v*v*w, www=w*w*w,  # third-order moments for turbulent transport term
    νₑ³ = νₑ^3,                         # SGS dissipation -- note: |S̄|² = νₑ² / (Cₛ Δ)⁴) with Smagorinsky model
)
outputs = merge(model.velocities, model.tracers, avg_outputs_varlist)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

avg_filename = "abl_averages.jld2"
avg_output_interval = 10minutes
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs;
                                                  filename = avg_filename,
                                                  schedule = AveragedTimeInterval(avg_output_interval),
                                                  overwrite_existing = true)

# ## Output horizontal slices for animation

# Find the `k`-index closest to z = 100 m.

z = znodes(grid, Center())
k₁₀₀ = searchsortedfirst(z, 100)
@info "Saving slices at z = $(z[k₁₀₀]) m (k = $k₁₀₀)"

# Find the `j`-index closest to the domain center.

y = ynodes(grid, Center())
jmid = Ny ÷ 2
@info "Saving slices at y = $(y[jmid]) m (j = $jmid)"

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

# Now we are ready to run the simulation.

run!(simulation)

# ## Load output and visualize

# Let's load the saved output.

uts  = FieldTimeSeries(avg_filename, "u")
vts  = FieldTimeSeries(avg_filename, "v")
wts  = FieldTimeSeries(avg_filename, "w")
θts  = FieldTimeSeries(avg_filename, "θ")
uuts = FieldTimeSeries(avg_filename, "uu")
vvts = FieldTimeSeries(avg_filename, "vv")
wwts = FieldTimeSeries(avg_filename, "ww")
uwts = FieldTimeSeries(avg_filename, "uw")
vwts = FieldTimeSeries(avg_filename, "vw")
θwts = FieldTimeSeries(avg_filename, "θw")
νₑts = FieldTimeSeries(avg_filename, "νₑ")

times = uts.times
Nt = length(times)
zᶜ = znodes(uts.grid, Center())  # cell centers (Nz)
zᶠ = znodes(uts.grid, Face())    # face centers (Nz+1)

Nz = uts.grid.Nz
Δz = zspacings(uts.grid, Center())[:]
nothing #hide

# Compute diagnostics at each saved time
WS_mean = zeros(Nz, Nt)
WD_mean = zeros(Nz, Nt)
θ_mean  = zeros(Nz, Nt)

uu_var = zeros(Nz, Nt)
vv_var = zeros(Nz, Nt)
ww_var = zeros(Nz+1, Nt)

uw_res = zeros(Nz, Nt)
vw_res = zeros(Nz, Nt)
θw_res = zeros(Nz, Nt)
uw_sgs = zeros(Nz, Nt)
vw_sgs = zeros(Nz, Nt)
θw_sgs = zeros(Nz, Nt)

ρᵣ_vec = Array(interior(ρᵣ, 1, 1, :))

for n in 1:Nt
    u_n    = Array(interior(uts[n],  1, 1, :))
    v_n    = Array(interior(vts[n],  1, 1, :))
    w_raw  = Array(interior(wts[n],  1, 1, :))
    θ_n    = Array(interior(θts[n],  1, 1, :))
    uu_n   = Array(interior(uuts[n], 1, 1, :))
    vv_n   = Array(interior(vvts[n], 1, 1, :))
    ww_raw = Array(interior(wwts[n], 1, 1, :))
    uw_n   = Array(interior(uwts[n], 1, 1, :))
    vw_n   = Array(interior(vwts[n], 1, 1, :))
    θw_n   = Array(interior(θwts[n], 1, 1, :))
    νₑ_n   = Array(interior(νₑts[n], 1, 1, :))

    ## Interpolate face fields to cell centers
    w_n = @views (w_raw[1:end-1] .+ w_raw[2:end]) ./ 2

    ## Fig 1: Mean profiles
    WS_mean[:, n] .= @. sqrt(u_n^2 + v_n^2)
    WD_mean[:, n] .= mod.(270 .- atand.(v_n, u_n), 360)
    θ_mean[:, n]  .= θ_n

    ## Fig 2: Velocity variances normalized by u★²
    uu_var[:, n] .= (uu_n .- u_n.^2) ./ u★^2
    vv_var[:, n] .= (vv_n .- v_n.^2) ./ u★^2
    ww_var[:, n] .= (ww_raw .- w_raw.^2) ./ u★^2

    ## Vertical derivatives for SGS fluxes
    ∂z_u = similar(u_n)
    ∂z_v = similar(v_n)
    ∂z_θ = similar(θ_n)
    ∂z_u[1] = (u_n[2] - u_n[1]) / Δz[1]
    ∂z_v[1] = (v_n[2] - v_n[1]) / Δz[1]
    ∂z_θ[1] = (θ_n[2] - θ_n[1]) / Δz[1]
    for k in 2:Nz-1
        ∂z_u[k] = (u_n[k+1] - u_n[k-1]) / (Δz[k-1] + Δz[k])
        ∂z_v[k] = (v_n[k+1] - v_n[k-1]) / (Δz[k-1] + Δz[k])
        ∂z_θ[k] = (θ_n[k+1] - θ_n[k-1]) / (Δz[k-1] + Δz[k])
    end
    ∂z_u[end] = (u_n[end] - u_n[end-1]) / Δz[end]
    ∂z_v[end] = (v_n[end] - v_n[end-1]) / Δz[end]
    ∂z_θ[end] = (θ_n[end] - θ_n[end-1]) / Δz[end]

    ## Fig 3: Resolved fluxes (momentum normalized by ρ₀u★²)
    uw_res[:, n] .= ρᵣ_vec .* (uw_n .- u_n .* w_n) ./ (ρ₀ * u★^2)
    vw_res[:, n] .= ρᵣ_vec .* (vw_n .- v_n .* w_n) ./ (ρ₀ * u★^2)
    θw_res[:, n] .= θw_n .- θ_n .* w_n

    ## Fig 3: SGS modeled fluxes (ρᵣ νₑ dU/dz normalized by ρ₀u★²; (νₑ/Pr_t) dθ/dz)
    uw_sgs[:, n] .= -ρᵣ_vec .* νₑ_n .* ∂z_u ./ (ρ₀ * u★^2)
    vw_sgs[:, n] .= -ρᵣ_vec .* νₑ_n .* ∂z_v ./ (ρ₀ * u★^2)
    θw_sgs[:, n] .= -νₑ_n ./ closure.Pr .* ∂z_θ
end

# Define a colormap for each time.

cmap = cgrad(:viridis)
colors = [cmap[(n-1)/max(Nt-1, 1)] for n in 1:Nt]
labels = [n == 1 ? "0–1 hr" : "$(n-1)–$n hr" for n in 1:Nt]

# Finally, we are ready to plot.

plot_interval = Int(1hour / avg_output_interval)

# First we plot the mean profiles (wind speed, wind direction, potential temperature).

fig1 = Figure(size=(800, 400), fontsize=14)

ax1a = Axis(fig1[1, 1], xlabel="√(U² + V²) (m/s)", ylabel="z (m)",title="Horizontal wind speed")
ax1b = Axis(fig1[1, 2], xlabel="WD (° from N)", ylabel="z (m)",title="Wind direction")
ax1c = Axis(fig1[1, 3], xlabel="θ (K)", ylabel="z (m)",title="Potential temperature")

for n in 1:plot_interval:Nt
    lines!(ax1a, WS_mean[:, n], zᶜ, color=colors[n], label=labels[n])
    lines!(ax1b, WD_mean[:, n], zᶜ, color=colors[n])
    lines!(ax1c,  θ_mean[:, n], zᶜ, color=colors[n])
end

linkyaxes!(ax1a, ax1b, ax1c)
hideydecorations!(ax1b, grid=false)
hideydecorations!(ax1c, grid=false)

Legend(fig1[1, 4], ax1a, "Time", framevisible=false)
fig1

# Next, we plot the velocity variances normalized by ``u_★^2``

fig2 = Figure(size=(800, 400), fontsize=14)

ax2a = Axis(fig2[1, 1], xlabel="⟨u′u′⟩ / u_★²", ylabel="z (m)", title="u variance")
ax2b = Axis(fig2[1, 2], xlabel="⟨v′v′⟩ / u_★²", ylabel="z (m)", title="v variance")
ax2c = Axis(fig2[1, 3], xlabel="⟨w′w′⟩ / u_★²", ylabel="z (m)", title="w variance")

for n in 1:plot_interval:Nt
    lines!(ax2a, uu_var[:, n], zᶜ, color=colors[n], label=labels[n])
    lines!(ax2b, vv_var[:, n], zᶜ, color=colors[n])
    lines!(ax2c, ww_var[:, n], zᶠ, color=colors[n])
end

linkyaxes!(ax2a, ax2b, ax2c)
hideydecorations!(ax2b, grid=false)
hideydecorations!(ax2c, grid=false)

Legend(fig2[1, 4], ax2a, "Time", framevisible=false)
fig2

# Last, we plot the resolved and the SGS fluxes.

fig3 = Figure(size=(1200, 450), fontsize=14)

ax3a = Axis(fig3[1, 1], xlabel="τˣ / ρ₀u_★²", ylabel="z (m)", title="x-momentum flux")
ax3b = Axis(fig3[1, 2], xlabel="τʸ / ρ₀u_★²", ylabel="z (m)", title="y-momentum flux")
ax3c = Axis(fig3[1, 3], xlabel="Jᶿ (K m/s)", ylabel="z (m)", title="Potential temperature flux")

for n in 1:plot_interval:Nt
    lines!(ax3a, uw_res[:, n] .+ uw_sgs[:, n], zᶜ, color=colors[n], label=labels[n])
    lines!(ax3a, uw_res[:, n], zᶜ, color=colors[n], linestyle=:dash)
    lines!(ax3a, uw_sgs[:, n], zᶜ, color=colors[n], linestyle=:dot)

    lines!(ax3b, vw_res[:, n] .+ vw_sgs[:, n], zᶜ, color=colors[n])
    lines!(ax3b, vw_res[:, n], zᶜ, color=colors[n], linestyle=:dash)
    lines!(ax3b, vw_sgs[:, n], zᶜ, color=colors[n], linestyle=:dot)

    lines!(ax3c, θw_res[:, n] .+ θw_sgs[:, n], zᶜ, color=colors[n])
    lines!(ax3c, θw_res[:, n], zᶜ, color=colors[n], linestyle=:dash)
    lines!(ax3c, θw_sgs[:, n], zᶜ, color=colors[n], linestyle=:dot)
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
