# # [Nocturnal marine stratocumulus (DYCOMS-II RF01)](@id dycoms_example)
#
# This example simulates a nocturnal marine stratocumulus-topped boundary layer following
# the first research flight (RF01) of the second Dynamics and Chemistry of Marine
# Stratocumulus field study [Stevens2005](@cite). DYCOMS-II RF01 is the canonical test
# case for large eddy simulation of stratocumulus.
#
# Stratocumulus is a far more delicate target than the shallow cumulus of
# [BOMEX](@ref bomex_example) or [RICO](@ref rico_example): the cloud layer is maintained
# by longwave cooling concentrated in the few tens of meters below cloud top, and it is
# eroded by entrainment of warm, dry free-tropospheric air across a very sharp inversion.
# The balance between the two is delicate enough that the liquid water paths reported by
# the sixteen models in the [Stevens2005](@citet) intercomparison spanned nearly a factor
# of twelve. Resolution near the inversion, and the numerics of the advection scheme,
# matter more here than in almost any other boundary layer case.
#
# Initial conditions for this case are provided by the wonderfully useful package
# [AtmosphericProfilesLibrary.jl](https://github.com/CliMA/AtmosphericProfilesLibrary.jl).

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using CairoMakie
using CUDA
using Printf
using Statistics
using Random

Random.seed!(123)
if CUDA.functional()
    CUDA.seed!(123)
end

# ## Domain and grid
#
# The DYCOMS-II RF01 domain is 3.36 km × 3.36 km horizontally with 35 m grid spacing,
# corresponding to 96 points in each horizontal direction ([Stevens2005](@citet); Section 3).
# The protocol asks for a vertical extent of at least 1250 m and a vertical spacing of 5 m
# or finer in the vicinity of the inversion, which we satisfy with a uniform 5 m grid up
# to 1500 m.
#
# As for the other boundary layer examples we reduce the numerical precision to Float32.

Oceananigans.defaults.FloatType = Float32

Nx = Ny = 96
Nz = 300

x = y = (0, 3360)
z = (0, 1500)

grid = RectilinearGrid(GPU(); x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

# ## Reference state and formulation
#
# We use the anelastic formulation with a reference state built from the surface values
# specified by [Stevens2005](@citet): a surface pressure of 1017.8 hPa and a mixed-layer
# liquid-ice potential temperature of 289 K.

# [Stevens2005](@citet); Table 2 recommends an isobaric heat capacity of 1015 J/kg/K for
# this case -- an intermediate value between dry air and the moist boundary layer air,
# chosen so that the specified ``θ^{ℓi}`` profile reproduces the observed cloud base. We
# adopt it here, which places the initial cloud base within the tolerance the protocol
# asks for; Breeze's default of 1005 J/kg/K puts it about 13 m too low.

constants = ThermodynamicConstants(dry_air_heat_capacity=1015)

reference_state = ReferenceState(grid, constants,
                                 base_pressure = 101780,
                                 potential_temperature = 289)

dynamics = AnelasticDynamics(reference_state)

# ## Surface fluxes
#
# The RF01 protocol prescribes fixed surface sensible and latent heat fluxes of 15 W/m²
# and 115 W/m² ([Stevens2005](@citet); Section 2, "Plan of Attack"). These are supplied
# under the interface keys `ρE` and `ρqᵗ`, which Breeze routes onto whichever thermodynamic
# and moisture variables the model evolves.

sensible_heat_flux = 15   # W m⁻²
latent_heat_flux = 115    # W m⁻²

# The latent heat flux is converted to a moisture mass flux with the enthalpy of
# condensation recommended for this case, ``ℒˡ = 2.47`` MJ/kg ([Stevens2005](@citet); Table 2).

ℒˡ = 2.47e6  # J kg⁻¹
moisture_flux = latent_heat_flux / ℒˡ  # kg m⁻² s⁻¹

ρE_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(sensible_heat_flux))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(moisture_flux))

# Surface momentum is removed by a bulk drag with the drag coefficient
# ``C^D = 0.0011`` specified by [Stevens2005](@citet); Section 2a.

Cᴰ = 0.0011

ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))
ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=Cᴰ))

boundary_conditions = (ρE=ρE_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs)

# ## Large-scale subsidence
#
# The large-scale horizontal divergence is ``D = 3.75 × 10^{-6}`` s⁻¹, which corresponds
# to a subsidence velocity that increases linearly with height, ``w^s = -D z``
# ([Stevens2005](@citet); Section 2a).

D = 3.75e-6  # s⁻¹

FT = eltype(grid)
wˢ = Field{Nothing, Nothing, Face}(grid)
set!(wˢ, z -> -D * z)

subsidence = SubsidenceForcing(wˢ)

# This is what it looks like:

lines(wˢ; axis = (xlabel = "wˢ (m/s)", ylabel = "z (m)"))

# ## Geostrophic forcing
#
# The geostrophic wind is uniform, ``u^g = 7`` m/s and ``v^g = -5.5`` m/s
# ([Stevens2005](@citet); Section 2a). RF01 was flown off the coast of California near
# 31.5°N, so the Coriolis parameter is ``f = 2 Ω \sin(31.5°) ≈ 7.62 × 10^{-5}`` s⁻¹.

coriolis = FPlane(latitude=31.5)

uᵍ = AtmosphericProfilesLibrary.Dycoms_RF01_u0(FT)
vᵍ = AtmosphericProfilesLibrary.Dycoms_RF01_v0(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))

# ## Sponge layer
#
# The domain top at 1500 m sits well inside the strongly stratified free troposphere, so
# we damp vertical velocity in the upper 250 m to keep gravity waves radiated by the cloud
# top from reflecting off the rigid lid. This is a numerical device, not part of the case
# protocol.

sponge_mask = GaussianMask{:z}(center=1500, width=150)
sponge = Relaxation(rate=1/60, mask=sponge_mask)

forcing = (; u = (subsidence, geostrophic.u),
             v = (subsidence, geostrophic.v),
             w = sponge,
             θ = subsidence,
             qᵉ = subsidence)
nothing #hide

# ## Radiation
#
# The cloud layer is driven by longwave cooling at cloud top, which is far too important
# to leave out and far too expensive to compute with a full radiative transfer solver in
# every column. [Stevens2005](@citet) instead calibrate a three-component fit to
# δ-four-stream calculations, in which the net upward radiative flux
#
# ```math
# ℐ(z) = ℐ₀ \, \mathrm{e}^{-τ^{ℓw}(z, H)} + ℐ₁ \, \mathrm{e}^{-τ^{ℓw}(0, z)}
#      + ρᵢ \, cᵖ D \left [ \frac{(z - zᵢ)^{4/3}}{4} + zᵢ (z - zᵢ)^{1/3} \right ]
# ```
#
# combines cloud-top cooling, cloud-base warming, and a free-tropospheric term above the
# inversion. Here ``τ^{ℓw}(a, b) = κˡ ∫_a^b ρ \, qˡ \, \mathrm{d} z`` is the longwave
# optical thickness of the intervening liquid water, so the first two terms respond to the
# cloud the simulation actually produces. Breeze implements this as `DYCOMSRadiation`,
# whose defaults are the parameters recommended by [Stevens2005](@citet); Table 3.

radiation = DYCOMSRadiation(grid; divergence = D,
                            heat_capacity = constants.dry_air.heat_capacity)

# ## Model setup
#
# RF01 was selected in part because it was *not* complicated by significant drizzle, so
# warm-phase saturation adjustment is all the microphysics we need.

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
advection = WENO(order=9)

model = AtmosphereModel(grid; dynamics, coriolis, microphysics, advection,
                        thermodynamic_constants = constants,
                        radiation, forcing, boundary_conditions)

# ## Initial conditions
#
# The mean state is idealized as a two-layer structure in liquid-ice potential temperature
# and total water, with the inversion at ``z_i = 840`` m ([Stevens2005](@citet); Eqs. 1-2):
#
# ```math
# θ^{ℓi} = \begin{cases} 289 \, \mathrm{K} & z ≤ z_i \\
#                        297.5 + (z - z_i)^{1/3} \, \mathrm{K} & z > z_i \end{cases}
# \qquad
# q^t = \begin{cases} 9 \, \mathrm{g/kg} & z ≤ z_i \\
#                     1.5 \, \mathrm{g/kg} & z > z_i \end{cases}
# ```
#
# These are implemented in [AtmosphericProfilesLibrary](https://github.com/CliMA/AtmosphericProfilesLibrary.jl).

θˡⁱ₀ = AtmosphericProfilesLibrary.Dycoms_RF01_θ_liq_ice(FT)
qᵗ₀ = AtmosphericProfilesLibrary.Dycoms_RF01_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.Dycoms_RF01_u0(FT)
v₀ = AtmosphericProfilesLibrary.Dycoms_RF01_v0(FT)

# Turbulence is triggered with a random perturbation to the potential temperature in the
# lowest 200 m, uniform on ``[-0.05, 0.05)`` K -- that is, 0.1 K peak-to-peak.

δθ = 0.1  # K, peak-to-peak
zδ = 200  # m

ϵ() = rand() - 1/2
θᵢ(x, y, z) = θˡⁱ₀(z) + δθ * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀(z)
uᵢ(x, y, z) = u₀(z)
vᵢ(x, y, z) = v₀(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ, v=vᵢ)

# The protocol asks us to check the initial state before integrating: cloud base should
# be within 10 m of 600 m, and the cloud top liquid water content near 0.45 g/kg
# ([Stevens2005](@citet); Section 4).

qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ

qˡ₀ = Field(Average(qˡ, dims=(1, 2)))
compute!(qˡ₀)

zᶜ = Oceananigans.Grids.znodes(grid, Center())
qˡ_profile = Array(view(qˡ₀, 1, 1, :))
cloudy = findall(q -> q > 1e-6, qˡ_profile)
@info @sprintf("Initial cloud base: %.1f m, cloud top: %.1f m, max qˡ: %.2e kg/kg",
               zᶜ[first(cloudy)], zᶜ[last(cloudy)], maximum(qˡ_profile))

# ## Simulation
#
# The case is integrated for four hours ([Stevens2005](@citet); Section 3).

simulation = Simulation(model; Δt=1, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.7, max_Δt=2)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

# ## Output and progress
#
# The liquid water path is the headline diagnostic for this case, so we track it alongside
# the usual suspects.

ρ = total_density(model.dynamics)
LWP = Field(Integral(ρ * qˡ, dims=3))

wall_clock = Ref(time_ns())

function progress(sim)
    compute!(LWP)
    wmax = maximum(abs, sim.model.velocities.w)
    qˡmax = maximum(qˡ)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, wall: %s, max|w|: %.2f m/s, max(qˡ): %.2e, ⟨LWP⟩: %.3f kg/m²",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   wmax, qˡmax, mean(LWP))

    @info msg
    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, IterationInterval(500))

# We output horizontally-averaged profiles every 30 minutes, plus the radiative flux so we
# can see where the cooling is concentrated.

θ = liquid_ice_potential_temperature(model)
ℐ = radiation.net_upward_flux
zᵢ = radiation.inversion_height

u, v, w = model.velocities

outputs = (; u, v, w, θ, qˡ, qᵛ)
avg_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))
avg_outputs = merge(avg_outputs, (; ℐ = Average(ℐ, dims=(1, 2)),
                                    zᵢ = Average(zᵢ, dims=(1, 2))))

filename = "dycoms.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, avg_outputs; filename,
                                                  schedule = TimeInterval(30minutes),
                                                  overwrite_files = true)

# And slices for animation. Cloud top sits near 840 m, so we take a horizontal slice just
# below it where the liquid water is largest.

k = searchsortedfirst(zᶜ, 800)
@info "Saving horizontal slices at z = $(zᶜ[k]) m (k = $k)"

slice_outputs = (
    wxy = view(w, :, :, k),
    qˡxy = view(qˡ, :, :, k),
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "dycoms_slices.jld2",
                                                schedule = TimeInterval(1minute),
                                                overwrite_files = true)

simulation.output_writers[:lwp] = JLD2Writer(model, (; LWP);
                                             filename = "dycoms_lwp.jld2",
                                             schedule = TimeInterval(1minute),
                                             overwrite_files = true)

@info "Running DYCOMS-II RF01 simulation..."
run!(simulation)

# ## Results: mean profile evolution
#
# The hallmark of a well-simulated stratocumulus layer is that it stays well-mixed: ``θ^{ℓi}``
# and ``q^t`` remain nearly uniform from the surface to the inversion, with all of the
# structure concentrated in the few tens of meters at cloud top.

θt = FieldTimeSeries(filename, "θ")
qᵛt = FieldTimeSeries(filename, "qᵛ")
qˡt = FieldTimeSeries(filename, "qˡ")
ut = FieldTimeSeries(filename, "u")
vt = FieldTimeSeries(filename, "v")
ℐt = FieldTimeSeries(filename, "ℐ")

times = θt.times
Nt = length(times)

fig = Figure(size=(1000, 760), fontsize=14)

axθ = Axis(fig[1, 1], xlabel="θˡⁱ (K)", ylabel="z (m)")
axq = Axis(fig[1, 2], xlabel="qᵛ (kg/kg)", ylabel="z (m)")
axl = Axis(fig[1, 3], xlabel="qˡ (kg/kg)", ylabel="z (m)")
axuv = Axis(fig[2, 1], xlabel="u, v (m/s)", ylabel="z (m)")
axℐ = Axis(fig[2, 2], xlabel="ℐ (W/m²)", ylabel="z (m)")

default_colours = Makie.wong_colors()
colors = [default_colours[mod1(i, length(default_colours))] for i in 1:Nt]

for n in 1:Nt
    label = n == 1 ? "initial" : @sprintf("%.1f hr", times[n] / hour)

    lines!(axθ, θt[n], color=colors[n], label=label)
    lines!(axq, qᵛt[n], color=colors[n])
    lines!(axl, qˡt[n], color=colors[n])
    lines!(axuv, ut[n], color=colors[n], linestyle=:solid)
    lines!(axuv, vt[n], color=colors[n], linestyle=:dash)
    lines!(axℐ, ℐt[n], color=colors[n])
end

for ax in (axθ, axq, axl, axuv, axℐ)
    ylims!(ax, 0, 1200)
end

xlims!(axθ, 287, 302)

axislegend(axθ, position=:rb, labelsize=10)
text!(axuv, 0.05, 0.95, text="solid: u\ndashed: v", fontsize=12, space=:relative, align=(:left, :top))

fig[0, :] = Label(fig, "DYCOMS-II RF01: mean profile evolution (Stevens et al., 2005)",
                  fontsize=18, tellwidth=false)

save("dycoms_profiles.png", fig) #src
fig

# The net upward radiative flux ``ℐ`` shows the structure the parameterization is built to
# capture: a sharp drop across cloud top, where the ``ℐ₀ = 70`` W/m² of cloud-top cooling
# is deposited into a very thin layer, and the weaker cloud-base warming below.

# ## Liquid water path
#
# The liquid water path is the quantity that varied by a factor of twelve across the
# intercomparison, so it is the natural scalar to watch.

LWPt = FieldTimeSeries("dycoms_lwp.jld2", "LWP")
lwp_times = LWPt.times
lwp_mean = [mean(LWPt[n]) for n in 1:length(lwp_times)]

fig = Figure(size=(700, 400), fontsize=14)
ax = Axis(fig[1, 1], xlabel="time (hours)", ylabel="⟨LWP⟩ (kg/m²)",
          title="DYCOMS-II RF01: domain-mean liquid water path")
lines!(ax, lwp_times ./ hour, lwp_mean, linewidth=3)

save("dycoms_lwp.png", fig) #src
fig

# ## Animation of the cloud layer
#
# Finally we animate vertical velocity and liquid water, in a vertical slice through the
# domain and in a horizontal slice just below cloud top.

wxz_ts = FieldTimeSeries("dycoms_slices.jld2", "wxz")
qˡxz_ts = FieldTimeSeries("dycoms_slices.jld2", "qˡxz")
wxy_ts = FieldTimeSeries("dycoms_slices.jld2", "wxy")
qˡxy_ts = FieldTimeSeries("dycoms_slices.jld2", "qˡxy")

times = wxz_ts.times
Nt = length(times)

xᶜ = Oceananigans.Grids.xnodes(grid, Center())

fig = Figure(size=(1000, 760), fontsize=14)

axwxz = Axis(fig[2, 2], aspect=2, xaxisposition=:top, xlabel="x (m)", ylabel="z (m)", title="Vertical velocity w")
axqxz = Axis(fig[2, 3], aspect=2, xaxisposition=:top, xlabel="x (m)", ylabel="z (m)", title="Liquid water qˡ")
axwxy = Axis(fig[3, 2], aspect=1, xlabel="x (m)", ylabel="y (m)", title="@ z = $(zᶜ[k]) m")
axqxy = Axis(fig[3, 3], aspect=1, xlabel="x (m)", ylabel="y (m)", title="@ z = $(zᶜ[k]) m")

wlim = maximum(abs, wxz_ts) / 2
qˡlim = maximum(qˡxz_ts)

n = Observable(1)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
wxy_n = @lift wxy_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
title = @lift "DYCOMS-II RF01 at t = " * prettytime(times[$n])

hmw = heatmap!(axwxz, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmq = heatmap!(axqxz, qˡxz_n, colormap=Reverse(:Blues_4), colorrange=(0, qˡlim))
heatmap!(axwxy, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axqxy, qˡxy_n, colormap=Reverse(:Blues_4), colorrange=(0, qˡlim))

for ax in (axwxz, axqxz)
    ylims!(ax, 0, 1200)
    lines!(ax, xᶜ, fill(zᶜ[k], length(xᶜ)), color=:grey, linestyle=:dash)
end

Colorbar(fig[2:3, 1], hmw, label="w (m/s)", tellheight=false, height=Relative(0.7), flipaxis=false)
Colorbar(fig[2:3, 4], hmq, label="qˡ (kg/kg)", tellheight=false, height=Relative(0.7))

fig[1, :] = Label(fig, title, fontsize=18, tellwidth=false)

rowgap!(fig.layout, 1, -40)
rowgap!(fig.layout, 2, -40)

CairoMakie.record(fig, "dycoms_slices.mp4", 1:Nt; framerate=12, compression=23) do nn
    n[] = nn
end
nothing #hide

# ![](dycoms_slices.mp4)
