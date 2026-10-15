# # Kinematic driver: cloud and precipitation with P3
#
# This example uses [`PrescribedDynamics`](@ref) to lift moist air through a
# vertical column at 2 m/s. [`P3Microphysics`](@ref) predicts cloud droplets,
# rain, ice, and liquid water carried on ice, with **prognostic supersaturation**
# and aerosol activation. Velocity is prescribed while transport, thermodynamics,
# and microphysics evolve. We use liquid–ice potential temperature, the default
# thermodynamic formulation, explicitly selected below for clarity.
#
# This is an open column: moist air enters at the bottom, air leaves at the top,
# and hydrometeors can sediment out. The rising-parcel example instead retains
# all water in a closed moving parcel. This setup demonstrates the ingredients
# of a kinematic microphysics comparison; it is not a reproduction of the
# upstream P3 `kin1d` benchmark.

using Oceananigans
using Oceananigans.Units
using Breeze
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode
using CairoMakie
using Printf

# ## Grid and background atmosphere
#
# The column extends through the troposphere into the lower stratosphere.
# We construct hydrostatic pressure and density using the same potential
# temperature profile used to initialize the model. The atmosphere starts at
# 80% relative humidity over liquid water, with no condensate.

Nz = 100
Lz = 20kilometers
grid = RectilinearGrid(CPU(); size=Nz, z=(0, Lz), topology=(Flat, Flat, Bounded))
constants = ThermodynamicConstants()
θ₀ = 300
p₀ = 1e5
ℋ₀ = 0.8
zᵗʳ = 12kilometers
θᵗʳ = 343
Tᵗʳ = 213
g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_initial(z)
    θ_troposphere = θ₀ + (θᵗʳ - θ₀) * (z / zᵗʳ)^(5/4)
    θ_stratosphere = θᵗʳ * exp(g / (cᵖᵈ * Tᵗʳ) * (z - zᵗʳ))
    return ifelse(z ≤ zᵗʳ, θ_troposphere, θ_stratosphere)
end

reference_state = ReferenceState(grid, constants;
                                 base_pressure = p₀,
                                 potential_temperature = θ_initial)

# ## Prescribed ascent and P3
#
# Uniform vertical velocity does not satisfy ∂z(ρw) = 0 for a stratified density
# profile. `divergence_correction=true` compensates for that mass-flux divergence
# in the transported specific quantities. The flow remains prescribed even as
# latent heating changes the temperature.
#
# P3's `predict_supersaturation=true` adds `ρsᵛ⁺ˡ`, the density-weighted vapor
# excess over liquid saturation. Aerosol activation also predicts cloud droplet
# number and the remaining aerosol reservoir. Number mixing ratios are in kg⁻¹.

W₀ = 2
nᵃ₀ = 3e8
dynamics = PrescribedDynamics(reference_state; divergence_correction=true)
aerosol = AerosolActivation(AerosolMode(number_mixing_ratio=nᵃ₀); prognostic=true)
microphysics = P3Microphysics(; aerosol, predict_supersaturation=true)

# ## Incoming air and precipitation boundaries
#
# At the surface, pressure equals the standard potential-temperature reference
# pressure, so the condensate-free inflow has T = θ₀. Its moisture and
# supersaturation are calculated from the same relative humidity. Fresh aerosol
# enters with the air; all incoming hydrometeor mass and number are zero.
# P3's default open precipitation boundary allows sedimentation out of the column.

ρ₀ = surface_density(reference_state)
qᵛ⁺ˡ₀ = saturation_specific_humidity(θ₀, ρ₀, constants, PlanarLiquidSurface())
qᵗ₀ = ℋ₀ * qᵛ⁺ˡ₀
sᵛ⁺ˡ₀ = qᵗ₀ - qᵛ⁺ˡ₀

microphysical_bcs = NamedTuple(name => FieldBoundaryConditions(bottom=ValueBoundaryCondition(0))
                              for name in prognostic_field_names(microphysics))
boundary_conditions = merge(microphysical_bcs,
    (; ρθ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * θ₀)),
       ρqᵗ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * qᵗ₀)),
       ρsᵛ⁺ˡ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * sᵛ⁺ˡ₀)),
       ρnᵃ = FieldBoundaryConditions(bottom=ValueBoundaryCondition(ρ₀ * nᵃ₀)),
       w = FieldBoundaryConditions(bottom=NormalFlowBoundaryCondition(W₀),
                                   top=NormalFlowBoundaryCondition(W₀))))

model = AtmosphereModel(grid; dynamics, microphysics, boundary_conditions,
                        formulation = :LiquidIcePotentialTemperature,
                        advection = WENO(order=5),
                        thermodynamic_constants = constants)

# ## Initializing supersaturation
#
# First initialize the thermodynamic state. Then initialize the supersaturation
# density from that state, so the prognostic starts with the actual vapor deficit.
# This second `set!` depends on the temperature and moisture established by the first.

set!(model; θ=θ_initial, ℋ=ℋ₀, w=W₀)
μ = model.microphysical_fields
ρ = reference_state.density
qᵛ⁺ˡ = SaturationSpecificHumidity(model)
set!(model; ρsᵛ⁺ˡ=ρ * (μ.qᵛ - qᵛ⁺ˡ))

# ## Simulation and output
#
# Run for 3600 s, saving profiles every 30 s. The output retains both prognostic
# and diagnosed supersaturation, along with mass and number concentrations, for
# subsequent comparisons. Supersaturation sᵛ⁺ˡ has units of kg/kg; it is not the
# dimensionless relative supersaturation qᵛ / qᵛ⁺ˡ − 1.

simulation = Simulation(model; Δt=1, stop_time=60minutes)
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

diagnosed_sᵛ⁺ˡ = Field(μ.qᵛ - qᵛ⁺ˡ)
outputs = (; T = model.temperature,
             μ.qᵛ, μ.qᶜˡ, μ.qʳ, μ.qⁱ, μ.qʷⁱ,
             μ.nᶜˡ, μ.nʳ, μ.nⁱ, μ.nᵃ, μ.sᵛ⁺ˡ, diagnosed_sᵛ⁺ˡ)
filename = "kinematic_driver_p3.jld2"
simulation.output_writers[:profiles] = JLD2Writer(model, outputs;
                                                 filename,
                                                 schedule = TimeInterval(30),
                                                 overwrite_files = true)

function progress(sim)
    μ = sim.model.microphysical_fields
    @info @sprintf("t = %s, max cloud = %.2e, rain = %.2e, ice = %.2e kg/kg",
                   prettytime(sim), maximum(μ.qᶜˡ), maximum(μ.qʳ), maximum(μ.qⁱ))
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10minutes))
run!(simulation)
nothing #hide

# ## Cloud and precipitation through time
#
# Each panel shows the full time history of one condensate. Color scales are
# labeled separately because cloud, rain, ice, and liquid coating can have very
# different abundances. All mass fractions retain their model units of kg/kg.
# Figures are saved as high-resolution PNGs and zoomable SVGs.

qᶜˡts = FieldTimeSeries(filename, "qᶜˡ")
qʳts = FieldTimeSeries(filename, "qʳ")
qⁱts = FieldTimeSeries(filename, "qⁱ")
qʷⁱts = FieldTimeSeries(filename, "qʷⁱ")
Tts = FieldTimeSeries(filename, "T")
qᵛts = FieldTimeSeries(filename, "qᵛ")
sᵛ⁺ˡts = FieldTimeSeries(filename, "sᵛ⁺ˡ")
diagnosed_sᵛ⁺ˡts = FieldTimeSeries(filename, "diagnosed_sᵛ⁺ˡ")
nothing #hide

# The time series is indexed over the physical column cells, excluding halos.

function plot_time_height!(layout, field_ts; title, label, colormap,
                           colorbar_ticks=Makie.automatic, kwargs...)
    z = znodes(field_ts.grid, Center()) / kilometers
    data = permutedims(Array(field_ts[1, 1, 1:Nz, :]))
    ax = Axis(layout[1, 1]; xlabel="Time (s)", ylabel="Height (km)", title)
    plot = heatmap!(ax, field_ts.times, z, data; colormap, rasterize=2, kwargs...)
    Colorbar(layout[1, 2], plot; label, ticks=colorbar_ticks)
    return ax
end
nothing #hide

set_theme!(fontsize=20, linewidth=3)
condensate_figure = Figure(size=(1200, 1000))
Label(condensate_figure[0, 1:2], "P3 cloud and precipitation · prescribed ascent at 2 m/s"; fontsize=24)
plot_time_height!(condensate_figure[1, 1], qᶜˡts;
                  title = "Cloud liquid", label = "qᶜˡ (kg/kg)", colormap = :Greens)
plot_time_height!(condensate_figure[1, 2], qʳts;
                  title = "Rain", label = "qʳ (kg/kg)", colormap = :Oranges)
plot_time_height!(condensate_figure[2, 1], qⁱts;
                  title = "Ice", label = "qⁱ (kg/kg)", colormap = :Blues)
plot_time_height!(condensate_figure[2, 2], qʷⁱts;
                  title = "Liquid on ice", label = "qʷⁱ (kg/kg)", colormap = :Purples)
save("kinematic_driver_p3_condensate.png", condensate_figure; px_per_unit=2)
save("kinematic_driver_p3_condensate.svg", condensate_figure)
condensate_figure

# [Download the condensate figure as SVG](kinematic_driver_p3_condensate.svg).
#
# ## Temperature, vapor, and supersaturation
#
# The two supersaturation panels share a symmetric logarithmic color scale,
# linear within ±10⁻⁵ kg/kg. This resolves small vapor excesses and larger deficits
# on the same scale. The prognostic and diagnosed values can differ while
# transport, cooling, and microphysical relaxation act.

supersaturation_limit = max(maximum(abs, sᵛ⁺ˡts[1, 1, 1:Nz, :]),
                            maximum(abs, diagnosed_sᵛ⁺ˡts[1, 1, 1:Nz, :]))
supersaturation_colors = (; colormap = :balance,
                            colorscale = Makie.Symlog10(1e-5),
                            colorrange = (-supersaturation_limit, supersaturation_limit),
                            colorbar_ticks = ([-1e-3, -1e-4, 0, 1e-4, 1e-3],
                                              ["−10⁻³", "−10⁻⁴", "0", "10⁻⁴", "10⁻³"]))
thermodynamics_figure = Figure(size=(1200, 1000))
plot_time_height!(thermodynamics_figure[1, 1], Tts;
                  title = "Temperature", label = "T (K)", colormap = :thermal)
plot_time_height!(thermodynamics_figure[1, 2], qᵛts;
                  title = "Water vapor", label = "qᵛ (kg/kg)", colormap = :Blues)
plot_time_height!(thermodynamics_figure[2, 1], sᵛ⁺ˡts;
                  title = "Prognostic supersaturation", label = "sᵛ⁺ˡ (kg/kg)", supersaturation_colors...)
plot_time_height!(thermodynamics_figure[2, 2], diagnosed_sᵛ⁺ˡts;
                  title = "Diagnosed supersaturation", label = "qᵛ − qᵛ⁺ˡ (kg/kg)", supersaturation_colors...)
save("kinematic_driver_p3_thermodynamics.png", thermodynamics_figure; px_per_unit=2)
save("kinematic_driver_p3_thermodynamics.svg", thermodynamics_figure)
thermodynamics_figure

# [Download the thermodynamics figure as SVG](kinematic_driver_p3_thermodynamics.svg).
#
# Unlike a closed parcel, the column's water inventory changes through inflow,
# outflow, sedimentation, and the prescribed divergence correction. A quantitative
# comparison with another kinematic driver must match those terms as well as the
# sounding, velocity profile, P3 options, and numerical resolution.
