# # Rising dry and cloudy parcels
#
# A rising air parcel expands and cools. Once it reaches saturation, cloud droplets
# form, precipitation develops, and latent heating changes the ascent. We begin
# with five demonstrations: dry ascent; prescribed ascent with one-moment,
# Kessler, and two-moment microphysics; and buoyancy-driven two-moment ascent.
# We then extend the comparison with prescribed and buoyant P3 parcels using
# **prognostic supersaturation** and aerosol activation.
#
# All seven parcels use liquid–ice potential temperature, the default formulation.
# It is conserved during closed adiabatic ascent; temperature responds to pressure
# and the evolving liquid and ice fractions. Each case gets its own readable figure.

using Oceananigans
using Oceananigans: interpolate
using Oceananigans.Units
using Breeze
using Breeze: DCMIP2016KesslerMicrophysics, TetensFormula
using Breeze.Microphysics.PredictedParticleProperties: AerosolActivation, AerosolMode
using CloudMicrophysics
using CairoMakie

# ## Background atmosphere
#
# The grid holds environmental profiles; the parcel itself is a single moving
# thermodynamic state. We use a constant-potential-temperature reference atmosphere
# and an environmental humidity that decreases with height. Each parcel starts
# at the surface with the same temperature and total water mass fraction.

grid = RectilinearGrid(size=700, z=(0, 7kilometers), topology=(Flat, Flat, Bounded))
constants = ThermodynamicConstants()

qᵗ₀ = 0.015    # Surface specific humidity [kg/kg]
Hq = 2500      # Humidity scale height [m]
qᵗ(z) = qᵗ₀ * exp(-z / Hq)

# ## Recording a parcel trajectory
#
# A shared helper initializes and runs each parcel. Snapshots copy the scalar
# thermodynamic and microphysical states, including P3's prognostic `ρsᵛ⁺ˡ`.
# Unlike the dimensionless relative supersaturation S, the specific prognostic
# sᵛ⁺ˡ = qᵛ − qᵛ⁺ˡ has units of kg/kg. We initialize it consistently with the
# parcel's vapor and saturation mass fractions.
#
# Parcel dynamics retains all condensate: precipitation forms but does not fall
# out of this closed parcel. The environmental humidity profile is not entrained.

function run_parcel(; microphysics=nothing, buoyant=false, stop_time=90minutes, Δt=1,
                      thermodynamic_constants=constants)
    dynamics = if buoyant
        ParcelDynamics(vertical_velocity_formulation=PrognosticVerticalVelocity())
    else
        ParcelDynamics()
    end
    model = AtmosphereModel(grid; dynamics, microphysics,
                            formulation = :LiquidIcePotentialTemperature,
                            thermodynamic_constants)
    reference_state = ReferenceState(grid, thermodynamic_constants;
                                     base_pressure = 101325,
                                     potential_temperature = 300)
    set!(model; qᵗ, z = 0, w = 1, w_parcel = 1,
         θ = reference_state.potential_temperature,
         p = reference_state.pressure,
         ρ = reference_state.density)

    state = model.dynamics.state
    if !isnothing(state.μ) && haskey(state.μ, :ρsᵛ⁺ˡ)
        qᵛ⁺ˡ = saturation_specific_humidity(state.𝒰, thermodynamic_constants, PlanarLiquidSurface())
        sᵛ⁺ˡ = state.𝒰.moisture_mass_fractions.vapor - qᵛ⁺ˡ
        state.μ = merge(state.μ, (; ρsᵛ⁺ˡ=state.ρ * sᵛ⁺ˡ))
    end

    snapshot() = (; t = model.clock.time, z = state.z, w = state.w,
                   ρ = state.ρ, 𝒰 = state.𝒰, μ = state.μ)
    snapshots = typeof(snapshot())[]
    simulation = Simulation(model; Δt, stop_time)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    add_callback!(simulation, sim -> push!(snapshots, snapshot()), TimeInterval(5))
    run!(simulation)

    return (; model, snapshots)
end
nothing #hide

# ## Dry adiabatic ascent
#
# With no microphysics, water remains vapor even after saturation is reached.
# This provides a reference for the cooling that would occur without latent heating.

dry = run_parcel(stop_time=30minutes)
nothing #hide

# Use two generous panels rather than compressing all the cases into one figure.
# Every figure is also saved as a vector SVG for zooming and a high-resolution PNG.

set_theme!(fontsize=20, linewidth=3)
dry_z = [s.z / kilometers for s in dry.snapshots]
dry_T = [temperature(s.𝒰, constants) for s in dry.snapshots]
dry_Tₑ = [interpolate(s.z, dry.model.temperature) for s in dry.snapshots]
dry_S = [s.𝒰.moisture_mass_fractions.vapor /
         saturation_specific_humidity(s.𝒰, constants, PlanarLiquidSurface()) - 1
         for s in dry.snapshots]

dry_figure = Figure(size=(1000, 600))
ax = Axis(dry_figure[1, 1]; xlabel = "Temperature (K)", ylabel = "Height (km)",
          title = "Dry adiabatic cooling")
lines!(ax, dry_T, dry_z; color=:magenta, label="Parcel")
lines!(ax, dry_Tₑ, dry_z; color=:gray, linestyle=:dash, label="Environment")
axislegend(ax; position=:lb)
ax = Axis(dry_figure[1, 2]; xlabel = "Relative supersaturation S", ylabel = "Height (km)",
          title = "Approach to saturation")
vlines!(ax, [0]; color=:gray, linestyle=:dash)
lines!(ax, dry_S, dry_z; color=:dodgerblue)
save("rising_parcels_dry.svg", dry_figure)
save("rising_parcels_dry.png", dry_figure; px_per_unit=2)
dry_figure

# [Download the dry-ascent figure as SVG](rising_parcels_dry.svg).
#
# ## Plotting cloudy parcels
#
# Cloudy parcels share temperature, supersaturation, and condensate diagnostics.
# Number-predicting schemes get a fourth panel for aerosols and hydrometeors.
# Here supersaturation means qᵛ − qᵛ⁺ˡ in kg/kg, using each model's saturation law.
# P3 also carries this quantity prognostically; both curves will be shown there.

function plot_cloudy_parcel(parcel, title)
    constants = parcel.model.thermodynamic_constants
    snapshots = parcel.snapshots
    prognostics = first(snapshots).μ
    predicts_supersaturation = haskey(prognostics, :ρsᵛ⁺ˡ)
    predicts_number = haskey(prognostics, :ρnᶜˡ)
    z = [s.z / kilometers for s in snapshots]
    T = [temperature(s.𝒰, constants) for s in snapshots]
    Tₑ = [interpolate(s.z, parcel.model.temperature) for s in snapshots]
    diagnosed_sᵛ⁺ˡ = [s.𝒰.moisture_mass_fractions.vapor -
                      saturation_specific_humidity(s.𝒰, constants, PlanarLiquidSurface())
                      for s in snapshots]

    fig = Figure(size=predicts_number ? (1000, 1000) : (1500, 600))
    columns = predicts_number ? 2 : 3
    Label(fig[0, 1:columns], title; fontsize=24)
    ax = Axis(fig[1, 1]; xlabel = "Temperature (K)", ylabel = "Height (km)",
              title = "Latent heating")
    lines!(ax, T, z; color=:magenta, label="Parcel")
    lines!(ax, Tₑ, z; color=:gray, linestyle=:dash, label="Environment")
    axislegend(ax; position=:lb)

    ax = Axis(fig[1, 2]; xlabel = "Supersaturation (kg/kg)", ylabel = "Height (km)",
              title = "Supersaturation",
              xscale = Makie.Symlog10(1e-5),
              xticks = ([-1e-2, -1e-3, -1e-4, 0, 1e-4, 1e-3],
                        ["−10⁻²", "−10⁻³", "−10⁻⁴", "0", "10⁻⁴", "10⁻³"]))
    vlines!(ax, [0]; color=:gray, linestyle=:dot)
    if predicts_supersaturation
        sᵛ⁺ˡ = [s.μ.ρsᵛ⁺ˡ / s.ρ for s in snapshots]
        lines!(ax, sᵛ⁺ˡ, z; color=:dodgerblue, label="Prognostic")
    end
    lines!(ax, diagnosed_sᵛ⁺ˡ, z; color=:black, linestyle=:dash, label="Diagnosed")
    axislegend(ax; position=:rb)

    mass_panel = predicts_number ? fig[2, 1] : fig[1, 3]
    ax = Axis(mass_panel; xlabel = "Mass fraction (kg/kg)", ylabel = "Height (km)",
              title = "Condensate")
    for (name, label, color) in ((:ρqᶜˡ, "Cloud liquid", :lime),
                                (:ρqʳ, "Rain", :orangered),
                                (:ρqᶜⁱ, "Cloud ice", :deepskyblue4),
                                (:ρqⁱ, "Ice", :deepskyblue4),
                                (:ρqʷⁱ, "Liquid on ice", :purple))
        haskey(prognostics, name) || continue
        q = [getproperty(s.μ, name) / s.ρ for s in snapshots]
        lines!(ax, q, z; color, label)
    end
    axislegend(ax; position=:rb)

    if predicts_number
        ax = Axis(fig[2, 2]; xlabel = "Particle number (kg⁻¹)", ylabel = "Height (km)",
                  title = "Aerosol and hydrometeors", xscale = log10,
                  xticks = (10.0 .^ (-3:3:9), ["10⁻³", "10⁰", "10³", "10⁶", "10⁹"]))
        for (name, label, color) in ((:ρnᵃ, "Aerosol", :gray),
                                    (:ρnᶜˡ, "Cloud droplets", :lime),
                                    (:ρnʳ, "Rain drops", :orangered),
                                    (:ρnⁱ, "Ice particles", :deepskyblue4))
            haskey(prognostics, name) || continue
            n = [getproperty(s.μ, name) / s.ρ for s in snapshots]
            ## NaNs leave gaps where a population is absent, without joining across them.
            n = [value > 1e-3 ? value : NaN for value in n]
            lines!(ax, n, z; color, label)
        end
        axislegend(ax; position=:lb)
    end
    return fig
end
nothing #hide

function plot_parcel_dynamics(parcel, title)
    t = [s.t for s in parcel.snapshots]
    z = [s.z / kilometers for s in parcel.snapshots]
    w = [s.w for s in parcel.snapshots]
    fig = Figure(size=(1000, 600))
    Label(fig[0, 1:2], title; fontsize=24)
    ax = Axis(fig[1, 1]; xlabel="Time (s)", ylabel="Height (km)", title="Parcel ascent")
    lines!(ax, t, z; color=:black, label="Buoyant")
    lines!(ax, t, t / kilometers; color=:gray, linestyle=:dash, label="Prescribed 1 m/s")
    axislegend(ax; position=:lt)
    ax = Axis(fig[1, 2]; xlabel="Time (s)", ylabel="Vertical velocity (m/s)",
              title="Buoyancy changes the ascent rate")
    lines!(ax, t, w; color=:black)
    hlines!(ax, [1]; color=:gray, linestyle=:dash)
    return fig
end
nothing #hide

# ## Cloudy ascent with one-moment microphysics
#
# The one-moment scheme tracks condensate mass. Non-equilibrium cloud formation
# relaxes supersaturation through condensation, while autoconversion and accretion
# transfer cloud water into rain following [Morrison and Grabowski (2008)](@cite Morrison2008novel).
# As condensate forms, latent heating reduces the cooling during ascent.
# The prescribed ascent is 1 m/s for 90 minutes.

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

one_moment = run_parcel(microphysics=OneMomentCloudMicrophysics())
one_moment_figure = plot_cloudy_parcel(one_moment, "One-moment microphysics · prescribed ascent at 1 m/s")
save("rising_parcels_one_moment.svg", one_moment_figure)
save("rising_parcels_one_moment.png", one_moment_figure; px_per_unit=2)
one_moment_figure

# [Download the one-moment figure as SVG](rising_parcels_one_moment.svg).
#
# ## Cloudy ascent with Kessler microphysics
#
# The DCMIP2016 Kessler scheme includes saturation adjustment, autoconversion,
# accretion, and rain evaporation following [Kessler (1969)](@cite Kessler1969). We use its Tetens
# saturation-vapor-pressure formula, including when diagnosing supersaturation.
# Comparing this case with the one-moment parcel shows how their different
# treatments of cloud formation affect the vapor deficit and condensate.

kessler_constants = ThermodynamicConstants(saturation_vapor_pressure=TetensFormula())
kessler = run_parcel(microphysics=DCMIP2016KesslerMicrophysics(),
                     thermodynamic_constants=kessler_constants)
kessler_figure = plot_cloudy_parcel(kessler, "Kessler microphysics · prescribed ascent at 1 m/s")
save("rising_parcels_kessler.svg", kessler_figure)
save("rising_parcels_kessler.png", kessler_figure; px_per_unit=2)
kessler_figure

# [Download the Kessler figure as SVG](rising_parcels_kessler.svg).
#
# ## Cloudy ascent with two-moment microphysics
#
# The [Seifert and Beheng (2006)](@cite SeifertBeheng2006) scheme predicts both
# mass and number for cloud droplets and rain. Aerosol activation follows
# [Abdul-Razzak and Ghan (2000)](@cite AbdulRazzakGhan2000): supersaturation
# activates droplets from the aerosol reservoir. The number panel reveals
# activation, aerosol depletion, and changes in the particle populations as
# droplets collide and rain develops.

two_moment = run_parcel(microphysics=TwoMomentCloudMicrophysics())
two_moment_figure = plot_cloudy_parcel(two_moment, "Two-moment microphysics · prescribed ascent at 1 m/s")
save("rising_parcels_two_moment.svg", two_moment_figure)
save("rising_parcels_two_moment.png", two_moment_figure; px_per_unit=2)
two_moment_figure

# [Download the two-moment figure as SVG](rising_parcels_two_moment.svg).
#
# ## Buoyancy-driven ascent with two-moment microphysics
#
# Now the vertical velocity responds to buoyancy:
# dw/dt = −g(ρ_parcel − ρ_env)/ρ_env. We use the same two-moment
# scheme as the preceding prescribed ascent.
# The parcel starts at 1 m/s and runs for 300 s. Condensation warms the parcel,
# changes its density, and feeds back on its ascent speed.

buoyant_two_moment = run_parcel(microphysics=TwoMomentCloudMicrophysics(),
                                buoyant=true, stop_time=5minutes)
buoyant_two_moment_figure = plot_cloudy_parcel(buoyant_two_moment, "Two-moment microphysics · buoyant ascent")
save("rising_parcels_buoyant_two_moment.svg", buoyant_two_moment_figure)
save("rising_parcels_buoyant_two_moment.png", buoyant_two_moment_figure; px_per_unit=2)
buoyant_two_moment_figure

# [Download the buoyant two-moment figure as SVG](rising_parcels_buoyant_two_moment.svg).

buoyant_two_moment_dynamics = plot_parcel_dynamics(buoyant_two_moment, "Two-moment buoyant ascent")
save("rising_parcels_two_moment_dynamics.svg", buoyant_two_moment_dynamics)
save("rising_parcels_two_moment_dynamics.png", buoyant_two_moment_dynamics; px_per_unit=2)
buoyant_two_moment_dynamics

# [Download the two-moment ascent history as SVG](rising_parcels_two_moment_dynamics.svg).
#
# ## Prescribed ascent with P3 and prognostic supersaturation
#
# We now extend the demonstrations with [`P3Microphysics`](@ref), which also
# follows the properties of rimed and partially melted ice. See [P3: Theory](@ref p3_theory).
# `predict_supersaturation=true` adds the prognostic density `ρsᵛ⁺ˡ`.
# Aerosol activation predicts cloud droplet number, and `prognostic=true` also
# tracks the remaining unactivated aerosol reservoir. The same P3 configuration
# is used for both P3 trajectories.

microphysics = P3Microphysics(; predict_supersaturation = true,
                              aerosol = AerosolActivation(AerosolMode(); prognostic=true))
cloudy = run_parcel(; microphysics)
nothing #hide

# ## Cloud and precipitation evolution
#
# Each P3 case gets a full four-panel figure. Supersaturation has its own
# panel: we plot the prognostic sᵛ⁺ˡ alongside qᵛ − qᵛ⁺ˡ diagnosed from the
# thermodynamic state, using kg/kg for both. Their agreement checks the coupling
# of prognostic supersaturation to expansion and latent heating.
# A symmetric logarithmic axis, linear within
# ±10⁻⁵ kg/kg, resolves small supersaturations alongside the initial vapor deficit.
#
# Condensate mass and particle number occupy separate panels. The number axis
# is logarithmic; absent populations are omitted. Rime is part of the ice mass
# and is not counted again as an independent water reservoir.

cloudy_figure = plot_cloudy_parcel(cloudy, "P3 parcel · θ_li · prescribed ascent at 1 m/s")
save("rising_parcels_p3.svg", cloudy_figure)
save("rising_parcels_p3.png", cloudy_figure; px_per_unit=2)
cloudy_figure

# [Download the prescribed-ascent figure as SVG](rising_parcels_p3.svg).
#
# ## Buoyancy-driven ascent with P3
#
# Now the parcel's velocity evolves according to dw/dt = −g(ρ_parcel − ρ_env)/ρ_env.
# Condensation and freezing release latent heat, changing parcel density and
# therefore buoyancy. The parcel starts at 1 m/s, as in the prescribed case,
# but we follow it for only 300 s because its ascent can accelerate rapidly.

buoyant = run_parcel(; microphysics, buoyant=true, stop_time=5minutes)
buoyant_figure = plot_cloudy_parcel(buoyant, "P3 parcel · θ_li · buoyancy-driven ascent")
save("rising_parcels_buoyant_p3.svg", buoyant_figure)
save("rising_parcels_buoyant_p3.png", buoyant_figure; px_per_unit=2)
buoyant_figure

# [Download the buoyant-parcel figure as SVG](rising_parcels_buoyant_p3.svg).
#
# Plot the dynamics separately so that neither time history has to fit into a
# narrow extra column. The dashed curves show the prescribed 1 m/s ascent over
# the same time interval.

dynamics_figure = plot_parcel_dynamics(buoyant, "P3 buoyant ascent")
save("rising_parcels_dynamics.svg", dynamics_figure)
save("rising_parcels_dynamics.png", dynamics_figure; px_per_unit=2)
dynamics_figure

# [Download the ascent-history figure as SVG](rising_parcels_dynamics.svg).
#
# The prescribed and buoyant parcels illustrate the feedback between ascent,
# supersaturation, aerosol activation, and latent heating. Their particle populations
# need not agree at the same height: different ascent speeds allow different amounts
# of time for condensation, collisions, and ice processes. Total water remains in
# each parcel even as its partition among vapor and condensate changes.
