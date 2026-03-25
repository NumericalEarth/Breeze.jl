# # P3 parcel explorer in a deep-convection sounding
#
# This example tells the P3 parcel story in two steps, both using the
# idealized supercell sounding of [Klemp et al. (2015)](@cite KlempEtAl2015)
# and a buoyancy-driven parcel model.
#
# 1. **P3 fidelity check**: the same warm-bubble parcel is launched with
#    either P3 or Kessler microphysics. The parcel starts from the same sounding,
#    with the same thermal perturbation and the same initial moisture. This lets
#    us compare temperature, vertical velocity, and hydrometeor partition without
#    changing the launch conditions.
# 2. **P3 feature exploration**: the same sounding and the same total moisture
#    are reused, but the parcel is seeded with the same ice population and
#    different cloud/rain partitions. This isolates the P3 idea that one ice
#    category can move smoothly through a continuum of rime fraction, rime
#    density, size, and fall speed.
#
# Kessler remains useful as a warm-rain reference, but only P3 can evolve a
# mixed-phase continuum of ice properties.

using Breeze
using CairoMakie
using Oceananigans
using Oceananigans: interpolate
using Oceananigans.Units

using Breeze: DCMIP2016KesslerMicrophysics, PrognosticVerticalVelocity,
              TetensFormula, ThermodynamicConstants

const AtmosphereModels = Breeze.AtmosphereModels
const PredictedParticleProperties = Breeze.Microphysics.PredictedParticleProperties
const Thermodynamics = Breeze.Thermodynamics

# ## Idealized supercell sounding
#
# We borrow the background potential-temperature, relative-humidity, and wind
# profiles from the splitting-supercell example. The parcel launch uses the same
# 0.5 K warm perturbation at 1.5 km, but here the vertical motion is carried by
# the parcel itself via `PrognosticVerticalVelocity`.

grid = RectilinearGrid(size = 15000, z = (0, 15kilometers), topology = (Flat, Flat, Bounded))

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 100000,
                                 potential_temperature = 300)

θ₀ = 300
θᵖ = 343
zᵖ = 12kilometers
Tᵖ = 213

zˢ = 5kilometers
uˢ = 30
uᶜ = 15

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_background(z)
    if z <= zᵖ
        return θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5 / 4)
    else
        return θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    end
end

function ℋ_background(z)
    if z <= zᵖ
        return 1 - 3 / 4 * (z / zᵖ)^(5 / 4)
    else
        return 1 / 4
    end
end

function u_background(z)
    if z < zˢ - 1000
        return uˢ * (z / zˢ) - uᶜ
    elseif abs(z - zˢ) <= 1000
        return (-4 / 5 + 3 * (z / zˢ) - 5 / 4 * (z / zˢ)^2) * uˢ - uᶜ
    else
        return uˢ - uᶜ
    end
end

launch_height = 1500
launch_θ_perturbation = 0.5
initial_vertical_velocity = 5

plot_top = 10kilometers
stop_time = 18minutes
Δt = 1
record_interval = 1

seeded_ice_mass = 1e-4
seeded_ice_number = 5e5
cloud_partition = 8e-4
rain_partition = 3e-4
rain_number_partition = 1e5

height_profile = collect(range(0, plot_top, length = 400))
background_T_profile = [Thermodynamics.temperature_from_potential_temperature(θ_background(z),
                                                                              interpolate(z, reference_state.pressure),
                                                                              constants;
                                                                              pˢᵗ = reference_state.standard_pressure)
                        for z in height_profile]
nothing #hide

# ## Helper functions
#
# The helpers below set up the buoyant parcel launch, seed optional
# microphysics partitions, and convert the P3 moments into the continuous ice
# diagnostics that we want to visualize.

function p3_ice_diagnostics(p3, ρ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ)
    FT = typeof(ρ)

    if qⁱ <= p3.minimum_mass_mixing_ratio || nⁱ <= p3.minimum_number_mixing_ratio
        return (; Fᶠ = zero(FT),
                  ρᶠ = zero(FT),
                  Fˡ = zero(FT),
                  mean_diameter = zero(FT),
                  fall_speed = zero(FT),
                  reflectivity = zero(FT))
    end

    rime_state = PredictedParticleProperties.consistent_rime_state(p3, qⁱ, qᶠ, bᶠ)
    Fᶠ = rime_state.Fᶠ
    ρᶠ = rime_state.ρᶠ
    Fˡ = qʷⁱ > 0 ? qʷⁱ / (qⁱ + qʷⁱ) : zero(FT)

    params = PredictedParticleProperties.distribution_parameters(ρ * qⁱ, ρ * nⁱ, Fᶠ, ρᶠ)

    state = PredictedParticleProperties.IceSizeDistributionState(FT;
        intercept = params.N₀,
        shape = params.μ,
        slope = params.λ,
        rime_fraction = Fᶠ,
        liquid_fraction = Fˡ,
        rime_density = ρᶠ,
        air_density = ρ)

    return (; Fᶠ,
              ρᶠ,
              Fˡ,
              mean_diameter = PredictedParticleProperties.evaluate(PredictedParticleProperties.MeanDiameter(), state),
              fall_speed = PredictedParticleProperties.evaluate(PredictedParticleProperties.MassWeightedFallSpeed(), state),
              reflectivity = PredictedParticleProperties.evaluate(PredictedParticleProperties.Reflectivity(), state) / ρ)
end

function initialize_p3_state(p3, ρ; qᶜˡ = 0, qʳ = 0, nʳ = 0,
                               qⁱ = 0, nⁱ = 0, qᶠ = 0, bᶠ = 0, qʷⁱ = 0)
    reflectivity = p3_ice_diagnostics(p3, ρ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ).reflectivity

    return (; ρqᶜˡ = ρ * qᶜˡ,
              ρqʳ = ρ * qʳ,
              ρnʳ = ρ * nʳ,
              ρqⁱ = ρ * qⁱ,
              ρnⁱ = ρ * nⁱ,
              ρqᶠ = ρ * qᶠ,
              ρbᶠ = ρ * bᶠ,
              ρzⁱ = ρ * reflectivity,
              ρqʷⁱ = ρ * qʷⁱ)
end

prognostic_parcel_dynamics() = ParcelDynamics(vertical_velocity_formulation = PrognosticVerticalVelocity())

supercell_parcel_model(microphysics) = AtmosphereModel(grid;
                                                       dynamics = prognostic_parcel_dynamics(),
                                                       microphysics = microphysics,
                                                       thermodynamic_constants = constants)

function apply_warm_bubble_perturbation!(model; Δθ = launch_θ_perturbation)
    state = model.dynamics.state
    p = state.𝒰.reference_pressure
    θ = θ_background(state.z) + Δθ
    T = Thermodynamics.temperature_from_potential_temperature(θ, p, model.thermodynamic_constants;
                                                              pˢᵗ = model.dynamics.standard_pressure)

    state.𝒰 = Thermodynamics.with_temperature(state.𝒰, T, model.thermodynamic_constants)
    state.ℰ = state.𝒰.static_energy
    state.ρℰ = state.ρ * state.ℰ

    return nothing
end

function rebuild_parcel_thermodynamics!(model; temperature_target = nothing)
    state = model.dynamics.state
    microphysics = model.microphysics
    zero_velocities = (; u = zero(state.ρ), v = zero(state.ρ), w = zero(state.ρ))

    ℳ = AtmosphereModels.microphysical_state(microphysics, state.ρ, state.μ, state.𝒰, zero_velocities)
    qᵛᵉ = AtmosphereModels.specific_prognostic_moisture_from_total(microphysics, state.qᵗ, ℳ)
    q = AtmosphereModels.moisture_fractions(microphysics, ℳ, qᵛᵉ)

    state.𝒰 = Thermodynamics.with_moisture(state.𝒰, q)

    if !isnothing(temperature_target)
        state.𝒰 = Thermodynamics.with_temperature(state.𝒰, temperature_target, model.thermodynamic_constants)
    end

    state.ℰ = state.𝒰.static_energy
    state.ρℰ = state.ρ * state.ℰ

    return nothing
end

function initialize_supercell_parcel!(model; z = launch_height, w = initial_vertical_velocity)
    set!(model,
         θ = θ_background,
         ℋ = ℋ_background,
         p = reference_state.pressure,
         ρ = reference_state.density,
         u = u_background,
         z = z,
         w_parcel = w)

    apply_warm_bubble_perturbation!(model)

    return nothing
end

function stop_at_plot_top!(simulation)
    function maybe_stop(sim)
        if sim.model.dynamics.state.z >= plot_top
            sim.stop_time = sim.model.clock.time
        end

        return nothing
    end

    add_callback!(simulation, maybe_stop, IterationInterval(1))

    return nothing
end

function seed_p3_parcel!(model, p3; qᶜˡ = 0, qʳ = 0, nʳ = 0, qⁱ = 0, nⁱ = 0)
    state = model.dynamics.state
    T₀ = temperature(state.𝒰, model.thermodynamic_constants)

    state.μ = initialize_p3_state(p3, state.ρ;
                                  qᶜˡ = qᶜˡ,
                                  qʳ = qʳ,
                                  nʳ = nʳ,
                                  qⁱ = qⁱ,
                                  nⁱ = nⁱ)

    rebuild_parcel_thermodynamics!(model; temperature_target = T₀)

    return nothing
end

function seed_kessler_parcel!(model; qᶜˡ = 0, qʳ = 0)
    state = model.dynamics.state
    T₀ = temperature(state.𝒰, model.thermodynamic_constants)

    state.μ = (; ρqᶜˡ = state.ρ * qᶜˡ,
                ρqʳ = state.ρ * qʳ)

    rebuild_parcel_thermodynamics!(model; temperature_target = T₀)

    return nothing
end

function ascending_branch(values, z)
    ascending_indices = Int[1]
    z_max = first(z)

    for i in 2:length(z)
        if z[i] > z_max
            push!(ascending_indices, i)
            z_max = z[i]
        end
    end

    return values[ascending_indices], z[ascending_indices]
end

function run_p3_case(; label, color, qᶜˡ = 0, qʳ = 0, nʳ = 0, qⁱ = 0, nⁱ = 0)
    microphysics = PredictedParticlePropertiesMicrophysics()
    model = supercell_parcel_model(microphysics)

    initialize_supercell_parcel!(model)

    if qᶜˡ > 0 || qʳ > 0 || nʳ > 0 || qⁱ > 0 || nⁱ > 0
        seed_p3_parcel!(model, microphysics; qᶜˡ, qʳ, nʳ, qⁱ, nⁱ)
    end

    simulation = Simulation(model; Δt, stop_time, verbose = false)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    stop_at_plot_top!(simulation)

    z = Float64[]
    w = Float64[]
    T = Float64[]
    qᶜˡ_ts = Float64[]
    qʳ_ts = Float64[]
    qⁱ_ts = Float64[]
    Fᶠ_ts = Float64[]
    ρᶠ_ts = Float64[]
    mean_diameter_ts = Float64[]
    fall_speed_ts = Float64[]

    function record_state(sim)
        state = sim.model.dynamics.state
        μ = state.μ

        qⁱₙ = μ.ρqⁱ / state.ρ
        nⁱₙ = μ.ρnⁱ / state.ρ
        qᶠₙ = μ.ρqᶠ / state.ρ
        bᶠₙ = μ.ρbᶠ / state.ρ
        qʷⁱₙ = μ.ρqʷⁱ / state.ρ

        diagnostics = p3_ice_diagnostics(microphysics, state.ρ, qⁱₙ, nⁱₙ, qᶠₙ, bᶠₙ, qʷⁱₙ)

        push!(z, state.z)
        push!(w, state.w)
        push!(T, temperature(state.𝒰, sim.model.thermodynamic_constants))
        push!(qᶜˡ_ts, μ.ρqᶜˡ / state.ρ)
        push!(qʳ_ts, μ.ρqʳ / state.ρ)
        push!(qⁱ_ts, qⁱₙ)
        push!(Fᶠ_ts, diagnostics.Fᶠ)
        push!(ρᶠ_ts, diagnostics.ρᶠ)
        push!(mean_diameter_ts, diagnostics.mean_diameter)
        push!(fall_speed_ts, diagnostics.fall_speed)

        return nothing
    end

    record_state(simulation)
    add_callback!(simulation, record_state, IterationInterval(record_interval))
    run!(simulation)

    return (; label,
              color,
              z,
              w,
              T,
              qᶜˡ = qᶜˡ_ts,
              qʳ = qʳ_ts,
              qⁱ = qⁱ_ts,
              Fᶠ = Fᶠ_ts,
              ρᶠ = ρᶠ_ts,
              mean_diameter = mean_diameter_ts,
              fall_speed = fall_speed_ts)
end

function run_kessler_case(; label, color, qᶜˡ = 0, qʳ = 0)
    microphysics = DCMIP2016KesslerMicrophysics()
    model = supercell_parcel_model(microphysics)

    initialize_supercell_parcel!(model)

    if qᶜˡ > 0 || qʳ > 0
        seed_kessler_parcel!(model; qᶜˡ, qʳ)
    end

    simulation = Simulation(model; Δt, stop_time, verbose = false)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    stop_at_plot_top!(simulation)

    z = Float64[]
    w = Float64[]
    T = Float64[]
    qᶜˡ_ts = Float64[]
    qʳ_ts = Float64[]

    function record_state(sim)
        state = sim.model.dynamics.state
        μ = state.μ

        push!(z, state.z)
        push!(w, state.w)
        push!(T, temperature(state.𝒰, sim.model.thermodynamic_constants))
        push!(qᶜˡ_ts, μ.ρqᶜˡ / state.ρ)
        push!(qʳ_ts, μ.ρqʳ / state.ρ)

        return nothing
    end

    record_state(simulation)
    add_callback!(simulation, record_state, IterationInterval(record_interval))
    run!(simulation)

    return (; label,
              color,
              z,
              w,
              T,
              qᶜˡ = qᶜˡ_ts,
              qʳ = qʳ_ts)
end

# ## Section 1: P3 vs. Kessler parcels
#
# Both parcels are launched from the same sounding,
# same 0.5 K thermal perturbation, same initial vertical velocity, and no seeded
# condensate. Differences aloft therefore come from the microphysics, not the
# initialization.

p3_reference = run_p3_case(label = "P3", color = :dodgerblue)
kessler_reference = run_kessler_case(label = "Kessler", color = :black)
nothing #hide

set_theme!(fontsize = 16, linewidth = 2.5)

fig1 = Figure(size = (1250, 450))

ax11 = Axis(fig1[1, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Same launch, different microphysics")

ax12 = Axis(fig1[1, 2];
    xlabel = "Vertical velocity (m/s)",
    ylabel = "Height (km)",
    title = "Buoyancy-driven updraft")

ax13 = Axis(fig1[1, 3];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Hydrometeor partition")

lines!(ax11, background_T_profile, height_profile ./ 1000;
       color = :gray40, linestyle = :dot, label = "Environment")
lines!(ax11, ascending_branch(p3_reference.T, p3_reference.z ./ 1000)...;
       color = :magenta, label = p3_reference.label)
lines!(ax11, ascending_branch(kessler_reference.T, kessler_reference.z ./ 1000)...;
       color = :black, linestyle = :dash, label = kessler_reference.label)

vlines!(ax12, [0]; color = :gray40, linestyle = :dot)
lines!(ax12, ascending_branch(p3_reference.w, p3_reference.z ./ 1000)...;
       color = p3_reference.color, label = p3_reference.label)
lines!(ax12, ascending_branch(kessler_reference.w, kessler_reference.z ./ 1000)...;
       color = kessler_reference.color, linestyle = :dash, label = kessler_reference.label)

lines!(ax13, ascending_branch(p3_reference.qᶜˡ, p3_reference.z ./ 1000)...;
       color = :lime, label = "P3 qᶜˡ")
lines!(ax13, ascending_branch(p3_reference.qʳ, p3_reference.z ./ 1000)...;
       color = :orangered, label = "P3 qʳ")
lines!(ax13, ascending_branch(p3_reference.qⁱ, p3_reference.z ./ 1000)...;
       color = :dodgerblue, label = "P3 qⁱ")

lines!(ax13, ascending_branch(kessler_reference.qᶜˡ, kessler_reference.z ./ 1000)...;
       color = :lime, linestyle = :dash, label = "Kessler qᶜˡ")
lines!(ax13, ascending_branch(kessler_reference.qʳ, kessler_reference.z ./ 1000)...;
       color = :orangered, linestyle = :dash, label = "Kessler qʳ")

ylims!(ax11, 0, plot_top / 1000)
ylims!(ax12, 0, plot_top / 1000)
ylims!(ax13, 0, plot_top / 1000)

axislegend(ax11; position = :lb, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax12; position = :lt, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax13; position = :rb, labelsize = 11, nbanks = 2, backgroundcolor = (:white, 0.8))

fig1

# ## Section 2: the P3 continuum with fixed total water
#
# We now reuse the same sounding and parcel launch, but we keep the parcel
# temperature and total water fixed while repartitioning that water among vapor,
# cloud, and rain. All three P3 parcels share the same seeded ice distribution.
# The only change is the initial cloud/rain partition available for riming.

p3_feature_cases = [
    run_p3_case(;
        label = "Deposition only",
        color = :dodgerblue,
        qⁱ = seeded_ice_mass,
        nⁱ = seeded_ice_number),

    run_p3_case(;
        label = "Cloud riming",
        color = :lime,
        qᶜˡ = cloud_partition,
        qⁱ = seeded_ice_mass,
        nⁱ = seeded_ice_number),

    run_p3_case(;
        label = "Cloud + rain riming",
        color = :orangered,
        qᶜˡ = cloud_partition,
        qʳ = rain_partition,
        nʳ = rain_number_partition,
        qⁱ = seeded_ice_mass,
        nⁱ = seeded_ice_number),
]
nothing #hide

fig2 = Figure(size = (1250, 950))

ax21 = Axis(fig2[1, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Same launch temperature, different partitions")

ax22 = Axis(fig2[1, 2];
    xlabel = "Vertical velocity (m/s)",
    ylabel = "Height (km)",
    title = "Updraft responds to latent heating")

ax23 = Axis(fig2[1, 3];
    xlabel = "Ice mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Liquid availability changes ice growth")

ax24 = Axis(fig2[2, 1];
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Rime density ρᶠ [kg/m³]",
    title = "P3 moves through a continuum")

ax25 = Axis(fig2[2, 2];
    xlabel = "Mean diameter [m]",
    ylabel = "Mass-weighted fall speed [m/s]",
    xscale = log10,
    yscale = log10,
    title = "Bulk fall speed evolves continuously")

ax26 = Axis(fig2[2, 3];
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Height (km)",
    title = "Riming turns on smoothly")

lines!(ax21, background_T_profile, height_profile ./ 1000;
       color = :gray40, linestyle = :dot, label = "Environment")
vlines!(ax22, [0]; color = :gray40, linestyle = :dot)

for case in p3_feature_cases
    lines!(ax21, ascending_branch(case.T, case.z ./ 1000)...; color = case.color, label = case.label)
    lines!(ax22, ascending_branch(case.w, case.z ./ 1000)...; color = case.color, label = case.label)
    lines!(ax23, ascending_branch(case.qⁱ, case.z ./ 1000)...; color = case.color, label = case.label)

    lines!(ax24, case.Fᶠ, case.ρᶠ; color = case.color, label = case.label)
    scatter!(ax24, [first(case.Fᶠ)], [first(case.ρᶠ)];
             color = case.color, marker = :circle, markersize = 10)
    scatter!(ax24, [last(case.Fᶠ)], [last(case.ρᶠ)];
             color = case.color, marker = :utriangle, markersize = 12)

    lines!(ax25, case.mean_diameter, case.fall_speed; color = case.color)
    scatter!(ax25, [first(case.mean_diameter)], [first(case.fall_speed)];
             color = case.color, marker = :circle, markersize = 10)
    scatter!(ax25, [last(case.mean_diameter)], [last(case.fall_speed)];
             color = case.color, marker = :utriangle, markersize = 12)

    lines!(ax26, ascending_branch(case.Fᶠ, case.z ./ 1000)...; color = case.color)
end

ylims!(ax21, 0, plot_top / 1000)
ylims!(ax22, 0, plot_top / 1000)
ylims!(ax23, 0, plot_top / 1000)
ylims!(ax26, 0, plot_top / 1000)

axislegend(ax21; position = :lb, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax22; position = :lt, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax23; position = :rb, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax24; position = :lt, labelsize = 12, backgroundcolor = (:white, 0.8))

fig2

# ## Discussion
#
# - In the first section, the P3 and Kessler parcels start from the same
#   sounding, the same 0.5 K warm-bubble perturbation, and the same prognostic
#   updraft. Any separation aloft therefore comes from the microphysics.
# - Kessler can only move water between vapor, cloud, and rain. P3 begins from
#   the same warm-rain launch, but it can also create and evolve ice once the
#   parcel enters the mixed-phase part of the sounding.
# - In the second section, all three P3 parcels share the same temperature
#   perturbation, the same total water, and the same seeded ice. Changing only
#   the liquid partition pushes that same ice population along different
#   continuous trajectories in ``(Fᶠ, ρᶠ)`` and ``(D, V)`` space.
# - That is the main P3 idea: there is no handoff between cloud ice, snow,
#   graupel, and hail categories. One prognostic ice species changes its bulk
#   properties continuously as the parcel experiences deposition, riming, and
#   melting.
