# # P3 parcel explorer in a deep-convection sounding
#
# This example tells the P3 parcel story in two steps, both using the
# idealized supercell sounding of [Klemp et al. (2015)](@cite KlempEtAl2015)
# and a parcel model rising at a fixed updraft of 1 m/s.
#
# 1. **P3 fidelity check**: identical parcels are launched with either P3 or
#    Kessler microphysics. The parcel starts from the same sounding with the
#    same initial moisture and ascends at the same prescribed 1 m/s. This lets
#    us compare temperature and hydrometeor partition without confounding
#    differences in updraft strength.
# 2. **P3 feature exploration**: the parcel now launches from ~6 km, where the
#    sounding is already subfreezing, and is seeded with the same ice population
#    but different cloud/rain partitions. This isolates the P3 idea that one ice
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
using SpecialFunctions: gamma

using Breeze: DCMIP2016KesslerMicrophysics,
              TetensFormula, ThermodynamicConstants
using Breeze.Thermodynamics: hydrostatic_density, hydrostatic_temperature

const AtmosphereModels = Breeze.AtmosphereModels
const PredictedParticleProperties = Breeze.Microphysics.PredictedParticleProperties
const Thermodynamics = Breeze.Thermodynamics

# ## Idealized supercell sounding
#
# We borrow the background potential-temperature and relative-humidity profiles
# from the splitting-supercell example. The parcel launches from the prescribed
# height and rises at a fixed 1 m/s updraft, so any thermodynamic evolution
# along the trajectory is set entirely by the microphysics.

grid = RectilinearGrid(size = 15000, z = (0, 15kilometers), topology = (Flat, Flat, Bounded))

constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula())

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = 100000,
                                 potential_temperature = 300)

θ₀ = 300
θᵖ = 343
zᵖ = 12kilometers
Tᵖ = 213
qᵛ_max = 0.014 # kg/kg — well-mixed boundary-layer cap from Klemp et al. (2015)

g = constants.gravitational_acceleration
cᵖᵈ = constants.dry_air.heat_capacity

function θ_background(z)
    θᵗ = θ₀ + (θᵖ - θ₀) * (z / zᵖ)^(5/4)
    θˢ = θᵖ * exp(g / (cᵖᵈ * Tᵖ) * (z - zᵖ))
    return (z ≤ zᵖ) * θᵗ + (z > zᵖ) * θˢ
end

function qᵛ_background(z)
    ℋ = (1 - 3/4 * (z / zᵖ)^(5/4)) * (z ≤ zᵖ) + 1/4 * (z > zᵖ)
    p₀ = reference_state.surface_pressure
    pˢᵗ = reference_state.standard_pressure
    T = hydrostatic_temperature(z, p₀, θ_background, pˢᵗ, constants)
    ρ = hydrostatic_density(z, p₀, θ_background, pˢᵗ, constants)
    qᵛ⁺ = saturation_specific_humidity(T, ρ, constants, PlanarLiquidSurface())
    return min(ℋ * qᵛ⁺, qᵛ_max)
end

launch_height = 1500
fixed_updraft = 1

plot_top = 10kilometers
stop_time = 180minutes
Δt = 1
record_interval = 1

## Section 2 parameters — launch from ~6 km where T ≈ 256 K
cold_launch_height = 6000
cold_seeded_ice_mass = 1e-4
cold_seeded_ice_number = 1e5
cold_cloud_partition = 2e-3
cold_rain_partition = 1e-3
cold_rain_number_partition = 1e4

height_profile = collect(range(0, plot_top, length = 400))
background_T_profile = [Thermodynamics.temperature_from_potential_temperature(θ_background(z),
                                                                              interpolate(z, reference_state.pressure),
                                                                              reference_state.standard_pressure,
                                                                              constants)
                        for z in height_profile]
nothing #hide

# ## Helper functions
#
# The helpers below set up the parcel launch, seed optional microphysics
# partitions, and convert the P3 moments into the continuous ice diagnostics
# that we want to visualize.

function p3_ice_diagnostics(p3, ρ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ)
    FT = typeof(ρ)

    if qⁱ <= p3.minimum_mass_mixing_ratio || nⁱ <= p3.minimum_number_mixing_ratio
        return (; Fᶠ = zero(FT), ρᶠ = zero(FT), Fˡ = zero(FT), reflectivity = zero(FT))
    end

    rime_state = PredictedParticleProperties.consistent_rime_state(p3, qⁱ, qᶠ, bᶠ, FT(0))
    Fᶠ = rime_state.Fᶠ
    ρᶠ = rime_state.ρᶠ
    Fˡ = qʷⁱ > 0 ? qʷⁱ / (qⁱ + qʷⁱ) : zero(FT)

    ## The lambda solver needs a nonzero rime density even when rime fraction is
    ## zero; use the IceSizeDistributionState default (400 kg/m³) in that case.
    ρᶠ_for_psd = ρᶠ > 0 ? ρᶠ : FT(400)

    params = PredictedParticleProperties.distribution_parameters(ρ * qⁱ, ρ * nⁱ, Fᶠ, ρᶠ_for_psd)

    ## Sixth moment of the gamma PSD: Z = N₀ Γ(7+μ) / λ^(7+μ).
    ## Convert from per-volume (m⁶/m³) to per-mass (m⁶/kg); P3 stores
    ## the dynamics variable z̃ⁱ = √(zⁱ nⁱ) in the prognostic ρz̃ⁱ field.
    Z_per_volume = params.N₀ * gamma(7 + params.μ) / params.λ^(7 + params.μ)

    return (; Fᶠ, ρᶠ, Fˡ, reflectivity = Z_per_volume / ρ)
end

function initialize_p3_state(p3, ρ; qᶜˡ = 0, nᶜˡ = 0, qʳ = 0, nʳ = 0,
                               qⁱ = 0, nⁱ = 0, qᶠ = 0, bᶠ = 0, qʷⁱ = 0)
    base = (; ρqᶜˡ = ρ * qᶜˡ,
              ρnᶜˡ = ρ * nᶜˡ,
              ρqʳ = ρ * qʳ,
              ρnʳ = ρ * nʳ,
              ρqⁱ = ρ * qⁱ,
              ρnⁱ = ρ * nⁱ,
              ρqᶠ = ρ * qᶠ,
              ρbᶠ = ρ * bᶠ,
              ρqʷⁱ = ρ * qʷⁱ,
              ρsˢᵃᵗ = zero(ρ))

    PredictedParticleProperties.is_three_moment_ice(p3) || return base

    reflectivity = p3_ice_diagnostics(p3, ρ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ).reflectivity
    z̃ⁱ = sqrt(max(0, reflectivity * nⁱ))
    return merge(base, (; ρz̃ⁱ = ρ * z̃ⁱ))
end

supercell_parcel_model(microphysics) = AtmosphereModel(grid;
                                                       dynamics = ParcelDynamics(),
                                                       microphysics = microphysics,
                                                       thermodynamic_constants = constants)

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

function initialize_supercell_parcel!(model; z = launch_height)
    set!(model,
         θ = θ_background,
         qᵗ = qᵛ_background,
         p = reference_state.pressure,
         ρ = reference_state.density,
         w = fixed_updraft,
         z = z)

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

function run_p3_case(; label, color, qᶜˡ = 0, qʳ = 0, nʳ = 0, qⁱ = 0, nⁱ = 0,
                       launch_z = launch_height)
    microphysics = PredictedParticlePropertiesMicrophysics()
    model = supercell_parcel_model(microphysics)

    initialize_supercell_parcel!(model; z = launch_z)

    if qᶜˡ > 0 || qʳ > 0 || nʳ > 0 || qⁱ > 0 || nⁱ > 0
        seed_p3_parcel!(model, microphysics; qᶜˡ, qʳ, nʳ, qⁱ, nⁱ)
    end

    simulation = Simulation(model; Δt, stop_time, verbose = false)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)
    stop_at_plot_top!(simulation)

    z = Float64[]
    T = Float64[]
    qᶜˡ_ts = Float64[]
    qʳ_ts = Float64[]
    qⁱ_ts = Float64[]
    Fᶠ_ts = Float64[]
    ρᶠ_ts = Float64[]

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
        push!(T, temperature(state.𝒰, sim.model.thermodynamic_constants))
        push!(qᶜˡ_ts, μ.ρqᶜˡ / state.ρ)
        push!(qʳ_ts, μ.ρqʳ / state.ρ)
        push!(qⁱ_ts, qⁱₙ)
        push!(Fᶠ_ts, diagnostics.Fᶠ)
        push!(ρᶠ_ts, diagnostics.ρᶠ)

        return nothing
    end

    record_state(simulation)
    add_callback!(simulation, record_state, IterationInterval(record_interval))
    run!(simulation)

    return (; label,
              color,
              z,
              T,
              qᶜˡ = qᶜˡ_ts,
              qʳ = qʳ_ts,
              qⁱ = qⁱ_ts,
              Fᶠ = Fᶠ_ts,
              ρᶠ = ρᶠ_ts)
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
    T = Float64[]
    qᶜˡ_ts = Float64[]
    qʳ_ts = Float64[]

    function record_state(sim)
        state = sim.model.dynamics.state
        μ = state.μ

        push!(z, state.z)
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
              T,
              qᶜˡ = qᶜˡ_ts,
              qʳ = qʳ_ts)
end

# ## Section 1: P3 vs. Kessler parcels
#
# Both parcels are launched from the same sounding, ascend at the same fixed
# 1 m/s, and start with no seeded condensate. Differences aloft therefore come
# from the microphysics, not the initialization.

p3_reference = run_p3_case(label = "P3", color = :dodgerblue)
kessler_reference = run_kessler_case(label = "Kessler", color = :black)
nothing #hide

set_theme!(fontsize = 16, linewidth = 2.5)

fig1 = Figure(size = (900, 450))

ax11 = Axis(fig1[1, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Same launch, different microphysics")

ax12 = Axis(fig1[1, 2];
    xlabel = "Mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Hydrometeor partition")

lines!(ax11, background_T_profile, height_profile ./ 1000;
       color = :gray40, linestyle = :dot, label = "Environment")
lines!(ax11, ascending_branch(p3_reference.T, p3_reference.z ./ 1000)...;
       color = :magenta, label = p3_reference.label)
lines!(ax11, ascending_branch(kessler_reference.T, kessler_reference.z ./ 1000)...;
       color = :black, linestyle = :dash, label = kessler_reference.label)

lines!(ax12, ascending_branch(p3_reference.qᶜˡ, p3_reference.z ./ 1000)...;
       color = :lime, label = "P3 qᶜˡ")
lines!(ax12, ascending_branch(p3_reference.qʳ, p3_reference.z ./ 1000)...;
       color = :orangered, label = "P3 qʳ")
lines!(ax12, ascending_branch(p3_reference.qⁱ, p3_reference.z ./ 1000)...;
       color = :dodgerblue, label = "P3 qⁱ")

lines!(ax12, ascending_branch(kessler_reference.qᶜˡ, kessler_reference.z ./ 1000)...;
       color = :lime, linestyle = :dash, label = "Kessler qᶜˡ")
lines!(ax12, ascending_branch(kessler_reference.qʳ, kessler_reference.z ./ 1000)...;
       color = :orangered, linestyle = :dash, label = "Kessler qʳ")

ylims!(ax11, 0, plot_top / 1000)
ylims!(ax12, 0, plot_top / 1000)

axislegend(ax11; position = :lb, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax12; position = :rb, labelsize = 11, nbanks = 2, backgroundcolor = (:white, 0.8))

fig1

# ## Section 2: the P3 ice continuum from a cold launch
#
# Now the parcel launches from ~6 km where the sounding temperature is already
# about 256 K — well below freezing. Ice seeded at this altitude persists and
# grows by vapor deposition, while any supercooled liquid enables riming.
# All three cases share the same seeded ice population; only the liquid
# partition differs, pushing the ice along different continuous trajectories
# in (Fᶠ, ρᶠ) space and (D, V) space.

p3_feature_cases = [
    run_p3_case(;
        label = "Deposition only",
        color = :dodgerblue,
        qⁱ = cold_seeded_ice_mass,
        nⁱ = cold_seeded_ice_number,
        launch_z = cold_launch_height),

    run_p3_case(;
        label = "Cloud riming",
        color = :lime,
        qᶜˡ = cold_cloud_partition,
        qⁱ = cold_seeded_ice_mass,
        nⁱ = cold_seeded_ice_number,
        launch_z = cold_launch_height),

    run_p3_case(;
        label = "Cloud + rain riming",
        color = :orangered,
        qᶜˡ = cold_cloud_partition,
        qʳ = cold_rain_partition,
        nʳ = cold_rain_number_partition,
        qⁱ = cold_seeded_ice_mass,
        nⁱ = cold_seeded_ice_number,
        launch_z = cold_launch_height),
]
nothing #hide

fig2 = Figure(size = (1250, 950))

ax21 = Axis(fig2[1, 1];
    xlabel = "Temperature (K)",
    ylabel = "Height (km)",
    title = "Cold launch, different liquid partitions")

ax22 = Axis(fig2[1, 2];
    xlabel = "Ice mixing ratio (kg/kg)",
    ylabel = "Height (km)",
    title = "Liquid availability changes ice growth")

ax23 = Axis(fig2[1, 3];
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Height (km)",
    title = "Riming turns on smoothly")

ax24 = Axis(fig2[2, 1];
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Rime density ρᶠ [kg/m³]",
    title = "P3 moves through a continuum")

lines!(ax21, background_T_profile, height_profile ./ 1000;
       color = :gray40, linestyle = :dot, label = "Environment")

for case in p3_feature_cases
    lines!(ax21, ascending_branch(case.T, case.z ./ 1000)...; color = case.color, label = case.label)
    lines!(ax22, ascending_branch(case.qⁱ, case.z ./ 1000)...; color = case.color, label = case.label)
    lines!(ax23, ascending_branch(case.Fᶠ, case.z ./ 1000)...; color = case.color)

    lines!(ax24, case.Fᶠ, case.ρᶠ; color = case.color, label = case.label)
    scatter!(ax24, [first(case.Fᶠ)], [first(case.ρᶠ)];
             color = case.color, marker = :circle, markersize = 10)
    scatter!(ax24, [last(case.Fᶠ)], [last(case.ρᶠ)];
             color = case.color, marker = :utriangle, markersize = 12)
end

ylims!(ax21, cold_launch_height / 1000, plot_top / 1000)
ylims!(ax22, cold_launch_height / 1000, plot_top / 1000)
ylims!(ax23, cold_launch_height / 1000, plot_top / 1000)

axislegend(ax21; position = :lb, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax22; position = :rb, labelsize = 12, backgroundcolor = (:white, 0.8))
axislegend(ax24; position = :lt, labelsize = 12, backgroundcolor = (:white, 0.8))

fig2

# ## Discussion
#
# - In the first section, the P3 and Kessler parcels start from the same
#   sounding and ascend at the same fixed 1 m/s updraft. Any separation aloft
#   therefore comes from the microphysics, not the thermodynamics of buoyancy.
# - Kessler can only move water between vapor, cloud, and rain. P3 begins from
#   the same warm-rain launch, but it can also create and evolve ice once the
#   parcel enters the mixed-phase part of the sounding.
# - In the second section, all three P3 parcels launch from ~6 km where the
#   sounding is already subfreezing. They share the same seeded ice but differ
#   in their initial liquid partition. Changing only the liquid available for
#   riming pushes the ice along different continuous trajectories in
#   ``(Fᶠ, ρᶠ)`` and ``(D, V)`` space.
# - That is the main P3 idea: there is no handoff between cloud ice, snow,
#   graupel, and hail categories. One prognostic ice species changes its bulk
#   properties continuously as the parcel experiences deposition, riming, and
#   melting.
