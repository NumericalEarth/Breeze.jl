# # P3 parcel explorer: a continuum of ice properties
#
# This example uses three parcel model simulations to show the central P3
# idea: there is only **one** ice category, but its properties evolve
# continuously. Instead of converting
#
# ```text
# cloud ice → snow → graupel
# ```
#
# P3 predicts the **rime fraction** ``Fᶠ`` and **rime density** ``ρᶠ`` of the
# same ice population, so bulk quantities like fall speed and reflectivity move
# smoothly through property space [Morrison and Milbrandt (2015a)](@cite
# Morrison2015parameterization), [Milbrandt et al. (2021)](@cite
# MilbrandtEtAl2021).
#
# We start each parcel from the same seeded ice distribution and only change how
# much supercooled liquid is available:
#
# - **Vapor growth**: no cloud or rain, so the ice mainly grows by vapor deposition.
# - **Cloud riming**: supercooled cloud water is available for collection.
# - **Cloud + rain riming**: both cloud water and rain accelerate riming and densification.
#
# The point is not to create three different hydrometeor categories. The point
# is that the same P3 ice population slides continuously from lightly rimed to
# heavily rimed states as the forcing changes.

using Breeze
using CairoMakie
using Oceananigans

using Breeze.Microphysics.PredictedParticleProperties:
    IceSizeDistributionState,
    MeanDiameter,
    MassWeightedFallSpeed,
    Reflectivity,
    consistent_rime_state,
    distribution_parameters,
    evaluate

# ## Helper functions
#
# We run a stationary one-cell atmosphere model with P3 microphysics. The helper
# below converts the prognostic P3 moments ``qⁱ``, ``nⁱ``, ``qᶠ``, and ``ρbᶠ``
# into intuitive diagnostics: ``Fᶠ``, ``ρᶠ``, mean diameter, and mass-weighted
# fall speed.

function p3_ice_diagnostics(p3, ρ, qⁱ, nⁱ, qᶠ, bᶠ, qʷⁱ)
    FT = typeof(ρ)

    if qⁱ ≤ p3.minimum_mass_mixing_ratio || nⁱ ≤ p3.minimum_number_mixing_ratio
        return (; Fᶠ = zero(FT),
                  ρᶠ = p3.ice.minimum_rime_density,
                  Fˡ = zero(FT),
                  mean_diameter = zero(FT),
                  fall_speed = zero(FT),
                  reflectivity = zero(FT))
    end

    rime_state = consistent_rime_state(p3, qⁱ, qᶠ, bᶠ)
    Fᶠ = rime_state.Fᶠ
    ρᶠ = rime_state.ρᶠ
    Fˡ = qʷⁱ > 0 ? qʷⁱ / (qⁱ + qʷⁱ) : zero(FT)

    params = distribution_parameters(ρ * qⁱ, ρ * nⁱ, Fᶠ, ρᶠ)

    state = IceSizeDistributionState(FT;
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
              mean_diameter = evaluate(MeanDiameter(), state),
              fall_speed = evaluate(MassWeightedFallSpeed(), state),
              reflectivity = evaluate(Reflectivity(), state) / ρ)
end

function run_continuum_case(; label, color, qᵗ, qᶜˡ = 0, qʳ = 0, nʳ = 0,
                              stop_time = 30, Δt = 0.25, record_interval = 4)
    grid = RectilinearGrid(CPU();
        size = (1, 1, 1),
        extent = (1, 1, 1),
        topology = (Periodic, Periodic, Bounded))

    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants;
                                     surface_pressure = 70000,
                                     potential_temperature = 268)

    microphysics = PredictedParticlePropertiesMicrophysics(;
        precipitation_boundary_condition = ImpenetrableBoundaryCondition())

    model = AtmosphereModel(grid;
        dynamics = AnelasticDynamics(reference_state),
        thermodynamic_constants = constants,
        microphysics)

    ρ = first(model.dynamics.reference_state.density)

    ## All parcels start from the same seeded ice population.
    qⁱ₀ = 1.5e-4
    nⁱ₀ = 4e5

    ## The sixth moment is initialized consistently with qⁱ and nⁱ so the
    ## three-moment closure starts from a self-consistent PSD.
    initial = p3_ice_diagnostics(microphysics, ρ, qⁱ₀, nⁱ₀, 0.0, 0.0, 0.0)

    set!(model;
         θ = 268,
         qᵗ,
         qᶜˡ,
         qʳ,
         nʳ,
         qⁱ = qⁱ₀,
         nⁱ = nⁱ₀,
         ρzⁱ = ρ * initial.reflectivity)

    simulation = Simulation(model; Δt, stop_time, verbose = false)
    Oceananigans.Diagnostics.erroring_NaNChecker!(simulation)

    t = Float64[]
    Fᶠ = Float64[]
    ρᶠ = Float64[]
    mean_diameter = Float64[]
    fall_speed = Float64[]
    qⁱ = Float64[]
    qᶠ = Float64[]

    function record_state(sim)
        μ = sim.model.microphysical_fields

        qⁱₙ = first(μ.ρqⁱ) / ρ
        nⁱₙ = first(μ.ρnⁱ) / ρ
        qᶠₙ = first(μ.ρqᶠ) / ρ
        bᶠₙ = first(μ.ρbᶠ) / ρ
        qʷⁱₙ = first(μ.ρqʷⁱ) / ρ

        diagnostics = p3_ice_diagnostics(microphysics, ρ, qⁱₙ, nⁱₙ, qᶠₙ, bᶠₙ, qʷⁱₙ)

        push!(t, sim.model.clock.time)
        push!(Fᶠ, diagnostics.Fᶠ)
        push!(ρᶠ, diagnostics.ρᶠ)
        push!(mean_diameter, diagnostics.mean_diameter)
        push!(fall_speed, diagnostics.fall_speed)
        push!(qⁱ, qⁱₙ)
        push!(qᶠ, qᶠₙ)

        return nothing
    end

    record_state(simulation)
    add_callback!(simulation, record_state, IterationInterval(record_interval))
    run!(simulation)

    return (; label, color, t, Fᶠ, ρᶠ, mean_diameter, fall_speed, qⁱ, qᶠ)
end

# ## Three forcing regimes
#
# The only thing that changes between these parcels is the access to liquid
# water. P3 responds by moving the same ice population through a continuous
# range of ``Fᶠ`` and ``ρᶠ`` values.

cases = [
    run_continuum_case(;
        label = "Vapor growth",
        color = :dodgerblue,
        qᵗ = 0.0018),

    run_continuum_case(;
        label = "Cloud riming",
        color = :limegreen,
        qᵗ = 0.0040,
        qᶜˡ = 8e-4),

    run_continuum_case(;
        label = "Cloud + rain riming",
        color = :orangered,
        qᵗ = 0.0040,
        qᶜˡ = 8e-4,
        qʳ = 3e-4,
        nʳ = 1e5),
]
nothing #hide

# ## Visualizing the P3 continuum
#
# Circles mark the initial state, triangles the final state. There are no
# category jumps anywhere in these curves: each trajectory is a smooth walk
# through P3 property space.

set_theme!(fontsize = 16, linewidth = 2.5)

fig = Figure(size = (1100, 900))

ax1 = Axis(fig[1, 1];
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Rime density ρᶠ [kg/m³]",
    title = "Continuous evolution in predicted rime properties")

ax2 = Axis(fig[1, 2];
    xlabel = "Mean diameter [m]",
    ylabel = "Mass-weighted fall speed [m/s]",
    xscale = log10,
    yscale = log10,
    title = "Bulk fall speed changes continuously")

ax3 = Axis(fig[2, 1];
    xlabel = "t [s]",
    ylabel = "Rime fraction Fᶠ",
    title = "Same ice seed, different forcing")

ax4 = Axis(fig[2, 2];
    xlabel = "t [s]",
    ylabel = "Ice mass mixing ratio qⁱ [kg/kg]",
    title = "More liquid access grows more ice mass")

for case in cases
    lines!(ax1, case.Fᶠ, case.ρᶠ; color = case.color, label = case.label)
    scatter!(ax1, [first(case.Fᶠ)], [first(case.ρᶠ)];
             color = case.color, marker = :circle, markersize = 10)
    scatter!(ax1, [last(case.Fᶠ)], [last(case.ρᶠ)];
             color = case.color, marker = :utriangle, markersize = 12)

    lines!(ax2, case.mean_diameter, case.fall_speed; color = case.color)
    scatter!(ax2, [first(case.mean_diameter)], [first(case.fall_speed)];
             color = case.color, marker = :circle, markersize = 10)
    scatter!(ax2, [last(case.mean_diameter)], [last(case.fall_speed)];
             color = case.color, marker = :utriangle, markersize = 12)

    lines!(ax3, case.t, case.Fᶠ; color = case.color, label = case.label)
    lines!(ax4, case.t, case.qⁱ; color = case.color, label = case.label)
end

axislegend(ax1; position = :lt, labelsize = 12)
axislegend(ax4; position = :rb, labelsize = 12)

fig

# ## What to notice
#
# The three curves all start from the same seeded ice population, but they do
# not branch into separate species. Instead:
#
# - Vapor growth nudges the parcel toward a lightly rimed state.
# - Supercooled cloud pushes the same ice population toward larger ``Fᶠ`` and
#   larger fall speed.
# - Adding rain extends that trend even farther, with a denser and faster-falling
#   final state.
#
# That smooth behavior is the feature we wanted to expose: P3 represents ice as
# a **continuum of predicted properties**, not a hand-tuned ladder of discrete
# hydrometeor categories.

nothing #hide
