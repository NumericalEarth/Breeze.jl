# # Cloud microphysics in a stationary parcel model
#
# This example demonstrates non-equilibrium cloud microphysics in a stationary
# parcel framework. We explore how vapor, cloud liquid, and rain evolve
# under different initial conditions, illustrating the key microphysical processes:
#
# - **Condensation**: Supersaturated vapor → cloud liquid (timescale τ ≈ 10 s)
# - **Autoconversion**: Cloud liquid → rain (timescale τ ≈ 1000 s)
# - **Rain evaporation**: Subsaturated rain → vapor
#
# We compare **one-moment** (mass only) and **two-moment** (mass + number)
# microphysics schemes. For two-moment, we use the [SeifertBeheng2006](@citet)
# scheme, which derives process rates from the evolving particle size distribution.
# Tracking droplet number concentration enables realistic representation of
# aerosol-cloud interactions.
#
# Stationary parcel models are classic tools in cloud physics, isolating microphysics
# from dynamics; see [rogers1989short](@citet).

using Breeze
using CloudMicrophysics
using CairoMakie

# ## Model setup
#
# A Lagrangian parcel is a closed system - rain doesn't "fall out" because
# the parcel moves with the hydrometeors. We use an `ImpenetrableBoundaryCondition()`
# to ensure moisture is conserved within the parcel.

grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                       topology=(Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
dynamics = AnelasticDynamics(reference_state)

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics
TwoMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.TwoMomentCloudMicrophysics

# ## Unified simulation helper
#
# A single function runs parcel simulations with either microphysics scheme.
# The function dynamically tracks number concentrations when using 2M microphysics.

function run_parcel_simulation(; microphysics, θ = 300, stop_time = 2000, Δt = 1,
                                 qᵗ = 0.020, qᶜˡ = 0, qʳ = 0,
                                 nᶜˡ = 0, nʳ = 0)

    model = AtmosphereModel(grid; dynamics, thermodynamic_constants=constants, microphysics)
    is_two_moment = microphysics isa TwoMomentCloudMicrophysics

    if is_two_moment
        set!(model; θ, qᵗ, qᶜˡ, nᶜˡ, qʳ, nʳ)
    else
        set!(model; θ, qᵗ, qᶜˡ, qʳ)
    end

    simulation = Simulation(model; Δt, stop_time, verbose=false)

    ## Time series storage
    t = Float64[]
    qᵛ, qᶜˡ, qʳ = Float64[], Float64[], Float64[]
    nᶜˡ, nʳ = Float64[], Float64[]
    T = Float64[]

    function record_time_series(sim)
        μ = sim.model.microphysical_fields
        push!(t, time(sim))
        push!(qᵛ, first(μ.qᵛ))
        push!(qᶜˡ, first(μ.qᶜˡ))
        push!(qʳ, first(μ.qʳ))
        push!(T, first(sim.model.temperature))
        if is_two_moment
            push!(nᶜˡ, first(μ.nᶜˡ))
            push!(nʳ, first(μ.nʳ))
        end
    end

    add_callback!(simulation, record_time_series)
    run!(simulation)

    if is_two_moment
        return (; t, qᵛ, qᶜˡ, qʳ, nᶜˡ, nʳ, T)
    else
        return (; t, qᵛ, qᶜˡ, qʳ, T)
    end
end
nothing #hide

# # Comparing one-moment and two-moment microphysics
#
# We run four cases to demonstrate the key differences between schemes:
# - **1M cases**: Show effect of varying condensation timescale
# - **2M cases**: Show effect of varying initial droplet number
#
# These comparisons illustrate the fundamental difference: one-moment schemes
# use prescribed timescales, while two-moment schemes derive rates from
# the evolving size distribution.

# ## Define microphysics schemes with different parameters

import CloudMicrophysics.Parameters as CMP
one_moment_cloud_microphysics_categories = BreezeCloudMicrophysicsExt.one_moment_cloud_microphysics_categories
precipitation_boundary_condition = ImpenetrableBoundaryCondition()

## First, a slow scheme
cloud_liquid_slow = CMP.CloudLiquid{Float64}(τ_relax=20.0, ρw=1000.0, r_eff=10e-6)
categories = one_moment_cloud_microphysics_categories(cloud_liquid = cloud_liquid_slow)
microphysics_1m_slow = OneMomentCloudMicrophysics(; categories, precipitation_boundary_condition)

# Then a fast scheme
cloud_liquid_fast = CMP.CloudLiquid{Float64}(τ_relax=2.0, ρw=1000.0, r_eff=10e-6)
categories = one_moment_cloud_microphysics_categories(cloud_liquid = cloud_liquid_fast)
microphysics_1m_fast = OneMomentCloudMicrophysics(; categories, precipitation_boundary_condition)

# And now a default two-moment scheme using Seifert and Beheng (2006)
microphysics_2m = TwoMomentCloudMicrophysics(; precipitation_boundary_condition)

# ## Run four comparison cases
#
# All cases start with the same supersaturated conditions (qᵗ = 0.030).

## One-moment cases: varying condensation timescale
stop_time = 1000
case_1m_slow = run_parcel_simulation(; microphysics = microphysics_1m_slow, qᵗ = 0.030, stop_time)
case_1m_fast = run_parcel_simulation(; microphysics = microphysics_1m_fast, qᵗ = 0.030, stop_time)

## Two-moment cases: varying initial droplet number
stop_time = 4000
case_2m_few  = run_parcel_simulation(; microphysics = microphysics_2m, qᵗ = 0.030, nᶜˡ = 100e6, stop_time)
case_2m_many = run_parcel_simulation(; microphysics = microphysics_2m, qᵗ = 0.030, nᶜˡ = 300e6, stop_time)
nothing #hide

# ## Visualization
#
# We compare all four cases side-by-side to highlight the key differences.

## Colorblind-friendly colors
c_cloud = :limegreen
c_rain  = :orangered
c_cloud_n = :purple
c_rain_n = :gold

fig = Figure(size=(1000, 700))
set_theme!(linewidth=2.5, fontsize=16)

## Row 1: One-moment comparison
Label(fig[1, 1:2], "One-moment microphysics: effect of condensation timescale")
ax1_q = Axis(fig[2, 1]; xlabel="t (s)", ylabel="q (kg/kg)", title="Mass mixing ratios")

t = case_1m_slow.t
lines!(ax1_q, t, case_1m_slow.qᶜˡ; color=c_cloud, label="qᶜˡ (τ = 10 s)")
lines!(ax1_q, t, case_1m_slow.qʳ; color=c_rain, label="qʳ (τ = 10 s)")
lines!(ax1_q, t, case_1m_fast.qᶜˡ; color=c_cloud, linestyle=:dash, label="qᶜˡ (τ = 2 s)")
lines!(ax1_q, t, case_1m_fast.qʳ; color=c_rain, linestyle=:dash, label="qʳ (τ = 2 s)")
axislegend(ax1_q; position=:rc, labelsize=11)

ax1_T = Axis(fig[2, 2]; xlabel="t (s)", ylabel="T (K)", title="Temperature")
lines!(ax1_T, t, case_1m_slow.T; color=:magenta, label="τ = 10 s")
lines!(ax1_T, t, case_1m_fast.T; color=:magenta, linestyle=:dash, label="τ = 2 s")
axislegend(ax1_T; position=:rb, labelsize=11)
xlims!(ax1_T, 0, 200)

## Row 2: Two-moment comparison
Label(fig[3, 1:2], "Two-moment microphysics: effect of initial droplet number")
ax2_q = Axis(fig[4, 1]; xlabel="t (s)", ylabel="q (kg/kg)", title="Mass mixing ratios")
t = case_2m_few.t
lines!(ax2_q, t, case_2m_few.qᶜˡ; color=c_cloud, label="qᶜˡ (nᶜˡ₀ = 100/mg)")
lines!(ax2_q, t, case_2m_few.qʳ; color=c_rain, label="qʳ (nᶜˡ₀ = 100/mg)")
lines!(ax2_q, t, case_2m_many.qᶜˡ; color=c_cloud, linestyle=:dash, label="qᶜˡ (nᶜˡ₀ = 300/mg)")
lines!(ax2_q, t, case_2m_many.qʳ; color=c_rain, linestyle=:dash, label="qʳ (nᶜˡ₀ = 300/mg)")
axislegend(ax2_q; position=:rc, labelsize=11)

ax2_n = Axis(fig[4, 2]; xlabel="t (s)", ylabel="n (1/kg)", title="Number concentrations")
lines!(ax2_n, t, case_2m_few.nᶜˡ; color=c_cloud_n, label="nᶜˡ (nᶜˡ₀ = 100/mg)")
lines!(ax2_n, t, case_2m_few.nʳ .* 1e6; color=c_rain_n, label="nʳ × 10⁶ (nᶜˡ₀ = 100/mg)")
lines!(ax2_n, t, case_2m_many.nᶜˡ; color=c_cloud_n, linestyle=:dash, label="nᶜˡ (nᶜˡ₀ = 300/mg)")
lines!(ax2_n, t, case_2m_many.nʳ .* 1e6; color=c_rain_n, linestyle=:dash, label="nʳ × 10⁶ (nᶜˡ₀ = 300/mg)")
axislegend(ax2_n; position=:rt, labelsize=11)

rowsize!(fig.layout, 1, Relative(0.05))
rowsize!(fig.layout, 3, Relative(0.05))

fig

# ## Discussion
#
# ### One-moment microphysics (top row)
#
# The condensation timescale τ controls how quickly supersaturated vapor
# converts to cloud liquid. With τ = 2 s (dashed), condensation happens
# 5× faster than with τ = 10 s (solid). However, once equilibrium is reached,
# both cases produce identical cloud liquid amounts and rain evolution.
# This illustrates a key limitation: **1M schemes prescribe process rates
# rather than deriving them from the microphysical state**.
#
# ### Two-moment microphysics (bottom row)
#
# Initial droplet number dramatically affects precipitation timing:
#
# - **Fewer droplets (100/mg)**: The same cloud water is distributed among
#   fewer, larger droplets. These large drops collide more efficiently,
#   accelerating autoconversion → rain forms faster.
#
# - **More droplets (300/mg)**: Cloud water is spread across many small
#   droplets. Small drops have low collision efficiency → rain is suppressed.
#   This is the "cloud lifetime effect" central to aerosol-cloud interactions.
#
# The number concentration panel reveals additional physics:
# - **Self-collection** reduces nᶜˡ as droplets merge
# - **Autoconversion** creates new rain drops (nʳ increases)
# - The ratio q/n determines mean particle size
#
# This sensitivity to droplet number is why **two-moment schemes are essential
# for studying aerosol effects on precipitation**. More aerosols → more CCN →
# more cloud droplets → smaller drops → less rain (the Twomey effect).
