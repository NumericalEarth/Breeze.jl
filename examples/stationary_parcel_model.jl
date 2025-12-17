# # Stationary parcel model
#
# This example demonstrates non-equilibrium cloud microphysics in a stationary
# parcel framework. We explore how vapor, cloud liquid, and rain evolve
# under different initial conditions, illustrating the key microphysical processes:
#
# - **Condensation**: Supersaturated vapor → cloud liquid (timescale τ ≈ 10 s)
# - **Autoconversion**: Cloud liquid → rain (timescale τ ≈ 1000 s)
# - **Rain evaporation**: Subsaturated rain → vapor
#
# Stationary parcel models are classic tools in cloud physics, isolating microphysics
# from dynamics. See Rogers & Yau (1989) "A Short Course in Cloud Physics".

using Breeze
using CloudMicrophysics
using CairoMakie

# ## Model setup

grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                       topology=(Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics
microphysics = OneMomentCloudMicrophysics()

τ = microphysics.cloud_formation.liquid.τ_relax  # Condensation timescale (~10 s)

# ## Simulation helper

function run_parcel_simulation(; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0, stop_time=10, Δt=0.1)
    model = AtmosphereModel(grid; formulation, thermodynamic_constants=constants, microphysics)
    set!(model; θ, qᵗ, qᶜˡ, qʳ)
    simulation = Simulation(model; Δt, stop_time)
    
    t, qᵛ, qᶜˡ, qʳ, T = Float64[], Float64[], Float64[], Float64[], Float64[]
    
    function record_time_series(sim)
        push!(t, time(sim))
        push!(qᵛ, first(sim.model.microphysical_fields.qᵛ))
        push!(qᶜˡ, first(sim.model.microphysical_fields.qᶜˡ))
        push!(qʳ, first(sim.model.microphysical_fields.qʳ))
        push!(T, first(sim.model.temperature))
    end
    
    add_callback!(simulation, record_time_series)
    run!(simulation)
    
    return (; t, qᵛ, qᶜˡ, qʳ, T)
end

# ## Four cases illustrating different regimes
#
# We run four simulations with different initial conditions to explore
# the full spectrum of microphysical behavior:

case1 = run_parcel_simulation(qᵗ=0.025, stop_time=5τ)                  # Supersaturated
case2 = run_parcel_simulation(qᵗ=0.030, stop_time=5τ)                  # Higher moisture
case3 = run_parcel_simulation(qᵗ=0.015, qᶜˡ=0.005, qʳ=0.002, stop_time=5τ)        # Subsaturated with rain
case4 = run_parcel_simulation(qᵗ=0.025, qʳ=0.001, stop_time=5τ)        # Supersaturated with rain
nothing #hide

# ## Visualization
#
# We plot the *change* in moisture mass fractions from initial conditions,
# keeping units consistent (no conversions).

fig = Figure(size=(900, 650), fontsize=13)

norm(t) = t ./ τ  # Normalize time by condensation timescale

c_vapor = :steelblue
c_cloud = :seagreen  
c_rain = :indianred
c_temp = :darkorange

# Helper to compute deviation from initial value
Δ(x) = x .- x[1]

# --- Row 1: Condensation ---
ax1a = Axis(fig[1,1]; title="(a) Condensation", ylabel="Δq", xticklabelsvisible=false)
lines!(ax1a, norm(case1.t), Δ(case1.qᵛ); color=c_vapor, linewidth=2, label="Δqᵛ")
lines!(ax1a, norm(case1.t), Δ(case1.qᶜˡ); color=c_cloud, linewidth=2, label="Δqᶜˡ")
lines!(ax1a, norm(case1.t), Δ(case1.qʳ); color=c_rain, linewidth=2, label="Δqʳ")
axislegend(ax1a; position=:rt, framevisible=false)

ax1b = Axis(fig[1,2]; title="(a) Temperature", ylabel="T (K)", xticklabelsvisible=false)
lines!(ax1b, norm(case1.t), case1.T; color=c_temp, linewidth=2)

# --- Row 2: Precipitation ---
ax2a = Axis(fig[2,1]; title="(b) High moisture", ylabel="Δq", xticklabelsvisible=false)
lines!(ax2a, norm(case2.t), Δ(case2.qᵛ); color=c_vapor, linewidth=2)
lines!(ax2a, norm(case2.t), Δ(case2.qᶜˡ); color=c_cloud, linewidth=2)
lines!(ax2a, norm(case2.t), Δ(case2.qʳ); color=c_rain, linewidth=2)

ax2b = Axis(fig[2,2]; title="(b) Temperature", ylabel="T (K)", xticklabelsvisible=false)
lines!(ax2b, norm(case2.t), case2.T; color=c_temp, linewidth=2)

# --- Row 3: Rain evaporation ---
ax3a = Axis(fig[3,1]; title="(c) Cloud + rain evaporation", ylabel="Δq", xticklabelsvisible=false)
lines!(ax3a, norm(case3.t), Δ(case3.qᵛ); color=c_vapor, linewidth=2)
lines!(ax3a, norm(case3.t), Δ(case3.qᶜˡ); color=c_cloud, linewidth=2)
lines!(ax3a, norm(case3.t), Δ(case3.qʳ); color=c_rain, linewidth=2)

ax3b = Axis(fig[3,2]; title="(c) Temperature", ylabel="T (K)", xticklabelsvisible=false)
lines!(ax3b, norm(case3.t), case3.T; color=c_temp, linewidth=2)

# --- Row 4: Mixed ---
ax4a = Axis(fig[4,1]; title="(d) Condensation + rain evaporation", 
            xlabel="t / τ", ylabel="Δq")
lines!(ax4a, norm(case4.t), Δ(case4.qᵛ); color=c_vapor, linewidth=2)
lines!(ax4a, norm(case4.t), Δ(case4.qᶜˡ); color=c_cloud, linewidth=2)
lines!(ax4a, norm(case4.t), Δ(case4.qʳ); color=c_rain, linewidth=2)

ax4b = Axis(fig[4,2]; title="(d) Temperature", xlabel="t / τ", ylabel="T (K)")
lines!(ax4b, norm(case4.t), case4.T; color=c_temp, linewidth=2)

fig

# ## Discussion
#
# - **(a) Condensation**: Supersaturated vapor condenses to cloud liquid,
#   releasing latent heat and warming the parcel. Equilibrium is reached in ~5τ.
#
# - **(b) High moisture**: Higher initial moisture creates more cloud liquid.
#   Autoconversion to rain is slow (τ_acnv ≈ 100τ) so rain remains negligible
#   on these short timescales.
#
# - **(c) Rain evaporation**: Subsaturated air with pre-existing rain.
#   Rain evaporates, cooling the parcel via latent heat absorption.
#
# - **(d) Mixed**: Supersaturated with rain. Cloud forms via condensation while
#   rain simultaneously evaporates. The net temperature change depends on the
#   balance between latent heat release (condensation) and absorption (evaporation).
