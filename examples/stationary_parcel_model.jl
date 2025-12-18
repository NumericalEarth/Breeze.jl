# # One-moment microphysics in a stationary parcel model
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
#
# A Lagrangian parcel is a closed system - rain doesn't "fall out" because
# the parcel moves with the hydrometeors. We use `ImpenetrableBottom()` to
# ensure moisture is conserved within the parcel.

grid = RectilinearGrid(CPU(); size=(1, 1, 1), x=(0, 1), y=(0, 1), z=(0, 1),
                       topology=(Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants; surface_pressure=101325, potential_temperature=300)
formulation = AnelasticFormulation(reference_state; thermodynamics=:LiquidIcePotentialTemperature)

BreezeCloudMicrophysicsExt = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
OneMomentCloudMicrophysics = BreezeCloudMicrophysicsExt.OneMomentCloudMicrophysics

# ImpenetrableBottom ensures rain collects in the parcel rather than exiting
microphysics = OneMomentCloudMicrophysics(precipitation_boundary_condition = ImpenetrableBottom())

τ = microphysics.cloud_formation.liquid.τ_relax  # Condensation timescale (~10 s)

# ## Simulation helper

function run_parcel_simulation(; θ=300, qᵗ=0.020, qᶜˡ=0, qʳ=0, stop_time=5τ, Δt=0.1)
    model = AtmosphereModel(grid; formulation, thermodynamic_constants=constants, microphysics)
    set!(model; θ, qᵗ, qᶜˡ, qʳ)
    simulation = Simulation(model; Δt, stop_time, verbose=false)
    
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

# ## Five cases illustrating different regimes
#
# We run five simulations with different initial conditions to explore
# the full spectrum of microphysical behavior:

case1 = run_parcel_simulation(qᵗ=0.025)                      # Supersaturated
case2 = run_parcel_simulation(qᵗ=0.030)                      # Higher moisture
case3 = run_parcel_simulation(qᵗ=0.015, qʳ=0.002)            # Subsaturated with rain
case4 = run_parcel_simulation(qᵗ=0.025, qʳ=0.001)            # Supersaturated with rain
case5 = run_parcel_simulation(qᵗ=0.030, stop_time=500τ)      # Long run for autoconversion
nothing #hide

# ## Visualization
#
# We plot the *change* in moisture mass fractions from initial conditions,
# keeping units consistent (no conversions).

fig = Figure(size=(900, 900), fontsize=16)

norm(t) = t ./ τ  # Normalize time by condensation timescale

# Bright, colorblind-friendly colors (Wong palette + vibrant choices)
c_vapor = :dodgerblue      # Bright blue
c_cloud = :lime            # Vivid green  
c_rain = :orangered        # Bright orange-red
c_temp = :magenta          # Vibrant magenta

Δ(x) = x .- x[1]  # Deviation from initial value

function plot_case!(fig, row, case, description; show_xlabel=false, xlims=(0, 20))
    # Label spanning both columns
    Label(fig[row, 1:2], description; fontsize=17, halign=:left, padding=(10, 0, 0, 0))
    
    # Moisture panel
    ax_q = Axis(fig[row+1, 1]; ylabel="Δq", limits=(xlims, nothing),
                xticklabelsvisible=show_xlabel, xlabel=show_xlabel ? "t / τ" : "")
    lines!(ax_q, norm(case.t), Δ(case.qᵛ);  color=c_vapor, linewidth=2.5, label="Δqᵛ")
    lines!(ax_q, norm(case.t), Δ(case.qᶜˡ); color=c_cloud, linewidth=2.5, label="Δqᶜˡ")
    lines!(ax_q, norm(case.t), Δ(case.qʳ);  color=c_rain,  linewidth=2.5, label="Δqʳ")
    
    # Temperature panel
    ax_T = Axis(fig[row+1, 2]; ylabel="T (K)", limits=(xlims, nothing),
                xticklabelsvisible=show_xlabel, xlabel=show_xlabel ? "t / τ" : "")
    lines!(ax_T, norm(case.t), case.T; color=c_temp, linewidth=2.5)
    
    return ax_q, ax_T
end

# Plot first 4 cases with xlims=(0, 20) for rapid evolution
ax1, _ = plot_case!(fig, 1, case1, "(a) Condensation: supersaturated vapor → cloud")
plot_case!(fig, 3, case2, "(b) High moisture: more condensation")
plot_case!(fig, 5, case3, "(c) Evaporation: subsaturated with pre-existing rain")
plot_case!(fig, 7, case4, "(d) Mixed: condensation + rain evaporation")

# Plot last case with xlims=(0, 500) for slow autoconversion
plot_case!(fig, 9, case5, "(e) Autoconversion: cloud liquid → rain (500τ run)";
           show_xlabel=true, xlims=(0, 500))

# Legend outside the figure
Legend(fig[0, 1:2], ax1; orientation=:horizontal, framevisible=false)

# Adjust row heights for labels vs axes
for i in 1:2:9
    rowsize!(fig.layout, i, Relative(0.02))
end

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
#
# - **(e) Autoconversion**: Running 500× longer reveals rain formation via
#   autoconversion. Cloud liquid slowly converts to rain on timescales of ~100τ.
#   Temperature remains nearly constant since autoconversion involves no phase change
#   (just redistribution of liquid water between cloud and rain categories).
