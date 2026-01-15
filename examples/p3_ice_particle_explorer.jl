# # P3 Ice Particle Physics Explorer
#
# This example explores the rich physics of ice particles using the P3
# (Predicted Particle Properties) scheme. Unlike traditional microphysics that
# categorizes ice into fixed species (cloud ice, snow, graupel), P3 treats ice
# as a **continuum** with properties that evolve based on the particle's history.
#
# We'll visualize how key ice integrals behave across the P3 parameter space:
# - **Slope parameter λ**: Controls mean particle size (small λ = large particles)
# - **Shape parameter μ**: Controls breadth of the size distribution
# - **Rime fraction Fᶠ**: Fraction of ice mass that is rimed (0 = pristine, 1 = graupel)
# - **Liquid fraction Fˡ**: Fraction of total mass that is liquid water on ice
#
# This exploration builds intuition for how P3 represents the full spectrum
# from delicate dendritic snowflakes to dense graupel hailstones.
#
# Reference: [MilbrandtMorrison2016](@citet) for the 3-moment P3 formulation.

using Breeze.Microphysics.PredictedParticleProperties
using CairoMakie

# ## The P3 Ice Size Distribution
#
# P3 uses a gamma size distribution for ice particles:
#
# ```math
# N'(D) = N_0 D^\mu \exp(-\lambda D)
# ```
#
# where:
# - N₀ is the intercept parameter
# - μ (mu) is the shape parameter (0 = exponential, larger = peaked)
# - λ (lambda) is the slope parameter (larger = smaller mean size)
#
# The **moments** of this distribution are:
# - M₀ = total number concentration
# - M₃ ∝ total ice mass
# - M₆ ∝ radar reflectivity
#
# P3's 3-moment scheme tracks all three, giving it unprecedented accuracy.

# ## Exploring Fall Speed Dependence on Particle Size
#
# Larger ice particles fall faster. The relationship depends on particle habit
# and riming. Let's see how the number-weighted, mass-weighted, and
# reflectivity-weighted fall speeds vary with the slope parameter λ.

λ_values = 10 .^ range(log10(100), log10(5000), length=50)  # 1/m
N₀ = 1e6  # m⁻⁴

V_number = Float64[]
V_mass = Float64[]
V_reflectivity = Float64[]

for λ in λ_values
    state = IceSizeDistributionState(Float64;
        intercept = N₀,
        shape = 0.0,
        slope = λ)

    push!(V_number, evaluate(NumberWeightedFallSpeed(), state))
    push!(V_mass, evaluate(MassWeightedFallSpeed(), state))
    push!(V_reflectivity, evaluate(ReflectivityWeightedFallSpeed(), state))
end

# Convert λ to mean diameter for intuition: D̄ = (μ+1)/λ for gamma distribution
mean_diameter_mm = 1000 .* (0.0 + 1) ./ λ_values  # μ = 0

fig = Figure(size=(1000, 800))

ax1 = Axis(fig[1, 1];
    xlabel = "Mean diameter D̄ (mm)",
    ylabel = "Fall speed integral",
    title = "Ice Fall Speed vs Particle Size",
    xscale = log10,
    yscale = log10)

lines!(ax1, mean_diameter_mm, V_number; color=:dodgerblue, linewidth=3, label="Number-weighted Vₙ")
lines!(ax1, mean_diameter_mm, V_mass; color=:limegreen, linewidth=3, label="Mass-weighted Vₘ")
lines!(ax1, mean_diameter_mm, V_reflectivity; color=:orangered, linewidth=3, label="Reflectivity-weighted V_z")

axislegend(ax1; position=:lt)

# The hierarchy Vᵢ > Vₘ > Vₙ reflects that larger particles (which dominate
# higher moments) fall faster.

# ## The Effect of Riming
#
# When ice particles collect supercooled cloud droplets, they become **rimed**.
# Rime fills in the dendritic structure, making particles denser and faster-falling.
# Let's see how the rime fraction affects fall speed.

rime_fractions = range(0, 1, length=40)
rime_densities = [400.0, 600.0, 800.0]  # kg/m³ - light, medium, heavy riming

ax2 = Axis(fig[1, 2];
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Mass-weighted fall speed",
    title = "Effect of Riming on Fall Speed")

colors = [:skyblue, :royalblue, :navy]
for (i, ρ_rime) in enumerate(rime_densities)
    V_rimed = Float64[]
    for F_rim in rime_fractions
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = 1000.0,  # Fixed size distribution
            rime_fraction = F_rim,
            rime_density = ρ_rime)

        push!(V_rimed, evaluate(MassWeightedFallSpeed(), state))
    end
    lines!(ax2, rime_fractions, V_rimed;
           color=colors[i], linewidth=3,
           label="ρ_rime = $(Int(ρ_rime)) kg/m³")
end

axislegend(ax2; position=:lt)

# Heavier riming (higher ρ_rime and F_rim) → faster fall speeds.
# This is why graupel falls faster than snow!

# ## Ventilation: Enhanced Mass Transfer
#
# As particles fall, air flow around them enhances vapor diffusion and heat
# transfer. This **ventilation effect** is crucial for depositional growth
# and sublimation. Larger, faster-falling particles have higher ventilation.

ax3 = Axis(fig[2, 1];
    xlabel = "Mean diameter D̄ (mm)",
    ylabel = "Ventilation factor",
    title = "Ventilation Enhancement",
    xscale = log10,
    yscale = log10)

V_vent = Float64[]
V_vent_enhanced = Float64[]

for λ in λ_values
    state = IceSizeDistributionState(Float64;
        intercept = N₀,
        shape = 0.0,
        slope = λ)

    push!(V_vent, evaluate(Ventilation(), state))
    push!(V_vent_enhanced, evaluate(VentilationEnhanced(), state))
end

lines!(ax3, mean_diameter_mm, V_vent;
       color=:purple, linewidth=3, label="Basic ventilation")
lines!(ax3, mean_diameter_mm, V_vent_enhanced;
       color=:magenta, linewidth=3, linestyle=:dash, label="Enhanced (D > 100 μm)")

axislegend(ax3; position=:lt)

# ## Meltwater Shedding: When Ice Starts to Melt
#
# Near 0°C, ice particles begin to melt. Liquid water accumulates on the
# particle surface. If there's too much liquid, it **sheds** as raindrops.
# The shedding rate depends on the liquid fraction.

liquid_fractions = range(0, 0.5, length=40)

ax4 = Axis(fig[2, 2];
    xlabel = "Liquid fraction Fˡ",
    ylabel = "Shedding rate integral",
    title = "Meltwater Shedding")

# Different particle sizes
λ_test = [500.0, 1000.0, 2000.0]
colors = [:coral, :crimson, :darkred]
D_labels = ["Large (D̄ ≈ 2 mm)", "Medium (D̄ ≈ 1 mm)", "Small (D̄ ≈ 0.5 mm)"]

for (i, λ) in enumerate(λ_test)
    shedding = Float64[]
    for F_liq in liquid_fractions
        state = IceSizeDistributionState(Float64;
            intercept = 1e6,
            shape = 0.0,
            slope = λ,
            liquid_fraction = F_liq)

        push!(shedding, evaluate(SheddingRate(), state))
    end
    lines!(ax4, liquid_fractions, shedding;
           color=colors[i], linewidth=3, label=D_labels[i])
end

axislegend(ax4; position=:lt)

# Larger particles (small λ) shed more water because they collect more
# meltwater before it can refreeze.

fig

# ## Reflectivity: What Radars See
#
# Weather radars measure the 6th moment of the drop size distribution (M₆).
# Ice particles contribute to reflectivity based on their size and density.
# Let's see how reflectivity varies with particle properties.

fig2 = Figure(size=(1000, 400))

ax5 = Axis(fig2[1, 1];
    xlabel = "Mean diameter D̄ (mm)",
    ylabel = "Reflectivity integral (log scale)",
    title = "Ice Reflectivity vs Size",
    xscale = log10,
    yscale = log10)

Z_pristine = Float64[]
Z_rimed = Float64[]

for λ in λ_values
    state_pristine = IceSizeDistributionState(Float64;
        intercept = 1e6, shape = 0.0, slope = λ,
        rime_fraction = 0.0)

    state_rimed = IceSizeDistributionState(Float64;
        intercept = 1e6, shape = 0.0, slope = λ,
        rime_fraction = 0.7, rime_density = 700.0)

    push!(Z_pristine, evaluate(Reflectivity(), state_pristine))
    push!(Z_rimed, evaluate(Reflectivity(), state_rimed))
end

lines!(ax5, mean_diameter_mm, Z_pristine;
       color=:cyan, linewidth=3, label="Pristine ice")
lines!(ax5, mean_diameter_mm, Z_rimed;
       color=:red, linewidth=3, label="Heavily rimed (graupel)")

axislegend(ax5; position=:lt)

# ## The Shape Parameter μ: Controlling Distribution Breadth
#
# While λ controls mean size, μ controls how spread out the distribution is.
# Higher μ → more peaked distribution (most particles near mean size).

ax6 = Axis(fig2[1, 2];
    xlabel = "Shape parameter μ",
    ylabel = "Normalized fall speed",
    title = "Effect of Distribution Shape")

μ_values = range(0, 6, length=30)
λ_fixed = 1000.0

V_vs_mu = Float64[]
Z_vs_mu = Float64[]

for μ in μ_values
    state = IceSizeDistributionState(Float64;
        intercept = 1e6, shape = μ, slope = λ_fixed)

    push!(V_vs_mu, evaluate(MassWeightedFallSpeed(), state))
    push!(Z_vs_mu, evaluate(Reflectivity(), state))
end

lines!(ax6, μ_values, V_vs_mu ./ maximum(V_vs_mu);
       color=:teal, linewidth=3, label="Mass-weighted velocity")
lines!(ax6, μ_values, Z_vs_mu ./ maximum(Z_vs_mu);
       color=:gold, linewidth=3, label="Reflectivity")

axislegend(ax6; position=:rt)

fig2

# ## Physical Interpretation
#
# These explorations reveal the physics encoded in P3's integral tables:
#
# 1. **Fall speed hierarchy**: V_z > V_m > V_n because larger particles
#    (which dominate higher moments) fall faster. This matters for
#    differential sedimentation of mass vs number.
#
# 2. **Riming densifies particles**: Rime fills in the fractal structure
#    of ice crystals, increasing density and fall speed. Graupel (F_rim ≈ 1)
#    falls much faster than pristine snow.
#
# 3. **Ventilation scales with size**: Large, fast-falling particles have
#    enhanced mass transfer with the environment. This accelerates both
#    growth (in supersaturated air) and sublimation (in subsaturated air).
#
# 4. **Melting leads to shedding**: As ice melts, liquid accumulates until
#    aerodynamic forces shed it as rain. This is why melting snow produces
#    a characteristic radar "bright band."
#
# 5. **Reflectivity is size-dominated**: The D⁶ weighting means a few large
#    particles dominate the radar signal. Rimed particles contribute more
#    because they're denser.
#
# ## Coming Soon: Full P3 Dynamics
#
# When the P3 microphysics tendencies are complete, we'll extend this
# to a full parcel simulation showing:
# - Ice nucleation and initial growth
# - The Bergeron-Findeisen process (ice stealing vapor from liquid)
# - Riming: supercooled droplet collection
# - Aggregation: snowflakes sticking together
# - Melting near 0°C and the transition to rain
#
# The stationary parcel framework is perfect for isolating these processes
# and understanding how P3's predicted properties evolve in time.

nothing #hide
