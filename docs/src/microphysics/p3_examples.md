# P3 Examples and Visualization

This section provides worked examples demonstrating P3 microphysics concepts
through visualization and analysis.

## Ice Particle Property Explorer

Let's explore how ice particle properties vary with size and riming state.

### Mass-Diameter Relationship

```@example p3_examples
using Breeze.Microphysics.PredictedParticleProperties
using CairoMakie

mass = IceMassPowerLaw()

# Diameter range: 10 μm to 10 mm
D = 10 .^ range(-5, -2, length=200)

# Compute mass for different riming states
m_unrimed = [ice_mass(mass, 0.0, 400.0, d) for d in D]
m_light = [ice_mass(mass, 0.2, 400.0, d) for d in D]
m_moderate = [ice_mass(mass, 0.5, 500.0, d) for d in D]
m_heavy = [ice_mass(mass, 0.8, 700.0, d) for d in D]

# Reference: pure ice sphere
m_sphere = @. 917 * π/6 * D^3

fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1],
    xlabel = "Maximum dimension D [m]",
    ylabel = "Particle mass m [kg]",
    xscale = log10,
    yscale = log10,
    title = "Ice Particle Mass vs Size")

lines!(ax, D, m_sphere, color=:gray, linestyle=:dash, linewidth=1, label="Pure ice sphere")
lines!(ax, D, m_unrimed, linewidth=2, label="Unrimed (Fᶠ = 0)")
lines!(ax, D, m_light, linewidth=2, label="Light rime (Fᶠ = 0.2)")
lines!(ax, D, m_moderate, linewidth=2, label="Moderate rime (Fᶠ = 0.5)")
lines!(ax, D, m_heavy, linewidth=2, label="Heavy rime (Fᶠ = 0.8)")

axislegend(ax, position=:lt)
fig
```

Notice how riming increases particle mass at a given size, with the effect most
pronounced for larger particles where the mass-diameter relationship transitions
from the aggregate power law to the graupel regime.

### Regime Transitions

```@example p3_examples
# Visualize regime thresholds
fig = Figure(size=(700, 400))
ax = Axis(fig[1, 1],
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Threshold diameter [mm]",
    title = "Ice Particle Regime Thresholds")

Ff_range = range(0.01, 0.99, length=50)
D_th = Float64[]
D_gr = Float64[]
D_cr = Float64[]

for Ff in Ff_range
    thresholds = ice_regime_thresholds(mass, Ff, 500.0)
    push!(D_th, thresholds.spherical * 1e3)
    push!(D_gr, min(thresholds.graupel * 1e3, 20))
    push!(D_cr, min(thresholds.partial_rime * 1e3, 20))
end

lines!(ax, Ff_range, D_th, label="D_th (spherical)", linewidth=2)
lines!(ax, Ff_range, D_gr, label="D_gr (graupel)", linewidth=2)
lines!(ax, Ff_range, D_cr, label="D_cr (partial rime)", linewidth=2)

axislegend(ax, position=:rt)
ylims!(ax, 0, 15)
fig
```

As rime fraction increases, the graupel regime expands to smaller sizes,
while the partial-rime regime contracts to larger sizes.

## Size Distribution Visualization

### Effect of Mass Content

```@example p3_examples
using SpecialFunctions: gamma

fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [mm]",
    ylabel = "N'(D) [m⁻⁴]",
    yscale = log10,
    title = "Ice Size Distributions for Different Mass Contents\n(N = 10⁵ m⁻³)")

D_mm = range(0.01, 8, length=300)
D_m = D_mm .* 1e-3

N_ice = 1e5

# Different mass contents
for (L, color, label) in [
    (1e-5, :blue, "L = 0.01 g/m³"),
    (5e-5, :green, "L = 0.05 g/m³"),
    (1e-4, :orange, "L = 0.1 g/m³"),
    (5e-4, :red, "L = 0.5 g/m³"),
    (1e-3, :purple, "L = 1.0 g/m³")
]
    params = distribution_parameters(L, N_ice, 0.0, 400.0)
    N_D = @. params.N₀ * D_m^params.μ * exp(-params.λ * D_m)
    lines!(ax, D_mm, N_D, color=color, linewidth=2, label=label)
end

axislegend(ax, position=:rt)
ylims!(ax, 1e2, 1e14)
fig
```

Higher mass content (at fixed number) shifts the distribution toward larger particles.

### Shape Parameter Effect

```@example p3_examples
fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [mm]",
    ylabel = "N'(D) / N₀",
    title = "Effect of Shape Parameter μ on Distribution Shape\n(λ = 2000 m⁻¹)")

D_mm = range(0.01, 3, length=200)
D_m = D_mm .* 1e-3
λ = 2000.0

for μ in [0, 1, 2, 4, 6]
    N_norm = @. D_m^μ * exp(-λ * D_m)
    N_norm ./= maximum(N_norm)  # Normalize to peak
    lines!(ax, D_mm, N_norm, linewidth=2, label="μ = $μ")
end

axislegend(ax, position=:rt)
fig
```

Higher ``μ`` produces a narrower distribution with a more pronounced mode.

## Fall Speed Analysis

### Weighted Fall Speeds vs Slope Parameter

```@example p3_examples
fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1],
    xlabel = "Slope parameter λ [m⁻¹]",
    ylabel = "Fall speed integral [m/s]",
    xscale = log10,
    title = "Fall Speed Integrals vs λ\n(larger λ → smaller particles)")

λ_values = 10 .^ range(2.5, 5, length=30)

V_n = Float64[]
V_m = Float64[]
V_z = Float64[]

for λ in λ_values
    state = IceSizeDistributionState(Float64; intercept=1e6, shape=0.0, slope=λ)
    push!(V_n, evaluate(NumberWeightedFallSpeed(), state))
    push!(V_m, evaluate(MassWeightedFallSpeed(), state))
    push!(V_z, evaluate(ReflectivityWeightedFallSpeed(), state))
end

lines!(ax, λ_values, V_n, linewidth=2, label="Vₙ (number-weighted)")
lines!(ax, λ_values, V_m, linewidth=2, label="Vₘ (mass-weighted)")
lines!(ax, λ_values, V_z, linewidth=2, label="Vᵤ (reflectivity-weighted)")

axislegend(ax, position=:rt)
fig
```

Larger particles (smaller ``λ``) fall faster, and the reflectivity-weighted velocity
(emphasizing large particles) exceeds the mass-weighted velocity, which in turn
exceeds the number-weighted velocity.

### Effect of Riming on Fall Speed

```@example p3_examples
fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1],
    xlabel = "Rime fraction Fᶠ",
    ylabel = "Mass-weighted fall speed [m/s]",
    title = "Effect of Riming on Ice Fall Speed")

Ff_range = range(0, 0.9, length=20)
λ = 1000.0

V_m_values = Float64[]

for Ff in Ff_range
    state = IceSizeDistributionState(Float64; 
        intercept=1e6, shape=0.0, slope=λ,
        rime_fraction=Ff, rime_density=500.0)
    push!(V_m_values, evaluate(MassWeightedFallSpeed(), state))
end

lines!(ax, Ff_range, V_m_values, linewidth=3, color=:blue)
scatter!(ax, Ff_range, V_m_values, markersize=8, color=:blue)

fig
```

Rimed particles fall faster due to higher density.

## Integral Comparison

### All Fall Speed Components

```@example p3_examples
# Compare different integral types at a fixed state
state = IceSizeDistributionState(Float64; 
    intercept=1e6, shape=2.0, slope=1500.0,
    rime_fraction=0.3, rime_density=500.0)

integrals = [
    ("NumberWeightedFallSpeed", evaluate(NumberWeightedFallSpeed(), state)),
    ("MassWeightedFallSpeed", evaluate(MassWeightedFallSpeed(), state)),
    ("ReflectivityWeightedFallSpeed", evaluate(ReflectivityWeightedFallSpeed(), state)),
    ("Ventilation", evaluate(Ventilation(), state)),
    ("EffectiveRadius [μm]", evaluate(EffectiveRadius(), state) * 1e6),
    ("MeanDiameter [mm]", evaluate(MeanDiameter(), state) * 1e3),
    ("MeanDensity [kg/m³]", evaluate(MeanDensity(), state)),
]

println("Integral values for state:")
println("  N₀ = 10⁶, μ = 2, λ = 1500 m⁻¹, Fᶠ = 0.3")
println()
for (name, value) in integrals
    println("  $name = $(round(value, sigdigits=4))")
end
```

## Lambda Solver Demonstration

### Convergence Visualization

```@example p3_examples
fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1],
    xlabel = "log₁₀(L/N) [kg per particle]",
    ylabel = "log₁₀(λ) [m⁻¹]",
    title = "Lambda Solver: λ vs Mean Particle Mass")

N_ice = 1e5  # Fixed number concentration
L_values = 10 .^ range(-6, -2, length=50)

λ_unrimed = Float64[]
λ_rimed = Float64[]

for L in L_values
    params_ur = distribution_parameters(L, N_ice, 0.0, 400.0)
    params_ri = distribution_parameters(L, N_ice, 0.5, 500.0)
    push!(λ_unrimed, params_ur.λ)
    push!(λ_rimed, params_ri.λ)
end

log_LN = log10.(L_values ./ N_ice)

lines!(ax, log_LN, log10.(λ_unrimed), linewidth=2, label="Unrimed")
lines!(ax, log_LN, log10.(λ_rimed), linewidth=2, label="Rimed (Fᶠ = 0.5)")

axislegend(ax, position=:rt)
fig
```

At the same L/N ratio, rimed particles have larger ``λ`` (smaller characteristic size)
because their higher mass-per-particle requires smaller particles to match the ratio.

## Quadrature Accuracy

### Convergence with Number of Points

```@example p3_examples
state = IceSizeDistributionState(Float64; 
    intercept=1e6, shape=0.0, slope=1000.0)

n_points = [8, 16, 32, 64, 128, 256]
V_values = Float64[]
reference = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=512)

for n in n_points
    V = evaluate(NumberWeightedFallSpeed(), state; n_quadrature=n)
    push!(V_values, V)
end

fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Number of quadrature points",
    ylabel = "Relative error",
    xscale = log10,
    yscale = log10,
    title = "Quadrature Convergence")

errors = abs.(V_values .- reference) ./ reference
lines!(ax, n_points, errors, linewidth=2)
scatter!(ax, n_points, errors, markersize=10)

fig
```

The Chebyshev-Gauss quadrature converges rapidly, with 64 points typically
sufficient for double precision.

## Summary Visualization

```@example p3_examples
# Create a comprehensive summary figure
fig = Figure(size=(900, 600))

# Mass-diameter (top left)
ax1 = Axis(fig[1, 1],
    xlabel = "D [mm]", ylabel = "Mass [mg]",
    title = "Mass vs Diameter")

D_mm = range(0.1, 5, length=100)
D_m = D_mm .* 1e-3

for (Ff, label) in [(0.0, "Fᶠ=0"), (0.5, "Fᶠ=0.5")]
    m = [ice_mass(mass, Ff, 500.0, d) for d in D_m]
    lines!(ax1, D_mm, m .* 1e6, label=label)
end
axislegend(ax1, position=:lt)

# Size distribution (top right)
ax2 = Axis(fig[1, 2],
    xlabel = "D [mm]", ylabel = "N'(D) [m⁻⁴]",
    yscale = log10, title = "Size Distribution")

for L in [1e-5, 1e-4, 1e-3]
    params = distribution_parameters(L, 1e5, 0.0, 400.0)
    N_D = @. params.N₀ * D_m^params.μ * exp(-params.λ * D_m)
    lines!(ax2, D_mm, N_D, label="L=$(L*1e3) g/m³")
end
ylims!(ax2, 1e3, 1e13)
axislegend(ax2, position=:rt, fontsize=10)

# Fall speed vs λ (bottom left)
ax3 = Axis(fig[2, 1],
    xlabel = "λ [m⁻¹]", ylabel = "Fall speed",
    xscale = log10, title = "Fall Speed Integrals")

λ_vals = 10 .^ range(2.5, 4.5, length=30)
V_n = [evaluate(NumberWeightedFallSpeed(), 
       IceSizeDistributionState(Float64; intercept=1e6, shape=0.0, slope=λ)) for λ in λ_vals]
V_m = [evaluate(MassWeightedFallSpeed(), 
       IceSizeDistributionState(Float64; intercept=1e6, shape=0.0, slope=λ)) for λ in λ_vals]

lines!(ax3, λ_vals, V_n, label="Vₙ")
lines!(ax3, λ_vals, V_m, label="Vₘ")
axislegend(ax3, position=:rt)

# μ-λ relationship (bottom right)
ax4 = Axis(fig[2, 2],
    xlabel = "λ [m⁻¹]", ylabel = "μ",
    xscale = log10, title = "μ-λ Relationship")

relation = ShapeParameterRelation()
λ_range = 10 .^ range(2, 5, length=100)
μ_vals = [shape_parameter(relation, log(λ)) for λ in λ_range]

lines!(ax4, λ_range, μ_vals, linewidth=2, color=:blue)
hlines!(ax4, [relation.μmax], linestyle=:dash, color=:gray)

fig
```

This figure summarizes the key relationships in P3:
1. **Top left**: Mass increases with size and riming
2. **Top right**: Size distribution shifts with mass content
3. **Bottom left**: Fall speed decreases with λ (smaller particles)
4. **Bottom right**: Shape parameter μ increases with λ up to a maximum

