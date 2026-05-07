# [P3 Examples and Visualization](@id p3_examples)

This section provides worked examples demonstrating P3 microphysics concepts
through visualization and analysis.

The examples illustrate key concepts from the P3 papers:
- Mass-diameter relationships from [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization)
- Size distribution from [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) and [Heymsfield (2003)](@cite Heymsfield2003)
- μ-λ relationship from [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 27

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

# μ-λ relationship (bottom)
ax4 = Axis(fig[2, 1:2],
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
3. **Bottom**: Shape parameter μ increases with λ up to a maximum
