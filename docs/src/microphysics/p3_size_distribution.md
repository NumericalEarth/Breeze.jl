# [Size Distribution](@id p3_size_distribution)

P3 assumes ice particles follow a **gamma size distribution**, with parameters
determined from prognostic moments and empirical closure relations.

## Gamma Size Distribution

The number concentration of ice particles per unit volume, as a function of
maximum dimension ``D``, follows ([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 19):

```math
N'(D) = N₀ D^μ e^{-λD}
```

where:
- ``N'(D)`` [m⁻⁴] is the number concentration per unit diameter
- ``N₀`` [m⁻⁵⁻μ] is the intercept parameter
- ``μ`` [-] is the shape parameter (≥ 0)
- ``λ`` [m⁻¹] is the slope parameter

The shape parameter ``μ`` controls the distribution width:
- ``μ = 0``: Exponential (Marshall-Palmer) distribution
- ``μ > 0``: Narrower distribution with a mode at ``D = μ/λ``

This form is standard in cloud microphysics and is discussed in
[Milbrandt & Yau (2005)](@cite MilbrandtYau2005) for multi-moment schemes.

## Moments of the Distribution

The ``k``-th moment of the size distribution is:

```math
M_k = \int_0^∞ D^k N'(D)\, dD = N₀ \int_0^∞ D^{k+μ} e^{-λD}\, dD
```

Using the gamma function identity ``\int_0^∞ x^{a-1} e^{-x} dx = Γ(a)``:

```math
M_k = N₀ \frac{Γ(k + μ + 1)}{λ^{k+μ+1}}
```

### Key Moments

**Number concentration** (0th moment):

```math
N = M_0 = N₀ \frac{Γ(μ + 1)}{λ^{μ+1}}
```

**Mean diameter** (1st moment / 0th moment):

```math
\bar{D} = \frac{M_1}{M_0} = \frac{μ + 1}{λ}
```

**Reflectivity** (6th moment) — this is the third prognostic variable in three-moment P3
([Milbrandt et al. (2021)](@cite MilbrandtEtAl2021)):

```math
Z ∝ M_6 = N₀ \frac{Γ(μ + 7)}{λ^{μ+7}}
```

## Shape-Slope (μ-λ) Relationship

In the officail P3 code, ``μ`` is diagnosed rather than set by a
single global power law. Define the mean-volume diameter estimate (in mm)

```math
D_{mvd} = 10^3 \left(\frac{L}{c_{gp}}\right)^{1/3},
```

where ``c_{gp}`` is the coefficient in the fully rimed mass law ``m(D) = c_{gp} D^3``.
Then:

```math
μ =
\begin{cases}
\text{clamp}\left(0.076 (0.01 λ)^{0.8} - 2,\ 0,\ 6\right), & D_{mvd} \le 0.2\,\text{mm} \\
\text{clamp}\left(0.25 (D_{mvd} - 0.2)\, f_ρ\, Fᶠ,\ 0,\ μ_{max}\right), & D_{mvd} > 0.2\,\text{mm}
\end{cases}
```

with

```math
f_ρ = \max\left(1,\ 1 + 0.00842(\bar{ρ}-400)\right),
\quad \bar{ρ} = \frac{6 c_{gp}}{π},
\quad μ_{max} = 20.
```

The first branch corresponds to the Heymsfield (2003) μ–λ fit (Eq. 27 in
[Morrison2015parameterization](@cite)), written with λ in m⁻¹ (the factor 0.01
converts to cm⁻¹). The second branch increases ``μ`` with particle size and riming
in the Fortran lookup-table generator.

!!! note "Breeze helper closure"
    The `TwoMomentClosure` / `ShapeParameterRelation` used in the examples implements
    only the Heymsfield power-law clamp (``μ_{max} = 6``). This matches the small-particle
    branch but omits the large-particle diagnostic used in the officail P3 code.

!!! note "Three-Moment Mode"
    In the officail P3 code, ``μ`` (and the bulk ice density used in rates) are obtained
    from lookup table 3 (`p3_lookupTable_3.dat-v1.4`) by interpolation in the Z/Q space,
    rime fraction, liquid fraction, and rime density. The analytic moment relations
    provide the conceptual basis for the table but are not solved directly at runtime.

```@example p3_psd
using Breeze.Microphysics.PredictedParticleProperties
using CairoMakie

# Compute μ vs λ
relation = ShapeParameterRelation()
λ_values = 10 .^ range(2, 5, length=100)
μ_values = [shape_parameter(relation, log(λ)) for λ in λ_values]

fig = Figure(size=(500, 350))
ax = Axis(fig[1, 1],
    xlabel = "Slope parameter λ [m⁻¹]",
    ylabel = "Shape parameter μ",
    xscale = log10,
    title = "μ-λ Relationship (Morrison & Milbrandt 2015a)")

lines!(ax, λ_values, μ_values, linewidth=2)
hlines!(ax, [relation.μmax], linestyle=:dash, color=:gray, label="μmax")

fig
```

## Determining Distribution Parameters

Given prognostic moments ``L`` (mass concentration) and ``N`` (number concentration),
plus predicted rime properties ``Fᶠ`` and ``ρᶠ``, we solve for the distribution
parameters ``(N₀, λ, μ)``.

In the official P3 lookup tables, rime and liquid fractions are tabulated on
discrete bins (0, 1/3, 2/3, 1) and interpolated during lookup.

### The Mass-Number Ratio

The ratio of ice mass to number concentration depends on the distribution parameters:

```math
\frac{L}{N} = \frac{\int_0^∞ m(D) N'(D)\, dD}{\int_0^∞ N'(D)\, dD}
```

For a power-law mass relationship ``m(D) = α D^β``, this simplifies to:

```math
\frac{L}{N} = α \frac{Γ(β + μ + 1)}{λ^β Γ(μ + 1)}
```

However, P3 uses a **piecewise** mass-diameter relationship with four regimes
(see [Particle Properties](@ref p3_particle_properties)), so the integral must
be computed over each regime separately.

### Lambda Solver

Finding ``λ`` requires solving:

```math
\log\left(\frac{L}{N}\right) = \log\left(\frac{\int_0^∞ m(D) N'(D)\, dD}{\int_0^∞ N'(D)\, dD}\right)
```

This is a nonlinear equation in ``λ`` (since ``μ = μ(λ)``). In the official P3
code, ``λ`` is determined during lookup-table generation by scanning over a
fixed range (roughly 10–10⁷ m⁻¹) and selecting the value that best matches L/N
for the current ``μ`` and piecewise ``m(D)``. The runtime then interpolates ``λ``
from the tables. The `distribution_parameters` helper in Breeze instead uses a
secant solver for direct evaluation.

```@example p3_psd
# Solve for distribution parameters
L_ice = 1e-4   # Ice mass concentration [kg/m³]
N_ice = 1e5    # Ice number concentration [1/m³]
rime_fraction = 0.0
rime_density = 400.0

params = distribution_parameters(L_ice, N_ice, rime_fraction, rime_density)

println("Distribution parameters:")
println("  N₀ = $(round(params.N₀, sigdigits=3)) m⁻⁵⁻μ")
println("  λ  = $(round(params.λ, sigdigits=3)) m⁻¹")
println("  μ  = $(round(params.μ, digits=2))")
```

### Computing ``N₀``

Once ``λ`` and ``μ`` are known, the intercept is found from normalization:

```math
N₀ = \frac{N λ^{μ+1}}{Γ(μ + 1)}
```

## Visualizing Size Distributions

```@example p3_psd
using SpecialFunctions: gamma

# Plot size distributions for different L/N ratios
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [mm]",
    ylabel = "N'(D) [m⁻⁴]",
    yscale = log10,
    title = "Ice Size Distributions")

D_mm = range(0.01, 5, length=200)
D_m = D_mm .* 1e-3

N_ice = 1e5
for (L, label, color) in [(1e-5, "L = 10⁻⁵ kg/m³", :blue),
                           (1e-4, "L = 10⁻⁴ kg/m³", :green),
                           (1e-3, "L = 10⁻³ kg/m³", :red)]
    params = distribution_parameters(L, N_ice, 0.0, 400.0)
    N_D = @. params.N₀ * D_m^params.μ * exp(-params.λ * D_m)
    lines!(ax, D_mm, N_D, label=label, color=color)
end

axislegend(ax, position=:rt)
ylims!(ax, 1e3, 1e12)
fig
```

## Effect of Rime Fraction

Riming changes particle mass at a given size, which affects the inferred distribution:

```@example p3_psd
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [mm]",
    ylabel = "N'(D) [m⁻⁴]",
    yscale = log10,
    title = "Effect of Riming on Size Distribution\n(L = 10⁻⁴ kg/m³, N = 10⁵ m⁻³)")

L_ice = 1e-4
N_ice = 1e5

for (Ff, label, color) in [(0.0, "Fᶠ = 0 (unrimed)", :blue),
                            (0.3, "Fᶠ = 0.3", :green),
                            (0.6, "Fᶠ = 0.6", :orange)]
    params = distribution_parameters(L_ice, N_ice, Ff, 500.0)
    N_D = @. params.N₀ * D_m^params.μ * exp(-params.λ * D_m)
    lines!(ax, D_mm, N_D, label=label, color=color)
end

axislegend(ax, position=:rt)
ylims!(ax, 1e3, 1e12)
fig
```

## Mass Integrals with Piecewise m(D)

The challenge in P3 is that the mass-diameter relationship is piecewise
(see [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eqs. 1-4):

```math
\int_0^∞ m(D) N'(D)\, dD = \sum_{i=1}^{4} \int_{D_{i-1}}^{D_i} a_i D^{b_i} N'(D)\, dD
```

Each piece has the form:

```math
\int_{D_1}^{D_2} a D^b N₀ D^μ e^{-λD}\, dD = a N₀ \int_{D_1}^{D_2} D^{b+μ} e^{-λD}\, dD
```

Using incomplete gamma functions:

```math
\int_{D_1}^{D_2} D^k e^{-λD}\, dD = \frac{1}{λ^{k+1}} \left[ Γ(k+1, λD_1) - Γ(k+1, λD_2) \right]
```

where ``Γ(a, x) = \int_x^∞ t^{a-1} e^{-t} dt`` is the upper incomplete gamma function.

## Numerical Stability

All computations are performed in **log space** for numerical stability:

```math
\log\left(\int_{D_1}^{D_2} D^k e^{-λD}\, dD\right) =
-(k+1)\log(λ) + \log Γ(k+1) + \log(q_1 - q_2)
```

where ``q_i = Γ(k+1, λD_i) / Γ(k+1)`` is the regularized incomplete gamma function.

## Three-Moment Extension

With three-moment ice ([Milbrandt et al. (2021)](@cite MilbrandtEtAl2021),
[Milbrandt et al. (2024)](@cite MilbrandtEtAl2024),
[Morrison et al. (2025)](@cite Morrison2025complete3moment)),
the 6th moment ``Z`` provides an additional constraint.
This allows independent determination of ``μ`` rather than using the μ-λ relationship:

```math
\frac{Z}{N} = \frac{Γ(μ + 7)}{λ^6 Γ(μ + 1)}
```

Combined with the L/N ratio, this gives two equations for two unknowns (``μ`` and ``λ``).
In the official P3 code, these constraints are used to build a lookup table that
returns ``μ`` (and bulk density) by interpolation; ``λ`` is then obtained from
the main table using the diagnosed ``μ``.

The benefit of three-moment ice is improved representation of:
- **Size sorting**: Large particles fall faster and separate from small ones
- **Hail formation**: Accurate simulation of heavily rimed particles
- **Radar reflectivity**: Direct prognostic variable rather than diagnosed

Both two-moment and three-moment solvers are implemented:

- **Two-moment**: Use `distribution_parameters(L, N, Fᶠ, ρᶠ)` with `TwoMomentClosure`
- **Three-moment**: Use `distribution_parameters(L, N, Z, Fᶠ, ρᶠ)` with `ThreeMomentClosure`

## Summary

The P3 size distribution closure proceeds as:

1. **Prognostic moments**: ``L``, ``N`` (and optionally ``Z``) are carried by the model
2. **Rime properties**: ``Fᶠ`` and ``ρᶠ`` determine the mass-diameter relationship
3. **Lambda solver**: ``λ`` is tabulated by scanning L/N in the reference Fortran (Breeze uses a secant solver in the helper)
4. **μ diagnosis**: Piecewise diagnostic for 2-moment, or lookup-table inversion for 3-moment
5. **Normalization**: Intercept ``N₀`` from number conservation

This provides the complete size distribution needed for computing microphysical rates.

## References for This Section

- [Morrison2015parameterization](@cite): PSD formulation and μ-λ relationship (Sec. 2b)
- [MilbrandtYau2005](@cite): Multimoment bulk microphysics and shape parameter analysis
- [Heymsfield2003](@cite): Ice size distribution observations used for μ-λ fit
- [MilbrandtEtAl2021](@cite): Three-moment ice with Z as prognostic
- [MilbrandtEtAl2024](@cite): Updated three-moment formulation
- [Morrison2025complete3moment](@cite): Complete three-moment implementation
