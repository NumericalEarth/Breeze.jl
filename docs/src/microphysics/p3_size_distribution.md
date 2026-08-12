# [Size Distribution](@id p3_size_distribution)

```@meta
CurrentModule = Breeze.Microphysics.PredictedParticleProperties
```

P3 assumes ice particles follow a **gamma size distribution**, with parameters
determined from prognostic moments and empirical closure relations.

## Gamma Size Distribution

The number concentration of ice particles per unit volume, as a function of
maximum dimension ``D``, follows ([Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eq. 2):

```math
N'(D) = N₀ D^μ e^{-λD}
```

where:
- ``N'(D)`` [m⁻⁴] is the number concentration per unit diameter
- ``N₀`` [m⁻⁴⁻μ] is the intercept parameter
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

**Reflectivity** (6th moment), diagnosed from the PSD:

```math
Z ∝ M_6 = N₀ \frac{Γ(μ + 7)}{λ^{μ+7}}
```

## Shape-Slope (μ-λ) Relationship

In two-moment P3, ``μ`` is diagnosed
rather than set by a single global power law. Define the mean-volume diameter
estimate (in mm) from the mean per-particle mass ``L/N``:

```math
D_{mvd} = 10^3 \left(\frac{L/N}{c_{gp}}\right)^{1/3},
```

where ``c_{gp} = (π/6) ρ^{gr}`` is the coefficient in the fully rimed mass law
``m(D) = c_{gp} D^3``. Then:

```math
μ =
\begin{cases}
\text{clamp}\left(0.076 (0.01 λ)^{0.8} - 2,\ 0,\ 6\right), & D_{mvd} \le 0.2\,\text{mm} \\
\text{clamp}\left(0.25 (D_{mvd} - 0.2)\, f_ρ\, F^f,\ 0,\ μ_{max}\right), & D_{mvd} > 0.2\,\text{mm}
\end{cases}
```

with

```math
f_ρ = \max\left(1,\ 1 + 0.00842(\bar{ρ}-400)\right),
\quad \bar{ρ} = \frac{6 c_{gp}}{π},
\quad μ_{max} = 20.
```

The first branch is the [Heymsfield (2003)](@cite Heymsfield2003) μ–λ fit
; the prefactor ``0.076 \cdot (0.01\, λ)^{0.8}``
embeds the cm⁻¹↔m⁻¹ unit conversion of the original form.
The second branch increases ``μ`` with particle size and riming in the
Fortran lookup-table generator.

When liquid fraction is active (``F^l > 0``), the bulk density used in
``D_{mvd}`` and ``f_ρ`` is blended with the liquid density:

```math
ρ^{gr} = (1 - F^l)\, ρ^{gr}_\text{dry} + F^l\, 1000\,\text{kg/m}^3.
```

When ``F^f = 0`` the lookup-table generator additionally substitutes
``ρ_{g,\text{dry}} \to ρ_\text{rime}`` (the rime-density axis of the table)
because the partially-rimed regime has zero mass at that point. Within this
diagnostic ``ρ^f`` is floored at 50 kg/m³, the first coordinate of Table 1's
rime-density axis, matching the runtime lookup's clamp of the canonical unrimed
``ρ^f = 0``.

!!! note "Two-Moment Mode"
    The piecewise closure above is the formula the Fortran table *generator*
    evaluates. Breeze's model path does not evaluate it per grid point: ``μ`` is
    read from Table 1's shape-parameter column (Fortran `mu_i_save`), which
    stores the generator's result, and is interpolated in the same
    ``(\log \bar{m}, F^f, F^l, ρ^f)`` space as every other Table 1 integral
    (`compute_ice_shape_parameter` in `process_rate_helpers.jl`).

The plots below read ``λ`` and ``μ`` straight out of Table 1, so they show the
closure exactly as the model sees it.

```@example p3_psd
using Breeze
using Breeze.Microphysics.PredictedParticleProperties
using Logging: NullLogger, with_logger
using SpecialFunctions: loggamma
using CairoMakie

p3 = with_logger(NullLogger()) do
    PredictedParticlePropertiesMicrophysics()
end

bulk = p3.ice.bulk_properties

# Table 1 is indexed by the mean particle mass m̄ = q/N. Liquid fraction and the
# μ axis are singleton coordinates here.
"Read (λ, μ) from Table 1 and rebuild N₀ = N λ^(μ+1) / Γ(μ+1)."
function psd_from_table(p3, q, N, Fᶠ, ρᶠ; Fˡ = 0.0, μ_axis = 0.0)
    bulk = p3.ice.bulk_properties
    log_m̄ = log10(q / N)
    λ = bulk.slope(log_m̄, Fᶠ, Fˡ, ρᶠ, μ_axis)
    μ = bulk.shape(log_m̄, Fᶠ, Fˡ, ρᶠ, μ_axis)
    log_N₀ = log(N) + (μ + 1) * log(λ) - loggamma(μ + 1)
    return (; λ, μ, log_N₀)
end

N_ice = 1e5
q_values = 10 .^ range(-7, -2, length=80)

fig = Figure(size=(500, 350))
ax = Axis(fig[1, 1],
    xlabel = "Slope parameter λ [m⁻¹]",
    ylabel = "Shape parameter μ",
    xscale = log10,
    title = "Tabulated μ-λ Relationship")

for (Fᶠ, label, color) in [(0.0, "Fᶠ = 0 (unrimed)", :blue),
                            (0.5, "Fᶠ = 0.5", :orange),
                            (1.0, "Fᶠ = 1.0 (fully rimed)", :red)]
    psds = [psd_from_table(p3, q, N_ice, Fᶠ, 500.0) for q in q_values]
    lines!(ax, getfield.(psds, :λ), getfield.(psds, :μ),
           linewidth=2, color=color, label=label)
end

axislegend(ax, position=:rt)
fig
```

## Dry Size Distribution (Liquid-Fraction Active)

When ``F^l > 0``, the official P3 generator solves a separate **dry** PSD from
the dry-only ice mass ``q^i`` for the four liquid-fraction melting integrals
(see [Cholette et al. (2019)](@cite Cholette2019parameterization) for the
rationale). Deposition / sublimation, collection, sedimentation, and
reflectivity use the wet PSD. Breeze inherits that split through the tables:
the melting rate reads the dry-PSD Fortran `f1pr24`–`f1pr27` columns, while
deposition / sublimation reads the wet-PSD `f1pr05` / `f1pr14` pair.

The dry parameters follow from rescaling the wet ones so the mass moment
matches ``q_\text{dry} = q_\text{total}(1 - F^l)``:

```math
λ_d = λ\,(1-F^l)^{-1/β},\qquad N_{0,d} = N_0\,(λ_d/λ)^{μ+1},
```

with ``β`` the effective mass–diameter exponent of the state. At ``F^l = 0`` the
dry and wet distributions coincide. Breeze never evaluates this rescaling at
runtime — it reads the dry-PSD columns straight out of the Fortran table.

## Determining Distribution Parameters

Given prognostic moments ``L`` (mass concentration) and ``N`` (number concentration),
plus predicted rime properties ``F^f`` and ``ρ^f``, we solve for the distribution
parameters ``(N₀, λ, μ)``.

In the official P3 lookup tables, rime fraction ``F^f`` and liquid fraction ``F^l``
are each tabulated on 4 discrete nodes (``\{0, 1/3, 2/3, 1\}``) and interpolated
during lookup.

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

This is a nonlinear equation in ``λ``, since ``μ = μ(λ)``. In the official P3
code, ``λ`` is determined during lookup-table generation by scanning over a
fixed range (roughly 10–10⁷ m⁻¹) and selecting the value that best matches L/N
for the current ``μ`` and piecewise ``m(D)``.

!!! note "The model path does not solve for ice ``λ``"
    Breeze never needs ice ``λ`` per grid point, because every Table 1 integral
    is indexed by the *mean particle mass* ``\log \bar{m}`` rather than by
    ``λ``, so the slope is already baked into the tabulated values. Table 1 does
    carry a slope-parameter column, which Breeze loads for diagnostics (the
    plots on this page) but no rate reads. Rain is the exception: its
    distribution is exponential with ``μ^r = 0``, so ``λ^r`` follows in closed
    form from ``q^r/n^r`` via `rain_slope_parameter`, and rain integrals *are*
    indexed by ``\log λ^r``.

```@example p3_psd
q_ice = 1e-4   # Ice mass concentration [kg/m³]
N_ice = 1e5    # Ice number concentration [1/m³]
rime_fraction = 0.0
rime_density = 400.0

psd = psd_from_table(p3, q_ice, N_ice, rime_fraction, rime_density)

println("Tabulated distribution parameters:")
println("  log N₀ = $(round(psd.log_N₀, digits=2))")
println("  λ  = $(round(psd.λ, sigdigits=3)) m⁻¹")
println("  μ  = $(round(psd.μ, digits=2))")
```

### Computing ``N₀``

Once ``λ`` and ``μ`` are known, the intercept follows from a normalization integral. Inverting
the zeroth moment normalizes on number,

```math
N₀ = \frac{N λ^{μ+1}}{Γ(μ + 1)},
```

which is what `psd_from_table` above evaluates. The Fortran table generator
instead normalizes on mass (`create_p3_lookupTable_1.f90:1054`):

```math
N₀ = \frac{L}{\int_0^∞ m(D)\, D^μ e^{-λD}\, dD}
```

The two coincide whenever ``λ`` satisfies the L/N constraint above, since that constraint is
exactly the statement that the two normalizations agree. They part company only where the
mean-diameter limiter clamps ``λ``: normalizing on mass keeps ``L`` exact and lets the
represented number concentration absorb the adjustment — P3's own policy, which adjusts ``N``
to keep the mean particle size physical — whereas normalizing on number would preserve ``N``
and misstate the mass.

## Visualizing Size Distributions

```@example p3_psd
# Plot size distributions for different q/N ratios
fig = Figure(size=(600, 400))
ax = Axis(fig[1, 1],
    xlabel = "Diameter D [mm]",
    ylabel = "N'(D) [m⁻⁴]",
    yscale = log10,
    title = "Ice Size Distributions")

D_mm = range(0.01, 5, length=200)
D_m = D_mm .* 1e-3

N_ice = 1e5
for (q, q_label, color) in [(1e-5, "q = 10⁻⁵ kg/m³", :blue),
                            (1e-4, "q = 10⁻⁴ kg/m³", :green),
                            (1e-3, "q = 10⁻³ kg/m³", :red)]
    psd = psd_from_table(p3, q, N_ice, 0.0, 400.0)
    N_D = @. exp(psd.log_N₀ + psd.μ * log(D_m) - psd.λ * D_m)
    label = q_label * "  (μ = $(round(psd.μ, digits=2)))"
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
    title = "Effect of Riming on Size Distribution\n(q = 10⁻⁴ kg/m³, N = 10⁵ m⁻³)")

q_ice = 1e-4
N_ice = 1e5

for (Ff, Ff_label, color) in [(0.0, "Fᶠ = 0 (unrimed)", :blue),
                               (0.3, "Fᶠ = 0.3", :green),
                               (0.6, "Fᶠ = 0.6", :orange)]
    psd = psd_from_table(p3, q_ice, N_ice, Ff, 500.0)
    N_D = @. exp(psd.log_N₀ + psd.μ * log(D_m) - psd.λ * D_m)
    label = Ff_label * "  (μ = $(round(psd.μ, digits=2)))"
    lines!(ax, D_mm, N_D, label=label, color=color)
end

axislegend(ax, position=:rt)
ylims!(ax, 1e3, 1e12)
fig
```

## Mass Integrals with Piecewise m(D)

The challenge in P3 is that the mass-diameter relationship is piecewise
(see [Morrison & Milbrandt (2015a)](@cite Morrison2015parameterization) Eqs. 6, 7, 12, and 13):

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

## Summary

The P3 size distribution closure proceeds as:

1. **Prognostic moments**: ``L`` and ``N`` are carried by the model
2. **Rime properties**: ``F^f`` and ``ρ^f`` determine the mass-diameter relationship
3. **Slope**: ``λ`` is absorbed into the tables, which the generator indexes by
   mean particle mass ``\bar{m} = L/N``; the model path interpolates on
   ``\log \bar{m}`` and never solves for ice ``λ`` itself
4. **μ diagnosis**: a Table 1 lookup (the tabulated `mu_i_save`, i.e.
   the piecewise closure evaluated at generation time)
5. **Normalization**: the generator fixes the intercept ``N₀`` from the mass
   integral, so ``L`` is preserved even where the ``λ`` limiter binds. Work in
   ``\log N₀`` rather than ``N₀``: its m^-(4+μ) units put it beyond Float32 range
   for narrow distributions of small particles

This provides the complete size distribution needed for computing microphysical rates.

## References for This Section

- [Morrison2015parameterization](@cite): PSD formulation and μ-λ relationship (Sec. 2b)
- [MilbrandtYau2005](@cite): Multimoment bulk microphysics and shape parameter analysis
- [Heymsfield2003](@cite): Ice size distribution observations used for μ-λ fit
- [Cholette2019parameterization](@cite): Predicted-liquid-fraction extension and dry-PSD branch for melting/deposition
