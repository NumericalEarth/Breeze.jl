# [Microphysics](@id section:microphysics-overview)

In atmospheric modeling, "microphysics" largely overlaps with ["cloud physics"](https://en.wikipedia.org/wiki/Cloud_physics) and concerns the formation, development, and precipitation of clouds.
More generally, "microphysics" encompasses panoply of physical processes associated with _(i)_ the conversion of water between vapor, liquid, and ice phases, and _(ii)_ the interaction and transformation of "cloud particles", which include droplets, ice particles, and aerosols.
For example: microphysical processes include droplet nucleation on aerosol particles; liquid freezing and vapor solidification; chemical and humidity-based transformations of aerosol particles; the agglomeration of ice particles into snowflakes; collisions and breakup of falling droplets (rain); collision between ice particles and droplets or wholesale freezing of droplets to form hail (e.g. [Straka2009](@citet)).

Breeze microphysics is under active development.
Today, Breeze provides [warm-phase](@ref sec:warm-saturation-adjustment) and [mixed-phase](@ref section:mixed-phase-saturation-adjustment) saturation adjustment, the bulk schemes of [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl), and the Predicted Particle Properties (P3) mixed-phase scheme (see its [theory](@ref p3_theory) and [usage](@ref p3_usage) pages).
Breeze aims to eventually provide a wide range of microphysical models, ranging from simple warm-phase saturation adjustment, to the bulk schemes provided by the Climate Modeling Alliance's [CloudMicrophysics.jl](https://github.com/CliMA/CloudMicrophysics.jl), to superdroplet schemes, to spectral bin schemes that include a spectrum of droplet sizes, ice particle shapes and aerosol types.

## Advection of microphysical fields

Each microphysics scheme chooses the default advection scheme for its own prognostic fields
and for the moisture density it evolves, through
[`default_microphysics_advection`](@ref Breeze.AtmosphereModels.default_microphysics_advection).
For every scheme in Breeze (saturation adjustment, the one- and two-moment bulk schemes, Kessler, and P3)
these fields use a bounds-preserving `WENO` scheme (order 5, or order 3 on grids with a horizontal halo of 3)
with bounds `(0, ∞)`, combined with an `AdaptiveVerticallyImplicitDiscretization(cfl = 1/4)` that
treats the part of vertical transport, including sedimentation, above a Courant number of 1/4 implicitly.
The bounds limiter keeps fields non-negative only up to a Courant number of 5/18 per forward-Euler substep,
which is why the explicit part is capped at 1/4.
Fast-sedimenting species such as rain therefore stay non-negative and stable at time steps where
explicit centered advection would fail.
P3's prognostic vapor excess, which is negative in subsaturated air, is the exception and uses the model-wide default.

Other scalars (the thermodynamic variable, user tracers) and momentum keep the model-wide default,
`Centered(order = 2)`, and so do all scalars when `microphysics = nothing`.
A single `advection` or `scalar_advection` scheme passed to `AtmosphereModel` applies to every scalar,
and an entry of a `scalar_advection` `NamedTuple` overrides the default for that field:

```julia
model = AtmosphereModel(grid; microphysics, scalar_advection = (; ρqʳ = Centered(order = 2)))
```

Bounded WENO with implicit vertical transport costs more per time step than `Centered(order = 2)`:
a wider stencil, a limiter evaluated each stage, and a tridiagonal solve per column and field.
