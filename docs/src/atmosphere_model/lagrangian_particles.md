# Lagrangian particles

[`AtmosphereModel`](@ref) can carry a set of Lagrangian particles that are advected
with the resolved flow. Particles are built with Oceananigans'
`LagrangianParticles` (see the Oceananigans documentation for its full keyword
interface, including `tracked_fields` and custom `dynamics`) and attached through
the `particles` keyword argument:

```julia
using Oceananigans
using Breeze

grid = RectilinearGrid(size=(16, 16, 16), x=(0, 1e4), y=(0, 1e4), z=(0, 4e3))

particles = LagrangianParticles(x=[5e3], y=[5e3], z=[1e3])
model = AtmosphereModel(grid; particles)
```

`model.particles` is the attached object, and `nothing` when no particles are
requested (the default). Particle positions live in `model.particles.properties`
and are written by output writers like any other model state.

## Which velocity advects the particles

Particles are advected with the **Cartesian** velocity components
`model.velocities`, which are diagnosed from the prognostic momentum as
``u = \rho u / \rho`` during `update_state!`. Because particle positions are
stored as physical coordinates ``(x, y, z)``, the trajectory equations are

```math
\frac{\mathrm{d} x}{\mathrm{d} t} = u , \qquad
\frac{\mathrm{d} y}{\mathrm{d} t} = v , \qquad
\frac{\mathrm{d} z}{\mathrm{d} t} = w .
```

On a terrain-following grid this is *not* the contravariant velocity
``\tilde{w}`` that carries vertical transport (see the
[terrain-following coordinates](@ref Terrain-following-section) page);
``\tilde{w}`` measures flow across the tilted coordinate surfaces, whereas a
particle's physical altitude changes at the rate ``w``.

## When particles are advected

Particles are advanced with the time-stepping scheme of the model, stage by stage, so particle
trajectories have the same order of accuracy in time as the dycore. At the beginning of each stage,
the particle velocity is interpolated with the fields at the beginning of the stage.
Both Breeze time steppers — `SSPRungeKutta3` and `AcousticRungeKutta3` — combine every stage with the
state at the beginning of the time step, so particle positions are advanced in terms of the position
``X^n`` at the beginning of the time step and the displacement ``D`` from ``X^n``,

```math
D^{(k)} = a_k D^{(k-1)} + b_k \, \Delta t \, \boldsymbol{u}(X^{(k-1)}) , \qquad X^{(k)} = X^n + D^{(k)} , \qquad D^{(0)} = 0 .
```

With ``a_k = b_k = α_k = (1, 1/4, 2/3)``, this is the Shu–Osher form of SSP RK3,
``X^{(k)} = (1 - α_k) X^n + α_k (X^{(k-1)} + Δt \, \boldsymbol{u}(X^{(k-1)}))``.
With ``a_k = 0`` and ``b_k = β_k = (1/3, 1/2, 1)``, this is the Wicker–Skamarock scheme,
``X^{(k)} = X^n + β_k Δt \, \boldsymbol{u}(X^{(k-1)})``.
Boundary conditions are applied to ``X^{(k)}`` after every stage, but not to the displacement, so that
particles crossing a periodic boundary during a time step are advanced consistently.

At the beginning of each time step, particles are sorted (if `sorting` is specified) and their
`dynamics` is called once with the full time step. Particles absorbed at an `AbsorbingBoundary` are
removed at the end of the time step. Any `tracked_fields` attached to the particles are sampled after
each stage.

## Boundary conditions

Particles reflect off `Bounded` walls or are absorbed by them, according to the `ParticleBoundaryConditions`
of `LagrangianParticles` (by default, `ReflectingBoundary(1)`, an elastic bounce;
`ReflectingBoundary(0)` makes particles settle onto the wall), and wrap across `Periodic` boundaries.

## Terrain-following grids

On a [`TerrainFollowingVerticalDiscretization`](@ref) grid the physical altitude
and the grid's vertical coordinate differ,

```math
z(x, y, r) = r + \sum_n h_n(x, y) \, b_n(r) ,
```

and interpolation is performed against ``r``. Breeze therefore inverts
``z(x, y, \cdot)`` for ``r`` at the particle position before interpolating. The
inversion is exact for [`LinearDecay`](@ref) (the map is affine in ``r``) and uses
a bracketed Newton solve for [`TwoLevelDecay`](@ref), which converges to roundoff
in a handful of iterations. It is evaluated once per particle per stage and reused
for all three velocity components.

The terrain components ``h_n`` are interpolated horizontally to the particle
position, consistently with the staggered interpolation that `znode` applies at
velocity points, so a particle sitting exactly on a grid node recovers exactly
that node's reference coordinate.

The vertical walls are the bounding coordinate surfaces evaluated above the
particle's horizontal position: the local terrain ``z(x, y, r_\text{bottom})``
below and the flat lid ``z(x, y, r_\text{top})`` above. Because the ``\rho w``
bottom boundary condition on terrain grids is kinematic —
``w|_1 = \partial_x h \, u + \partial_y h \, v``, described under the boundary
conditions on the [terrain-following coordinates](@ref Terrain-following-section)
page — a particle resting on the surface slides along the terrain rather than
sticking.

Two limitations are worth noting:

  * The wall bounce reflects the vertical coordinate alone. On a sloping surface
    that is a vertical bounce, not a specular reflection about the surface normal:
    neither the horizontal position nor the velocity direction is modified. This
    is exact for the flat lid, and for `ReflectingBoundary(0)`, where the particle
    simply settles onto the ground; it is approximate for an elastic bounce off a
    steep slope.

  * Particles are **not** supported on an `ImmersedBoundaryGrid` built on a
    terrain-following grid — the terrain coordinate map and the immersed-boundary
    bounce have not been reconciled. Constructing such a model with `particles`
    throws an `ArgumentError`. Use the terrain-following grid directly; its lower
    boundary already follows the terrain.
