# [Terrain-following coordinates](@id Terrain-following-section)

Terrain-following coordinates map the irregular physical domain above topography onto a
regular computational domain. This avoids the need for immersed boundaries or cut cells
at the surface, simplifying the application of boundary conditions and ensuring smooth
coordinate surfaces near the ground.

The implementation is built on Oceananigans' [`MutableVerticalDiscretization`](@ref),
which provides the column-wise scaling factors ``\sigma(x, y)`` and surface displacement
``\eta(x, y)`` needed to deform the vertical coordinate. This page describes the
coordinate transformation, the metric corrections required for the dynamical equations,
and the user-facing API.

## Coordinate transformation

### Physical and computational coordinates

Let ``\zeta \in [0, z_{top}]`` denote the computational (reference) vertical coordinate and
``z(x, y, \zeta)`` the physical height. The relationship between them is

```math
z = \zeta \, \sigma(x, y) + \eta(x, y) ,
```

where ``\sigma`` is a column-wide vertical scaling factor and ``\eta`` is the
surface displacement. Oceananigans stores ``\sigma`` and ``\eta`` in the
[`MutableVerticalDiscretization`](@ref) and uses them to compute all vertical spacings and
node positions:

```math
\Delta z_{i,j,k} = \Delta \zeta_k \, \sigma_{i,j} , \qquad z_{i,j,k} = \zeta_k \, \sigma_{i,j} + \eta_{i,j} .
```

### Basic terrain following (Gal-Chen & Somerville)

The simplest terrain-following coordinate, introduced by [Gal-Chen and Somerville (1975)](@cite GalChen1975),
uses a linear decay of terrain influence with height:

```math
z(x, y, \zeta) = \zeta + h(x, y) \left(1 - \frac{\zeta}{z_{top}}\right) ,
```

where ``h(x, y)`` is the terrain height. Comparing with the general form above gives

```math
\sigma = \frac{z_{top} - h}{z_{top}} , \qquad \eta = h .
```

Coordinate surfaces are flat at the model top (``\zeta = z_{top} \Rightarrow z = z_{top}``)
and conform to the terrain at the surface (``\zeta = 0 \Rightarrow z = h``). This is the
[`BasicTerrainFollowing`](@ref) option in Breeze.

### Other formulations

More sophisticated coordinate formulations decay the terrain influence more rapidly with
height, reducing the distortion of coordinate surfaces in the upper atmosphere. Examples
include:

- The **SLEVE** (smooth level vertical) coordinate of [Schär et al. (2002)](@cite Schar2002),
  which splits the topography into large-scale and small-scale components with separate
  decay scales.
- The **hybrid terrain-following** coordinate of [Klemp (2011)](@cite Klemp2011), which uses
  a smooth transition from terrain-following surfaces near the ground to pure height
  surfaces aloft.

These are not yet implemented in Breeze but can be added by defining new smoothing types
that set ``\sigma`` and ``\eta`` accordingly.

## Metric corrections for the equations of motion

When the computational grid is not aligned with the Cartesian coordinate system, derivative
operators on the computational grid do not correspond directly to Cartesian derivatives.
Three corrections are needed, described by [Gal-Chen and Somerville (1975)](@cite GalChen1975)
and reviewed in [Durran (2010)](@cite Durran2010):

### Contravariant vertical velocity

Vertical transport through ``\zeta``-surfaces is governed by the **contravariant vertical
velocity** ``\tilde{\Omega}``, not the Cartesian vertical velocity ``w``. The contravariant
velocity is the component of motion normal to the (tilted) coordinate surfaces:

```math
\tilde{\Omega} = w
    - \left(\frac{\partial z}{\partial x}\right)_\zeta u
    - \left(\frac{\partial z}{\partial y}\right)_\zeta v .
```

For the basic terrain-following coordinate, the terrain slopes appearing here are

```math
\left(\frac{\partial z}{\partial x}\right)_\zeta
= \frac{\partial h}{\partial x} \left(1 - \frac{\zeta}{z_{top}}\right) , \qquad
\left(\frac{\partial z}{\partial y}\right)_\zeta
= \frac{\partial h}{\partial y} \left(1 - \frac{\zeta}{z_{top}}\right) .
```

The slopes decay linearly from their surface values to zero at the model top. Similarly,
the contravariant vertical momentum is

```math
\rho \tilde{\Omega} = \rho w
    - \left(\frac{\partial z}{\partial x}\right)_\zeta \rho u
    - \left(\frac{\partial z}{\partial y}\right)_\zeta \rho v .
```

### Horizontal pressure gradient correction

The horizontal pressure gradient at constant ``z`` differs from the gradient at constant
``\zeta``. The correction is

```math
\left(\frac{\partial p}{\partial x}\right)_z
= \left(\frac{\partial p}{\partial x}\right)_\zeta
- \left(\frac{\partial z}{\partial x}\right)_\zeta \frac{\partial p}{\partial \zeta} ,
```

and similarly for ``y``. Oceananigans' finite-difference operators compute
``(\partial p / \partial x)_\zeta`` on the computational grid, so the second term must
be explicitly subtracted.

### Continuity equation

The continuity equation in terrain-following coordinates replaces the Cartesian vertical
momentum ``\rho w`` with the contravariant vertical momentum ``\rho \tilde{\Omega}``:

```math
\partial_t \rho + \frac{\partial (\rho u)}{\partial x}
                + \frac{\partial (\rho v)}{\partial y}
                + \frac{\partial (\rho \tilde{\Omega})}{\partial \zeta} = 0 .
```

The Jacobian of the coordinate transformation enters implicitly through the modified
vertical spacings ``\Delta z = \sigma \, \Delta \zeta`` used by Oceananigans.

### Scalar transport

Advection of density-weighted scalars ``\rho c`` uses the same contravariant velocity
for vertical transport:

```math
\partial_t (\rho c) + \boldsymbol{\nabla}_\zeta \cdot (\rho c \, \tilde{\boldsymbol{U}}) = S_c ,
```

where ``\tilde{\boldsymbol{U}} = (u, v, \tilde{\Omega})``. This is handled automatically
by the transport dispatch mechanism described below.

## Implementation

### Setting up a terrain-following grid

The user-facing entry point is [`follow_terrain!`](@ref), which:

1. Evaluates the topography function ``h(x, y)`` at each grid column.
2. Sets ``\sigma_{i,j} = (z_{top} - h_{i,j}) / z_{top}`` and ``\eta_{i,j} = h_{i,j}``
   on the grid's [`MutableVerticalDiscretization`](@ref).
3. Computes terrain slopes ``\partial h / \partial x`` and ``\partial h / \partial y``
   using finite differences.
4. Returns a [`TerrainMetrics`](@ref) object storing the slopes and model-top height.

```@example terrain
using Breeze
using Oceananigans.Grids: MutableVerticalDiscretization

z_faces = MutableVerticalDiscretization(collect(range(0, 10000, length=41)))
grid = RectilinearGrid(size=(64, 40), x=(-50000, 50000), z=z_faces,
                       topology=(Periodic, Flat, Bounded))

h(x, y) = 500 * exp(-x^2 / 5000^2)
metrics = follow_terrain!(grid, h)
```

### Connecting terrain to dynamics

Pass the [`TerrainMetrics`](@ref) to [`CompressibleDynamics`](@ref) via the
`terrain_metrics` keyword:

```@example terrain
dynamics = CompressibleDynamics(ExplicitTimeStepping(); terrain_metrics=metrics)
model = AtmosphereModel(grid; dynamics)
typeof(model.dynamics.terrain_metrics)
```

When `terrain_metrics` is present, the model automatically:

- Computes ``\tilde{\Omega}`` and ``\rho \tilde{\Omega}`` as auxiliary fields
  during each state update.
- Uses ``\tilde{\Omega}`` in place of ``w`` for vertical transport of momentum,
  scalars, and density (via the `transport_velocities` / `transport_momentum`
  dispatch mechanism).
- Corrects horizontal pressure gradients with the terrain slope terms.

Without `terrain_metrics`, the standard Cartesian physics are used unchanged.

### Transport dispatch

The terrain corrections are injected into the existing tendency machinery through two
overloadable functions:

- `transport_velocities(model)`: Returns `(u, v, w)` for standard models or
  `(u, v, Ω̃)` for terrain-following models.
- `transport_momentum(model)`: Returns `(ρu, ρv, ρw)` or `(ρu, ρv, ρΩ̃)`.

The momentum and scalar tendency kernels use these transport tuples for advective fluxes,
so all vertical transport automatically uses the contravariant velocity when terrain is present.

## API reference

```@docs
follow_terrain!
TerrainMetrics
BasicTerrainFollowing
Breeze.TerrainFollowingDiscretization.terrain_slope_x
Breeze.TerrainFollowingDiscretization.terrain_slope_y
```
