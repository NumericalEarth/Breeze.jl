# [Terrain-following coordinates](@id Terrain-following-section)

Terrain-following coordinates map the irregular physical domain above topography
onto a regular computational domain. The horizontal grid stays uniform; only the
vertical coordinate is deformed so that the lowest face follows ``h(x, y)``. This
avoids immersed boundaries and cut cells at the surface and makes the kinematic
boundary condition trivial to express.

The price is that derivatives in the computational frame are no longer Cartesian
derivatives, and the discrete equations have to carry metric correction terms.
Most of this page is about getting those corrections right — and about an
initial-condition gotcha that bites every user the first time they set up a
stratified resting atmosphere over terrain.

The user-facing implementation is [`TerrainFollowingVerticalDiscretization`](@ref)
(TFVD). TFVD stores the reference vertical coordinate and a formulation
(`LinearDecay` or `TwoLevelDecay`) that evaluates the physical height, vertical
Jacobian, and terrain slopes used by the dynamics.

## Continuous formulation

### The coordinate mapping

Let ``\zeta \in [0, z_\text{top}]`` be the computational vertical coordinate and
``z(x, y, \zeta)`` the physical height. We need a smooth invertible mapping
that satisfies

```math
z(x, y, 0) = h(x, y) , \qquad z(x, y, z_\text{top}) = z_\text{top} ,
```

so coordinate surfaces conform to the terrain at the ground and to a flat lid
at the top. All formulations in Breeze take the additive form

```math
\boxed{\, z(x, y, \zeta) = \zeta + h(x, y) \, b(\zeta) \,}
```

with a *basis function* ``b(\zeta)`` satisfying ``b(0) = 1`` and
``b(z_\text{top}) = 0``. Different choices of ``b`` give different formulations
(see [Coordinate formulations](#coordinate-formulations) below).

### Vertical Jacobian and spacings

The vertical Jacobian of the coordinate map is

```math
\sigma(x, y, \zeta) \;:=\; \frac{\partial z}{\partial \zeta}
                = 1 + h(x, y) \, b'(\zeta) .
```

Physical vertical spacings inherit this Jacobian:

```math
\Delta z = \sigma \, \Delta \zeta .
```

For a stable, monotone coordinate we need ``\sigma > 0`` everywhere, which
constrains how steep ``h`` can be relative to the vertical decay scale of ``b``.
A negative ``\sigma`` means coordinate surfaces fold back on themselves —
the dynamics will blow up.

### Generalised horizontal derivatives

A horizontal derivative at constant physical altitude differs from one at
constant ``\zeta``. Apply the chain rule on ``\phi(x, y, z(x, y, \zeta))``:

```math
\left.\frac{\partial \phi}{\partial x}\right|_z
\;=\;
\left.\frac{\partial \phi}{\partial x}\right|_\zeta
\;-\;
\left(\frac{\partial z}{\partial x}\right)_\zeta \frac{\partial \phi}{\partial z} .
```

The terrain slope ``(\partial z / \partial x)_\zeta`` is

```math
\left(\frac{\partial z}{\partial x}\right)_\zeta
= \frac{\partial h}{\partial x} \, b(\zeta) ,
```

which is the surface slope of ``h`` times the basis function. The slope inherits
the vertical decay of ``b`` — at the lid, ``b(z_\text{top}) = 0``, the coordinate
surface is flat and the chain-rule correction vanishes.

Oceananigans' finite-difference operators on a `RectilinearGrid` naturally
compute ``\partial / \partial x|_\zeta``, so to get the physical horizontal
derivative we have to *subtract* the slope term. This subtraction is the source
of most discretisation pain in terrain-following dynamics.

### Contravariant vertical velocity

The velocity normal to a coordinate surface ``\zeta = \text{const}`` is

```math
\boxed{\,
\tilde{w} \;:=\; w
\;-\; \left(\frac{\partial z}{\partial x}\right)_\zeta u
\;-\; \left(\frac{\partial z}{\partial y}\right)_\zeta v
\,}
```

— the **contravariant vertical velocity**. ``\tilde{w}`` is what governs
transport *through* ``\zeta``-surfaces (advection, mass conservation, scalar
transport), while ``w`` is the Cartesian-frame vertical velocity (used for
buoyancy and physical interpretation).

The kinematic boundary condition at the ground

```math
w \big|_{z = h} = u \, \frac{\partial h}{\partial x} + v \, \frac{\partial h}{\partial y}
```

becomes, in contravariant form,

```math
\boxed{\, \tilde{w} \big|_{\zeta = 0} = 0 \,} .
```

This is the killer feature of terrain-following coordinates: the no-flow-through
condition is just ``\tilde{w} = 0`` at the bottom face. Breeze enforces it
explicitly each substep (see [Boundary conditions](#boundary-conditions)).

### Conservation form on the computational grid

The continuity equation in the computational frame is

```math
\partial_t \rho
+ \partial_x|_\zeta(\rho u)
+ \partial_y|_\zeta(\rho v)
+ \partial_\zeta(\rho \tilde{w})
\;=\; 0 ,
```

i.e. the ``\zeta``-frame divergence of the contravariant mass flux ``(\rho u, \rho v,
\rho \tilde{w})``. Scalar transport for any density-weighted quantity ``\rho c``
takes the same form,

```math
\partial_t (\rho c)
+ \boldsymbol{\nabla}_\zeta \cdot \big( \rho c \, (u, v, \tilde{w}) \big)
\;=\; S_c .
```

Vertical advection inside the dycore therefore uses ``\tilde{w}``, not ``w``.

## Coordinate formulations

The choice of basis function ``b(\zeta)`` controls how rapidly the influence of
terrain decays with altitude.

### LinearDecay (Gal-Chen & Somerville, 1975)

The simplest choice is

```math
b(\zeta) = 1 - \frac{\zeta}{z_\text{top}} , \qquad
b'(\zeta) = -\frac{1}{z_\text{top}} ,
```

so

```math
z = \zeta + h \left(1 - \frac{\zeta}{z_\text{top}}\right) ,
\qquad
\sigma = 1 - \frac{h}{z_\text{top}} ,
```

and the slope decays linearly to zero at the lid:

```math
\left(\frac{\partial z}{\partial x}\right)_\zeta = \frac{\partial h}{\partial x}
\left(1 - \frac{\zeta}{z_\text{top}}\right) .
```

LinearDecay is exact, simple, and bombproof. Its drawback is that small-scale
topographic features stay imprinted on coordinate surfaces all the way to the
model top — at high altitude, fast atmospheric flow over even mild terrain has
to traverse rapidly varying coordinate surfaces, generating spurious numerical
mixing.

### TwoLevelDecay (Schär et al., 2002)

The [`TwoLevelDecay`](@ref) coordinate — the "Smooth LEvel VErtical" (SLEVE)
coordinate of Schär et al. (2002) — splits the topography into a large-scale
part ``h_1`` and a small-scale residual ``h_2 = h - h_1``, and uses *different*
decay basis functions for each:

```math
z(x, y, \zeta)
= \zeta
+ h_1(x, y) \, b_1(\zeta)
+ h_2(x, y) \, b_2(\zeta) .
```

The bases are hyperbolic:

```math
b_n(\zeta) = \frac{\sinh\!\big((z_\text{top} - \zeta)/s_n\big)}{\sinh(z_\text{top}/s_n)} ,
\qquad
b_n'(\zeta) = -\frac{\cosh\!\big((z_\text{top} - \zeta)/s_n\big)}{s_n \, \sinh(z_\text{top}/s_n)} ,
```

with *decay scales* ``s_1 \gg s_2``. Large-scale features (mountains) decay
slowly via ``s_1``; small-scale features (sub-mountain undulations) decay fast
via ``s_2``. By the time you reach the upper atmosphere the small-scale terrain
has effectively disappeared from the coordinate surfaces, eliminating its
spurious imprint on the wave field aloft.

In Breeze the `h_1, h_2` split is done automatically: `materialize_terrain!`
evaluates the total ``h``, then smooths it with a low-pass filter to get
``h_1``, taking ``h_2 = h - h_1``. The split, slopes ``\partial_x h_1, \partial_x
h_2``, and basis functions are stored on the discretisation.

Choice of decay scales is problem-dependent. For Schär mountain waves a typical
choice is ``s_1 = z_\text{top}/2`` (large-scale slow decay) and ``s_2 = 2.5 \text{ km}``
(small-scale fast decay) — the small scale is the cos² modulation, the large
scale is the Gaussian envelope.

### Hybrid / Klemp (2011)

Not yet implemented but easy to add as a new `AbstractTerrainFormulation` (see
[Adding a new formulation](#adding-a-new-formulation)). The hybrid coordinate
forces ``b(\zeta) \to 0`` at a finite altitude ``\zeta_\text{flat}`` below the
lid, so coordinate surfaces are *exactly* flat above ``\zeta_\text{flat}``. It
removes all terrain influence in the upper atmosphere.

## The discrete grid

[`TerrainFollowingVerticalDiscretization`](@ref) stores both the reference
``\zeta``-coordinate spacings *and* the formulation-specific terrain fields
(``h``, ``\partial_x h``, etc.) needed to materialise the physical grid.

### Two vertical coordinates: `rnode` vs `znode`

This is the foundational distinction that trips users up:

| function | symbol | meaning |
|----------|--------|---------|
| `rnode(i, j, k, grid, ℓz)` | ``\zeta_k`` | **reference** vertical coordinate — uniform, terrain-independent |
| `znode(i, j, k, grid, ℓx, ℓy, ℓz)` | ``z_{i,j,k}`` | **physical** altitude at cell ``(i, j, k)`` — varies with ``h(x, y)`` |

The relationship is

```math
z_{i, j, k} = \zeta_k + \Delta z^\text{surface}_{i, j, k} , \qquad
\Delta z^\text{surface}_{i, j, k} = h(x_i, y_j) \, b(\zeta_k) .
```

`rnode` is what you store in your spec when you write
`TerrainFollowingVerticalDiscretization(range(0, 30e3, length=Nz+1))`. `znode`
is what the cell physically *is* in the atmosphere. Any field that depends on
altitude — potential temperature ``\theta(z)``, reference pressure ``p_\text{ref}(z)``,
moisture profile ``q_v(z)`` — must be evaluated at `znode`, not `rnode`.

```julia
using Oceananigans.Grids: rnode, znode

# At cell (i, j, k) on a TFVD grid with mountain peak at (0, 0):
rnode(k, grid, Center())                                    # → ζ_k, independent of x,y
znode(i, j, k, grid, Center(), Center(), Center())          # → z_phys at this cell
```

Implementation in `src/TerrainFollowingDiscretization/terrain_following_vertical_discretization.jl`:

```julia
@inline znode(i, j, k, grid::TFVDRG, ℓx, ℓy, ℓz) =
    rnode(i, j, k, grid, ℓx, ℓy, ℓz) +
    terrain_following_Δz_surface(i, j, k, grid, grid.z.formulation, ℓx, ℓy, ℓz)
```

The `terrain_following_Δz_surface` term is `h * b(ζ)`, dispatched on the
formulation.

### Vertical Jacobian σ and spacings

The grid Jacobian appears as `σⁿ(i, j, k, grid, ℓx, ℓy, ℓz)`. For LinearDecay
the formula is

```julia
@inline function terrain_following_σ(i, j, k, grid, f::LinearDecay, ℓx, ℓy, ℓz)
    h = _h(i, j, grid, f.h, ℓx, ℓy)
    return 1 + h * _b′_linear(f.z_top)
end
```

(`_b′_linear(z_top) = -1/z_top`). For TwoLevelDecay it's the same with two contributions
``h_1 b_1'(\zeta) + h_2 b_2'(\zeta)``.

All vertical spacings are derived through the same Jacobian:

```julia
@inline Oceananigans.Operators.Δzᶜᶜᶜ(i, j, k, grid::TFVDRG) =
    Oceananigans.Operators.Δrᶜᶜᶜ(i, j, k, grid) *
    σⁿ(i, j, k, grid, Center(), Center(), Center())
```

— and analogously for every `(ℓx, ℓy, ℓz)` stagger combination.

### Slope arrays

The formulation stores the precomputed horizontal slopes of ``h`` (or of
``h_1, h_2`` for TwoLevelDecay) at the appropriate horizontal stagger:

```julia
# In LinearDecay:
∂x_h :: SX   # (Face,   Center) — slope at u-faces
∂y_h :: SY   # (Center, Face)   — slope at v-faces

# In TwoLevelDecay:
∂x_h₁ :: SX  ; ∂x_h₂ :: SX
∂y_h₁ :: SY  ; ∂y_h₂ :: SY
```

These are filled by a kernel (in `src/TerrainFollowingDiscretization/materialize_terrain.jl`):

```julia
@kernel function _compute_terrain_slopes!(∂x_h, ∂y_h, grid, h_field)
    i, j = @index(Global, NTuple)
    @inbounds ∂x_h[i, j, 1] = δxᶠᶜᶜ(i, j, 1, grid, h_field) * Δx⁻¹ᶠᶜᶜ(i, j, 1, grid)
    @inbounds ∂y_h[i, j, 1] = δyᶜᶠᶜ(i, j, 1, grid, h_field) * Δy⁻¹ᶜᶠᶜ(i, j, 1, grid)
end
```

i.e. a 2-point centred difference of ``h`` to face position. The source uses
`δxᶠᶜᶜ` directly instead of the higher-level `∂x` because the terrain slope is a
purely horizontal surface quantity; it should not include the coordinate
chain-rule correction used for atmospheric fields.

The runtime slope used by the dynamics, ``\partial z / \partial x|_\zeta``, is
the precomputed surface slope times the basis function:

```julia
@inline function terrain_following_∂z∂x(i, j, k, grid, f::LinearDecay, ℓz)
    ζ = rnode(k, grid, ℓz)
    @inbounds return f.∂x_h[i, j, 1] * _b_linear(ζ, f.z_top)
end
```

For TwoLevelDecay both ``\partial_x h_1`` and ``\partial_x h_2`` are blended with their
respective bases:

```julia
@inline function terrain_following_∂z∂x(i, j, k, grid, f::TwoLevelDecay, ℓz)
    ζ = rnode(k, grid, ℓz)
    @inbounds return f.∂x_h₁[i, j, 1] * _b_sleve(ζ, f.z_top, f.large_scale_height) +
                     f.∂x_h₂[i, j, 1] * _b_sleve(ζ, f.z_top, f.small_scale_height)
end
```

### Building a terrain-following grid

```julia
using Oceananigans
using Breeze.TerrainFollowingDiscretization

# 1. Specify reference ζ faces and the formulation
z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, 30e3, length = Nz + 1));
    formulation = TwoLevelDecay(large_scale_height = 15e3,
                        small_scale_height = 2.5e3),
)

# 2. Build the grid as usual
grid = RectilinearGrid(
    size = (Nx, Ny, Nz),
    halo = (5, 5, 5),
    x = (-Lx/2, Lx/2),
    y = (-Ly/2, Ly/2),
    z = z_faces,
    topology = (Periodic, Periodic, Bounded),
)

# 3. Materialise the terrain. This:
#    - evaluates h(x, y) at each column,
#    - smooths into h₁ and computes h₂ = h - h₁ (TwoLevelDecay only),
#    - fills the precomputed slope arrays,
#    - sets z_top on the formulation.
hill(x, y) = 250 * exp(-(x / 5e3)^2) * cos(π * x / 4e3)^2
materialize_terrain!(grid, hill)
```

The pressure-gradient stencil ([`TerrainMetrics`](@ref)) is built automatically by
`CompressibleDynamics` from the grid; the default stencil is
[`SlopeOutsideInterpolation`](@ref). Override with
`CompressibleDynamics(...; slope_stencil = SlopeInsideInterpolation())` when desired.

## Discrete operators

### Generalised ∂x — two stencils

The chain-rule generalisation

```math
\left.\frac{\partial \phi}{\partial x}\right|_z
= \left.\frac{\partial \phi}{\partial x}\right|_\zeta
- \left(\frac{\partial z}{\partial x}\right)_\zeta \frac{\partial \phi}{\partial z}
```

admits two natural discretisations on a staggered grid. Both compute the
result at the same destination stagger — `(Face, Center, Center)` for a `u`-PGF
— but they choose different stencil positions for the slope and ``\partial_z
\phi`` factors.

**[`SlopeOutsideInterpolation`](@ref)** uses the grid's generalised ``\partial_x``
operator:

```julia
@inline function Oceananigans.Operators.∂xᶠᶜᶜ(i, j, k, grid::TFVDRG, ϕ)
    ∂x_at_r = δxᶠᶜᶜ(i, j, k, grid, ϕ) * Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    ∂z_ϕ    = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)
    ∂x_z    = Oceananigans.Operators.∂x_zᶠᶜᶜ(i, j, k, grid)
    return ∂x_at_r - ∂x_z * ∂z_ϕ
end
```

Read: take ``\delta_x \phi`` at the face, interpolate ``\partial_z \phi`` to
that same face position (the 4-point ``\mathcal{I}_{xz}^{F,C}`` average from
``(C, C, F)`` to ``(F, C, C)``), multiply by the grid metric ``\partial_x z`` *at*
the face, subtract.

**[`SlopeInsideInterpolation`](@ref)** computes the slope-weighted vertical
derivative *first*, on the cell-centered/face-centered grid where the slope
lives, then interpolates the product:

```julia
@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeInsideInterpolation, p_ref)
    ∂x_p′ = δxᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref) *
            Δx⁻¹ᶠᶜᶜ(i, j, k, grid)
    correction = ℑzᵃᵃᶜ(i, j, k, grid, ℑxᶠᵃᵃ, slope_x_times_∂z_p′,
                       d.terrain_metrics, d.pressure, p_ref)
    return ∂x_p′ - correction
end
```

where `slope_x_times_∂z_p′(i, j, k, grid, metrics, p, p_ref) = slope_x_ccf * ∂zᶜᶜᶠ(p − p_ref)`.

The two are mathematically equivalent in the continuum limit but differ at
``O(\Delta x^2, \Delta z^2)`` for a non-uniform topographic slope. In our
experience neither version cures the kind of imbalance described in
[Pitfalls](#pitfalls-the-ic-gotcha-that-bites-everyone) below: that pathology
lives upstream of the PGF stencil choice. SlopeOutside is the default and is
fine for production.

### Contravariant momentum

The transformation between ``\rho w`` (Cartesian, stored on prognostic-momentum
fields) and ``\rho \tilde{w}`` (contravariant, used by transport) is computed
each `update_state!`:

```julia
@kernel function _compute_contravariant_velocity!(w̃, ρw̃, grid, momentum, density, metrics)
    i, j, k = @index(Global, NTuple)
    slope_x = terrain_slope_x_ccf(i, j, k, grid, metrics)
    slope_y = terrain_slope_y_ccf(i, j, k, grid, metrics)

    ρu_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, momentum.ρu)
    ρv_ccf = ℑzᵃᵃᶠ(i, j, k, grid, ℑyᵃᶜᵃ, momentum.ρv)
    ρw_ccf = @inbounds momentum.ρw[i, j, k]

    ρw̃_ijk = ρw_ccf - slope_x * ρu_ccf - slope_y * ρv_ccf
    ρ_ccf  = ℑzᵃᵃᶠ(i, j, k, grid, density)
    w̃_ijk  = ρw̃_ijk / ρ_ccf

    @inbounds w̃[i, j, k]  = w̃_ijk
    @inbounds ρw̃[i, j, k] = ρw̃_ijk
end
```

then the bottom face is zeroed:

```julia
launch!(arch, grid, :xy, _zero_bottom_face!, w̃)
launch!(arch, grid, :xy, _zero_bottom_face!, ρw̃)
```

The horizontal stagger of `slope_x_ccf` is `(Center, Center, Face)`, matching
`ρw̃`'s stagger. It is obtained by interpolating the face-staggered ``\partial_z /
\partial_x``:

```julia
@inline terrain_slope_x_ccf(i, j, k, grid::TFVDRG, metrics) =
    ℑxᶜᵃᵃ(i, j, k, grid, ∂z∂x, Face())
```

The `Face()` argument selects the ``\zeta``-face position for `b(ζ)`, so the
slope is sampled at the same vertical position as ``\rho \tilde{w}``.

### Transport dispatch

The terrain corrections are injected into the existing tendency machinery via
two overloadable functions:

  - `transport_velocities(model)` returns `(u, v, w)` for flat-terrain models
    and `(u, v, w̃)` for terrain-following models.
  - `advecting_momentum(model)` returns `(ρu, ρv, ρw)` flat or `(ρu, ρv, ρw̃)`
    terrain.

Momentum and scalar tendency kernels use these tuples for advective fluxes, so
vertical transport automatically uses the contravariant velocity when terrain
metrics are attached.

## The reference state and well-balancing

### Why a reference state at all

The vertical momentum equation,

```math
\partial_t (\rho w) = - \partial_z p - g \rho - \boldsymbol{\nabla}\cdot(\rho w \boldsymbol{u}) ,
```

contains a near-cancellation: in hydrostatic balance ``\partial_z p \approx -g\rho``
and the two terms are each ``\mathcal{O}(\rho g) \approx 12\text{ Pa/m}`` while
the physical signal we care about (a mountain wave, a thermal) is much smaller.
The ``\mathcal{O}(\Delta z^2)`` truncation error from this cancellation can
dominate the answer.

The standard fix is to introduce a hydrostatically balanced reference state
``(p_\text{ref}, \rho_\text{ref})`` and split:

```math
p = p_\text{ref}(x, y, z) + p' , \qquad \rho = \rho_\text{ref}(x, y, z) + \rho' ,
```

with ``\partial_z p_\text{ref} + g \rho_\text{ref} = 0`` *discretely*. The vertical PGF
and buoyancy then operate on the perturbations:

```math
\partial_t (\rho w) \;\supset\; -\partial_z p' - g \rho' ,
```

which is small both in physical magnitude and in cancellation pressure — no
near-cancellation problem.

### Construction over terrain

On a flat grid the reference state is a function of ``z`` only and Breeze
stores it as the 1D `ExnerReferenceState`. Over terrain, each column has its
own ``z(x, y, \zeta)`` profile, so a single 1D reference profile would not be
hydrostatically balanced *per column*. Breeze therefore builds a fully 3D
reference field via per-column discrete Exner integration:

```julia
function compute_terrain_reference_state!(p_ref, ρ_ref, grid, p₀, θᵣ, pˢᵗ, constants)
    Nx, Ny, Nz = size(grid)
    c = Center()
    Rᵈ  = dry_air_gas_constant(constants)
    cᵖᵈ = constants.dry_air.heat_capacity
    κ   = Rᵈ / cᵖᵈ
    g   = constants.gravitational_acceleration

    for j in 1:Ny, i in 1:Nx
        πₖ = zero(κ)  # initialised below at k = 1
        for k in 1:Nz
            z_phys = znode(i, j, k, grid, c, c, c)
            θₖ     = θᵣ isa Number ? θᵣ : θᵣ(z_phys)

            if k == 1
                p_hydro = hydrostatic_pressure(z_phys, p₀, θᵣ, pˢᵗ, constants)
                πₖ      = (p_hydro / pˢᵗ)^κ
            else
                z_below = znode(i, j, k - 1, grid, c, c, c)
                θ_below = θᵣ isa Number ? θᵣ : θᵣ(z_below)
                θ_face  = (θₖ + θ_below) / 2
                Δz      = Δzᶜᶜᶠ(i, j, k, grid)
                πₖ      = πₖ - g * Δz / (cᵖᵈ * θ_face)
            end

            pₖ = pˢᵗ * πₖ^(1 / κ)
            ρₖ = pₖ / (Rᵈ * θₖ * πₖ)
            @inbounds p_ref[i, j, k] = pₖ
            @inbounds ρ_ref[i, j, k] = ρₖ
        end
    end
end
```

Two things to note:

1. Each column starts the discrete Exner integration from the *physical* height
   ``z_\text{phys}`` of the lowest cell — which is over the terrain, not at sea
   level. We evaluate the continuous hydrostatic pressure at that height and
   seed ``\pi`` from it.
2. The march upward is the *discrete* Exner relation,
   ``\pi_k = \pi_{k-1} - g \Delta z / (c_p \theta_\text{face})``, which satisfies the
   discrete hydrostatic balance ``\delta_z p_\text{ref} = -g \, \mathcal{I}_z \rho_\text{ref}``
   to machine precision per column.

Critically, **all ``z``'s in this construction are `znode`**: the *physical*
altitude of each cell, accounting for terrain deformation.

### Perturbation pressure gradient

With the reference state available, the slow horizontal PGF subtracts
``p_\text{ref}`` before taking the generalised derivative:

```julia
@inline function terrain_x_pressure_gradient(i, j, k, grid, d, ::SlopeOutsideInterpolation, p_ref)
    return ∂xᶠᶜᶜ(i, j, k, grid, perturbation_pressure, d.pressure, p_ref)
end

@inline perturbation_pressure(i, j, k, grid, p, p_ref) = @inbounds p[i, j, k] - p_ref[i, j, k]
```

i.e. compute ``\partial_x|_z (p - p_\text{ref})``. If the *actual* discrete
``p`` equals ``p_\text{ref}`` cell-by-cell (atmosphere at rest), then ``p - p_\text{ref}
= 0`` and the chain-rule derivative is exactly zero — the discrete state is
**well-balanced**.

The vertical PGF in the slow tendency uses the same trick:

```julia
@kernel function _assemble_slow_vertical_momentum_tendency!(Gˢρw, Gⁿρw, pᴸ, ρᴸ, pᵣ, ρᵣ, grid, g)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        ∂z_p′  = ∂zᶜᶜᶠ(i, j, k, grid, p_perturbation, pᴸ, pᵣ)
        ρ′ᶜᶜᶠ = ℑzᵃᵃᶠ(i, j, k, grid, ρ_perturbation, ρᴸ, ρᵣ)
        Gˢρw[i, j, k] = (Gⁿρw[i, j, k] - ∂z_p′ - g * ρ′ᶜᶜᶠ) * (k > 1)
    end
end
```

For ``p = p_\text{ref}, \rho = \rho_\text{ref}`` the right-hand side is identically
zero (modulo any horizontal-momentum-driven ``G^n_{\rho w}``), so an exactly
balanced state remains at rest at machine precision — *provided* the IC actually
satisfies ``p_\text{IC} = p_\text{ref}``. The next section is about how to
ensure that.

## Boundary conditions

### Bottom: kinematic via ``\tilde{w} = 0``

The continuous boundary condition ``w = u\,\partial_x h + v\,\partial_y h``
becomes ``\tilde{w} = 0`` at ``\zeta = 0``. Breeze enforces this *exactly* each
time ``\tilde{w}`` is computed:

```julia
launch!(arch, grid, :xy, _zero_bottom_face!, w̃)
launch!(arch, grid, :xy, _zero_bottom_face!, ρw̃)
```

(The bottom face is at ``k = 1`` in Oceananigans' Center-Face convention.) This
means the prognostic state at the surface is the *Cartesian* ``\rho w``,
determined diagnostically from the contravariant constraint:

```math
\rho w \big|_\text{surface}
= \left(\frac{\partial h}{\partial x}\right) \rho u
+ \left(\frac{\partial h}{\partial y}\right) \rho v ,
```

while what propagates through the substep loop is ``\rho \tilde{w}``, anchored
at zero on the ground.

For initialisation, the recommended pattern is the same: zero
``\rho \tilde{w}`` at the surface and set ``\rho w`` from the kinematic
constraint. The validation script provides a one-shot kernel:

```julia
@kernel function _init_terrain_bottom_face_w!(ρw, w, ρ, ρu, grid)
    i, j = @index(Global, NTuple)
    k = 1
    slope_x = ℑxᶜᵃᵃ(i, j, k, grid, ∂z∂x, Oceananigans.Face())
    ρu_ccf  = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu)
    @inbounds begin
        ρw_target = slope_x * ρu_ccf
        ρ_ccf     = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρw[i, j, k] = ρw_target
        w[i, j, k]  = ρw_target / ρ_ccf
    end
end
```

Without this initialisation, a uniform background flow over terrain has the
right ``\rho w = 0`` interior but a non-zero ``\rho \tilde{w}`` at the surface
(because the substepper computes ``\rho \tilde{w}`` from ``\rho w - \text{slope}
\cdot \rho u``), so the first substep starts with a wrong (non-zero) boundary value.

### Top: sponge

The lid at ``\zeta = z_\text{top}`` is a closed boundary, but for gravity-wave
problems we want to absorb rather than reflect upgoing energy. Breeze provides
[`UpperSponge`](@ref) (`damp_opt = 3` in WRF-speak) — an implicit Rayleigh
sponge folded into the column tridiag of the acoustic substep. See
[`SplitExplicitTimeDiscretization`](@ref) for the keyword interface.

## Time integration

Terrain dynamics inherit Breeze's
[`SplitExplicitTimeDiscretization`](@ref): a Wicker–Skamarock RK3 outer step
with acoustic substepping. The relevant terrain-specific bits are:

  - All vertical transport inside the substep loop uses ``\tilde{w}``, not ``w``.
  - The slow tendency for ``\rho w`` is the perturbation form
    ``-\partial_z p' - g \rho'``, so a well-balanced reference state suppresses
    the dominant ``\mathcal{O}(\rho g)`` truncation error.
  - The horizontal PGF in the substep is also perturbation-form
    (``\partial_x (p - p_\text{ref})``), so a well-balanced state generates
    zero horizontal acceleration from PGF.
  - The Crank–Nicolson forward weight ``\omega \in [0.5, 1]`` off-centres the
    implicit vertical solve; the default ``\omega = 0.65`` adds modest
    damping over centred CN.

The split-explicit machinery is the same as in flat-terrain models; what
changes is *which* fields go into the kernels (contravariant vs Cartesian
momentum) and *which* references are subtracted (3D ``p_\text{ref}, \rho_\text{ref}``
vs 1D background).

## Pitfalls: altitude-dependent initial conditions

> *This is the most important section on the page. Read it before you write any
> validation script that initialises a stratified resting atmosphere over
> terrain.*

On a TFVD grid, `rnode` and `znode` are different. A profile such as
``\theta(z)``, ``q_v(z)``, or a height-dependent background wind must be
evaluated at the physical altitude `znode`, not at the reference coordinate
``\zeta``. Breeze extends Oceananigans' `node` for TFVD grids so user-facing
`set!` calls receive the physical altitude.

The recommended hydrostatic thermal IC is therefore:

```julia
θ_profile(x, z) = θ₀ * exp(N² * z / g)

set!(model,
     ρ = model.dynamics.terrain_reference_density,
     θ = θ_profile,
     u = U,
     v = 0,
     w = 0,
     enforce_mass_conservation = false)
```

This keeps ``\rho``, ``\rho\theta``, and the terrain reference pressure
consistent with the same physical-height profile. After `update_state!`, a rest
state should satisfy

```julia
using Oceananigans.Fields: interior

p     = interior(model.dynamics.pressure)
p_ref = interior(model.dynamics.terrain_reference_pressure)

isapprox(p, p_ref; atol = 1e-9)   # → true for a well-balanced IC
```

`isapprox` (with a small absolute tolerance, since the perturbation should be
machine zero) is the idiomatic way to test the balance — equality `==` is too
strict for floating-point round-off, and a hand-rolled
`maximum(abs, p .- p_ref) < tol` requires you to pick `tol` and read out a
scalar. If this check returns `false` for a nominally resting hydrostatic IC,
the first thing to suspect is whether some altitude-dependent quantity
bypassed `set!` or was evaluated from `rnode`.

### The failure mode this avoids

If ``\rho = \rho_\text{ref}`` but ``\theta`` is evaluated at ``\zeta`` instead
of ``z_\text{phys}``, then ``\rho\theta`` no longer matches the terrain
reference state. Over the summit of a 250 m hill, ``\zeta_1`` and
``z_\text{phys}`` can differ by hundreds of meters; for an
``N = 0.01 \text{ s}^{-1}`` stratification this is an ``O(1 \text{ K})``
potential-temperature error in the lowest cell.

After `update_state!` applies the local equation of state, the IC pressure no
longer equals ``p_\text{ref}``. The slow horizontal pressure-gradient force sees
``\partial_x(p - p_\text{ref}) \neq 0`` and drives a surface-bound velocity
column before the intended mountain wave has had time to develop.

The same diagnostic should expose this immediately:

```julia
p     = interior(model.dynamics.pressure)
p_ref = interior(model.dynamics.terrain_reference_pressure)

isapprox(p, p_ref; atol = 1e-9)   # → false when the IC is out of balance
```

Both the pressure mismatch and the corresponding resting-state PGF should be
machine-zero for a balanced IC; `isapprox(...; atol=…)` flips to `false` once
the imbalance exceeds round-off.

### Quantitative impact in a real validation

Schär mountain wave, ``N = 0.01 \text{ s}^{-1}``, ``U = 10 \text{ m/s}``,
``h_0 = 250 \text{ m}``, TwoLevelDecay 400×200, ``t = 600 \text{ s}``:

| quantity | broken IC | fixed IC | CM1 reference |
|----------|-----------|----------|---------------|
| ``\max\|w\|`` (m/s) | 1.45 | 1.54 | 2.03 |
| 99th-pct ``\|w\|`` (m/s) | **0.26** | **0.07** | 0.10 |
| summit-zone energy fraction | 6.6 % | 24 % | 33 % |
| spurious ``\partial_x p\|_\text{rest}`` | 0.26 Pa/m | 1.5e-13 Pa/m | — |

The broken IC has both (a) the real Schär surface signal and (b) a broad
spurious imbalance signal that smears across the whole domain. Fixing the IC
removes (b), leaving (a) — which matches CM1's structure.

### Why the same issue doesn't bite flat models

On a flat-terrain grid, `rnode == znode` at every cell, so `set!(field, f(z))`
and `set!(model, f(z))` are equivalent. The issue is specific to
terrain-following coordinates and to quantities whose values depend on physical
altitude.

## Adding a new formulation

To add e.g. the Klemp (2011) hybrid coordinate, define a new struct
`<: AbstractTerrainFormulation` and dispatch the four key methods:

```julia
struct Hybrid{ZT, FT, H, SX, SY} <: AbstractTerrainFormulation
    z_top   :: ZT
    z_flat  :: FT       # height above which b(ζ) = 0
    h       :: H
    ∂x_h    :: SX
    ∂y_h    :: SY
end

@inline _b_hybrid(ζ, z_top, z_flat) =
    ifelse(ζ >= z_flat, zero(ζ), (1 - ζ/z_flat)^6)
@inline _b′_hybrid(ζ, z_top, z_flat) =
    ifelse(ζ >= z_flat, zero(ζ), -6 * (1 - ζ/z_flat)^5 / z_flat)

# σⁿ, Δz_surface, ∂z∂x, ∂z∂y — copy the LinearDecay implementations and
# substitute the new b, b′.
```

Then `materialize_terrain!` followed by `CompressibleDynamics` will dispatch the
standard pipeline (evaluate ``h``, fill ``\partial_x h, \partial_y h``, build the
PGF stencil, attach to dynamics) automatically.

## Worked example: Schär mountain wave

The full validation script is
`validation_output/substepper/terrain_schar_mountain_wave_validation.jl` (kept locally; not part of the repo).
Here's the minimal stand-alone version:

```julia
using Oceananigans, Breeze
using Breeze.TerrainFollowingDiscretization:
    TerrainFollowingVerticalDiscretization, TwoLevelDecay, materialize_terrain!,
    SlopeOutsideInterpolation, ∂z∂x
using Breeze.AtmosphereModels: AtmosphereModel
using Breeze.CompressibleEquations: CompressibleDynamics, SplitExplicitTimeDiscretization
using Oceananigans: Center, Face
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑzᵃᵃᶠ
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

# ---- problem parameters ----
const Lx, Lz = 200e3, 30e3
const Nx, Nz = 400, 200
const U      = 10.0           # background wind, m/s
const N²     = 1e-4           # stratification, s⁻²
const g      = 9.81
const θ₀     = 280.0
const p₀     = 1e5
const h₀     = 250.0
const a      = 5e3
const λ      = 4e3
θ_profile(x, z) = θ₀ * exp(N² * z / g)
hill(x, y)   = h₀ * exp(-(x / a)^2) * cos(π * x / λ)^2

# ---- grid ----
z_faces = TerrainFollowingVerticalDiscretization(
    collect(range(0, Lz, length = Nz + 1));
    formulation = TwoLevelDecay(large_scale_height = Lz / 2,
                        small_scale_height = 2.5e3),
)
grid = RectilinearGrid(
    size = (Nx, Nz),
    halo = (5, 5),
    x = (-Lx/2, Lx/2),
    z = z_faces,
    topology = (Periodic, Flat, Bounded),
)
materialize_terrain!(grid, hill)

# ---- dynamics ----
# `CompressibleDynamics` auto-builds the PGF stencil from the TFVD grid; default
# is `SlopeOutsideInterpolation()`, override with `slope_stencil = ...`.
td = SplitExplicitTimeDiscretization(acoustic_cfl = 0.5)
dyn = CompressibleDynamics(td;
    reference_potential_temperature  = θ_profile,
    surface_pressure                 = p₀,
)
model = AtmosphereModel(grid; dynamics = dyn, advection = WENO(order = 9),
                        timestepper = :AcousticRungeKutta3)

# ---- IC: at-rest plus uniform U ----
@kernel function _init_terrain_bottom_face_w!(ρw, w, ρ, ρu, grid)
    i, j = @index(Global, NTuple)
    k = 1
    slope_x = ℑxᶜᵃᵃ(i, j, k, grid, ∂z∂x, Face())
    ρu_ccf  = ℑzᵃᵃᶠ(i, j, k, grid, ℑxᶜᵃᵃ, ρu)
    @inbounds begin
        ρw_target = slope_x * ρu_ccf
        ρ_ccf     = ℑzᵃᵃᶠ(i, j, k, grid, ρ)
        ρw[i, j, k] = ρw_target
        w[i, j, k]  = ρw_target / ρ_ccf
    end
end

set!(model,
     ρ = model.dynamics.terrain_reference_density,
     θ = θ_profile,
     u = U,
     v = 0,
     w = 0,
     enforce_mass_conservation = false)

launch!(architecture(grid), grid, :xy, _init_terrain_bottom_face_w!,
        model.momentum.ρw, model.velocities.w,
        model.dynamics.density, model.momentum.ρu, grid)

update_state!(model)

# ---- run ----
simulation = Simulation(model; Δt = 2.0, stop_time = 600.0)
run!(simulation)
```

The two patterns to copy verbatim into your own scripts are:

  1. `set!(model, ρ = model.dynamics.terrain_reference_density,
     θ = θ_profile, ...)` — the hydrostatic thermal IC path.
  2. `_init_terrain_bottom_face_w!` — the kinematic BC, applied once at IC so
     ``\rho \tilde{w} = 0`` from the very first substep.

The bottom-face kernel belongs in your script (or in a project-local utility
module), after the `set!` call has filled the density and horizontal momentum.

## API reference

  - [`TerrainFollowingVerticalDiscretization`](@ref) — terrain-following vertical coordinate
  - [`LinearDecay`](@ref), [`TwoLevelDecay`](@ref) — basis formulations
  - [`materialize_terrain!`](@ref) — evaluate ``h``, fill slopes
  - [`TerrainMetrics`](@ref) — PGF stencil object (built automatically by
    `CompressibleDynamics` on TFVD grids; `build_terrain_metrics` is also
    exported as an advanced/manual entry point)
  - [`SlopeOutsideInterpolation`](@ref), [`SlopeInsideInterpolation`](@ref) —
    PGF stencil flavours (pass via `slope_stencil = ...` to `CompressibleDynamics`)
  - `compute_terrain_reference_state!` — discrete hydrostatic reference (internal)

## References

  - [Gal-Chen and Somerville (1975)](@cite GalChen1975) — original
    terrain-following coordinate.
  - [Schär et al. (2002)](@cite Schar2002) — TwoLevelDecay coordinate; the standard
    Schär mountain-wave test case.
  - [Klemp (2011)](@cite Klemp2011) — hybrid terrain-following / height
    coordinate.
  - [Durran (2010)](@cite Durran2010), Chapter 8 — clear textbook treatment of
    terrain-following metric corrections.
