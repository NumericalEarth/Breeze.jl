---
title: "Compressible WENO Vector-Invariant Momentum Advection on an Orthogonal C-grid"
subtitle: "Implementation design with Hollingsworth-instability risk controls"
author: "Working development note"
date: "2026-07-01"
geometry: margin=1in
fontsize: 11pt
---

# Purpose and scope

This document describes a first implementation target for **fully compressible WENO vector-invariant momentum advection on a purely orthogonal C-grid**. The goal is to implement and test the compressible vector-invariant algorithm before introducing non-orthogonal metric, Hodge-map, cubed-sphere seam, or `SphericalShellGrid` complications.

The document deliberately excludes:

- non-orthogonal metric conversion,
- staggered off-diagonal Hodge maps,
- cubed-sphere seams and vector rotations,
- non-orthogonal `SphericalShellGrid` metric placement,
- hydrostatic free-surface coupling.

The target grid is an orthogonal C-grid such as a rectilinear grid or an orthogonal curvilinear grid. Velocity and mass-flux components are normal to coordinate faces. No off-diagonal metric terms appear. This orthogonal implementation should become the trusted reference before we return to non-orthogonal grids.

The central conservative compressible vector-invariant tendency is

$$
\partial_t (\rho \boldsymbol u)
=
\boldsymbol U \times \boldsymbol Z
-
\rho \nabla K
-
\boldsymbol u\, \nabla\!\cdot \boldsymbol U
+ \text{non-advective forces},
$$

where

$$
\boldsymbol U = \rho \boldsymbol u,
\qquad
K = \frac12 |\boldsymbol u|^2,
\qquad
\boldsymbol Z = \boldsymbol \zeta + \boldsymbol f,
\qquad
\boldsymbol \zeta = \nabla \times \boldsymbol u.
$$

The design principles are:

$$
\boxed{\text{use the same mass flux } \boldsymbol U \text{ in continuity, scalar transport, momentum, and energy diagnostics,}}
$$

$$
\boxed{\text{use the full 3D compressible mass-flux divergence } \nabla\!\cdot\boldsymbol U,}
$$

$$
\boxed{\text{WENO reconstruct relative vorticity } \boldsymbol\zeta \text{ and add smooth planetary vorticity } \boldsymbol f,}
$$

and

$$
\boxed{\text{audit the centered vector-invariant core for discrete kinetic-energy and balance defects before enabling WENO.}}
$$

# Why Hollingsworth-like instability matters

A compressible vector-invariant C-grid scheme can suffer from a Hollingsworth-like instability if the discrete Lamb-vector term, kinetic-energy-gradient term, mass-divergence term, and pressure/energy coupling are not compatible in the discrete kinetic-energy or entropy budget.

This risk is higher for compressible dynamics than for incompressible or shallow-water dynamics because:

1. density varies,
2. acoustic modes are present,
3. the mass matrix or kinetic-energy inner product is density-weighted,
4. the mass-flux divergence is not constrained to vanish,
5. nearly balanced low-Mach states can expose small indefinite energy residuals.

A recent vertical-slice compressible frontogenesis test reports a Hollingsworth-like instability when a vector-invariant form is used for the advective nonlinearity. A recent discrete exterior calculus analysis argues that density-independent mass matrices in compressible vector-invariant discretizations can carry an indefinite energy residual and identifies a density-weighted mass matrix as the structural remedy. These sources motivate treating Hollingsworth stability as a first-class design requirement rather than an after-the-fact tuning issue.

The practical implication is:

$$
\boxed{\text{do not rely on hyperviscosity or divergence damping to hide a bad vector-invariant discretization.}}
$$

Instead, build a centered density-consistent core, audit its energy residual, then add WENO dissipation deliberately.

# Notation

Let

$$
\boldsymbol u = (u, v, w),
\qquad
\boldsymbol U = (U,V,W) = \rho (u,v,w)
$$

be velocity and mass flux density. In a finite-volume implementation, face areas are applied by divergence operators. Thus the mathematical flux density component $U=\rho u$ corresponds in code to a face-located mass or momentum field that is multiplied by face area by the existing divergence operator.

The full mass-flux divergence is

$$
\mathcal D_{\boldsymbol U}
\equiv
\nabla\!\cdot\boldsymbol U
=
\partial_x U + \partial_y V + \partial_z W.
$$

For a fully compressible model,

$$
\partial_t\rho + \mathcal D_{\boldsymbol U}=0,
\qquad
\mathcal D_{\boldsymbol U}\ne 0
\quad\text{in general.}
$$

The relative vorticity components are

$$
\zeta_x = \partial_y w - \partial_z v,
$$

$$
\zeta_y = \partial_z u - \partial_x w,
$$

$$
\zeta_z = \partial_x v - \partial_y u.
$$

The absolute-vorticity vector is

$$
\boldsymbol Z = \boldsymbol\zeta + \boldsymbol f.
$$

For the vector-invariant momentum identity, $\boldsymbol Z$ is the absolute-vorticity vector. It should not be confused with normalized Ertel potential vorticity.

# Orthogonal C-grid locations

Use Oceananigans-style staggered location superscripts.

| Quantity | Location | Meaning |
|---|---:|---|
| $\rho$ | `ccc` | density at cell centers |
| $u$, $U=\rho u$ | `fcc` | x-face velocity / mass flux |
| $v$, $V=\rho v$ | `cfc` | y-face velocity / mass flux |
| $w$, $W=\rho w$ | `ccf` | z-face velocity / mass flux |
| $K$ | `ccc` | kinetic energy, usually center-collocated |
| $\mathcal D_{\boldsymbol U}$ | `ccc` | full mass-flux divergence |
| $\zeta_x, Z_x$ | `cff` | x-vorticity edge location |
| $\zeta_y, Z_y$ | `fcf` | y-vorticity edge location |
| $\zeta_z, Z_z$ | `ffc` | z-vorticity edge location |

On an orthogonal C-grid, no off-diagonal Hodge map is required. Computing velocities from mass-weighted momentum requires only density interpolation to faces. The mass flux components are usually the prognostic momentum fields themselves.

# Continuous derivation

Start from the conservative compressible momentum equation without pressure, gravity, diffusion, or forcing:

$$
\partial_t(\rho \boldsymbol u)
+
\nabla\!\cdot(\boldsymbol U\otimes\boldsymbol u)
+
\rho\boldsymbol f\times\boldsymbol u
=0.
$$

The advective flux divergence satisfies

$$
\nabla\!\cdot(\boldsymbol U\otimes\boldsymbol u)
=
\rho (\boldsymbol u\cdot\nabla)\boldsymbol u
+
\boldsymbol u\,\nabla\!\cdot\boldsymbol U.
$$

Using

$$
(\boldsymbol u\cdot\nabla)\boldsymbol u
=\nabla K - \boldsymbol u\times\boldsymbol\zeta,
$$

we obtain

$$
\nabla\!\cdot(\boldsymbol U\otimes\boldsymbol u)
=
\rho\nabla K
-
\boldsymbol U\times\boldsymbol\zeta
+
\boldsymbol u\,\mathcal D_{\boldsymbol U}.
$$

Including Coriolis in absolute vorticity gives

$$
\boxed{
\partial_t(\rho\boldsymbol u)
=
\boldsymbol U\times\boldsymbol Z
-
\rho\nabla K
-
\boldsymbol u\,\mathcal D_{\boldsymbol U}
+
\text{non-advective forces}.
}
$$

In components,

$$
T_u = VZ_z - WZ_y - \rho\partial_xK - u\mathcal D_{\boldsymbol U},
$$

$$
T_v = WZ_x - UZ_z - \rho\partial_yK - v\mathcal D_{\boldsymbol U},
$$

$$
T_w = UZ_y - VZ_x - \rho\partial_zK - w\mathcal D_{\boldsymbol U}.
$$

These are tendencies for $\rho u$, $\rho v$, and $\rho w$, not for velocity alone.

# Discrete orthogonal C-grid algorithm

The recommended first implementation is an **unsplit fully 3D compressible vector-invariant advection**. The hydrostatic-style split is useful for comparison and for later HFSM work, but the first compressible implementation should keep all three momentum components symmetric.

## Diagnostic fields

Given the prognostic fields

$$
\rho^{ccc},
\qquad
(\rho u)^{fcc},
\qquad
(\rho v)^{cfc},
\qquad
(\rho w)^{ccf},
$$

compute velocities by density interpolation:

$$
u^{fcc} = \frac{(\rho u)^{fcc}}{\rho^{fcc}_{\mathrm{interp}}},
$$

$$
v^{cfc} = \frac{(\rho v)^{cfc}}{\rho^{cfc}_{\mathrm{interp}}},
$$

$$
w^{ccf} = \frac{(\rho w)^{ccf}}{\rho^{ccf}_{\mathrm{interp}}}.
$$

The mass flux components are

$$
U^{fcc}=(\rho u)^{fcc},
\qquad
V^{cfc}=(\rho v)^{cfc},
\qquad
W^{ccf}=(\rho w)^{ccf}.
$$

Compute

$$
\mathcal D_{\boldsymbol U}^{ccc}
=
D_x U^{fcc}+D_y V^{cfc}+D_z W^{ccf}.
$$

Compute kinetic energy at centers, for example

$$
K^{ccc}=\frac12\left[
\mathcal I_{fcc\to ccc}(u^2)
+
\mathcal I_{cfc\to ccc}(v^2)
+
\mathcal I_{ccf\to ccc}(w^2)
\right].
$$

Compute relative vorticity components:

$$
\zeta_x^{cff}=D_y w^{ccf}-D_z v^{cfc},
$$

$$
\zeta_y^{fcf}=D_z u^{fcc}-D_x w^{ccf},
$$

$$
\zeta_z^{ffc}=D_x v^{cfc}-D_y u^{fcc}.
$$

Then form absolute vorticity by adding smooth planetary vorticity components:

$$
Z_i = \zeta_i + f_i.
$$

For WENO, do not reconstruct $Z_i$ directly. Reconstruct $\zeta_i$, then add the smoothly reconstructed $f_i$.

## Momentum tendencies

The u-momentum tendency at `fcc` is

$$
T_u^{fcc}
=
\left(VZ_z\right)^{fcc}
-
\left(WZ_y\right)^{fcc}
-
\rho^{fcc}(D_xK)^{fcc}
-
u^{fcc}\left(\mathcal D_{\boldsymbol U}\right)^{fcc}.
$$

The v-momentum tendency at `cfc` is

$$
T_v^{cfc}
=
\left(WZ_x\right)^{cfc}
-
\left(UZ_z\right)^{cfc}
-
\rho^{cfc}(D_yK)^{cfc}
-
v^{cfc}\left(\mathcal D_{\boldsymbol U}\right)^{cfc}.
$$

The w-momentum tendency at `ccf` is

$$
T_w^{ccf}
=
\left(UZ_y\right)^{ccf}
-
\left(VZ_x\right)^{ccf}
-
\rho^{ccf}(D_zK)^{ccf}
-
w^{ccf}\left(\mathcal D_{\boldsymbol U}\right)^{ccf}.
$$

The interpolation or reconstruction to each tendency location is handled by the centered or WENO sub-schemes.

# Hollingsworth-aware discrete energy design

The central risk is that the discretized form

$$
\boldsymbol U\times\boldsymbol Z
-
\rho\nabla K
-
\boldsymbol u\mathcal D_{\boldsymbol U}
$$

may not be neutral in the discrete kinetic-energy budget. If the residual is of indefinite sign, it can drive grid-scale velocity and pressure noise, especially in low-Mach balanced states.

## Density-weighted kinetic energy

The compressible kinetic energy is

$$
E_K = \int \frac12\rho |\boldsymbol u|^2\,dV.
$$

The discrete kinetic-energy diagnostic should therefore use the same density interpolation and face velocities used in the momentum equation. A representative discrete kinetic-energy diagnostic is

$$
E_{K,h}
=
\sum_{ccc}
\rho^{ccc} K^{ccc}\,\Delta V^{ccc},
$$

with

$$
K^{ccc}=\frac12\left[
\mathcal I_{fcc\to ccc}(u^2)
+
\mathcal I_{cfc\to ccc}(v^2)
+
\mathcal I_{ccf\to ccc}(w^2)
\right].
$$

This does not by itself guarantee stability. But it gives a concrete inner product in which to audit the VI operator.

## Energy-production diagnostic

For inviscid, periodic, no-pressure tests, define the discrete advective kinetic-energy production

$$
P_{\mathrm{VI}}
=
\sum
\boldsymbol u_h\cdot
\left(
\boldsymbol U\times\boldsymbol Z
-
\rho\nabla K
-
\boldsymbol u\mathcal D_{\boldsymbol U}
\right)_h
\Delta V_h.
$$

The subscript $h$ denotes the actual staggered-grid quadrature and interpolation used by the implementation.

For a centered, nondissipative core, $P_{\mathrm{VI}}$ should either be roundoff-small or match the kinetic-energy flux balance implied by conservative flux-form momentum transport. If it is resolution-independent, localized at grid scale, or has indefinite sign around balanced states, the scheme is not ready for WENO.

The pass/fail rule is:

$$
\boxed{\text{centered compressible VI must pass an energy-production audit before WENO is enabled.}}
$$

## Same mass flux everywhere

The mass flux used in continuity must be the mass flux used in momentum, scalar, and energy diagnostics:

$$
\partial_t\rho = -D_iU_i,
$$

$$
\partial_t(\rho u_i)
=
\text{VI}_i(\boldsymbol U,\boldsymbol Z,K,D_jU_j)+\cdots.
$$

Do not recompute a slightly different mass divergence inside momentum, scalar, or energy terms. Store or compute once:

```julia
mass_flux_divergence = div(U, V, W)
```

and reuse it everywhere that represents $\mathcal D_{\boldsymbol U}$.

# WENO reconstruction targets

## Relative-vorticity fluxes

For every rotational flux product, reconstruct relative vorticity with WENO and add planetary vorticity smoothly:

$$
\widehat Z_i
=
\mathcal R_{\mathrm{WENO}}[\zeta_i]
+
\mathcal R_{\mathrm{smooth}}[f_i].
$$

Examples:

$$
(VZ_z)^{fcc}
\approx
V^{fcc}_{\mathrm{adv}}
\left(
\mathcal R^{\operatorname{bias}(V)}_{ffc\to fcc}[\zeta_z]
+
\mathcal R_{ffc\to fcc}^{\mathrm{smooth}}[f_z]
\right),
$$

$$
(WZ_y)^{fcc}
\approx
W^{fcc}_{\mathrm{adv}}
\left(
\mathcal R^{\operatorname{bias}(W)}_{fcf\to fcc}[\zeta_y]
+
\mathcal R_{fcf\to fcc}^{\mathrm{smooth}}[f_y]
\right).
$$

The bias should be determined by the transporting mass flux or transport velocity at the target momentum location.

## Kinetic-energy gradient

The KE-gradient term is

$$
-\rho\nabla K.
$$

For the first implementation, use a centered KE-gradient path and verify the centered compressible VI identity against flux-form momentum advection. Then add WENO KE-gradient reconstruction as a second stage.

The natural WENO targets are directional KE differences, especially self-contributions such as

$$
D_x(u^2/2),
\qquad
D_y(v^2/2),
\qquad
D_z(w^2/2),
$$

with bias determined by the corresponding component $U,V,W$. Cross contributions may be centered initially. A full WENO KE-gradient design must be checked against the conservative flux-form identity and the energy-production diagnostic.

## Full 3D mass-divergence upwinding

The compressible divergence term is

$$
-u_i\mathcal D_{\boldsymbol U},
\qquad
\mathcal D_{\boldsymbol U}=D_xU+D_yV+D_zW.
$$

Unlike hydrostatic-incompressible WENO VI, the divergence to upwind is **not horizontal only**. It is the full 3D mass-flux divergence. For example, for u momentum,

$$
-u^{fcc}\left(\mathcal D_{\boldsymbol U}\right)^{fcc}
$$

should reconstruct or interpolate the full center divergence to `fcc`, with self-upwinding determined by $u$ or $U$. Similar formulas hold for v and w.

This term must be included **exactly once**. Double counting it is a credible route to Hollingsworth-like or purely compressible instability.

# Optional hydrostatic-style split

Although the recommended first compressible implementation is the unsplit 3D form, it is useful to record the algebraic split used in hydrostatic-style WENO VI.

Let

$$
\mathcal D_h = \partial_xU+\partial_yV.
$$

Then for horizontal momentum,

$$
-W\zeta_y
-
\rho\partial_x\left(\frac12 w^2\right)
-
u\mathcal D_{\boldsymbol U}
=
-
\partial_z(Wu)
-
u\mathcal D_h,
$$

and

$$
W\zeta_x
-
\rho\partial_y\left(\frac12 w^2\right)
-
v\mathcal D_{\boldsymbol U}
=
-
\partial_z(Wv)
-
v\mathcal D_h.
$$

Thus the vertical rotational flux, vertical kinetic-energy gradient, and full divergence contribution can be rearranged into a vertical momentum flux divergence plus a horizontal divergence flux.

The split form and the unsplit 3D form are alternatives. They must not be added independently.

# Implementation architecture

## Scheme names

Use verbose type names for user-facing structs.

Suggested first milestone:

```julia
CompressibleVectorInvariant
CompressibleWENOVectorInvariant
```

or, if extending existing Oceananigans `VectorInvariant`, add explicit compressible dispatch while keeping the existing names. The safer development path is to create an experimental scheme family first, then merge abstractions later if the design converges.

## Struct fields

For a stateless orthogonal-grid implementation, follow the current `WENOVectorInvariant` style and store reconstruction choices rather than runtime scratch fields:

```julia
struct CompressibleVectorInvariant{N, FT, VR, PR, DR, KR, MR, U} <: AbstractAdvectionScheme{N, FT}
    vorticity_reconstruction :: VR
    planetary_vorticity_reconstruction :: PR
    divergence_reconstruction :: DR
    kinetic_energy_gradient_reconstruction :: KR
    mass_flux_reconstruction :: MR
    upwinding :: U
end
```

A WENO convenience constructor should build this generic scheme:

```julia
CompressibleWENOVectorInvariant(;
    vorticity_order = 5,
    divergence_order = 5,
    kinetic_energy_gradient_order = 5,
    mass_flux_order = 5,
    planetary_vorticity_reconstruction = Centered(order = 2),
)
```

The important rule is that relative vorticity and planetary vorticity have separate reconstruction paths.

## Internal functions

Use verbose names for public functions and compact mathematical notation only for small internal pointwise helpers.

Suggested helper names:

```julia
compute_compressible_velocity!
compressible_mass_flux_divergence
relative_vorticity_x
relative_vorticity_y
relative_vorticity_z
absolute_vorticity_x
absolute_vorticity_y
absolute_vorticity_z
kinetic_energy
compressible_vector_invariant_u_tendency
compressible_vector_invariant_v_tendency
compressible_vector_invariant_w_tendency
```

For mathematical pointwise kernels, compact names are acceptable if they stay internal:

```julia
zeta_x_cff
zeta_y_fcf
zeta_z_ffc
mass_flux_divergence_ccc
```

# Minimal implementation roadmap

## Stage 0: flux-form baseline

Ensure that compressible flux-form momentum advection works on the same orthogonal C-grid test cases. This is the reference implementation.

Tests:

- density conservation,
- momentum conservation on periodic grids,
- agreement with analytic manufactured flux divergences,
- stable smooth acoustic/vortex tests.

## Stage 1: centered compressible vector-invariant identity

Implement centered versions of all terms:

$$
\boldsymbol U\times\boldsymbol Z,
\qquad
-\rho\nabla K,
\qquad
-\boldsymbol u\mathcal D_{\boldsymbol U}.
$$

Compare pointwise tendencies against flux-form advection for smooth random fields on periodic orthogonal grids. The difference should converge at the expected order.

Add the energy-production diagnostic $P_{\mathrm{VI}}$ in this stage. This stage should be complete before any WENO logic is added.

## Stage 2: WENO divergence/self-upwinding

Add WENO reconstruction for the full 3D mass-flux divergence term.

Acceptance criterion:

- using WENO divergence with centered vorticity and centered KE remains stable,
- divergence-only WENO agrees with centered divergence on smooth fields at the expected order,
- the vertical contribution $D_zW$ is included and tested independently,
- balanced-state tests do not show grid-scale energy growth.

## Stage 3: WENO relative-vorticity reconstruction

Add WENO reconstruction of $\zeta_x,\zeta_y,\zeta_z$ in the rotational fluxes.

Acceptance criterion:

- WENO acts on $\zeta$, not $Z=\zeta+f$,
- small-Rossby tests show dissipation targets relative-vorticity variance rather than smooth planetary vorticity,
- vorticity-only WENO is stable with centered KE and centered/divergence WENO,
- no Hollingsworth-like grid-scale growth appears in balanced rotating tests.

## Stage 4: WENO KE-gradient reconstruction

Add WENO KE-gradient reconstruction after the centered KE path is well tested.

Acceptance criterion:

- KE-only WENO is stable with centered vorticity and centered/divergence WENO,
- smooth manufactured tests show expected order,
- energy diagnostics do not show uncontrolled production,
- balanced low-Mach tests remain quiet.

## Stage 5: full compressible WENO VI

Enable all WENO pieces together:

$$
\text{WENO vorticity}
+
\text{WENO divergence}
+
\text{WENO KE gradient}.
$$

Acceptance criterion:

- stable smooth compressible tests,
- stable forced/decaying turbulence tests,
- conservation of total mass to roundoff,
- no obvious grid-scale noise in velocity, density, or relative-vorticity spectra,
- agreement with flux-form baseline in smooth regimes,
- no Hollingsworth-like growth in the balanced-state suite.

# Must-pass tests

## Continuous/discrete identity test

1. Generate smooth periodic fields $\rho,u,v,w$.
2. Compute flux-form advection:

$$
-\nabla\!\cdot(\boldsymbol U\otimes\boldsymbol u).
$$

3. Compute centered VI advection:

$$
\boldsymbol U\times\boldsymbol\zeta
-
\rho\nabla K
-
\boldsymbol u\mathcal D_{\boldsymbol U}.
$$

4. Verify convergence of the difference under grid refinement.

This is the most important pre-WENO correctness test.

## Full divergence test

Construct fields with

$$
\partial_xU+\partial_yV=0,
\qquad
\partial_zW\ne 0.
$$

The divergence term must not vanish. This catches accidental reuse of horizontal-only divergence logic.

## Hydrostatic split equivalence test

For horizontal momentum, verify that the unsplit vertical terms equal the split terms:

$$
-W\zeta_y
-
\rho\partial_x(w^2/2)
-
u\mathcal D_{\boldsymbol U}
=
-
\partial_z(Wu)
-
u\mathcal D_h,
$$

and similarly for v. This should be a unit test if both unsplit and split paths exist.

## Energy-production and Hollingsworth screen

Use smooth, periodic, inviscid states and compute

$$
P_{\mathrm{VI}}
=
\sum
\boldsymbol u_h\cdot
\left(
\boldsymbol U\times\boldsymbol Z
-
\rho\nabla K
-
\boldsymbol u\mathcal D_{\boldsymbol U}
\right)_h
\Delta V_h.
$$

Run this diagnostic on:

1. random smooth velocity and density fields,
2. low-Mach perturbations,
3. rotating balanced states,
4. stratified hydrostatic states,
5. shear/baroclinic states.

The centered scheme should not produce an indefinite grid-scale residual. If it does, do not enable WENO; fix the centered discretization first.

## Balanced-state perturbation test

Construct a nearly balanced low-Mach rotating state with a tiny perturbation. Run without physical diffusion initially.

Watch:

- velocity grid-scale energy,
- density grid-scale energy,
- pressure/acoustic noise,
- relative-vorticity high-wavenumber spectrum,
- growth localized to C-grid computational branches.

A Hollingsworth-like failure often appears here before it appears in violent turbulence or shock tests.

## Density-weighting A/B test

Compare variants:

1. density-independent velocity mass matrix or kinetic-energy diagnostic,
2. density-weighted kinetic-energy diagnostic and mass-flux-consistent divergence.

The density-weighted/mass-flux-consistent version should have lower energy residuals and better balanced-state behavior.

## Relative-vorticity WENO test

Run a small-Rossby test with $|f|\gg |\zeta|$. Compare:

1. WENO on $Z=f+\zeta$,
2. WENO on $\zeta$ plus centered/smooth reconstruction of $f$.

The accepted implementation should use the second form.

## Decomposition stability tests

Run each WENO component in isolation:

- WENO divergence only,
- WENO relative vorticity only,
- WENO KE-gradient only,
- full WENO.

This prevents a passing full run from hiding an unstable component behind extra dissipation elsewhere.

## Conservation tests

On periodic orthogonal grids:

- total mass must be conserved to roundoff,
- total momentum should be conserved when pressure/forcing/boundaries permit,
- energy should be monitored; centered VI should have no large spurious energy source,
- WENO should dissipate or control small scales rather than inject grid-scale energy.

## Smooth convergence tests

Use manufactured smooth fields and compare against analytic derivatives or high-resolution references. Test all terms separately:

- rotational fluxes,
- kinetic-energy gradients,
- divergence term,
- full tendency.

## Frontogenesis / Hollingsworth stress test

Add a vertical-slice frontogenesis or frontogenesis-like test to the long-running suite. This kind of test has exposed Hollingsworth-like instability in compressible vector-invariant settings and is a good stress test for balanced, rotating, compressible dynamics.

# First recommended model target

The first full model target should be the compressible `Breeze.AtmosphereModel` or an isolated compressible model kernel on a simple orthogonal C-grid. Avoid HFSM/free-surface coupling for this milestone.

A good first model suite is:

1. passive density/velocity manufactured tendency test,
2. smooth acoustic wave,
3. compressible isentropic vortex or smooth vortex-like flow,
4. low-Mach balanced rotating state,
5. vertical-slice frontogenesis-like stress test,
6. forced 3D turbulence on a periodic box,
7. comparison with flux-form compressible advection.

# What success means

The first implementation is successful when:

1. centered compressible VI agrees with flux-form advection on smooth orthogonal-grid tests,
2. centered compressible VI passes an energy-production audit,
3. the same mass flux is used in continuity, scalar transport, momentum, and diagnostics,
4. WENO divergence uses the full 3D mass-flux divergence,
5. WENO relative-vorticity reconstruction is separated from smooth planetary vorticity,
6. WENO vorticity-only, divergence-only, and KE-only decomposition tests pass,
7. full WENO VI is stable in compressible periodic-box tests,
8. balanced low-Mach tests and frontogenesis-like tests do not show Hollingsworth-like growth,
9. the implementation requires no non-orthogonal Hodge or seam machinery.

Only after these conditions are met should the design be generalized to non-orthogonal `SphericalShellGrid` metrics and staggered Hodge maps.

# Implementation warnings

Do not claim the scheme is Hollingsworth-safe just because a turbulent or acoustic test runs. The failure mode can be subtle and state-dependent.

Do not add WENO to an energy-inconsistent centered operator and hope WENO dissipation fixes it.

Do not use horizontal divergence where the compressible derivation requires full 3D mass-flux divergence.

Do not WENO-reconstruct absolute vorticity directly in small-Rossby regimes.

Do not double count the compressible divergence term when using the hydrostatic-style vertical split.

# References and context

- Yamazaki and Cotter, "A vertical slice frontogenesis test case for compressible nonhydrostatic dynamical cores of atmospheric models," arXiv:2501.09752, 2025. Reports a Hollingsworth-like instability in a vector-invariant compressible comparison case.
- Korn, "A no-go theorem and its resolution for the discrete compressible barotropic Navier--Stokes equations," arXiv:2605.16554, 2026. Argues that density-independent mass matrices carry an indefinite energy residual and that a density-weighted construction removes the Hollingsworth residual structurally.
- Existing Oceananigans WENO vector-invariant experience suggests decomposing WENO components independently: divergence/self-upwinding, relative-vorticity reconstruction, and KE-gradient reconstruction should each pass isolation tests before enabling the full scheme.
