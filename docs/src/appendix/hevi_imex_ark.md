# HEVI IMEX-ARK for compressible dynamics

This appendix derives the horizontally explicit, vertically implicit (HEVI)
splitting used by `IMEXRungeKuttaSSP3332` for `CompressibleDynamics` with
`VerticallyImplicitTimeStepping`.

## Governing equations

The fully compressible Euler equations in conservation form are

```math
\partial_t \rho   + \nabla \cdot (\rho \mathbf{u}) = 0
```
```math
\partial_t (\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \mathbf{u})
  + \nabla p + \rho g \hat{z} = \rho \mathbf{f}
```
```math
\partial_t (\rho\theta) + \nabla \cdot (\rho\theta \mathbf{u}) = S_\theta
```

where ``\rho`` is density, ``\mathbf{u} = (u, v, w)`` is velocity,
``p = p(\rho, \rho\theta)`` is the equation-of-state (EOS) pressure,
``\theta`` is potential temperature, ``g`` is gravitational acceleration,
and ``S_\theta`` collects microphysical, radiative, and diffusive sources.

The prognostic variables are ``(\rho, \rho u, \rho v, \rho w, \rho\theta)``.

### Equation of state

For dry air the pressure is

```math
p = p^{st} \left( \frac{\rho\theta\, R^d}{p^{st}} \right)^{c_p^d / c_v^d}
```

where ``p^{st}`` is the standard pressure, ``R^d`` the dry-air gas constant,
and ``c_p^d, c_v^d`` the heat capacities at constant pressure and volume.

The **acoustic speed squared** is

```math
\mathfrak{C}^2 = \gamma^m R^m T = \frac{c_p^m}{c_v^m} R^m T
```

and the **linearized pressure gradient** satisfies

```math
\frac{\partial p}{\partial z}
  \approx \frac{\mathfrak{C}^2}{\theta} \frac{\partial (\rho\theta)}{\partial z}
```

This linearization is exact in the continuum but introduces an ``O(\Delta z^2)``
error at the discrete level.

## HEVI splitting

The IMEX Additive Runge–Kutta (ARK) framework splits the tendency into
explicit ``f^E`` and implicit ``f^I`` parts, with ``f^E + f^I = f``.

For vertical momentum ``\rho w``:

```math
f^E_{\rho w} = -\nabla \cdot (\rho w \mathbf{u})
             - \mathbf{f} \times (\rho \mathbf{u}) \cdot \hat{z}
             + F_{\rho w}
```

```math
f^I_{\rho w} = -\frac{\partial p}{\partial z} - \rho g
```

The vertical pressure gradient and gravity are removed from ``f^E`` (returned
as zero by `explicit_z_pressure_gradient` and `explicit_buoyancy_forceᶜᶜᶠ`)
and supplied through ``f^I``.

For potential temperature density ``\rho\theta``:

```math
f^E_{\rho\theta} = -\nabla \cdot (\rho\theta \mathbf{u})
                  + S_\theta
                  \underbrace{- f^I_{\rho\theta}}_{\text{subtracted}}
```

```math
f^I_{\rho\theta} = -\frac{1}{V} \delta_z (A_z\, \theta\, \rho w)
```

The implicit ``f^I_{\rho\theta}`` is the vertical ``\rho\theta`` flux (acoustic
compressibility coupling). The explicit tendency includes the full 3D advection,
so ``f^I_{\rho\theta}`` must be **subtracted** from ``f^E_{\rho\theta}`` to avoid
double-counting. This is implemented by `remove_vertical_ρθ_flux_from_explicit!`.

For density ``\rho``:

```math
f^E_\rho = -\nabla_{xy} \cdot (\rho u, \rho v)
\qquad
f^I_\rho = -\frac{1}{V} \delta_z (A_z\, \rho w)
```

The explicit density tendency uses horizontal-only divergence (`div_xyᶜᶜᶜ`),
with the vertical divergence in ``f^I_\rho``.

## Helmholtz solve at each implicit stage

Given the predictor state ``(\rho w^*, \rho\theta^*, \rho^*, p^*)``, the
implicit system couples ``\rho w`` and ``\rho\theta`` through the vertical
acoustic mode:

```math
\rho\theta^+ = \rho\theta^* - \tau\, \theta_0\, \partial_z (\rho w^+)
```
```math
\rho w^+ = \rho w^* - \tau\, \frac{\mathfrak{C}^2}{\theta}\,
           \partial_z (\rho\theta^+) - \tau\, \rho^+ g
```

where ``\tau = \gamma h`` is the SDIRK diagonal element times the time step.

### Perturbation form

Define ``\delta(\rho\theta) = \rho\theta^+ - \rho\theta^*`` and
``\delta(\rho w) = \rho w^+ - \rho w^*``. Using
``\delta\rho \approx \delta(\rho\theta) / \theta_0`` (linearization of
``\rho = \rho\theta / \theta``):

```math
\delta(\rho\theta) = -\tau\, \theta_0\, \partial_z (\rho w^*)
                   - \tau\, \theta_0\, \partial_z (\delta(\rho w))
```

```math
\delta(\rho w) = -\tau\, \frac{\mathfrak{C}^2}{\theta}\,
                 \partial_z (\delta(\rho\theta))
               - \tau\, \frac{g}{\theta_0}\, \delta(\rho\theta)_{\text{face}}
               - \tau\, \left[\frac{\partial p^*}{\partial z} + \rho^* g\right]
```

The three terms in the ``\delta(\rho w)`` equation are:
1. **Perturbation PGF** from the linearized Jacobian
2. **Gravity buoyancy** from ``\delta\rho = \delta(\rho\theta)/\theta_0``
3. **Mean PGF+gravity** at the predictor state

### Schur complement

Substituting the ``\delta(\rho w)`` expression into the
``\delta(\rho\theta)`` equation eliminates ``\delta(\rho w)``:

```math
\left[ I - \tau^2 L - \tau^2 G \right] \delta(\rho\theta) = \text{RHS}
```

where

- ``L = V^{-1} \delta_z (A_z\, \mathfrak{C}^2\, \delta_z(\cdot) / \Delta z^f)``
  is the **acoustic operator** (symmetric second derivative)

- ``G = g\, \partial / \partial z`` is the **gravity operator**
  (skew-symmetric first derivative, from the Schur complement of the
  buoyancy feedback)

The RHS has two terms:

```math
\text{RHS} = \underbrace{-\tau\, \theta_0\, \text{div}_z(\rho w^*)}_{\text{acoustic flux}}
           + \underbrace{\tau^2\, \text{div}_z\!\left(\theta\,
             \left[\frac{\partial (p^* - \bar{p})}{\partial z}
             + g (\rho^* - \bar{\rho})\right]\right)}_{\text{gravity (mean-subtracted)}}
```

## The horizontal-mean reference state

The gravity RHS involves the hydrostatic residual
``R = \partial p^*/\partial z + \rho^* g``. If this residual has a nonzero
horizontal mean, the Helmholtz produces a correction ``\delta(\rho\theta)``
that is constant in ``x``, creating a spurious mean ``\theta`` drift.

### Why constant-in-x residuals cause θ drift

The Helmholtz correction ``\delta(\rho\theta)`` modifies ``\rho\theta`` but
not ``\rho`` (the density is updated separately from ``\rho w``). The
back-solve adds a gravity contribution ``\delta(\rho w)_{\text{grav}}``
that modifies ``\rho w``, which then updates ``\rho`` through the continuity
equation. Since the Helmholtz ``\delta(\rho\theta)`` does not account for
this gravity-driven ``\rho w`` change, the updates to ``\rho\theta`` and
``\rho`` become inconsistent, and ``\theta = \rho\theta / \rho`` drifts.

When the residual ``R`` is constant in ``x``, the inconsistency is
systematic: every column gets the same bias, producing a uniform
``\theta`` offset.

### Solution: subtract the horizontal mean

Define the reference state as the horizontal average of the current fields:

```math
\bar{p}(z) = \langle p^* \rangle_{x,y}, \qquad
\bar{\rho}(z) = \langle \rho^* \rangle_{x,y}
```

The perturbation ``p' = p^* - \bar{p}`` and ``\rho' = \rho^* - \bar{\rho}``
satisfy **``\langle p' \rangle_{x,y} = 0``** and
**``\langle \rho' \rangle_{x,y} = 0``** by construction.

The hydrostatic residual decomposes as

```math
R = \underbrace{\frac{\partial \bar{p}}{\partial z} + \bar{\rho}\, g}_{R_0(z)\text{: constant in }x}
  + \underbrace{\frac{\partial p'}{\partial z} + \rho'\, g}_{R'(x,y,z)\text{: zero horizontal mean}}
```

The gravity RHS in the Helmholtz uses only the perturbation part ``R'``:

```math
\text{gravity RHS} = \tau^2\, \text{div}_z(\theta\, R')
```

This has **zero horizontal mean** because ``\langle R' \rangle_{x,y} = 0``.
The constant-in-``x`` part ``R_0`` is excluded from the Helmholtz entirely.

### Where does ``R_0`` go?

The mean hydrostatic residual ``R_0 = \partial \bar{p}/\partial z + \bar{\rho} g``
is an ``O(\Delta z^2)`` truncation error from the discrete EOS. It represents
the residual of the horizontally-averaged state, which is the same at every
column.

In the SSP3(3,3,2) method, the implicit tendency ``f^I_{\rho\theta}`` enters
the final combination as

```math
\rho\theta_{n+1} = \rho\theta_n + h \sum_j \left[ b^E_j\, f^E_j + b^I_j\, f^I_j \right]
```

Since ``f^E_{\text{adj}} = f^E - f^I_{\rho\theta}`` (after the subtraction),
and ``b^E = b^I = (1/6, 1/6, 2/3)`` for SSP3(3,3,2):

```math
\rho\theta_{n+1} = \rho\theta_n + h \sum_j b^E_j\, f^E_j
                 + h \sum_j \underbrace{(b^I_j - b^E_j)}_{= 0}\, f^I_j
```

The implicit tendency **cancels exactly** in the final combination. The
effect of ``R_0`` on intermediate predictors is ``O(\Delta t^2)``
(the order of the method), compared to the ``O(\Delta t)`` accumulation
that occurred with a column-by-column ``p_h`` approach.

## Discrete stencil

### Tridiagonal coefficients at cell center ``k``

**Acoustic** (symmetric):

```math
Q_{\text{bot}} = \frac{\tau^2 A_z^{k}\, \mathfrak{C}^2_{k}}
                      {\Delta z^f_k\, V_k}, \qquad
Q_{\text{top}} = \frac{\tau^2 A_z^{k+1}\, \mathfrak{C}^2_{k+1}}
                      {\Delta z^f_{k+1}\, V_k}
```

**Gravity** (skew-symmetric):

```math
G = \frac{\tau^2 g}{2 \Delta z^c_k}
```

**Combined**:

```math
a_k = -Q_{\text{lower}} + G, \qquad
b_k = 1 + Q_{\text{bot}} + Q_{\text{top}}, \qquad
c_k = -Q_{\text{top}} - G
```

where ``a_k`` is the sub-diagonal (solver convention: coefficient of
``\delta(\rho\theta)_{k}`` in the equation for row ``k+1``).

### Back-solve at face ``k``

```math
\delta(\rho w)_k = -\frac{\mathfrak{C}^2_k}{\theta_k}
                    \frac{\delta(\rho\theta)_k - \delta(\rho\theta)_{k-1}}{\Delta z^f_k}
                  - \frac{g}{\theta_k}\, \overline{\delta(\rho\theta)}^z_k
                  - \left[\frac{p'_k - p'_{k-1}}{\Delta z^f_k}
                    + g (\rho'_k)^f\right]
```

where ``p' = p - \bar{p}``, ``\rho' = \rho - \bar{\rho}``, and
``\overline{(\cdot)}^z`` denotes vertical interpolation to the face.

### Density and ``\rho\theta`` updates

After the back-solve gives ``\rho w^+``:

```math
\rho^+ = \rho^* - \tau\, V^{-1} \delta_z (A_z\, \rho w^+)
```

The ``\rho\theta`` update uses the Helmholtz result:
``\rho\theta^+ = \rho\theta^* + \delta(\rho\theta)``.

## SSP3(3,3,2) tableau

The method uses the Pareschi–Russo (2005) SSP3(3,3,2) tableau with
constant SDIRK diagonal ``\gamma = 1 - 1/\sqrt{2} \approx 0.293``.

**Explicit** (``a^E``):

| | 1 | 2 | 3 |
|---|---|---|---|
| 1 | 0 | | |
| 2 | 1 | 0 | |
| 3 | 1/4 | 1/4 | 0 |
| ``b^E`` | 1/6 | 1/6 | 2/3 |

**Implicit** (``a^I``):

| | 1 | 2 | 3 |
|---|---|---|---|
| 1 | ``\gamma`` | | |
| 2 | ``1-2\gamma`` | ``\gamma`` | |
| 3 | ``1/2-\gamma`` | 0 | ``\gamma`` |
| ``b^I`` | 1/6 | 1/6 | 2/3 |

Key property: ``b^E = b^I``, so the implicit tendency cancels in the
final combination, and ``|R(i\omega)| \le 1`` for all ``\omega`` (the
method damps oscillatory implicit modes).

The method is **not** stiffly accurate, so a final combination step
is required: ``y_{n+1} = y_n + h \sum_j (b^E_j f^E_j + b^I_j f^I_j)``.

## Summary

The HEVI IMEX-ARK implementation:

1. Zeros vertical PGF+buoyancy from explicit tendencies
2. At each implicit stage, computes horizontal-mean ``\bar{p}``, ``\bar{\rho}``
3. Solves the acoustic+gravity Helmholtz for ``\delta(\rho\theta)`` using
   mean-subtracted perturbation ``p' = p - \bar{p}``
4. Back-solves for ``\delta(\rho w)`` including perturbation PGF, gravity
   buoyancy, and mean-subtracted PGF+gravity
5. Updates ``\rho`` from the vertical divergence of the corrected ``\rho w``
6. Stores ``f^I`` from the solve residual and subtracts ``f^I_{\rho\theta}``
   from ``f^E_{\rho\theta}``
